import argparse
import time
import datetime
import torch_ac
import sys
import wandb
import os
import torch

import src.utils as utils
from src.utils import device
from src.policies import get_acmodel

from src.reward_machines.rm_env import RewardMachineEnv
import src.envs as envs
from src.vlm_models import get_foundation_model

if __name__ == '__main__':
    """
    ======================================================
    ===== LOAD RUN CONFIGURATION =========================
    ======================================================
    """
    from config import CONFIG, SETTINGS

    env_name = SETTINGS['env']
    config = CONFIG[env_name]
    vlm_model_type = SETTINGS['vlm_model_type']
    rm_file = SETTINGS['rm_file']
    rm_discount = config['rl_train']['rm_discount']
    average_transition_length = config['rl_train']['average_transition_length']
    rm_algo = config['rl_train']['rm_algo']
    rm_sync_freq = config['rl_train']['rm_sync_freq']

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1,
                        help="random seed (default: 1)")
    parser.add_argument("--model", default=None,
                        help="name of the model")
    parser.add_argument("--log-interval", type=int, default=1,
                        help="number of updates between two logs (default: 1)")
    parser.add_argument("--save-interval", type=int, default=10,
                        help="number of updates between two saves (default: 10, 0 means no saving)")
    parser.add_argument("--procs", type=int, default=16,
                        help="number of processes (default: 16)")
    parser.add_argument("--frames", type=int, default=15*10**6,
                        help="number of frames of training (default: 1e7)")
    parser.add_argument("--wandb", action="store_true", default=False,
                        help="Log the experiment with weights & biases")

    # OVERRIDE arguments from command line
    parser.add_argument("--rm-file", default=None)
    parser.add_argument("--rm-algo", default=None)

    args = parser.parse_args()

    if args.rm_file:
        rm_file = args.rm_file
    if args.rm_algo:
        rm_algo = args.rm_algo

    rm_task_name = rm_file.split('/')[-1].removesuffix(".txt")
    TASK_CONFIG = config['rl_train'][rm_task_name]

    epochs = TASK_CONFIG['epochs']
    batch_size = TASK_CONFIG['batch_size']
    frames_per_proc = TASK_CONFIG['frames_per_proc']
    discount = TASK_CONFIG['discount']
    lr = TASK_CONFIG['lr']
    entropy_coef = TASK_CONFIG['entropy_coef']
    potential_scale = TASK_CONFIG['potential_scale']
    gae_lambda = TASK_CONFIG['gae_lambda']

    value_loss_coef = 0.5
    optim_eps = 1e-8
    optim_alpha = 0.99
    clip_eps = 0.2
    recurrence = 1

    max_frames = {"sequence": 2600000, "loop": 4100000, "logic": 15000000, "sequence_safety": 10000000, "lift_red_box": 15000000, "pickup_each_box": 15000000, "show_green": 20000000} 
    assert rm_task_name in max_frames

    # Set run dir
    date = datetime.datetime.now().strftime("%y-%m-%d-%H-%M-%S")
    model_dir = utils.get_model_dir(env_name, vlm_model_type, rm_task_name, args.model if args.model else rm_algo, str(args.seed))

    # Load loggers and Tensorboard writer
    txt_logger = utils.get_txt_logger(model_dir)
    csv_file, csv_logger = utils.get_csv_logger(model_dir)

    if not args.wandb:
        os.environ['WANDB_MODE'] = 'disabled'
    wandb.init(project='natural-ltl')
    wandb.run.name = "-".join([env_name, vlm_model_type, rm_task_name, args.model if args.model else rm_algo, str(args.seed)])
    wandb.run.save()
    conf = wandb.config
    conf.update(args)

    # Log command and all script arguments
    txt_logger.info("{}\n".format(" ".join(sys.argv)))
    txt_logger.info("{}\n".format(args))

    # Set seed for all randomness sources
    utils.seed(args.seed)

    # Set device
    txt_logger.info(f"Device: {device}\n")

    # Load training status
    try:
        status = utils.get_status(model_dir)
    except OSError:
        status = {"num_frames": 0, "update": 0}
    txt_logger.info("Training status loaded\n")

    """
    ======================================================
    ===== LOAD REWARD MACHINE ==============
    ======================================================
    """

    # Load observations preprocessor
    preprocess_obss = utils.get_obss_preprocessor(env_name)
    txt_logger.info("Observations preprocessor loaded")


    """
    ======================================================
    ===== LOAD POLICY + VALUE MODELS =====================
    ======================================================
    """

    acmodel = get_acmodel(env_name, rm_file)

    if "model_state" in status:
        acmodel.load_state_dict(status["model_state"])
    acmodel.to(device)
    txt_logger.info("Actor-critic model loaded\n")
    txt_logger.info("{}\n".format(acmodel))

    """
    ======================================================
    ===== LOAD VLM MODELS =====================
    ======================================================
    """

    foundation_model = get_foundation_model(env_name, vlm_model_type, device=device)
    txt_logger.info("Foundation model loaded\n")
    txt_logger.info("{}\n".format(foundation_model))

    """
    ======================================================
    ===== Launch PPO =====================================
    ======================================================
    """
    algo = torch_ac.PPOAlgo(env_name, rm_file, average_transition_length, rm_algo, rm_discount, args.seed, args.procs, acmodel, foundation_model, device, frames_per_proc, discount, lr, potential_scale, gae_lambda,
                            entropy_coef, value_loss_coef,
                            optim_eps, clip_eps, epochs, batch_size, preprocess_obss, rm_sync_freq=rm_sync_freq)

    if "optimizer_state" in status:
        algo.optimizer.load_state_dict(status["optimizer_state"])
    txt_logger.info("Optimizer loaded\n")

    # Train model

    num_frames = status["num_frames"]
    update = status["update"]
    start_time = time.time()

    while num_frames < max_frames[rm_task_name]:
        # Update model parameters
        update_start_time = time.time()
        exps, logs1 = algo.collect_experiences()
        logs2 = algo.update_parameters(exps)
        logs = {**logs1, **logs2}
        update_end_time = time.time()

        num_frames += logs["num_frames"]
        update += 1

        # Print logs

        if update % args.log_interval == 0:
            fps = logs["num_frames"] / (update_end_time - update_start_time)
            duration = int(time.time() - start_time)
            true_return_per_episode = utils.synthesize(logs["true_return_per_episode"])
            agent_return_per_episode = utils.synthesize(logs["agent_return_per_episode"])
            agent_sparse_return_per_episode = utils.synthesize(logs["agent_sparse_return_per_episode"])
            num_frames_per_episode = utils.synthesize(logs["num_frames_per_episode"])

            header = ["update", "frames", "FPS", "duration"]
            data = [update, num_frames, fps, duration]
            header += ["true_return_" + key for key in true_return_per_episode.keys()]
            data += true_return_per_episode.values()
            header += ["agent_return_" + key for key in agent_return_per_episode.keys()]
            data += agent_return_per_episode.values()
            header += ["agent_sparse_return_" + key for key in agent_sparse_return_per_episode.keys()]
            data += agent_sparse_return_per_episode.values()
            header += ["num_frames_" + key for key in num_frames_per_episode.keys()]
            data += num_frames_per_episode.values()
            header += ["entropy", "value", "policy_loss", "value_loss", "grad_norm"]
            data += [logs["entropy"], logs["value"], logs["policy_loss"], logs["value_loss"], logs["grad_norm"]]

            txt_logger.info(
                "U {} | F {:06} | FPS {:04.0f} | D {} | true R:μσmM {:.2f} {:.2f} {:.2f} {:.2f} | agent R:μσmM {:.2f} {:.2f} {:.2f} {:.2f} | agent sparse R:μσmM {:.2f} {:.2f} {:.2f} {:.2f}  | F:μσmM {:.1f} {:.1f} {} {} | H {:.3f} | V {:.3f} | pL {:.3f} | vL {:.3f} | ∇ {:.3f}"
                .format(*data))



            if status["num_frames"] == 0:
                csv_logger.writerow(header)
            csv_logger.writerow(data)
            csv_file.flush()

            for field, value in zip(header, data):
                wandb.log({field: value})

        # Save status

        if args.save_interval > 0 and update % args.save_interval == 0:
            status = {"num_frames": num_frames, "update": update,
                      "model_state": acmodel.state_dict(), "optimizer_state": algo.optimizer.state_dict()}
            if hasattr(preprocess_obss, "vocab"):
                status["vocab"] = preprocess_obss.vocab.vocab
            utils.save_status(status, model_dir)
            txt_logger.info("Status saved")