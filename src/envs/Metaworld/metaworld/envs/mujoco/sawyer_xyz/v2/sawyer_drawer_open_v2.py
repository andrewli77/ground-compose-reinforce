from __future__ import annotations
from typing import Any

import numpy as np
import numpy.typing as npt
from gymnasium.spaces import Box

import mujoco
from metaworld.envs.asset_path_utils import full_v2_path_for
from metaworld.envs.mujoco.sawyer_xyz.sawyer_xyz_env import RenderMode, SawyerXYZEnv
from metaworld.envs.mujoco.utils import reward_utils
from metaworld.types import InitConfigDict

from scipy.spatial.transform import Rotation



class SawyerDrawerOpenEnvV2(SawyerXYZEnv):
    def __init__(
        self,
        render_mode: RenderMode | None = None,
        camera_name: str | None = None,
        camera_id: int | None = None,
        initialize_random_positions = True,
    ) -> None:
        hand_low = (-0.5, 0.20, 0.05)
        hand_high = (0.5, 1.2, 0.5)
        self.num_objects = 5
        self.time_limit = self.max_path_length = 2000
        

        super().__init__(
            hand_low=hand_low,
            hand_high=hand_high,
            render_mode=render_mode,
            camera_name=camera_name,
            camera_id=camera_id,
            frame_skip=5,
        )

        self.init_config: InitConfigDict = {
            "drawer1_init_angle": 0.3,
            "drawer1_init_pos": np.array([-0.2, 0.9, 0.0], dtype=np.float32),
            "drawer2_init_angle": 0.3,
            "drawer2_init_pos": np.array([0.2, 0.9, 0.0], dtype=np.float32),
            "rbox_init_angle": 0.3,
            "rbox_init_pos": np.array([0.3, 0.5, 0.02], dtype=np.float32),
            "gbox_init_angle": 0.3,
            "gbox_init_pos": np.array([-0.3, 0.5, 0.02], dtype=np.float32),
            "bbox_init_angle": 0.3,
            "bbox_init_pos": np.array([0., 0.5, 0.02], dtype=np.float32),
            "hand_init_pos": np.array([0, 0.6, 0.2], dtype=np.float32),
        }
        self.drawer1_init_pos = self.init_config["drawer1_init_pos"]
        self.drawer2_init_pos = self.init_config["drawer2_init_pos"]
        self.rbox_init_pos = self.init_config["rbox_init_pos"]
        self.gbox_init_pos = self.init_config["gbox_init_pos"]
        self.bbox_init_pos = self.init_config["bbox_init_pos"]

        self.hand_init_pos = self.init_config["hand_init_pos"]

        self.maxDist = 0.2
        self.target_reward = 1000 * self.maxDist + 1000 * 2

        self.initialize_random_positions = initialize_random_positions

    @property
    def model_name(self) -> str:
        return full_v2_path_for("sawyer_xyz/sawyer_drawer.xml")

    @SawyerXYZEnv._Decorators.assert_task_is_set
    def evaluate_state(
        self, obs: npt.NDArray[np.float64], action: npt.NDArray[np.float32]
    ) -> tuple[float, dict[str, Any]]:
        labels = self.compute_labels(action, obs)
        return 0, {'labels': labels}

    def _get_pos_objects(self) -> npt.NDArray[Any]:
        drawer1_pos = self.get_body_com("drawer_link1") + np.array([0.0, -0.16, 0.0])
        drawer2_pos = self.get_body_com("drawer_link2") + np.array([0.0, -0.16, 0.0])
        rbox_pos = self.get_body_com("rbox")
        gbox_pos = self.get_body_com("gbox")
        bbox_pos = self.get_body_com("bbox")

        return np.concatenate([drawer1_pos, drawer2_pos, rbox_pos, gbox_pos, bbox_pos])

    def _get_quat_objects(self) -> npt.NDArray[Any]:
        drawer1_quat = self.data.body("drawer_link1").xquat
        drawer2_quat = self.data.body("drawer_link2").xquat

        rbox_xmat = self.data.geom("rboxGeom").xmat.reshape(3, 3)
        rbox_quat = Rotation.from_matrix(rbox_xmat).as_quat()
    
        gbox_xmat = self.data.geom("gboxGeom").xmat.reshape(3, 3)
        gbox_quat = Rotation.from_matrix(gbox_xmat).as_quat()

        bbox_xmat = self.data.geom("bboxGeom").xmat.reshape(3, 3)
        bbox_quat = Rotation.from_matrix(bbox_xmat).as_quat()

        return np.concatenate([drawer1_quat, drawer2_quat, rbox_quat, gbox_quat, bbox_quat])

    def reset_model(self) -> npt.NDArray[np.float64]:
        self._reset_hand()
        self.prev_obs = self._get_curr_obs_combined_no_goal()
        
        if self.initialize_random_positions:
            # Randomize drawer 1 open or closed
            drawer1_state = np.random.randint(2)
            if drawer1_state == 1:
                slide1_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, 'goal_slidey1')
                qpos_slide1 = self.model.jnt_qposadr[slide1_id]
                self.data.qpos[qpos_slide1] = -0.16 + np.random.rand() * 0.03

            # Randomize drawer 2 open or closed
            drawer2_state = np.random.randint(2)
            if drawer2_state == 1:
                slide2_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, 'goal_slidey2')
                qpos_slide2 = self.model.jnt_qposadr[slide2_id]
                self.data.qpos[qpos_slide2] = -0.16 + np.random.rand() * 0.03

            # Randomize rbox location (on ground, in drawer1, or in drawer2)
            rbox_state = np.random.randint(3)
            if rbox_state == 1:
                rbox_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'rbox')
                rbox_joint_id = self.model.body_jntadr[rbox_id]
                qpos_rbox = self.model.jnt_qposadr[rbox_joint_id]
                self.data.qpos[qpos_rbox: qpos_rbox+3] = self.get_body_com("drawer_link1") + drawer1_state * np.array([0., -0.16, 0.]) + 0.01 * np.random.uniform(size=(3,))
            elif rbox_state == 2:
                rbox_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'rbox')
                rbox_joint_id = self.model.body_jntadr[rbox_id]
                qpos_rbox = self.model.jnt_qposadr[rbox_joint_id]
                self.data.qpos[qpos_rbox: qpos_rbox+3] = self.get_body_com("drawer_link2") + drawer2_state * np.array([0., -0.16, 0.]) + 0.01 * np.random.uniform(size=(3,))

            # Randomize gbox location (on ground, in drawer1, or in drawer2)
            gbox_state = np.random.randint(3)
            if gbox_state == 1:
                gbox_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'gbox')
                gbox_joint_id = self.model.body_jntadr[gbox_id]
                qpos_gbox = self.model.jnt_qposadr[gbox_joint_id]
                self.data.qpos[qpos_gbox: qpos_gbox+3] = self.get_body_com("drawer_link1") + drawer1_state * np.array([0., -0.16, 0.]) + 0.01 * np.random.uniform(size=(3,))
            elif gbox_state == 2:
                gbox_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'gbox')
                gbox_joint_id = self.model.body_jntadr[gbox_id]
                qpos_gbox = self.model.jnt_qposadr[gbox_joint_id]
                self.data.qpos[qpos_gbox: qpos_gbox+3] = self.get_body_com("drawer_link2") + drawer2_state * np.array([0., -0.16, 0.]) + 0.01 * np.random.uniform(size=(3,))

            # Randomize bbox location (on ground, in drawer1, or in drawer2)
            bbox_state = np.random.randint(3)
            if bbox_state == 1:
                bbox_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'bbox')
                bbox_joint_id = self.model.body_jntadr[bbox_id]
                qpos_bbox = self.model.jnt_qposadr[bbox_joint_id]
                self.data.qpos[qpos_bbox: qpos_bbox+3] = self.get_body_com("drawer_link1") + drawer1_state * np.array([0., -0.16, 0.]) + 0.01 * np.random.uniform(size=(3,))
            elif bbox_state == 2:
                bbox_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'bbox')
                bbox_joint_id = self.model.body_jntadr[bbox_id]
                qpos_bbox = self.model.jnt_qposadr[bbox_joint_id]
                self.data.qpos[qpos_bbox: qpos_bbox+3] = self.get_body_com("drawer_link2") + drawer2_state * np.array([0., -0.16, 0.]) + 0.01 * np.random.uniform(size=(3,))

        mujoco.mj_forward(self.model, self.data)
        return self._get_obs()

    def compute_labels(
        self, action: npt.NDArray[Any], obs: npt.NDArray[np.float64]
    ) -> tuple[float, float, float, float, float, float]:
        gripper = obs[:3]
        handle1 = obs[4:7]
        handle2 = obs[11:14]
        rbox = obs[18:21]
        gbox = obs[25:28]
        bbox = obs[32:35]

        slide1_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, 'goal_slidey1')
        qpos_slide1 = self.model.jnt_qposadr[slide1_id]
        handle1_jointpos = self.data.qpos[qpos_slide1]

        slide2_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, 'goal_slidey2')
        qpos_slide2 = self.model.jnt_qposadr[slide2_id]
        handle2_jointpos = self.data.qpos[qpos_slide2]

        drawer1_open = handle1_jointpos < -0.12
        drawer2_open = handle2_jointpos < -0.12

        rbox_lifted = (rbox[2] - 0.02 > 0.04) and self.touching_object(self.data.geom("rboxGeom").id)
        gbox_lifted = (gbox[2] - 0.02 > 0.04) and self.touching_object(self.data.geom("gboxGeom").id)
        bbox_lifted = (bbox[2] - 0.02 > 0.04) and self.touching_object(self.data.geom("bboxGeom").id)

        def in_drawer(obj_pos, drawer_pos):
            xyz_in_drawer = -0.07 <= obj_pos[0] - drawer_pos[0] < 0.07 and -0.06 <= obj_pos[1] - drawer_pos[1] <= 0.05 and -0.03 <= obj_pos[2] - 0.07 <= 0.055
            return xyz_in_drawer

        return np.array([
            drawer1_open, drawer2_open, rbox_lifted, gbox_lifted, bbox_lifted, 
            in_drawer(rbox, handle1 + np.array([0.,0.16,0.])),
            in_drawer(gbox, handle1 + np.array([0.,0.16,0.])),
            in_drawer(bbox, handle1 + np.array([0.,0.16,0.])),
            in_drawer(rbox, handle2 + np.array([0.,0.16,0.])),
            in_drawer(gbox, handle2 + np.array([0.,0.16,0.])),
            in_drawer(bbox, handle2 + np.array([0.,0.16,0.])),
        ], dtype=np.uint8)
