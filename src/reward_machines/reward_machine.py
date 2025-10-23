from src.reward_machines.reward_functions import *
import time
import torch
import itertools

def fast_hash(arr):
    if type(arr) == torch.Tensor:
        return int("".join(map(str, arr.int().tolist())), 2)
    else:
        return int("".join(map(str, arr)), 2)  # Convert binary to integer


class RewardMachine:
    def __init__(self, file, rm_algo):
        # <U,u0,delta_u,delta_r>
        self.U  = []         # list of non-terminal RM states
        self.u0 = None       # initial state
        self.delta_u    = {} # state-transition function
        self.delta_r    = {} # reward-transition function
        self.terminal_u = -1  # All terminal states are sent to the same terminal state with id *-1*

        self.propositions = []
        self.propositions_to_idx = {}
        self.literals = []
        self.literals_to_idx = {}
        self.conjunctions = []
        self.conjunctions_to_idx = {} 

        self._load_reward_machine(file)
        self.known_transitions = {} # Auxiliary variable to speed up computation of the next RM state

        self.rm_algo = rm_algo

    # Public methods -----------------------------------
    def compute_intrastate_potential(self, u1, true_props, values):
        if self.rm_algo == 'none':
            return 0
        elif self.rm_algo == 'u' or u1 == -1:
            return self.potentials[u1]
        else:
            best_transition_potential = 0.

            for dnf_formula, u2 in self.delta_u[u1].items():
                v = self.evaluate_value_dnf(dnf_formula, true_props, values, self.rm_algo)
                transition_potential = v * (self._get_reward(u1, dnf_formula, None, False, False) + self.gamma * self.potentials[u2]) + self.u_max_self_loop_reward[u1] * (1 - v) / (1 - self.gamma)
                best_transition_potential = max(best_transition_potential, transition_potential)

            return best_transition_potential

    def add_reward_shaping(self, gamma, average_transition_length):
        """
        It computes the potential values for shaping the reward function:
            - gamma(float):    this is the gamma from the environment
            - average_transition_length(int): this represents the approximate number of environment steps each transition takes on average
        """
        self.gamma = gamma
        self.compute_max_self_loop_rewards()
        self.potentials = self.value_iteration(self.U, self.delta_u, self.delta_r, self.terminal_u, gamma, average_transition_length)

        # for u in self.potentials:
        #     self.potentials[u] = -self.potentials[u]


    def reset(self):
        # Returns the initial state
        return self.u0

    def _compute_next_state(self, u1, true_props):
        for dnf_formula, u2  in self.delta_u[u1].items():
            if self.evaluate_dnf(dnf_formula, true_props):
                return u2, dnf_formula

        # Change the default behaviour so that the reward machine stays in the same state if none of the transitions match
        return u1, None
        #assert False, f"No transition defined for {u1}, {true_props}"

    def get_next_state(self, u1, true_props):
        true_props_hash = fast_hash(true_props)
        
        if (u1,true_props_hash) not in self.known_transitions:
            u2, dnf = self._compute_next_state(u1, true_props)
            self.known_transitions[(u1,true_props_hash)] = u2, dnf
        return self.known_transitions[(u1,true_props_hash)]

    def step(self, u1, true_props, s_info, add_rs=False, env_done=False):
        """
        Emulates an step on the reward machine from state *u1* when observing *true_props*.
        The rest of the parameters are for computing the reward when working with non-simple RMs: s_info (extra state information to compute the reward).
        """

        # Computing the next state in the RM and checking if the episode is done
        assert u1 != self.terminal_u, "the RM was set to a terminal state!"
        u2, dnf = self.get_next_state(u1, true_props)
        done = (u2 == self.terminal_u)
        # Getting the reward
        if dnf is None:
            rew = 0
        else:
            rew = self._get_reward(u1,dnf,s_info,add_rs, env_done)

        return u2, rew, done


    def get_states(self):
        return self.U

    def get_useful_transitions(self, u1):
        # This is an auxiliary method used by the HRL baseline to prune "useless" options
        return [self.delta_u[u1][u2].split("&") for u2 in self.delta_u[u1] if u1 != u2]


    # Private methods -----------------------------------

    def _get_reward(self,u1,dnf,s_info,add_rs,env_done):
        """
        Returns the reward associated to this transition.
        """
        # Getting reward from the RM
        reward = 0 # NOTE: if the agent falls from the reward machine it receives reward of zero
        assert (u1, dnf) in self.delta_r
        reward += self.delta_r[(u1, dnf)].get_reward(s_info)
        # Adding the reward shaping (if needed)
        rs = 0.0
        if add_rs:
            un = self.terminal_u if env_done else u2 # If the env reached a terminal state, we have to use the potential from the terminal RM state to keep RS optimality guarantees
            rs = self.gamma * self.potentials[un] - self.potentials[u1]
        # Returning final reward
        return reward + rs


    def _load_reward_machine(self, file):
        """
        Example:
            e,g,n # list all of the propositions for indexing
            0      # initial state
            [2]    # terminal state
            (0,0,'!e&!n',ConstantRewardFunction(0))
            (0,1,'e&!g&!n',ConstantRewardFunction(0))
            (0,2,'e&g&!n',ConstantRewardFunction(1))
            (1,1,'!g&!n',ConstantRewardFunction(0))
            (1,2,'g&!n',ConstantRewardFunction(1))
        """
        # Reading the file
        f = open(file)
        lines = [l.rstrip() for l in f]
        f.close()

        # get the indexes
        self.propositions = list(eval(lines[0]))
        self.literals = self.propositions + [f'!{x}' for x in self.propositions]
        for i, x in enumerate(self.propositions):
            self.propositions_to_idx[x] = i
        for i, x in enumerate(self.literals):
            self.literals_to_idx[x] = i

        self.conjunctions = list(eval(lines[1]))
        for i, x in enumerate(self.conjunctions):
            self.conjunctions_to_idx[x] = i + len(self.literals)
        self.conjunctions_to_idx.update(self.literals_to_idx)

        # setting the DFA
        self.u0 = eval(lines[2])
        terminal_states = eval(lines[3])
        # adding transitions
        for e in lines[4:]:
            # Reading the transition
            u1, u2, dnf_formula, reward_function = eval(e)
            # terminal states
            if u1 in terminal_states:
                continue
            if u2 in terminal_states:
                u2  = self.terminal_u
            # Adding machine state
            self._add_state([u1,u2])
            # Adding state-transition to delta_u
            if u1 not in self.delta_u:
                self.delta_u[u1] = {}
            self.delta_u[u1][dnf_formula] = u2
            # Adding reward-transition to delta_r
            self.delta_r[(u1,dnf_formula)] = reward_function
        # Sorting self.U... just because... 
        self.U = sorted(self.U)

    def _add_state(self, u_list):
        for u in u_list:
            if u not in self.U and u != self.terminal_u:
                self.U.append(u)

    def evaluate_dnf(self, formula, true_props):
        """
        Evaluates 'formula' assuming 'true_props' are the only true propositions and the rest are false. 
        e.g. evaluate_dnf("a&b|!c&d","d") returns True 
        """
        # ORs
        if "|" in formula:
            for f in formula.split("|"):
                if self.evaluate_dnf(f,true_props):
                    return True
            return False
        # ANDs
        if "&" in formula:
            for f in formula.split("&"):
                if not self.evaluate_dnf(f,true_props):
                    return False
            return True
        # NOT
        if formula.startswith("!"):
            return not self.evaluate_dnf(formula[1:],true_props)

        # Base cases
        if formula == "True":  return True
        if formula == "False": return False

        idx = self.propositions_to_idx[formula]
        return bool(true_props[idx])

    def evaluate_value_dnf(self, formula, true_props, value_scores, rm_algo):
        # ORs
        if "|" in formula:
            return max([self.evaluate_value_dnf(f, true_props, value_scores, rm_algo) for f in formula.split("|")])
        # ANDs
        if "&" in formula:
            if rm_algo == "conjunctive":
                return value_scores[self.conjunctions_to_idx[formula]]
            else:
                return min([self.evaluate_value_dnf(f, true_props, value_scores, rm_algo) for f in formula.split("&")])

        # literal
        if self.evaluate_dnf(formula, true_props):
            return 1.
        else:
            return value_scores[self.literals_to_idx[formula]]

    def compute_max_self_loop_rewards(self):
        self.u_max_self_loop_reward = {}

        for u in self.U:
            # Check if delta_u[u] is a complete list. If not, that means there's an implicit self-loop with reward 0.
            complete = True
            all_truth_assignments = itertools.product([0, 1], repeat=len(self.propositions))

            for truth_assignment in all_truth_assignments:
                some_dnf_satisfied = False
                for dnf, u2 in self.delta_u[u].items():
                    if self.evaluate_dnf(dnf, truth_assignment):
                        some_dnf_satisfied = True
                        break
                if not some_dnf_satisfied:
                    complete = False
                    break
            if not complete:
                self.u_max_self_loop_reward[u] = 0.
            else:
                self.u_max_self_loop_reward[u] = -1000000000.

            for dnf, u2 in self.delta_u[u].items():
                if u2 == u and self.delta_r[(u,dnf)].get_type() == "constant":
                    r = self.delta_r[(u,dnf)].get_reward(None)
                    self.u_max_self_loop_reward[u] = max(self.u_max_self_loop_reward[u], r)


    def value_iteration(self, U, delta_u, delta_r, terminal_u, gamma, average_transition_length):
        """
        Standard value iteration approach. 
        We use it to compute the potential function for the automated reward shaping
        """
        rs_gamma = gamma**average_transition_length

        V = dict([(u,0) for u in U])
        V[terminal_u] = 0
        V_error = 1
        while V_error > 0.0000001:
            V_error = 0
            for u1 in U:
                q_u2 = []
                for f, u2 in delta_u[u1].items():
                    if delta_r[(u1,f)].get_type() == "constant": 
                        r = delta_r[(u1,f)].get_reward(None)
                    else:
                        r = 0 # If the reward function is not constant, we assume it returns a reward of zero
                    #q_u2.append(r+gamma*V[u2])
                    q_u2.append(rs_gamma*(r+V[u2]) + self.u_max_self_loop_reward[u1] * (1 - rs_gamma)/(1-gamma) )
                v_new = max(q_u2)
                V_error = max([V_error, abs(v_new-V[u1])])
                V[u1] = v_new
        return V

