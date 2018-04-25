import numpy as np
from ipdb import set_trace

class AlphaZero:
    def __init__(self, max_depth, turns_until_tau0=15, alpha=.8, epsilon=.2, c=1):
        self.turns_until_tau0 = turns_until_tau0
        self.alpha = alpha
        self.epsilon = epsilon
        self.c = c
        self.max_depth = max_depth

        self.curr_node = {
            "children": None,
            "parent": None,
            "N": 0,
            "d": 0
        }

        self.turn = 1

        self.T = 1

    def select(self, starting_indices, decision_list, embeddings, controller, cont_out):
        decision_idx = 0

        while self.curr_node["children"] is not None and self.curr_node["d"] < self.max_depth:
            choice_idx = self.curr_node["max_uct_idx"]
            self.curr_node = self.curr_node["children"][choice_idx]

            d = self.curr_node["d"]
            decision_idx = d % len(decision_list)
            starting_idx = starting_indices[decision_idx]
            emb = embeddings[starting_idx + choice_idx].view(1, 1, -1)
            cont_out = controller(emb)[0].squeeze(0)

        return cont_out, decision_idx

    def select_real(self, stochastic=True):
        visits = np.array([child["N"] for child in self.curr_node["children"]])

        if self.T != 0:
            visits_sum = (1.0 * visits.sum())

            assert visits_sum != 0
            
            # if visits_sum == 0 and self.curr_node["d"] == self.max_depth-1:
            #     idx = 0
            # else:
            visits = visits / visits_sum
            idx = np.random.choice(len(visits), p=visits)
        else:
            idx = np.argmax(visits)

        # if self.turn == self.turns_until_tau0:
        #     self.T = 0

        self.turn += 1

        self.curr_node = self.curr_node["children"][idx]
        self.curr_node["parent"] = None

        return idx, visits

    #so we can mask the policy before we get in, and it'll be correct
    #then we also pass 

    #we should try to calculate as muchas possible outside this class
    #so lets try to only pass what we need
    def expand(self, policy):
        self.curr_node["children"] = []

        for p in policy:
            child = {
                "N": 0,
                "W": 0,
                "Q": 0,
                "U": p,
                "P": p,
                "d": self.curr_node["d"]+1,
                "children": None,
                "parent": self.curr_node
            }

            self.curr_node["children"].extend([child])

        if self.curr_node["parent"] is None:
            self.add_dirichlet_noise()

        return self.curr_node

    def backup(self, value):
        value += 1
        value *= .5
        while self.curr_node["parent"] is not None:
            self.update_node(value)
            self.curr_node = self.curr_node["parent"]
            self.curr_node["max_uct"] = -1
            self.curr_node["max_uct_idx"] = -1
            for i, child in enumerate(self.curr_node["children"]):
                child["U"] = self.c*child["P"] * \
                    (1 + np.log(self.curr_node["N"])/(1 + child["N"]))
                child["UCT"] = child["Q"] + child["U"]
                if child["UCT"] > self.curr_node["max_uct"]:
                    self.curr_node["max_uct"] = child["UCT"]
                    self.curr_node["max_uct_idx"] = i

        #update root visits
        self.curr_node["N"] += 1

    def update_node(self, value):
        self.curr_node["N"] += 1
        self.curr_node["W"] += value
        self.curr_node["Q"] = self.curr_node["W"]/self.curr_node["N"]

    def add_dirichlet_noise(self):
        nu = np.random.dirichlet([self.alpha] * len(self.curr_node["children"]))*self.epsilon

        for i, child in enumerate(self.curr_node["children"]):
            child["P"] = child["P"]*(1-self.epsilon) + nu[i]
