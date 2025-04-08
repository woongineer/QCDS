import numpy as np
import torch
import torch.distributions as tdist
import torch.nn as nn
import torch.nn.functional as F


class Controller(torch.nn.Module):
    def __init__(self, n_qubits, q_depth):
        torch.nn.Module.__init__(self)
        self.n_qubits = n_qubits
        self.q_depth = q_depth
        self.rotations = ['x', 'y', 'z']
        self.operations = ['H', 'Px', 'Py', 'Pz', 'CNot', 'CSwap', 'Tof', 'CZ']
        self.shared_fc1 = nn.Linear(1, 48)
        self.shared_fc2 = nn.Linear(48, 12)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.BN1 = nn.BatchNorm1d(48)
        self.BN2 = nn.BatchNorm1d(12)
        self.fcAction = nn.ModuleDict()

        for layer in range(self.q_depth):
            for node in range(self.n_qubits):
                self.fcAction[str(layer % 6) + str(node % 4) + '0'] = nn.Linear(12, 2)  # reUploading or Not
                self.fcAction[str(layer % 6) + str(node % 4) + '1'] = nn.Linear(12,
                                                                                len(self.rotations))  # which rotations
                self.fcAction[str(layer % 6) + str(node % 4) + '2'] = nn.Linear(12,
                                                                                len(self.operations))  # which operations

    def forward(self):
        design = np.empty([self.q_depth, self.n_qubits, 3])
        log_prob_list = []
        entropy_list = []
        x = self.shared_fc1(torch.tensor([[1.0]]))
        x = F.leaky_relu(self.BN1(x))
        x = self.dropout1(x)
        x = self.shared_fc2(x)
        x = F.leaky_relu(self.BN2(x))
        x = self.dropout2(x)
        for layer in range(self.q_depth):
            for node in range(self.n_qubits):
                for decision in range(3):
                    logits = self.fcAction[str(layer % 6) + str(node % 4) + str(decision)](x)
                    probs = F.softmax(logits, dim=1)
                    m = tdist.Categorical(probs)
                    action = m.sample()
                    instant_log_prob = m.log_prob(action)
                    instant_entropy = m.entropy()
                    design[layer, node, decision] = action
                    log_prob_list.append(instant_log_prob)
                    entropy_list.append(instant_entropy)
        design = torch.tensor(design)
        log_prob = torch.sum(torch.stack(log_prob_list))
        entropy = torch.sum(torch.stack(entropy_list))
        return self.post_process(design), log_prob, entropy  ##TODO check if it is correct

    def post_process(self, design):
        updated_design = {}
        for l in range(self.q_depth):
            for n in range(self.n_qubits):
                layer = str(l)
                node = str(n)
                if design[l, n, 0] == 0:
                    updated_design[layer + node + '0'] = False
                else:
                    updated_design[layer + node + '0'] = True

                if design[l, n, 1] == 0:
                    updated_design[layer + node + '1'] = 'x'
                elif design[l, n, 1] == 1:
                    updated_design[layer + node + '1'] = 'y'
                else:
                    updated_design[layer + node + '1'] = 'z'

                if design[l, n, 2] == 0:
                    updated_design[layer + node + '2'] = 'H'
                elif design[l, n, 2] == 1:
                    updated_design[layer + node + '2'] = 'Px'
                elif design[l, n, 2] == 2:
                    updated_design[layer + node + '2'] = 'Py'
                elif design[l, n, 2] == 3:
                    updated_design[layer + node + '2'] = 'Pz'
                elif design[l, n, 2] == 4:
                    updated_design[layer + node + '2'] = 'CNot'
                elif design[l, n, 2] == 5:
                    updated_design[layer + node + '2'] = 'CSwap'
                elif design[l, n, 2] == 6:
                    updated_design[layer + node + '2'] = 'Tof'
                else:
                    updated_design[layer + node + '2'] = 'CZ'

        return updated_design
