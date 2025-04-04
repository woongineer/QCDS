import os
import pickle

import torch

from ControllerNetwork import Controller
from datasets import IRISDataLoaders
from schemes import *


if __name__ == '__main__':
    path = 'Results'
    QuantumPATH = 'quantumWeights/QuantumWeightsSaved'
    batch_size = 64
    test_batch_size = 1000
    Cepochs = 7
    Qepochs = 11
    lr = 0.01
    Clr = 0.1
    entropy_weight = 0.01
    seed = 1
    n_qubits = 4
    n_output = 3
    q_depth = 6

    train_loader, val_loader, test_loader = IRISDataLoaders(batch_size, test_batch_size)

    ControllerModel = Controller(n_qubits, q_depth)
    controller_optimizer = torch.optim.Adam(ControllerModel.parameters(), lr=Clr, eps=1e-3)
    report = scheme(ControllerModel, train_loader, val_loader, test_loader, controller_optimizer,
                    Cepochs, Qepochs, lr, QuantumPATH, entropy_weight)

    with open(os.path.join(path, 'IrisResults{}{}'.format(str(Clr), str(entropy_weight))), 'wb') as file:
        pickle.dump(report, file)
    print("report: ", report)
