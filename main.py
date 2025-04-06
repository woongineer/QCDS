import os
import pickle
import warnings

warnings.filterwarnings(
        "ignore",
        message="Setuptools is replacing distutils",
        module="_distutils_hack.*",
        category=UserWarning
    )

from ControllerNetwork import Controller
from data import NQEDataLoaders
from schemes import *


if __name__ == '__main__':
    path = 'Results'
    QuantumPATH = 'quantumWeights/QuantumWeightsSaved'
    batch_size = 25
    test_batch_size = 1000
    Cepochs = 2
    Qepochs = 2
    lr = 0.01
    Clr = 0.1
    entropy_weight = 0.01
    seed = 1
    n_qubits = 4
    q_depth = 6

    train_loader, val_loader, test_loader = NQEDataLoaders(batch_size=batch_size, test_batch_size=test_batch_size,
                                                           dataset="kmnist", reduction_sz=n_qubits)

    ControllerModel = Controller(n_qubits, q_depth)
    controller_optimizer = torch.optim.Adam(ControllerModel.parameters(), lr=Clr, eps=1e-3)
    report = scheme(ControllerModel, train_loader, val_loader, test_loader, controller_optimizer,
                    Cepochs, Qepochs, lr, QuantumPATH, entropy_weight)

    with open(os.path.join(path, f'NQE_{Clr}_{entropy_weight}'), 'wb') as file:
        pickle.dump(report, file)
    print("report: ", report)
