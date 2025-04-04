from __future__ import print_function

import os
import pickle

import torch

from Arguments import get_args
from ControllerNetwork import Controller
from datasets import IRISDataLoaders
from schemes import *


def main(args):
    print("args:", args)
    train_loader, val_loader, test_loader = IRISDataLoaders(args)
    ControllerModel = Controller(args).to('cpu')
    controller_optimizer = torch.optim.Adam(ControllerModel.parameters(), lr=args.Clr, eps=1e-3)
    report = scheme(ControllerModel, train_loader, val_loader, test_loader, controller_optimizer, args)

    with open(os.path.join(args.path, 'IrisResults{}{}'.format(str(args.Clr), str(args.entropy_weight))), 'wb') as file:
        pickle.dump(report, file)
    print("report: ", report)


if __name__ == '__main__':
    args = get_args()
    main(args)
