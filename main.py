import yaml
import os
import torch
from solver import Solver


def main():
    # open yaml file
    with open('./config.yml', 'r') as f:
        opt = yaml.load(f, Loader=yaml.Loader)

    # basic settings
    torch.manual_seed(opt['seed'])
    torch.backends.cudnn.benchmark =True
    dev = torch.device("cuda" if opt['use_cuda'] else "cpu")
    os.environ["CUDA_VISIBLE_DEVICES"] = opt['device_idx']

    solver = Solver(opt, dev)
    solver.fit()


if __name__ == "__main__":
    main()


