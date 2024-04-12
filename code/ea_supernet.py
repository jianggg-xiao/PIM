import argparse
import logging
import sys
import time
import os

import matplotlib.pyplot as plt
import torch.nn as nn
import torch
from torch.nn.parallel import parallel_apply
import scipy.signal as signal
import numpy as np
from modules import ComplexConv, LUT1D
from utils import (data_process, freq_shift, optimizer_init, data_load, cal_power, get_performance)

from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.evaluator import Evaluator
from pymoo.core.problem import Problem
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.repair.rounding import RoundingRepair
from pymoo.operators.sampling.rnd import IntegerRandomSampling
from pymoo.problems.static import StaticProblem
from pymoo.visualization.scatter import Scatter
from pymoo.config import Config

Config.warnings['not_compiled'] = False

# get current project path
project_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

parser = argparse.ArgumentParser("pimc")
parser.add_argument('--addr', type=str, default=project_path + '/data/')
parser.add_argument('--name', type=str, default='pim_16t_221110_38dBm_fr4_rnd32_1.pth')
parser.add_argument('--lr', type=float, default=1e-3, help='init learning rate')
parser.add_argument('--step', type=int, default=50, help='decay step')
parser.add_argument('--lr_gamma', type=float, default=0.5, help='decay rate')
parser.add_argument('--epochs', type=int, default=300, help='num of training epochs')
parser.add_argument('--seed', type=int, default=2, help='random seed')
parser.add_argument('--pop_size', type=int, default=20)
parser.add_argument('--generation', type=int, default=100)
parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--bs', type=int, default=4096, help='batch size')
parser.add_argument('--bn', type=int, default=100, help='batch number')
parser.add_argument('--tr', type=int, default=0.8, help='train ratio')
parser.add_argument('--chnl', type=int, default=16, help='channel number')

args = parser.parse_args()

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(f'log-{time.strftime("%Y%m%d-%H%M%S")}.txt')
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)

np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

logging.info(f"args: {args}")

flt = freq_shift(signal.firwin(255, [50 * 1e6], fs=245.76e6), 0, 245.76)


class Block(nn.Module):
    def __init__(self, arg):
        super(Block, self).__init__()
        self.L1 = ComplexConv(16, 16 * arg[0], arg[3], 2, 16)
        self.L2 = ComplexConv(16 * arg[0], arg[1], 1, 2, 1)
        self.L3 = LUT1D(arg[4], arg[1])
        self.L4 = ComplexConv(arg[1], 16 * arg[2], 1, 2, 1)
        self.L5 = ComplexConv(16 * arg[2], 16, arg[3], 2, 16)

    def forward(self, x):
        x0 = x.cfloat()
        x1 = self.L1(x0)
        x2 = self.L2(x1)
        x3 = self.L3(x2)
        x4 = self.L4(x3)
        x5 = self.L5(x4)
        return x5.unsqueeze(0)


class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        self.args = args
        self.n = args.shape[0]
        self.blocks = nn.ModuleList([Block(args[i, :]) for i in range(self.n)])

    def forward(self, input, target):
        output = torch.cat(parallel_apply(self.blocks, [input for _ in range(self.n)]), dim=0)
        err = target.unsqueeze(0) - output
        err_flt = torch.conv1d(err.permute(0, 2, 1).cfloat(),
                               torch.flip(torch.from_numpy(flt).unsqueeze(0).unsqueeze(0)
                                          .repeat(err.shape[-1], 1, 1).to(args.device).cfloat(), [-1]),
                               padding='same', groups=err.shape[-1])
        loss = (err_flt.abs()**2).mean() * err.shape[0]
        return output, err, loss


def train(train_queue, model, optimizer):
    for input, target in train_queue:
        input, target = input.to(args.device), target.to(args.device)
        loss = model(input, target)[-1]
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def infer(valid_queue, model, scale):
    output_all, err_all = [], []
    for input, target in valid_queue:
        input, target = input.to(args.device), target.to(args.device)
        output, err, _ = model(input, target)
        output_all.append(output)
        err_all.append(err)

    output_all, err_all = torch.cat(output_all, dim=1), torch.cat(err_all, dim=1)
    return (output_all.permute(2, 1, 0).cpu().detach().numpy() * scale,
            err_all.permute(2, 1, 0).cpu().detach().numpy() * scale)


def supernet(pop):
    tx, rx, nf, txn, rxn, scale = data_process(args)
    model = Model(pop).to(args.device)
    param_num = np.zeros((pop.shape[0], 1))
    for i in range(pop.shape[0]):
        param_num[i, 0] = sum(p.numel() for p in model.blocks[i].parameters())

    optimizer, scheduler = optimizer_init(model, pop.shape[0], args)
    for epoch in range(args.epochs):
        tra_all_dl, tra_dl, val_dl, L0 = data_load(txn, rxn, args)
        start = time.time()
        train(tra_all_dl, model, optimizer)
        scheduler.step()
        y_val, e_val = infer(val_dl, model, scale)
        end = time.time()

        ape_all = np.zeros((pop.shape[0], 1))
        for i in range(pop.shape[0]):
            pim_val, res_val, per_val, ape_val = get_performance(cal_power(rx[:, L0:], args.chnl)[0],
                                                                 cal_power(nf[:, L0:], args.chnl)[0],
                                                                 cal_power(e_val[:, :, i], args.chnl)[0],
                                                                 args.chnl)
            ape_all[i, 0] = ape_val
        k = np.argmax(ape_all[:, 0])
        print('epoch:', epoch, 'cost time:', round(end - start, 2),
              'pim:', pim_val, 'max_ape:', ape_all[k, 0], 'best arg:', pop[k], 'param_num:', param_num[k])
    return np.hstack((-ape_all, param_num))


def ea():
    problem = Problem(n_var=5, n_obj=2, vtype=int, xl=np.ones(5), xu=np.array([4, 49, 4, 51, 20]))

    # create the algorithm object
    algorithm = NSGA2(pop_size=args.pop_size,
                      sampling=IntegerRandomSampling(),
                      crossover=SBX(prob=1.0, eta=3.0, vtype=float, repair=RoundingRepair()),
                      mutation=PM(prob=1.0, eta=3.0, vtype=float, repair=RoundingRepair()),
                      eliminate_duplicates=True)

    # create an algorithm object that never terminates
    algorithm.setup(problem)

    # until the algorithm has no terminated
    for n_gen in range(args.generation):
        # ask the algorithm for the next solution to be evaluated
        pop = algorithm.ask()

        # get the design space values of the algorithm
        X = pop.get("X")

        # implement your evluation
        F = supernet(X)

        static = StaticProblem(problem, F=F)
        Evaluator().eval(static, pop)

        # returned the evaluated individuals which have been evaluated or even modified
        algorithm.tell(infills=pop)

        # obtain the result objective from the algorithm
        res = algorithm.result()

        logging.info(
            f'---------------generation:{n_gen}------------------\n'
            f'----------------------res.X:-----------------------\n'
            f'{res.X}\n'
            f'----------------------res.F:-----------------------\n'
            f'{np.array2string(res.F, formatter={"float_kind": lambda x: "%.2f" % x})}\n')

        # calculate a hash to show that all executions end with the same result
        plt.scatter(res.F[:, 0], res.F[:, 1], label=f'gen_{n_gen}')
        plt.grid()
        plt.legend()
        plt.xlabel('-ape')
        plt.ylabel('param_num')
        plt.savefig(f'pareto_front.png')


if __name__ == '__main__':
    ea()
