import argparse
import logging
import sys
import time
import os
import torch.nn as nn
import torch
import scipy.signal as signal
import numpy as np
from modules import ComplexConv, LUT1D
from utils import (data_process, freq_shift, optimizer_init, data_load, cal_power, get_performance)
# get current project path
project_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

parser = argparse.ArgumentParser("pimc")
parser.add_argument('--addr', type=str, default= project_path + '/data/')
parser.add_argument('--name', type=str, default='pim_16t_221110_38dBm_fr4_rnd32_1.pth')
parser.add_argument('--lr', type=float, default=1e-2, help='init learning rate')
parser.add_argument('--step', type=int, default=50, help='decay step')
parser.add_argument('--lr_gamma', type=float, default=0.5, help='decay rate')
parser.add_argument('--epochs', type=int, default=1, help='num of training epochs')
parser.add_argument('--seed', type=int, default=2, help='random seed')
parser.add_argument('--pop_size', type=int, default=20)
parser.add_argument('--mut_rate', type=float, default=0.8)
parser.add_argument('--generation', type=int, default=100)
parser.add_argument('--device', type=str, default='cpu')
parser.add_argument('--bs', type=int, default=256, help='batch size')
parser.add_argument('--bn', type=int, default=100, help='batch number')
parser.add_argument('--tr', type=int, default=0.8, help='train ratio')
parser.add_argument('--chnl', type=int, default=16, help='channel number')
parser.add_argument('--save', type=str, default='EXP', help='experiment name')

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

flt = freq_shift(signal.firwin(255, [50 * 1e6], fs=245.76e6),  0, 245.76)

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
        return x5


class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        self.args = args
        self.n = args.shape[0]
        self.blocks = nn.ModuleList([Block(args[i, :]) for i in range(self.n)])

    def forward(self, input, target):
        logits_all = torch.zeros((input.shape[0], input.shape[1], self.n), dtype=torch.cfloat).to(args.device)
        err_all = torch.zeros((input.shape[0], input.shape[1], self.n), dtype=torch.cfloat).to(args.device)
        err_mean = torch.zeros(self.n).to(args.device)
        for i in range(self.n):
            logits = self.blocks[i](input)
            err = target - logits
            err_flt = torch.conv1d(err.T.unsqueeze(0).cfloat(),
                                   torch.flip(torch.from_numpy(flt).unsqueeze(0).unsqueeze(0).
                                              repeat(err.shape[1], 1, 1).to(args.device).cfloat(), [-1]),
                                   padding='same', groups=err.shape[1]).squeeze(0).T
            err_mean[i] = (err_flt.abs() ** 2).mean()

            logits_all[:, :, i] = logits.data
            err_all[:, :, i] = err_flt.data
        loss = err_mean.sum()
        return logits_all, err_all, loss


def train(train_queue, model, optimizer):
    for input, target in train_queue:
        input, target = input.to(args.device), target.to(args.device)
        loss = model(input, target)[-1]
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def infer(valid_queue, model, scale):
    logits_all, err_all = [], []
    for input, target in valid_queue:
        input, target = input.to(args.device), target.to(args.device)
        logits, err, _ = model(input, target)
        logits_all.append(logits)
        err_all.append(err)

    logits_all, err_all = torch.cat(logits_all, dim=0), torch.cat(err_all, dim=0)
    return (logits_all.permute(1, 0, 2).cpu().detach().numpy() * scale,
            err_all.permute(1, 0, 2).cpu().detach().numpy() * scale)


def main(argv):
    tx, rx, nf, txn, rxn, scale = data_process(args)
    model = Model(argv).to(args.device)
    optimizer, scheduler = optimizer_init(model, pop_size, args)
    for epoch in range(args.epochs):
        tra_all_dl, tra_dl, val_dl, L0 = data_load(txn, rxn, args)
        start = time.time()
        train(tra_all_dl, model, optimizer)
        scheduler.step()
        y_val, e_val = infer(val_dl, model, scale)
        end = time.time()

        ape_all = np.zeros((pop_size, 1))
        for i in range(pop_size):
            pim_val, res_val, per_val, ape_val = get_performance(cal_power(rx[:, L0:], args.chnl)[0],
                                                                 cal_power(nf[:, L0:], args.chnl)[0],
                                                                 cal_power(e_val[:, :, i], args.chnl)[0],
                                                                 args.chnl)
            ape_all[i, 0] = ape_val
        k = np.argmax(ape_all[:, 0])
        print('epoch:', epoch, 'cost time:', round(end - start, 2),
              'pim:', pim_val, 'max_ape:', ape_all[k, 0], 'best arg:', argv[k])
    return ape_all


def selection(pop, fit):
    num = pop.shape[0]
    child = np.zeros_like(pop)
    for i in range(child.shape[0]):
        candidates = np.random.choice(num, 3, replace=False)
        winner = np.argmin(fit[candidates, 0])
        child[i, :] = pop[candidates[winner], :]
    return child


def crossover(pop):
    num = pop.shape[0]
    child = np.zeros_like(pop)
    for i in range(child.shape[0]):
        r1, r2 = np.random.choice(num, 2, replace=False)
        for j in range(child.shape[1]):
            if np.random.rand() < 0.5:
                child[i, j] = pop[r1, j]
            else:
                child[i, j] = pop[r2, j]
    return child


def mutate(pop):
    num = pop.shape[0]
    child = np.zeros_like(pop)
    for i in range(child.shape[0]):
        r1 = np.random.choice(num, 1)[0]
        for j in range(child.shape[1]):
            if np.random.rand() < args.mut_rate:
                if j == 0 or j == 2:
                    child[i, j] = np.random.randint(1, 4)
                elif j == 1:
                    child[i, j] = np.random.randint(1, 49)
                elif j == 3:
                    child[i, j] = np.random.randint(21) * 2 + 1
                elif j == 4:
                    child[i, j] = np.random.randint(1, 17)
            else:
                child[i, j] = pop[r1, j]
    return child


if __name__ == "__main__":
    pop_size = args.pop_size
    inchnl = np.random.randint(1, 4, size=(pop_size, 1))
    branch = np.random.randint(1, 49, size=(pop_size, 1))
    outchnl = np.random.randint(1, 4, size=(pop_size, 1))
    kernel = np.random.randint(21, size=(pop_size, 1)) * 2 + 1
    nspline = np.random.randint(1, 17, size=(pop_size, 1))
    argv = np.hstack((inchnl, branch, outchnl, kernel, nspline))

    fit = main(argv)

    for gen in range(args.generation):
        child = selection(argv, fit)
        child = crossover(child)
        child = mutate(child)
        fit_child = main(child)

        pop_pool = np.concatenate([argv, child], axis=0)
        fit_pool = np.concatenate([fit, fit_child], axis=0)
        selected = np.argsort(-fit_pool[:, 0])[:pop_size]
        argv = pop_pool[selected, :]
        fit = fit_pool[selected, :]
        logging.info(f'gen: {gen}, ape: {fit[0, 0]}, argv: {argv[0]}')
