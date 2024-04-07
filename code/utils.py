import math
import os
import shutil

import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import scipy.signal as signal
import matplotlib.pyplot as plt
from collections import deque, defaultdict

class AvgrageMeter(object):

  def __init__(self):
    self.reset()

  def reset(self):
    self.avg = 0
    self.sum = 0
    self.cnt = 0

  def update(self, val, n=1):
    self.sum += val * n
    self.cnt += n
    self.avg = self.sum / self.cnt

def create_exp_dir(path, scripts_to_save=None):
  if not os.path.exists(path):
    os.mkdir(path)
  print('Experiment dir : {}'.format(path))

  if scripts_to_save is not None:
    os.mkdir(os.path.join(path, 'scripts'))
    for script in scripts_to_save:
      dst_file = os.path.join(path, 'scripts', os.path.basename(script))
      shutil.copyfile(script, dst_file)


def __sub_round__(x):
    if x > 0:
        if (((np.floor(x)) % 2) == 0) and (2 * (np.floor(x)) == (2 * x - 1)):
            y = round(x) + 1
        else:
            y = round(x)
    else:
        if ((np.floor(x)) % 2) == 0:
            if 2 * (np.floor(x)) == 2 * x:
                y = round(x) - 1
            else:
                y = round(x)
        else:
            if 2 * (np.floor(x)) == (2 * x - 1):
                y = round(x) - 1
            else:
                y = round(x)
    return y


def round_new(x):
    if isinstance(x, complex):
        y_real = __sub_round__(x.real)
        y_imag = __sub_round__(x.imag)
        y = complex(y_real, y_imag)
    else:
        y = __sub_round__(x)
    return y


def reg_filter_design_function(fs, Nfft, fc1, fc2, stop_band_att, L_ht, win_deta, bw_flt):
    # Magnitude response in stop-band target value
    stop_band_gain = 10 ** (-stop_band_att / 20)
    # Pass-band Magnitude Response (odd number of bins)
    H0 = signal.windows.tukey(1 + 2 * round_new(0.5 * Nfft * fc2 / fs), 1 - fc1 / fc2)
    L_H0 = len(H0)
    D_H0 = math.floor(L_H0 / 2)
    H = np.append(np.append(np.zeros(int(Nfft / 2 - D_H0)), H0), np.zeros(int(Nfft / 2 - D_H0 - 1)))
    H = ((1 - stop_band_gain) * H + stop_band_gain * np.ones(Nfft))
    h = np.fft.ifft(
        np.fft.fftshift(H * np.exp(-1j * (Nfft / 2 - 1) * np.arange(-Nfft / 2, Nfft / 2) * 2 * math.pi / Nfft)))
    n = (Nfft - L_ht + 1) / 2
    win_function = np.kaiser(L_ht, win_deta)
    ht = np.real(h[int(n - 1): int(Nfft - n)]) * win_function
    ht = ht / sum(ht)
    hti = np.array(list(map(round_new, (2 ** (bw_flt - 1) * ht)))) / 2 ** (bw_flt - 1)
    return ht, hti


def sig_SL_corr_anly(sig0, sig1, fs, fc, left_fre, right_fre, title=['TX', 'RX'], ch_bias=0):
    try:
        sig0_num = sig0.shape[0]
        sig0_len = sig0.shape[1]
        sig0_data = sig0 + 0
    except:
        sig0_num = 1
        sig0_len = sig0.shape[0]
        sig0_data = sig0.reshape(1, sig0_len)

    try:
        sig1_num = sig1.shape[0]
        sig1_len = sig1.shape[1]
        sig1_data = sig1 + 0
    except:
        sig1_num = 1
        sig1_len = sig1.shape[0]
        sig1_data = sig1.reshape(1, sig1_len)

    fft_N = 2048
    fft_leftFreq = left_fre
    fft_rightFreq = right_fre
    fft_tranband_BW = 0.6
    fft_passband_BW = fft_rightFreq - fft_leftFreq
    flt_center = (fft_rightFreq + fft_leftFreq) / 2
    fft_atten_dB = 80
    fft_n_flt = 510
    [flt_sig, hti] = reg_filter_design_function(fs, fft_N, fft_passband_BW, fft_passband_BW + 2 * fft_tranband_BW,
                                                fft_atten_dB, fft_n_flt + 1, 6, 15)
    flt_sft = freq_shift(flt_sig, flt_center, fs)
    sig0_imd3_filt = np.zeros(sig0_data.shape, dtype=complex)
    sig1_filt = np.zeros(sig1_data.shape, dtype=complex)

    for ii in range(sig0_num):
        imd3_tmp = sig0_data[ii] * np.abs(sig0_data[ii]) / 2 ** 15
        imd3_sft = freq_shift(imd3_tmp, fc, fs)
        sig0_imd3_filt[ii] = np.convolve(imd3_sft, flt_sft, 'same')

    for ii in range(sig1_num):
        sig1_filt[ii] = np.convolve(sig1_data[ii], flt_sft, 'same')

    dly_mtx = np.zeros((sig0_num, sig1_num), dtype=int)
    corr_par = np.zeros((sig0_num, sig1_num), dtype=float)

    for kk in range(sig0_num):
        for ii in range(sig1_num):
            acor_tmp_abs = np.abs(np.correlate(sig1_filt[ii], sig0_imd3_filt[kk], 'full'))
            I = np.argmax(acor_tmp_abs)
            acor_av = acor_tmp_abs + 0
            acor_av[I] = 0
            if np.real(np.mean(acor_av)) == 0:
                corr_par[kk, ii] = 0
            else:
                corr_par[kk, ii] = 10 * np.log10(acor_tmp_abs[I] / np.mean(acor_av))
            dly = I - sig0_len + 1
            dly_mtx[kk, ii] = dly

    plt.figure()
    plt.pcolor(corr_par, cmap='jet')
    plt.xlabel(title[0])
    plt.ylabel(title[1])
    plt.colorbar(shrink=.83)
    plt.show()
    return dly_mtx, corr_par


def freq_shift(x, f0, fs):
    L = x.shape[0]
    d = math.floor(L / 2)
    if d < L / 2:
        exp_w = np.exp(1j * 2 * math.pi * (f0 / fs) * np.arange(-d, d + 1, 1))
    else:
        exp_w = np.exp(1j * 2 * math.pi * (f0 / fs) * np.arange(-d, d, 1))
    y = x * exp_w
    return y


def psd(s, win=np.kaiser(1024, 10), noverlap=1, nfft=2048, fs=245.76, fc=0):
    l_s = len(s)
    Nsignals = 0
    s_repck = []
    for i in range(l_s):
        if len(s[i].shape) > 1:
            N_s = s[i].shape[0]
            Nsignals = Nsignals + N_s
            for j in range(N_s):
                s_repck.append(s[i][j])
        else:
            Nsignals = Nsignals + 1
            s_repck.append(s[i])
    framesize = len(win)
    win_energy = sum(win ** 2)

    SS = np.zeros([Nsignals, nfft])
    for nn in range(Nsignals):
        L = s_repck[nn].shape[0]
        Nframes = int(np.fix((L - noverlap) / (framesize - noverlap)))
        s_buff_str = np.arange(0, L, framesize - noverlap)
        S1 = np.zeros([1, nfft])
        for kk in range(Nframes):
            s1 = s_repck[nn][s_buff_str[kk]:s_buff_str[kk] + framesize]
            S1 += abs(np.fft.fft(win * s1, nfft)) ** 2
        SS[nn, :] = 10 * np.log10(np.fft.fftshift(S1 * (1 / (Nframes * win_energy))))
    f = fc + (np.arange(nfft) / nfft - 0.5) * fs
    return SS, f


def cal_power(sig, chnl):
    Pe = np.zeros(chnl)
    SS_psd = []
    for i in range(chnl):
        sig_psd, f = psd(sig[[i], :])
        SS_tmp = sig_psd[:, (f > -30) & (f < -25)]
        SS_tmp = np.power(10, SS_tmp / 10)
        Pe[i] = 10 * np.log10(np.mean(SS_tmp[0, :]))
        SS_psd.append(sig_psd[0, :])
    SS_psd = np.array(SS_psd)
    return Pe, SS_psd, f


def get_performance(Pe_rx, Pe_nf, Pe_err, chnl):
    pim, res, per, ape = np.zeros(chnl), np.zeros(chnl), np.zeros(chnl), np.zeros(chnl)
    for i in range(chnl):
        pim[i] = Pe_rx[i] - Pe_nf[i]
        res[i] = Pe_err[i] - Pe_nf[i]
        per[i] = pim[i] - res[i]
        ape[i] = 10 * np.log10(10 ** (pim[i] / 10) - 1) - 10 * np.log10(10 ** (res[i] / 10) - 1)
    return round(np.mean(pim), 2), round(np.mean(res), 2), round(np.mean(per), 2), round(np.mean(ape), 2)


def data_process(args):
    mat = torch.load(args.addr+args.name)
    tx = mat['Tx']
    rx = mat['Rx']
    nf = mat['nf'].detach().numpy()
    txn = tx / np.max(np.abs(tx))
    rxn = rx / np.max(np.abs(rx))
    scale = np.max(np.abs(rx))
    return tx, rx, nf, txn, rxn, scale


def data_load(tx, rx, args):
    x_tra_all, y_tra_all = [], []
    L0 = int(tx.shape[1] * args.tr)
    for i in range(args.bn):
        start = np.random.randint(L0 - args.bs)
        x_tra_all.append(tx[:, start:start + args.bs])
        y_tra_all.append(rx[:, start:start + args.bs])
    x_tra_all = np.hstack(x_tra_all)
    y_tra_all = np.hstack(y_tra_all)
    x_tra_all = torch.from_numpy(x_tra_all.T)
    y_tra_all = torch.from_numpy(y_tra_all.T)

    x_tra = torch.from_numpy(tx[:, :L0].T)
    y_tra = torch.from_numpy(rx[:, :L0].T)
    x_val = torch.from_numpy(tx[:, L0:].T)
    y_val = torch.from_numpy(rx[:, L0:].T)
    tra_all_dl = DataLoader(TensorDataset(x_tra_all, y_tra_all), batch_size=args.bs)
    tra_dl = DataLoader(TensorDataset(x_tra, y_tra), batch_size=args.bs)
    val_dl = DataLoader(TensorDataset(x_val, y_val), batch_size=y_val.shape[0])
    return tra_all_dl, tra_dl, val_dl, L0


def optimizer_init(model, n, args):
    weight = []
    weight.extend([
        {'params': [model.blocks[i].L1.conv.weight for i in range(n)], 'lr': 2 * args.lr},
        {'params': [model.blocks[i].L2.conv.weight for i in range(n)], 'lr': 2 * args.lr},
        {'params': [model.blocks[i].L3.weight for i in range(n)], 'lr': 3 * args.lr},
        {'params': [model.blocks[i].L4.conv.weight for i in range(n)], 'lr': 4 * args.lr},
        {'params': [model.blocks[i].L5.conv.weight for i in range(n)], 'lr': 4 * args.lr},
    ])
    optimizer = torch.optim.AdamW(weight)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step, gamma=args.lr_gamma)
    return optimizer, scheduler


def topo_sort_edges(adj_matrix):
    n = len(adj_matrix)  # 节点数量
    graph = defaultdict(list)  # 图的表示
    reverse_graph = defaultdict(list)  # 逆向图
    in_degree = [0] * n  # 节点的入度

    # 构建图和逆向图，以及计算入度
    for i in range(n):
        for j in range(i + 1, n):  # 只考虑上三角矩阵
            if adj_matrix[i][j] == 1:
                graph[i].append(j)
                reverse_graph[j].append(i)
                in_degree[j] += 1
            elif adj_matrix[i][j] == -1:
                graph[j].append(i)
                reverse_graph[i].append(j)
                in_degree[i] += 1

    # 拓扑排序
    queue = deque([i for i in range(n) if in_degree[i] == 0])
    edge_order = []  # 存储边的排序结果
    while queue:
        node = queue.popleft()
        for neighbor in graph[node]:
            edge_order.append((node, neighbor))  # 记录边的顺序
            in_degree[neighbor] -= 1  # 减少入度
            if in_degree[neighbor] == 0:
                queue.append(neighbor)
    return edge_order

# def plt_psd():
# # 训练集
# plt.figure(figsize=(20, 18))
# plt.suptitle('Train Set')
# for i in range(16):
#     plt.subplot(4, 4, i + 1)
#     plt.psd(rx[i, :L0], 512, 245.76e6, color='blue', label='RX')
#     plt.psd(out_tra[i, :], 512, 245.76e6, color='purple', label='PIMCOUT')
#     plt.psd(e_tra[i, :], 512, 245.76e6, color='tab:red', label='ERR')
#     plt.psd(nf[i, :L0], 512, 245.76e6, color='black', label='NF')
#     plt.legend()
#     plt.xlim([-50e6, 50e6])
#     plt.ylim([-50, -20])
#
# plt.show()
# # 测试集
# plt.figure(figsize=(20, 18))
# plt.suptitle('Test Set')
# for i in range(16):
#     plt.subplot(4, 4, i + 1)
#     plt.psd(rx[i, L0:], 512, 245.76e6, color='blue', label='RX')
#     plt.psd(out_val[i, :], 512, 245.76e6, color='purple', label='PIMCOUT')
#     plt.psd(e_val[i, :], 512, 245.76e6, color='tab:red', label='ERR')
#     plt.psd(nf[i, L0:], 512, 245.76e6, color='black', label='NF')
#     plt.legend()
#     plt.xlim([-50e6, 50e6])
#     plt.ylim([-50, -20])
# plt.show()
