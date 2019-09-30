#!/usr/bin/env python
from argparse import ArgumentParser
from sys import exit
from operator import itemgetter
import re
import torch
from torch.nn import functional as F
import numpy as np
from mish_cuda import MishCudaFunction

def scale(val, spec="#0.4G"):
    PREFIXES = np.array([c for c in u"yzafpnµm kMGTPEZY"])
    exp = np.int8(np.log10(np.abs(val)) // 3 * 3 * np.sign(val))
    val /= 10.**exp
    prefix = PREFIXES[exp//3 + len(PREFIXES)//2]
    return f"{val:{spec}}{prefix}"

def display_times(times):
    return f"{scale(times.mean())}s ± {scale(times.std())}s ({scale(times.min())}s - {scale(times.max())}s)"

def profile(func, inp, n_repeat=100, warmup=10):
    fwd_times,bwd_times = [],[]
    for i in range(n_repeat + warmup):
        start,end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
        start.record()
        res = func(inp)
        end.record()
        torch.cuda.synchronize()
        if i >= warmup: fwd_times.append(start.elapsed_time(end))
        start,end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
        inp = inp.clone().requires_grad_()
        y = func(inp)
        l = y.mean()
        start.record()
        _ = torch.autograd.grad(l, inp)
        end.record()
        torch.cuda.synchronize()
        if i >= warmup: bwd_times.append(start.elapsed_time(end))
    return (np.array(fwd_times)/1000, # Elapsed time is in ms
            np.array(bwd_times)/1000)

mish_pt = lambda x: x.mul(torch.tanh(F.softplus(x)))

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-d', '--device', type=int, default=0)
    parser.add_argument('-n', '--n_repeat', type=int, default=100)
    parser.add_argument('-w', '--warmup', type=int, default=10)
    parser.add_argument('-s', '--size', default="(16,10,256,256)")
    parser.add_argument('-b', '--baseline', action='store_true')
    parser.add_argument('-t', '--type', default='all')
    args = parser.parse_args()
    if args.type == 'all': dtypes = [torch.float16, torch.float32, torch.float64]
    else:
        if not hasattr(torch, args.type): exit("Invalid data type, expected torch type or 'all', got {args.type}")
        dtypes = [getattr(torch, args.type)]
    dev = torch.device(type='cuda', index=args.device)
    sz_str = args.size.replace(' ','')
    if not re.match(r"[\(\[]\d+(,\d+)*[\)\]]", sz_str):
        exit("Badly formatted size, should be a list or tuple such as \"(1,2,3)\".")
    sz = list(map(int, sz_str[1:-1].split(',')))
    print(f"Profiling over {args.n_repeat} runs after {args.warmup} warmup runs.")
    print(f"Profiling on {torch.cuda.get_device_name(dev)}")
    for dtype in dtypes:
        if len(dtypes) > 1:
            print(f"Testing on {dtype}:")
            ind = ' '
        else: ind = ''
        inp = torch.randn(*sz, dtype=dtype, device=dev)
        timings = []
        funcs = {}
        if args.baseline: funcs.update(relu=torch.nn.functional.relu, softplus=torch.nn.functional.softplus, mish_pt=mish_pt)
        funcs['mish_cuda'] = MishCudaFunction.apply
        max_name = max(map(len, funcs.keys())) + 6
        for (name,func) in funcs.items():
            fwd_times,bwd_times = profile(func, inp, args.n_repeat, args.warmup)
            print(ind+(name+'_fwd:').ljust(max_name) + display_times(fwd_times))
            print(ind+(name+'_bwd:').ljust(max_name) + display_times(bwd_times))
