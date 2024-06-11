import argparse
from typing import Callable, Dict, Tuple
import time
import numpy as np
import torch
from torch import nn
from fasterkan.fasterkan import FasterKAN, FasterKANvolver
from efficient_kan import KAN
from torchkan import KAL_Net
from fastkan.fastkan import FastKAN as FastKANORG # Ensure the correct import path based on your project structure
from torchkan import KANvolver

def create_dataset(f, 
                   n_var = 28*28, 
                   ranges = [0,1],
                   train_num=60000, 
                   test_num=10000,
                   normalize_input=False,
                   normalize_label=False,
                   device='cpu',
                   seed=0):
    '''
    create dataset
    
    Args:
    -----
        f : function
            the symbolic formula used to create the synthetic dataset
        ranges : list or np.array; shape (2,) or (n_var, 2)
            the range of input variables. Default: [-1,1].
        train_num : int
            the number of training samples. Default: 1000.
        test_num : int
            the number of test samples. Default: 1000.
        normalize_input : bool
            If True, apply normalization to inputs. Default: False.
        normalize_label : bool
            If True, apply normalization to labels. Default: False.
        device : str
            device. Default: 'cpu'.
        seed : int
            random seed. Default: 0.
        
    Returns:
    --------
        dataset : dic
            Train/test inputs/labels are dataset['train_input'], dataset['train_label'],
                        dataset['test_input'], dataset['test_label']
         
    Example
    -------
    >>> f = lambda x: torch.exp(torch.sin(torch.pi*x[:,[0]]) + x[:,[1]]**2)
    >>> dataset = create_dataset(f, n_var=2, train_num=100)
    >>> dataset['train_input'].shape
    torch.Size([100, 2])
    '''

    np.random.seed(seed)
    torch.manual_seed(seed)

    if len(np.array(ranges).shape) == 1:
        ranges = np.array(ranges * n_var).reshape(n_var,2)
    else:
        ranges = np.array(ranges)
        
    train_input = torch.zeros(train_num, n_var)
    test_input = torch.zeros(test_num, n_var)
    for i in range(n_var):
        train_input[:,i] = torch.rand(train_num,)*(ranges[i,1]-ranges[i,0])+ranges[i,0]
        test_input[:,i] = torch.rand(test_num,)*(ranges[i,1]-ranges[i,0])+ranges[i,0]
        
        
    train_label = f(train_input)
    test_label = f(test_input)
        
        
    def normalize(data, mean, std):
            return (data-mean)/std
            
    if normalize_input == True:
        mean_input = torch.mean(train_input, dim=0, keepdim=True)
        std_input = torch.std(train_input, dim=0, keepdim=True)
        train_input = normalize(train_input, mean_input, std_input)
        test_input = normalize(test_input, mean_input, std_input)
        
    if normalize_label == True:
        mean_label = torch.mean(train_label, dim=0, keepdim=True)
        std_label = torch.std(train_label, dim=0, keepdim=True)
        train_label = normalize(train_label, mean_label, std_label)
        test_label = normalize(test_label, mean_label, std_label)

    dataset = {}
    dataset['train_input'] = train_input.to(device)
    dataset['test_input'] = test_input.to(device)

    dataset['train_label'] = train_label.to(device)
    dataset['test_label'] = test_label.to(device)

    return dataset

class MLP(nn.Module):
    def __init__(self, layers: Tuple[int, int, int], device: str):
        super().__init__()
        self.layer1 = nn.Linear(layers[0], layers[1], device=device)
        self.layer2 = nn.Linear(layers[1], layers[2], device=device)

    def forward(self, x: torch.Tensor):
        x = self.layer1(x)
        x = nn.functional.relu(x)
        x = self.layer2(x)
        x = nn.functional.sigmoid(x)
        return x
    


def benchmark(
        dataset: Dict[str, torch.Tensor],
        device: str,
        bs: int,
        loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        model: nn.Module,
        reps: int
    ) -> Dict[str, float]:
    forward_times = []
    backward_times = []
    forward_mems = []
    backward_mems = []
    for k in range(1 + reps):
        train_id = np.random.choice(dataset['train_input'].shape[0], bs, replace=False)
        tensor_input = dataset['train_input'][train_id]
        tensor_input = tensor_input.to(device)

        tensor_output = dataset['train_label'][train_id]
        tensor_output = tensor_output.to(device)

        if device == 'cpu':
            t0 = time.time()
            pred = model(tensor_input)
            t1 = time.time()
            if k > 0:
                forward_times.append((t1 - t0) * 1000)
            train_loss = loss_fn(pred, tensor_output)
            t2 = time.time()
            train_loss.backward()
            t3 = time.time()
            if k > 0:
                backward_times.append((t3 - t2) * 1000)
        elif device == 'cuda':
            torch.cuda.reset_peak_memory_stats()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)

            start.record()
            pred = model(tensor_input)
            end.record()

            torch.cuda.synchronize()
            if k > 0:
                forward_times.append(start.elapsed_time(end))
                forward_mems.append(torch.cuda.max_memory_allocated())

            train_loss = loss_fn(pred, tensor_output)

            torch.cuda.reset_peak_memory_stats()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)

            start.record()
            train_loss.backward()
            end.record()

            torch.cuda.synchronize()
            if k > 0:
                backward_times.append(start.elapsed_time(end))
                backward_mems.append(torch.cuda.max_memory_allocated())
    return {
        'forward': np.mean(forward_times),
        'backward': np.mean(backward_times),
        'forward-memory': np.mean(forward_mems) / (1024 ** 3),
        'backward-memory': np.mean(backward_mems) / (1024 ** 3),
    }


def save_results(t: Dict[str, Dict[str, float]], out_path: str):
    maxlen = np.max([len(k) for k in t.keys()])
    with open(out_path, 'w') as f:
        print(f"| {' '*maxlen} | {'forward':>11}     | {'backward':>11}     | {'forward':>11}     | {'backward':>11}     | {'num params':>11}     | {'num trainable params':>20}     |", file=f)
        print(f"| {' '*maxlen} | {'forward':>11}     | {'backward':>11}     | {'forward':>11}     | {'backward':>11}     | {'num params':>11}     | {'num trainable params':>20}     |")
        print(f"|{'-'*121}|", file=f)
        print(f"|{'-'*121}|")
        for key in t.keys():
            print(f"| {key:<{maxlen}}     | {t[key]['forward']:8.2f} ms     | {t[key]['backward']:8.2f} ms     | {t[key]['forward-memory']:8.2f} GB     | {t[key]['backward-memory']:8.2f} GB     | {t[key]['params']:>11}     | {t[key]['train_params']:>20}     |", file=f)
            print(f"| {key:<{maxlen}}     | {t[key]['forward']:8.2f} ms     | {t[key]['backward']:8.2f} ms     | {t[key]['forward-memory']:8.2f} GB     | {t[key]['backward-memory']:8.2f} GB     | {t[key]['params']:>11}     | {t[key]['train_params']:>20}     |")
        #print(f"FasterKAN can be after small modifications 4.99x faster than FastKAN and {} slower from MLP in forward speed")
        #print(f"FastKAN can be after small modifications 4.93x faster than efficient_kan and 3.02 slower from MLP in backward speed")
        #print(f"FastKAN can be after small modifications 2.57x smaller than efficient_kan and 2 bigger from MLP in forbward memory")
        #print(f"FastKAN can be after small modifications 2.57x smaller than efficient_kan and 1.4 bigger from MLP in backward memory")

def count_params(model: nn.Module) -> Tuple[int, int]:
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    pytorch_total_params_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return pytorch_total_params, pytorch_total_params_train


def _create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-path', default='times.txt', type=str)
    parser.add_argument('--method', choices=['fastkan', 'mlp', 'all'], type=str)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--inp-size', type=int, default=28*28, help='The dimension of the input variables.')
    parser.add_argument('--hid-size', type=int, default=64, help='The dimension of the hidden layer.')
    parser.add_argument('--reps', type=int, default=int(60000/64), help='Number of times to repeat execution and average.')
    parser.add_argument('--just-cuda', action='store_true', help='Whether to only execute the cuda version.')
    parser.add_argument('--bool-flag', action='store_true', help='Whether train grid and inv_denominator.')
    return parser


def main():
    parser = _create_parser()
    args = parser.parse_args()

    f = lambda x: torch.exp(torch.sin(torch.pi*x[:,[0]]) + x[:,[1]]**2)
    dataset = create_dataset(
        f, 
        n_var=args.inp_size,
        ranges = [0,1],
        train_num=60000, 
        test_num=10000,
        normalize_input=False,
        normalize_label=False,
        device='cpu',
        seed=0
    )
    loss_fn = lambda x, y: torch.mean((x - y) ** 2)
    
    res = {}
    if args.method == 'fasterkan' or args.method == 'all':
        model = FasterKAN(layers_hidden=[args.inp_size, args.hid_size, 10], grid_min = -1.2, grid_max = 1.2, num_grids = 8, exponent = 2, inv_denominator = 0.5, train_grid = args.bool_flag, train_inv_denominator = args.bool_flag)
        if not args.just_cuda:
            model.to('cpu')
            res['fasterkan-cpu'] = benchmark(dataset, 'cpu', args.batch_size, loss_fn, model, args.reps)
            res['fasterkan-cpu']['params'], res['fasterkan-cpu']['train_params'] = count_params(model)
        model.to('cuda')
        res['fasterkan-gpu'] = benchmark(dataset, 'cuda', args.batch_size, loss_fn, model, args.reps)
        res['fasterkan-gpu']['params'], res['fasterkan-gpu']['train_params'] = count_params(model)
    if args.method == 'fastkanorg' or args.method == 'all':
        model = FastKANORG(layers_hidden=[args.inp_size, args.hid_size, 10], grid_min = -1.2, grid_max = 1.2, num_grids = 8)
        if not args.just_cuda:
            model.to('cpu')
            res['fastkanorg-cpu'] = benchmark(dataset, 'cpu', args.batch_size, loss_fn, model, args.reps)
            res['fastkanorg-cpu']['params'], res['fastkanorg-cpu']['train_params'] = count_params(model)
        model.to('cuda')
        res['fastkanorg-gpu'] = benchmark(dataset, 'cuda', args.batch_size, loss_fn, model, args.reps)
        res['fastkanorg-gpu']['params'], res['fastkanorg-gpu']['train_params'] = count_params(model)
    if args.method == 'mlp' or args.method == 'all':
        model = MLP(layers=[args.inp_size, args.hid_size*8 , 10], device='cpu')#int(np.rint(args.hid_size*7.5*10/10))
        if not args.just_cuda:
            res['mlp-cpu'] = benchmark(dataset, 'cpu', args.batch_size, loss_fn, model, args.reps)
            res['mlp-cpu']['params'], res['mlp-cpu']['train_params'] = count_params(model)
        model.to('cuda')
        res['mlp-gpu'] = benchmark(dataset, 'cuda', args.batch_size, loss_fn, model, args.reps)
        res['mlp-gpu']['params'], res['mlp-gpu']['train_params'] = count_params(model)
    if args.method == 'efficientkan' or args.method == 'all':
        model = KAN(layers_hidden=[args.inp_size, args.hid_size, 10], grid_size=5, spline_order=3)
        if not args.just_cuda:
            model.to('cpu')
            res['effkan-cpu'] = benchmark(dataset, 'cpu', args.batch_size, loss_fn, model, args.reps)
            res['effkan-cpu']['params'], res['effkan-cpu']['train_params'] = count_params(model)
        model.to('cuda')
        res['effkan-gpu'] = benchmark(dataset, 'cuda', args.batch_size, loss_fn, model, args.reps)
        res['effkan-gpu']['params'], res['effkan-gpu']['train_params'] = count_params(model)
    if args.method == 'kalnet' or args.method == 'all':
        model = KAL_Net(layers_hidden=[args.inp_size, args.hid_size, 10], polynomial_order=3, base_activation=nn.SiLU)
        if not args.just_cuda:
            model.to('cpu')
            res['kalnet-cpu'] = benchmark(dataset, 'cpu', args.batch_size, loss_fn, model, args.reps)
            res['kalnet-cpu']['params'], res['kalnet-cpu']['train_params'] = count_params(model)
        model.to('cuda')
        res['kalnet-gpu'] = benchmark(dataset, 'cuda', args.batch_size, loss_fn, model, args.reps)
        res['kalnet-gpu']['params'], res['kalnet-gpu']['train_params'] = count_params(model)
    if args.method == 'kanvolve' or args.method == 'all':
        model = KANvolver(layers_hidden=[args.hid_size, 10], polynomial_order=2, base_activation=nn.ReLU)
        if not args.just_cuda:
            model.to('cpu')
            res['kanvolve-cpu'] = benchmark(dataset, 'cpu', args.batch_size, loss_fn, model, args.reps)
            res['kanvolve-cpu']['params'], res['kanvolve-cpu']['train_params'] = count_params(model)
        model.to('cuda')
        res['kanvolve-gpu'] = benchmark(dataset, 'cuda', args.batch_size, loss_fn, model, args.reps)
        res['kanvolve-gpu']['params'], res['kanvolve-gpu']['train_params'] = count_params(model)
    if args.method == 'fasterkanvolver' or args.method == 'all':
        model = FasterKANvolver(layers_hidden=[ args.hid_size, 10], grid_min = -1.2, grid_max = 0.2, num_grids = 8, exponent = 2, inv_denominator = 0.5, train_grid = args.bool_flag, train_inv_denominator = args.bool_flag)
        if not args.just_cuda:
            model.to('cpu')
            res['fasterkanvolver-cpu'] = benchmark(dataset, 'cpu', args.batch_size, loss_fn, model, args.reps)
            res['fasterkanvolver-cpu']['params'], res['fasterkanvolver-cpu']['train_params'] = count_params(model)
        model.to('cuda')
        res['fasterkanvolver-gpu'] = benchmark(dataset, 'cuda', args.batch_size, loss_fn, model, args.reps)
        res['fasterkanvolver-gpu']['params'], res['fasterkanvolver-gpu']['train_params'] = count_params(model)
                
    save_results(res, args.output_path)

if __name__=='__main__':
    main()