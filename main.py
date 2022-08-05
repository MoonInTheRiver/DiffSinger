# coding=utf8
import argparse
import json
import os
import sys

import torch

from inference.svs.ds_e2e import DiffSingerE2EInfer

root_dir = os.path.dirname(__file__)
os.environ['PYTHONPATH'] = f'"{root_dir}"'

parser = argparse.ArgumentParser(description='Run DiffSinger inference')
parser.add_argument('proj', type=str, help='Path to the input file')
parser.add_argument('--out', type=str, default='./infer_out', required=False, help='Path of the output folder')
parser.add_argument('--title', type=str, required=False, help='Title of output file')
parser.add_argument('--num', type=int, default=1, help='Number of runs')
parser.add_argument('--seed', type=int, help='Random seed of the inference')
args = parser.parse_args()

with open(args.proj, 'r', encoding='utf-8') as f:
    c = json.load(f)

sys.argv = [
    f'{root_dir}/inference/svs/ds_e2e.py',
    '--config',
    f'{root_dir}/usr/configs/midi/e2e/opencpop/ds100_adj_rel.yaml',
    '--exp_name',
    '0228_opencpop_ds100_rel'
]

if args.seed:
    print(f'Setting random seed: {args.seed}')
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

name = os.path.basename(args.proj).split('.')[0] if not args.title else args.title
if args.num == 1:
    DiffSingerE2EInfer.example_run(c, target=os.path.join(args.out, f'{name}.wav'))
else:
    for i in range(1, args.num + 1):
        DiffSingerE2EInfer.example_run(c, target=os.path.join(args.out, f'{name}-{str(i).zfill(3)}.wav'))
