# coding=utf8

import os
import sys
import inference.svs.ds_e2e as e2e
from utils.audio import save_wav
from utils.hparams import set_hparams, hparams

import torch

root_dir = os.path.dirname(os.path.abspath(__file__))
os.environ['PYTHONPATH'] = f'"{root_dir}"'

sys.argv = [
    f'{root_dir}/inference/svs/ds_e2e.py',
    '--config',
    f'{root_dir}/usr/configs/midi/e2e/opencpop/ds100_adj_rel.yaml',
    '--exp_name',
    '0228_opencpop_ds100_rel'
]

if __name__ == '__main__':

    set_hparams(print_hparams=False)
    infer_ins = e2e.DiffSingerE2EInfer(hparams)

    infer_ins.vocoder.to('cpu')
    with torch.no_grad():
        x = torch.rand(1, 80, 100)
        f0 = torch.rand(1, 100)

        torch.onnx.export(
            infer_ins.vocoder,
            (
                x,
                f0
            ),
            "hifigan.onnx",
            input_names=["x", "f0"],
            output_names=["y"],
            dynamic_axes={
                "x": {
                    0: "hop_size",
                    1: "win_size",
                    2: "fft_size",
                },
                "f0": {
                    0: "len",
                    1: "frames"
                },
                "y": {
                    0: "len",
                    1: "frames",
                    2: "batch_size"
                }
            },
            opset_version=11
        )

    print("OK")
