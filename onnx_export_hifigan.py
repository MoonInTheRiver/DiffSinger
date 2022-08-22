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

    dev = 'cuda'

    infer_ins = e2e.DiffSingerE2EInfer(hparams)
    infer_ins.vocoder.to(dev)

    with torch.no_grad():
        x = torch.rand(1, 80, 2).to(dev)
        f0 = torch.rand(1, 2).to(dev)

        x = torch.load("c.pt").to(dev)
        f0 = torch.load("f0.pt").to(dev)

        print(x.shape)
        print(f0.shape)

        torch.onnx.export(
            infer_ins.vocoder,
            (
                x,
                f0
            ),
            "hifigan.onnx",
            verbose=True,
            input_names=["x", "f0"],
            dynamic_axes={
                "x": {
                    0: "batch_size",
                    1: "num_mel_bin",
                    2: "frames",
                },
                "f0": {
                    0: "batch_size",
                    1: "frames"
                }
            },
            opset_version=11,
        )

    print(infer_ins.vocoder)
    print("OK")
