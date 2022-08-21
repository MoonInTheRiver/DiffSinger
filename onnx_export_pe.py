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
    infer_ins.pe.to(dev)
    with torch.no_grad():
        mel_input = torch.rand(1, 4097, 80).to(dev)

        torch.onnx.export(
            infer_ins.pe,
            (
                mel_input
            ),
            "xiaoma_pe.onnx",
            verbose=True,
            input_names=["mel_input"],
            dynamic_axes={
                "mel_input": {
                    0: "batch_size",
                    1: "frames",
                    2: "num_mel_bin",
                }
            },
            opset_version=11
        )

    print("OK")
