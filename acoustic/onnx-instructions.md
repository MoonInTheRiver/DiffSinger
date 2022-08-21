# 合成简易版

暂不支持手动指定音素时长。

---

## 准备环境

+ 新建Conda环境
````
conda create -n dfs_onnx python=3.9
conda activate dfs_onnx
````

+ 安装依赖
````
pip install librosa pypinyin tqdm six pyyaml numpy scipy
````

+ 安装ONNXRuntime
````
pip install onnxruntime-gpu
````

+ 准备模型
    + 下载`hifigan.onnx`、`singer_denoise.onnx`、`singer_fs.onnx`、`xiaoma_pe.onnx`，将其移动到`acoustic/models`目录下

## 运行

+ 使用CPU
````
python my_numpy.py xxx.ds
````

+ 使用GPU（需要确保CUDA版本与ONNXRuntime版本兼容）
````
python my_numpy.py xxx.ds -d gpu
````

## 示例

+ 小酒窝.ds

````
{
    "text": "小酒窝长睫毛AP是你最美的记号",
    "notes": "C#4/Db4 | F#4/Gb4 | G#4/Ab4 | A#4/Bb4 F#4/Gb4 | F#4/Gb4 C#4/Db4 | C#4/Db4 | rest | C#4/Db4 | A#4/Bb4 | G#4/Ab4 | A#4/Bb4 | G#4/Ab4 | F4 | C#4/Db4",
    "notes_duration": "0.407140 | 0.376190 | 0.242180 | 0.509550 0.183420 | 0.315400 0.235020 | 0.361660 | 0.223070 | 0.377270 | 0.340550 | 0.299620 | 0.344510 | 0.283770 | 0.323390 | 0.360340",
    "input_type": "word"
}
````

+ 运行

````
> python my_numpy.py 小酒窝.ds
...
Pass word-notes check.
29 29 29
Pass word-notes check.
[Status] Preprocess
[Status] Run fs
[Status] Run sample
[Status] Sample step: 100%|███████████████████████████| 100/100 [00:09<00:00, 10.25it/s] 
[Status] Run pe
[Status] Run vocoder
[Status] Save audio: ./infer_out\小酒窝0.wav
OK
````