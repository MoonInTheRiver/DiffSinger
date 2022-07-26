## DiffSinger (SVS version)

### PART1. [Run DiffSinger on PopCS](README-SVS-popcs.md)
In this part, we only focus on spectrum modeling (acoustic model) and assume the ground-truth (GT) F0 to be given as the pitch information following these papers [1][2][3]. 

Thus, the pipeline of this part can be summarized as:

```
[lyrics] -> [linguistic representation] (Frontend)
[linguistic representation] + [GT F0] + [GT phoneme duration] -> [mel-spectrogram]  (Acoustic model)
[mel-spectrogram] + [GT F0] -> [waveform] (Vocoder)
```


[1] Adversarially trained multi-singer sequence-to-sequence singing synthesizer. Interspeech 2020.

[2] SEQUENCE-TO-SEQUENCE SINGING SYNTHESIS USING THE FEED-FORWARD TRANSFORMER. ICASSP 2020.

[3] DeepSinger : Singing Voice Synthesis with Data Mined From the Web. KDD 2020.

### PART2. [Run DiffSinger on Opencpop](README-SVS-opencpop-cascade.md)
Thanks [Opencpop team](https://wenet.org.cn/opencpop/) for releasing their SVS dataset with MIDI label, **Jan.20, 2022**. (Also thanks to my co-author [Yi Ren](https://github.com/RayeRen), who applied for the dataset and did some preprocessing works for this part).

Since there are elaborately annotated MIDI labels, we are able to supplement the pipeline in PART 1 by adding a naive melody frontend.

#### 2.1
Thus, the pipeline of [this part](README-SVS-opencpop-cascade.md) can be summarized as:

```
[lyrics] + [MIDI] -> [linguistic representation (with MIDI information)] + [predicted F0] + [predicted phoneme duration] (Melody frontend)
[linguistic representation] + [predicted F0] + [predicted phoneme duration] -> [mel-spectrogram]  (Acoustic model)
[mel-spectrogram] + [predicted F0] -> [waveform] (Vocoder)
```

#### 2.2
In 2.1, we find that if we predict F0 explicitly in the melody frontend, there will be many bad cases of uv/v prediction. Then, we abandon the explicit prediction of the F0 curve in the melody frontend but make a joint prediction with spectrograms.

Thus, the pipeline of [this part](README-SVS-opencpop-e2e.md) can be summarized as:
```
[lyrics] + [MIDI] -> [linguistic representation] + [predicted phoneme duration] (Melody frontend)
[linguistic representation (with MIDI information)] + [predicted phoneme duration] -> [mel-spectrogram]  (Acoustic model)
[mel-spectrogram] -> [predicted F0]  (Pitch extractor)
[mel-spectrogram] + [predicted F0] -> [waveform] (Vocoder)
```