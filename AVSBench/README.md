# Directory guide for AVSBench
```commandline
.
├── AVS1
│   ├── ms3
│   │   ├── audio_wav
│   │   │   ├── test  [64 entries exceeds filelimit, not opening dir]
│   │   │   ├── train  [296 entries exceeds filelimit, not opening dir]
│   │   │   └── val  [64 entries exceeds filelimit, not opening dir]
│   │   ├── gt_masks
│   │   │   ├── test  [64 entries exceeds filelimit, not opening dir]
│   │   │   ├── train  [296 entries exceeds filelimit, not opening dir]
│   │   │   └── val  [64 entries exceeds filelimit, not opening dir]
│   │   └── visual_frames  [424 entries exceeds filelimit, not opening dir]
│   └── s4
│       ├── audio_wav
│       │   ├── test  [23 entries exceeds filelimit, not opening dir]
│       │   ├── train  [23 entries exceeds filelimit, not opening dir]
│       │   └── val  [23 entries exceeds filelimit, not opening dir]
│       ├── gt_masks
│       │   ├── test  [23 entries exceeds filelimit, not opening dir]
│       │   ├── train  [23 entries exceeds filelimit, not opening dir]
│       │   └── val  [23 entries exceeds filelimit, not opening dir]
│       └── visual_frames
│           ├── test  [23 entries exceeds filelimit, not opening dir]
│           ├── train  [23 entries exceeds filelimit, not opening dir]
│           └── val  [23 entries exceeds filelimit, not opening dir]
├── AVSBench_Dataset.py
├── README.md
├── eval_utils.py
├── metadata
│   ├── avs1_ms3_test.csv
│   └── avs1_s4_test.csv
```
All .wav files sampled 16k

## Important
Fix bug in official test code (Issue: F-Score results vary depending on the batch number)

Considering the notable impact of this issue on the performance of self-supervised learning models, we suggest utilizing our updated test code.

We already discussed this issue with the author who released the official code.
