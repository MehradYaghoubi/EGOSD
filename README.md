# EGOSD

## Entropy-Gated Online Self-Distillation for Visual Speech Recognition Based on Connectionist Temporal Classification

Research code accompanying the work by **Mehrad Yaghoubi** and **Azam Bastanfard**.

EGOSD augments a lightweight visual speech recognition backbone with an auxiliary recurrent **Belief Module**. During training, the entropy of the Belief predictions controls the contribution of frame-wise knowledge distillation to the main CTC recognition path. The model recognizes phoneme sequences from video without requiring audio at inference.

**Paper:** [SSRN preprint](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=6831426)<br>
**Preprint DOI:** [10.2139/ssrn.6831426](https://doi.org/10.2139/ssrn.6831426)<br>
**Code:** [MehradYaghoubi/EGOSD](https://github.com/MehradYaghoubi/EGOSD)<br>
**Original anonymous repository:** [EGOSD-D2CC](https://anonymous.4open.science/r/EGOSD-D2CC/)<br>
**Lip extraction package:** [PyLipExtractor](https://github.com/MehradYaghoubi/pylipextractor) · [PyPI](https://pypi.org/project/pylipextractor/)

The DOI above identifies the SSRN preprint; it is not a DOI for a published journal version. The results below distinguish the original single-run findings from the updated multiple-run evaluation. The preprint may retain the earlier figures until its corresponding revision is uploaded.

## Contents

- [Method](#method)
- [Results](#results)
- [Experimental protocol](#experimental-protocol)
- [Model size and computational cost](#model-size-and-computational-cost)
- [Repository structure](#repository-structure)
- [Installation](#installation)
- [Lip preprocessing with PyLipExtractor](#lip-preprocessing-with-pylipextractor)
- [Dataset preparation](#dataset-preparation)
- [Training](#training)
- [Evaluation and analysis](#evaluation-and-analysis)
- [Implementation notes and troubleshooting](#implementation-notes-and-troubleshooting)
- [Citation](#citation)
- [Related resources](#related-resources)
- [Authors and contact](#authors-and-contact)
- [Licensing](#licensing)

## Method

The visual backbone contains a 3D convolutional frontend, spatial attention pooling, temporal convolutional blocks, a Transformer encoder and a CTC prediction head. The auxiliary Belief Module processes the contextual encoder features with a GRU and produces an additional distribution over phoneme classes at every time step.

For Belief probabilities `q_t`, predictive entropy and the gate are:

```text
H(q_t) = -sum_c q_t(c) log q_t(c)
w_t    = exp(-H(q_t))
```

The consistency objective is a valid-frame average of `w_t * KL(stop_gradient(q_t) || p_t)`, where `p_t` is the main prediction. A separate CTC objective supervises the Belief branch. Higher predictive entropy reduces the consistency weight.

Entropy is used here as **predictive uncertainty**. It is not, by itself, a calibrated measure of prediction error, intrinsic visual ambiguity or aleatoric uncertainty. The unidirectional GRU receives features from a non-causal encoder, so the complete pipeline should not be described as strictly causal with respect to raw video frames.

The `detach_hidden_every` setting truncates gradient propagation through the recurrent hidden state. It does not erase the hidden-state values or restrict the Transformer to the same number of frames.

## Results

All PER values are percentages; lower is better. Values with `±` are **mean ± sample standard deviation**, not confidence intervals. A single-run value has no estimated run-to-run standard deviation.

The numbers in this section are reported experimental results from the revised manuscript and the authors' experiment records. They were not newly measured while preparing this README. The original source snapshot is preserved, with the revised gate-control and adapted CTC-SKD trainers added separately. Not every historical training configuration or comparison implementation is included.

### Main results over three runs

The repeated-run EGOSD experiment uses **detach interval = 2**.

| Dataset | Baseline PER (%) | EGOSD PER (%) | Absolute reduction (percentage points) | Relative PER reduction (%) | Runs per method |
| --- | ---: | ---: | ---: | ---: | ---: |
| GRID | 7.240 ± 0.052 | **6.367 ± 0.283** | 0.873 | 12.063 | 3 |
| Lombard GRID, after adaptation from GRID | 11.920 ± 0.933 | **9.577 ± 0.045** | 2.343 | 19.659 | 3 |

The individual test observations underlying these summaries are:

| Dataset | Method | Run 1 PER (%) | Run 2 PER (%) | Run 3 PER (%) |
| --- | --- | ---: | ---: | ---: |
| GRID | Baseline | 7.270 | 7.270 | 7.180 |
| GRID | EGOSD, detach = 2 | 6.110 | 6.320 | 6.670 |
| Lombard GRID | Baseline | 10.850 | 12.560 | 12.350 |
| Lombard GRID | EGOSD, detach = 2 | 9.530 | 9.620 | 9.580 |

Statistics are calculated from the individual values before rounding:

```text
mean       = sum(x_i) / n
sample SD  = sqrt(sum((x_i - mean)^2) / (n - 1))
relative PER reduction (%) = 100 * (baseline mean - EGOSD mean) / baseline mean
```

For example, the GRID EGOSD mean is `(6.11 + 6.32 + 6.67) / 3 = 6.366666...`, which is displayed as `6.367%`. Using this unrounded mean and baseline mean `7.24%` gives the relative reduction `12.063%`.

### Additional GRID comparisons

| Method | Test PER (%) | Runs | Purpose |
| --- | ---: | ---: | --- |
| Original-capacity Baseline | 7.240 ± 0.052 | 3 | Main reference |
| Capacity-matched Baseline, approximately 1.292M parameters | 7.190 | 1 | Examine an increase in backbone capacity |
| [Fixed-weight online distillation](train_gate_comparison_matched.py) | 8.200 | 1 | Use weight 1 for every valid frame |
| [Max-softmax weighting](train_gate_comparison_matched.py) | 7.310 | 1 | Use the largest Belief class probability as the frame weight |
| [Adapted CTC-SKD](train_ctc_skd_comparison.py) | 7.020 | 1 | Compare an adapted CTC self-distillation approach |
| EGOSD, detach = 2 | **6.367 ± 0.283** | 3 | Use entropy-based frame weighting |

The adapted CTC-SKD comparison has **1,101,059 student parameters** and **1,405,029 total training parameters**. It is an adaptation to the visual backbone, not a reproduction of the reference paper's ASR experiment. Its reference DOI is [10.21437/Interspeech.2024-363](https://doi.org/10.21437/Interspeech.2024-363).

The single-run controls favor EGOSD in these observations, but they do not establish statistical significance or a distribution of results for those controls. Additional fixed-weight/max-softmax runs on Lombard are not reported.

### Detach-interval ablation

| Detach interval (frames) | GRID test PER (%) | GRID runs | Lombard GRID test PER (%) | Lombard runs |
| ---: | ---: | ---: | ---: | ---: |
| 2 | 6.367 ± 0.283 | 3 | 9.577 ± 0.045 | 3 |
| 3 | 6.010 | 1 | 9.580 | 1 |
| 5 | **5.920** | 1 | 11.260 | 1 |
| 10 | 6.270 | 1 | Not reported | — |

The Lombard detach-5 value follows the authors' confirmed manuscript value, `11.260%`. The detach-3/5/10 observations are exploratory single runs and should not be interpreted as multiple-run means. In particular, the best observed GRID single run at detach 5 is a separate result from the repeated detach-2 experiment.

### Earlier single-run findings

The original preprint reported the following representative single-run results:

| Dataset | Baseline PER (%) | EGOSD PER (%) | Relative reduction (%) |
| --- | ---: | ---: | ---: |
| GRID | 7.270 | 6.110 | 15.956 |
| Lombard GRID | 12.380 | 9.660 | 21.971 |

These reductions were rounded to **15.96%** and **21.97%** in the earlier abstract. They must not be substituted for the relative reductions calculated from the three-run means. Table 3 of the manuscript also reports Baseline CER `5.480%` and WER `10.280%`; these are separate reported metrics and are not three-run PER summaries.

### Isolated-word evaluation on LRW-Persian

| Method | PER (%) |
| --- | ---: |
| Baseline | **42.050** |
| EGOSD | 45.250 |

EGOSD increases PER by **3.200 percentage points**, corresponding to a **7.610% relative increase** over the Baseline in this supplementary evaluation.

These clips are short isolated words; the evaluated material had a median length of **13 frames**, or **0.520 s at 25 fps**. Limited temporal evidence is a possible explanation for reduced benefit from the recurrent auxiliary branch, but clip duration, language, video quality and other dataset/training factors have not been experimentally isolated as causes. This result limits generalization of the improvements observed on GRID and Lombard.

The LRW-Persian paper reports word-level recognition metrics. Its word accuracy/error values are not directly comparable with the phoneme-level PER above. See [LRW-Persian](https://doi.org/10.48550/arXiv.2510.22716).

## Experimental protocol

### Speaker-disjoint splits

| Dataset | Total speakers | Training speakers | Validation speakers | Test speakers |
| --- | ---: | ---: | ---: | ---: |
| GRID | 33 | 24 (72.727%) | 5 (15.152%) | 4 (12.121%) |
| Lombard GRID | 54 | 41 (75.926%) | 6 (11.111%) | 7 (12.963%) |

The same train/validation/test partitions were retained across the repeated experiments. Checkpoint selection uses validation performance; the reported test PER is obtained in a separate test evaluation. Changing the speaker partition is a different experimental protocol, not a way to select a new training seed.

The three repeated runs used seeds **42, 42 and 43**. The first run used a different data order. They are therefore three separate runs, not three distinct-seed runs. Using a repeated seed with a different manifest/sample order can still produce different training trajectories.

For Lombard results, both methods were first trained on GRID and then fine-tuned on Lombard. Training on Lombard from scratch is not the same experiment as this adaptation protocol.

The available GRID manifests contain **23,975 training**, **4,989 validation** and **3,865 test** clips. The corresponding Lombard research manifests contain **4,052 / 594 / 669** clips. The original repository snapshot includes GRID metadata; those figures do not imply that raw media or every later experiment's metadata are distributed here.

The reported experiments used PyTorch and an NVIDIA RTX 2060 Super. No complete historical environment lock is provided, so exact package versions and the actual checkpoint initialization chain should be recorded when repeating an experiment.

## Model size and computational cost

The historical profiling report gives:

| Quantity | Baseline | EGOSD | Relative increase (%) |
| --- | ---: | ---: | ---: |
| Parameters | 1,101,059 | 1,269,093 | 15.261 |
| MACs | 11,195,625,150 | 11,208,321,450 | 0.113 |
| FLOPs | 22,391,250,300 | 22,416,642,900 | 0.113 |

These are the reported counts for the corresponding profiling setup, not new timing measurements. The EGOSD count includes the structurally present Belief branch. MAC/FLOP overhead is not itself a measurement of wall-clock latency and depends on the input dimensions and profiling convention. The original full forward pass still evaluates the auxiliary branch; removing it for deployment would be a separate implementation change.

## Repository structure

The original repository layout is retained:

```text
EGOSD/
├── Configs/
│   ├── config.py
│   └── comparison_config.py     # research configuration for added comparisons
├── DataProcessor/
│   ├── ULBiTP_data_loader.py
│   ├── preprocess_data.py
│   ├── generate_character_map.py
│   ├── generate_phoneme_map.py
│   ├── generate_data_lists.py
│   ├── speaker_segmenter.py
│   ├── grid_dataset.py
│   ├── kornia_augs.py
│   ├── mfa_downloader.py
│   ├── phoneme_to_viseme.json
│   └── txtcleaner.py
├── Model/
│   ├── base.py
│   └── egosd.py
├── Utils/
│   ├── custom_loss.py
│   ├── logging_utils.py
│   └── metrics.py
├── dataset_processed/
│   ├── train.json
│   ├── val.json
│   ├── test.json
│   ├── speaker_meta.json
│   └── ... vocabulary maps and labels
├── ch/                         # checkpoint files in the original snapshot
├── train_base.py
├── train_EGOSD.py
├── train_EGOSD_NoGate.py
├── train_base_char.py
├── train_gate_comparison_matched.py
├── train_ctc_skd_comparison.py
├── compare_models.py
├── compare_models_with_flops.py
├── analyze_and_plot_mechanism.py
└── README.md
```

`Model/base.py` contains the lightweight Baseline. `Model/egosd.py` contains the Belief-augmented architecture and uncertainty-weighted loss. Original internal class names containing `VLBP` are retained for checkpoint/source compatibility.

The checkpoint names in `ch/` belong to the original snapshot. A filename containing a PER-like number is not sufficient evidence that it is the test PER or a multiple-run result. Match each checkpoint to its run configuration, validation selection and separate test evaluation.

## Installation

Clone the repository and work from its root:

```bash
 git clone https://github.com/MehradYaghoubi/EGOSD.git
 cd EGOSD
```

Use a Python environment with compatible PyTorch, NumPy and MediaPipe versions. Install the appropriate CPU/CUDA PyTorch build using the [official PyTorch installation instructions](https://pytorch.org/get-started/locally/), then install the Python dependencies used by the original trainers:

```bash
python -m pip install numpy tqdm matplotlib seaborn scikit-learn editdistance kornia
```

For data preprocessing:

```bash
python -m pip install pylipextractor opencv-python requests scipy g2p-en nltk
```

NLTK/G2P resources may need to be downloaded when first used. FFmpeg must be installed separately and be available on `PATH` when optional video conversion is enabled. The MFA helper requires a separate compatible Montreal Forced Aligner installation and its `mfa` command. These alignment tools are not required merely to train from already prepared arrays and labels.

The optional FLOP script imports `ptflops` inside its profiling path; install it if running that analysis:

```bash
python -m pip install ptflops
```

The original models tolerate a missing optional `dropblock` import; install `dropblock` only if using code that actually requires it. No `requirements.txt` or exact historical dependency lock is assumed to exist in the original snapshot. Record the environment used for a new run, for example with `python -m pip freeze > environment-run.txt`.

## Lip preprocessing with PyLipExtractor

[**PyLipExtractor**](https://github.com/MehradYaghoubi/pylipextractor), developed by Mehrad Yaghoubi, supplies the lip-region extraction stage. Its [PyPI package](https://pypi.org/project/pylipextractor/) uses MediaPipe Face Mesh landmarks, supports EMA crop stabilization and configurable margins, and provides optional illumination processing and FFmpeg conversion. Its role here is to prepare visual input; it does not perform EGOSD's phoneme recognition.

The project's `DataProcessor/preprocess_data.py` imports `LipExtractor` and forwards the dataset settings before processing clips. The configured output is a sequence of **70 × 105** RGB lip crops, saved as NumPy arrays. Landmark drawing on the final model input is disabled. Debug images with landmarks can be used separately to inspect extraction.

Install the package:

```bash
python -m pip install pylipextractor
```

A minimal example for the documented tuple-return API is:

```python
from pathlib import Path
from pylipextractor.lip_extractor import LipExtractor

output = Path("output/lip_frames.npy")
output.parent.mkdir(parents=True, exist_ok=True)
LipExtractor.config.IMG_H = 70
LipExtractor.config.IMG_W = 105
LipExtractor.config.INCLUDE_LANDMARKS_ON_FINAL_OUTPUT = False

extractor = LipExtractor()
frames, rtf = extractor.extract_lip_frames(
    Path("input_video.mp4"), output_npy_path=output
)
if frames is None:
    raise RuntimeError("Lip extraction failed or the clip was rejected")
```

This example only extracts one clip. Training additionally needs the correct split path, frame count and target phoneme IDs. Keep preprocessing consistent across compared models. Package versions can differ in return values and defaults; preserve the version and settings used for an experiment rather than assuming every release is interchangeable.

Package documentation: [DOCUMENTATION.md](https://github.com/MehradYaghoubi/pylipextractor/blob/main/DOCUMENTATION.md). PyLipExtractor is independently MIT-licensed; that does not determine the license of this EGOSD repository.

## Dataset preparation

Obtain GRID/Lombard through the dataset providers and retain their terms of use. The training input is prepared video arrays and phoneme sequences; audio is not a recognition input.

The GRID raw layout expected by the original utilities is:

```text
Dataset/
├── videos/
│   ├── s01/
│   └── ...
└── aligns/
    ├── s01/
    └── ...
```

After preprocessing, a sample path in `train.json`, such as `video/train/s01/lgiz4p.npy`, resolves to:

```text
dataset_processed/video/train/s01/lgiz4p.npy
```

Each sample contains `video_path`, `num_frames` and `phoneme_ids`; supplied entries also include text/phoneme strings and character targets. Arrays are expected as `(T, H, W, 3)` with image values in the 0--255 range. The loader normalizes them to [-1, 1], pads variable-length batches and uses manifest `num_frames` as the expected length.

The original configuration uses 25 fps, 70 × 105 crops and a maximum of 75 frames for its GRID setup. Do not impose that maximum on another dataset without matching the actual experiment. The loader can pad/truncate arrays and can return dummy input for missing files: verify every manifest path and frame count before using it for training or test reporting.

The preparation utilities have different roles:

| File | Role |
| --- | --- |
| `grid_dataset.py` | Download and organize GRID files |
| `speaker_segmenter.py` | Generate speaker metadata for a new partition |
| `preprocess_data.py` | Extract and save lip crops with PyLipExtractor |
| `generate_character_map.py` | Build character vocabulary |
| `generate_phoneme_map.py` | Build phoneme vocabulary |
| `generate_data_lists.py` | Build manifests from prepared clips and annotations |
| `mfa_downloader.py` | Obtain alignment resources and invoke MFA |
| `txtcleaner.py` | Original transcript-cleaning helper |

For the reported split, preserve the provided maps and speaker metadata. The legacy split-ratio fields in the original config are not the authoritative record of the reported 24/5/4 split. Re-running generators can overwrite mappings, change label IDs or change sample order. Do not regenerate a split for each seed.

Some original helpers expect additional files/configuration from the full research workspace; inspect their imports and configuration requirements before running them. In particular, `generate_data_lists.py` refers to `DataProcessor.generate_viseme_map` and an `lm_builder` configuration that are absent from the slimmed original repository snapshot. The original `grid_dataset.py` main block also deletes an existing `Dataset` directory before downloading. Run that downloader only in an isolated working directory, not over an existing prepared dataset.

## Training

### Original snapshot

Run commands from the repository root so that relative data paths resolve correctly:

```bash
python train_base.py
python train_EGOSD.py
```

Before the EGOSD command, see the [original class-name compatibility note](#original-class-name-compatibility) below. The original filenames and training implementations have been retained.

The original repository defaults include:

| Setting | Original snapshot value |
| --- | --- |
| Backbone configuration | `MainConfig.rlx_config` |
| Frontend output channels | 128 |
| Encoder dimension | 192 |
| Temporal blocks | 5 |
| Temporal kernels | [3, 5, 7] |
| Attention heads / Transformer layers | 2 / 2 |
| Batch size | 32 |
| Maximum epochs | 90 |
| Initial learning rate | 0.0002 |
| Gradient clipping | 10.0 |
| Validation batch size | 1 |
| EGOSD detach interval | 2 |
| Consistency warm-up | First 12 epochs |
| Belief CTC coefficient | 0.5 |
| Consistency coefficient after warm-up | 2.0 |
| Early stopping | 6 epochs without validation PER improvement |

Optimizer/scheduler behavior follows each entry point. For example, the original Baseline reads configured weight decay, while EGOSD's fresh optimizer hard-codes `1e-2`; setting an unused legacy config field does not override that code path. Several other legacy configuration fields are not active in these trainers.

The later GRID/Lombard research configurations used a maximum of 150 epochs and gradient clipping 3.0. They must not be silently treated as identical to the original 90-epoch/10.0 configuration. The two added comparison trainers import `Configs/comparison_config.py`, a separate copy of the full research configuration from `New_VSR`; they do not replace the original trainers' configuration. Not every historical initialization chain is reconstructed here.

### Revision comparison experiments

The added files retain the training logic of the corresponding `New_VSR` scripts. Only import paths were adapted to the published model filenames and the separate comparison configuration:

| Entry point | Mode | Reported GRID test PER (%) |
| --- | --- | ---: |
| `train_gate_comparison_matched.py` | `--gate fixed` | 8.200 |
| `train_gate_comparison_matched.py` | `--gate maxsoftmax` | 7.310 |
| `train_ctc_skd_comparison.py` | Adapted CTC-SKD | 7.020 |

These are single-run results. The commands below illustrate how to launch the scripts; they do not establish the historical seed or guarantee reproduction of the reported numbers.

```bash
python train_gate_comparison_matched.py --gate fixed --seed 42
python train_gate_comparison_matched.py --gate maxsoftmax --seed 42
python train_ctc_skd_comparison.py --seed 42
python train_ctc_skd_comparison.py --evaluate /path/to/best_student.pth --split test
```

Both trainers use `MainConfig.rlx_config` from `Configs/comparison_config.py`. Its defaults include batch size 32, maximum 150 epochs, initial learning rate 0.0002 and gradient clipping 3.0. Set data paths and check the existing split manifests before running; do not regenerate splits for each comparison.

The gate trainer imports `Model/egosd.py`, whose published detach default is **2 frames**. It retains the 12-epoch consistency warm-up, followed by fixed or max-softmax frame weighting. This is the matched fixed-weight control associated with 8.200% PER, rather than the historical no-gate entry point.

The CTC-SKD trainer imports the student from `Model/base.py`, selects checkpoints using student validation PER and requires the separate evaluation command above for test PER. It writes `best_student.pth`, `last_training.pth` and protocol/history records under its configured run directory. It does not implement resume.

The historical blank convention is preserved: main/student CTC and decoding use the vocabulary blank (33 in the distributed map), while auxiliary Belief/teacher CTC uses blank 0. These added files do not correct that mismatch; changing it defines a different experimental protocol.

### Seed and detach controls

The original Base/EGOSD trainers fall back to seed 42 and do not expose a `--seed` flag. For a chosen seed, set `SEED` in `TrainingConfig` consistently for the experiment and record it. Keep the maps, split membership and sample order fixed when repeating a protocol.

The original EGOSD detach default is defined in `Model/egosd.py`, in `VLBPModule_v3`. Record its value explicitly when doing detach ablations; it is separate from the probability normalization used to calculate entropy or max-softmax confidence.

### Continuing from a checkpoint

The original entry points accept a full checkpoint containing `model_state`:

```bash
python train_base.py --resume /path/to/full_checkpoint.pth
python train_EGOSD.py --resume /path/to/full_checkpoint.pth
```

A best-weight file containing only a raw state dictionary is not automatically interchangeable with the full checkpoint expected by `--resume`. Keep the configuration and vocabulary compatible with the checkpoint.

The original Baseline uses its stored start epoch. The original EGOSD code restores state but its epoch loop restarts at zero, which can repeat the consistency warm-up. Therefore, do not describe its `--resume` behavior as an exact uninterrupted continuation. Match the actual training stages used in a reported result.

Logs and checkpoints are written to the configured `logs/` and `checkpoints/` paths. Select the model using validation PER; report test PER only after a separate test evaluation.

### Other original entry points

`train_EGOSD_NoGate.py` retains the historical no-gate experiment. It applies consistency without the later matched-control warm-up and must not be substituted for `train_gate_comparison_matched.py --gate fixed` merely because both use equal frame weights. `train_base_char.py` is retained as a separate character-oriented experimental entry point; it is not the source of the repeated phoneme-PER results above.

## Evaluation and analysis

For the original two-model test comparison, edit `BASE_CKPT` and `BELIEF_CKPT` at the end of `compare_models.py` to the actual checkpoint paths, then run:

```bash
python compare_models.py
```

This script loads the test split, decodes main-head predictions and calculates `calculate_per(..., standard=True)`. The standard PER calculation ignores `sil` and `<wb>` and pools phoneme edit distances over reference phonemes. The output consists of phoneme sequences; this is not automatically readable word transcription or a WER result.

The supplied comparison script uses permissive checkpoint loading (`strict=False`). Inspect missing/unexpected keys and confirm the model definition rather than treating the absence of an exception as proof of compatibility. For variable-length clips, decode only valid frames and record evaluation batch size: padding can affect a Transformer that is not passed a key-padding mask.

To inspect complexity, set the checkpoint paths in the FLOP analysis script and run:

```bash
python compare_models_with_flops.py
```

To inspect the entropy/consistency relationship, set `CKPT_PATH` in the mechanism script and run:

```bash
python analyze_and_plot_mechanism.py
```

Its plot describes prediction entropy and consistency behavior. It is not, on its own, calibration analysis or proof that high-entropy frames are visually ambiguous or erroneous.

## Implementation notes and troubleshooting

### Original class-name compatibility

The original `train_EGOSD.py` imports the names `ULBiTP_R_v8_EGOSD_FullFeedback` and `EGOSD_Loss`, while the distributed `Model/egosd.py` defines `ULBiTP_R_v8_VLBP_FullFeedback` and `UncertaintyAwareLoss`. If this produces an import error, the compatible import is:

```python
from Model.egosd import (
    ULBiTP_R_v8_VLBP_FullFeedback as ULBiTP_R_v8_EGOSD_FullFeedback,
    UncertaintyAwareLoss as EGOSD_Loss,
    count_parameters,
)
```

This maps existing names without changing the architecture or loss formula. This README update documents the mismatch; it does not modify the source files.

### CTC blank convention

The distributed phoneme map assigns `<blank>` ID **33**. Main CTC and greedy decoding obtain their blank from this map. The historical auxiliary Belief CTC constructs `torch.nn.CTCLoss` with the default blank **0**, and the adapted comparison retained the corresponding branch convention.

This is an implementation mismatch retained in the reported code. Do not silently remap the vocabulary or replace one branch's blank setting and then describe the changed experiment as the same historical protocol. A corrected consistent-blank implementation requires a separately identified experiment.

### Common issues

| Issue | What to check |
| --- | --- |
| `ModuleNotFoundError` | Activate the intended environment; install the dependency; run from the project root |
| Missing arrays / zero-length or dummy inputs | Resolve every `video_path` relative to `dataset_processed`; confirm extraction succeeded |
| Different numbers of classes | Preserve the training phoneme map and its ID order |
| Checkpoint size mismatch | Match backbone, class count, auxiliary branch and checkpoint format |
| Different PER after changing a seed | Check data order, split files, augmentation, initialization, optimizer stages and environment as well as the seed |
| FFmpeg conversion failure | Check FFmpeg availability, codec support and chosen CPU/CUDA conversion device |
| Missing viseme/LM configuration | Match the helper's required files to the full research workspace before generating new metadata |

## Citation

If you use this research, please cite the paper. Citation is requested separately from software-license conditions.

```bibtex
@misc{yaghoubi2026egosd,
  author       = {Yaghoubi, Mehrad and Bastanfard, Azam},
  title        = {Entropy-Gated Online Self-Distillation for Visual Speech
                  Recognition Based on Connectionist Temporal Classification},
  year         = {2026},
  howpublished = {SSRN preprint},
  doi          = {10.2139/ssrn.6831426},
  url          = {https://papers.ssrn.com/sol3/papers.cfm?abstract_id=6831426}
}
```

The preprint DOI should be replaced or supplemented with final journal bibliographic information once available; do not label the preprint DOI as a journal DOI.

## Related resources

- [GRID corpus paper](https://doi.org/10.1121/1.2229005): *An audio-visual corpus for speech perception and automatic speech recognition*.
- [Lombard corpus paper](https://doi.org/10.1121/1.5042758): *A corpus of audio-visual Lombard speech with frontal and profile views*.
- [CTC self-distillation reference](https://doi.org/10.21437/Interspeech.2024-363): source of the adapted CTC-SKD comparison, not an exact ASR reproduction.
- [MixSKD](https://doi.org/10.1007/978-3-031-20053-3_31): related image-recognition self-distillation work; it is not the implemented CTC-SKD comparison.
- [Online knowledge distillation via mutual contrastive learning](https://doi.org/10.1109/TPAMI.2023.3257878): related online-distillation work, not an additional experiment claimed here.
- [LRW-Persian](https://doi.org/10.48550/arXiv.2510.22716): isolated-word dataset used in the supplementary evaluation.

## Authors and contact

**Mehrad Yaghoubi** — [me.yaghoubi@iau.ir](mailto:me.yaghoubi@iau.ir)<br>
**Azam Bastanfard**, corresponding author — [bastanfard@iau.ac.ir](mailto:bastanfard@iau.ac.ir)

Affiliations:

1. Department of Computer Engineering, Ka.C., Islamic Azad University, Karaj, Iran.
2. Research Center of Interactive Technologies and Extended Reality Technologies, Islamic Azad University, Karaj Branch, Karaj, Iran.

## Licensing

The original EGOSD repository does not currently declare a repository-wide software license. Making a repository public does not by itself grant an unrestricted right to reuse or redistribute its code. An explicit license can be added once the copyright holders have selected its terms.

PyLipExtractor is a separate MIT-licensed package; its license does not automatically license EGOSD. Dataset terms and third-party package licenses remain separate. The DOI above identifies the research preprint and is not a software license.
