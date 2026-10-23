# Four-Class White Blood Cell Classification with ResNet18

An MD6329 Deep Learning for Healthcare AI course project comparing final-block fine-tuning and light data augmentation for white blood cell image classification.

## Quick links

- [Project repository](https://github.com/EIEI886/MD6329-WBC-ResNet18)
- [Open the notebook in Google Colab](https://colab.research.google.com/github/EIEI886/MD6329-WBC-ResNet18/blob/main/MD6329_WBC_Colab.ipynb)
- [View the interactive LecTrace presentation](https://eiei886.github.io/MD6329-WBC-ResNet18/)
- [LecTrace Python source](MD6329_WBC_LecTrace/01_wbc_lectrace.py)

For immediate viewing, use the LecTrace website. For a fresh experiment, use the Colab notebook with a T4 GPU. For a presentation-only rebuild, run only the four code cells in notebook Section 10; no retraining, manual upload, or Drive mount is required.

## Repository contents

| Path | Purpose |
|---|---|
| `MD6329_WBC_Colab.ipynb` | Training, evaluation, analysis, and GitHub-based presentation setup |
| `MD6329_WBC_LecTrace/01_wbc_lectrace.py` | Four-state presentation source |
| `MD6329_WBC_LecTrace/_wbc_support.py` | State loading, optional checkpoint inference, and figure rendering |
| `MD6329_WBC_LecTrace/assets/` | Exported states, metrics, and figures used by the presentation |
| `MD6329_WBC_LecTrace/_site/` | Generated static viewer published through GitHub Pages |
| `MD6329_WBC_results.zip` | Supplementary full experiment exports, when uploaded |

The notebook's Section 10 is the current Colab setup guide; it supersedes older upload-and-Drive instructions in `README_FIRST.md`.

## Research question

Under a fixed training budget, does fine-tuning the final residual block of ResNet18 improve classification compared with training only the classification head? Does light augmentation provide an additional benefit?

## Dataset

We use a four-class subset of **BloodMNIST+ at 64×64 resolution**, derived from the Barcelona peripheral blood cell (PBC) dataset. The classes are eosinophil, lymphocyte, monocyte, and neutrophil. Images are resized to 224×224 and normalized using ImageNet channel statistics.

The training notebook downloads the dataset through the official MedMNIST package. Original dataset files do not need to be stored in this repository.

Original split assignments are retained after pixel-level SHA-256 deduplication. Consistently labeled duplicates retain one copy using the fixed priority **train → validation → test**; all copies with conflicting labels are excluded. Nine images were removed.

| Class | Training | Validation | Test |
|---|---:|---:|---:|
| Eosinophil | 2,178 | 310 | 622 |
| Lymphocyte | 849 | 122 | 243 |
| Monocyte | 992 | 143 | 284 |
| Neutrophil | 2,330 | 333 | 665 |
| **Total** | **6,349** | **908** | **1,814** |

These results describe a filtered four-class image-level benchmark, not the unmodified official benchmark.

## Experimental design

| Condition | Trainable modules | Random augmentation |
|---|---|---|
| A: `A_frozen_aug` | Classification head only | Enabled |
| B: `B_layer4_aug` | `layer4` and classification head | Enabled |
| C: `C_layer4_noaug` | `layer4` and classification head | Disabled |

- **Initialization:** ImageNet-pretrained ResNet18; matching randomly initialized classification heads within each seed. Runs are independent, rather than trained sequentially from one another.
- **Seeds:** 42 and 2026.
- **Training:** AdamW; batch size 64; weight decay 1e-4; classification-head learning rate 1e-3; unfrozen-layer learning rate 1e-4; no class weights.
- **Budget:** up to 8 epochs; early stopping after 3 consecutive epochs without improvement in validation macro-F1.
- **Augmentation:** random horizontal and vertical flips, each with probability 0.5, and rotation within ±15 degrees.
- **BatchNorm:** running statistics remain fixed in frozen modules and update in unfrozen `layer4`.
- **Selection:** best checkpoint per run by validation macro-F1; final condition by mean validation macro-F1 across seeds. Test results are not used to revise the configuration.

## Results

Test results are **mean ± sample standard deviation across two seeds**. Standard deviation describes training variability, not a confidence interval.

| Condition | Accuracy (%) | Macro-F1 (%) |
|---|---:|---:|
| A: Frozen backbone + augmentation | 94.29 ± 0.27 | 94.53 ± 0.17 |
| B: Final-block fine-tuning + augmentation | 99.48 ± 0.12 | 99.31 ± 0.14 |
| C: Final-block fine-tuning without augmentation | **99.53 ± 0.04** | **99.42 ± 0.04** |

Condition C was selected using validation performance. Under matching augmentation, B improved mean test macro-F1 over A by **4.78 percentage points**. C exceeded B by only **0.10 percentage points**; this small difference across two seeds does not establish statistical significance or a general disadvantage of augmentation. B had the highest mean AUROC, so C is not best on every metric.

Additional outputs include per-class recall, macro recall, macro one-vs-rest AUROC, learning curves, confusion matrices, individual predictions, and Grad-CAM error analysis.

## Run the training notebook

1. Open the project `.ipynb` notebook in Google Colab.
2. Select **Runtime → Change runtime type → T4 GPU**.
3. The notebook defaults to `SAVE_TO_DRIVE = True`, which connects your own Google Drive to retain training checkpoints and results. For a run without Drive authorization, change it to `False`; outputs then remain in the temporary Colab runtime.
4. Run the initial dependency installation cell and Sections 1–9 in order. Dependencies, the dataset, and ImageNet initialization weights are downloaded automatically.

Saved outputs come from the original Colab experiments and preview runs. The presentation download and build cells were updated for sharing; the training configuration and model implementation are unchanged. Rebuilding LecTrace does not require repeating the six training runs.

The notebook does not require access to the authors' private Drive. Fresh execution retrains the six predefined runs; previously saved runs are reused only when their files are available in the configured output directory. With Drive disabled, download results and checkpoints before the Colab runtime ends. Section 9 exports the results ZIP but excludes `.pt` weights; download any checkpoints you want to retain separately. A new training configuration requires a separate output directory.

## LecTrace presentation

The `MD6329_WBC_LecTrace` folder contains the presentation script, support module, real exported states, figures, instructions, and a prebuilt static presentation.

The presentation explains four selected forward-pass states:

1. Normalized input: `[1, 3, 224, 224]`.
2. Final convolutional features: `[1, 512, 7, 7]`.
3. Global average pooling: a 512-dimensional feature vector.
4. Classification: logits → softmax → argmax.

It concludes with the experimental comparison and fixed-rule error examples. The average feature map is distinct from predicted-class Grad-CAM.

From the repository root, run:

```bash
cd MD6329_WBC_LecTrace
python3 -m pip install -r requirements.txt
lectrace serve 01_wbc_lectrace.py --port 7000
```

Open `http://localhost:7000`. For Colab, follow Section 10 of `MD6329_WBC_Colab.ipynb`.

**Execution modes:** Without a checkpoint, the portable presentation replays actual recorded convolutional features and logits while recomputing pooling, softmax, and argmax. With the original C/seed42 checkpoint and compatible PyTorch/torchvision installed, it recomputes a CPU forward pass. The presentation explicitly labels the mode. The published viewer displays states captured at build time; opening the website does not rerun Python. A clean Colab rebuild without the original checkpoint uses recorded-state replay. Original-checkpoint CPU inference was verified in the authors' Colab preview, and recorded-state execution and static building were checked during package preparation. No retraining is needed to view the presentation.

## Limitations

This study uses one data source, four selected cell types, two seeds, and a limited training budget. Some best checkpoints occur at the epoch limit, so fully converged performance has not been established. Resizing cannot recover details lost at 64×64 resolution. Patient independence, near-duplicates, external generalization, and probability calibration have not been verified. Grad-CAM visualizations do not establish clinically valid reasoning. The project evaluates cell classification, not disease diagnosis or clinical deployment.

## Data, methods, and software sources

- Acevedo et al. (2020). *A dataset of microscopic peripheral blood cell images for development of automatic recognition systems.* https://doi.org/10.1016/j.dib.2020.105474
- Yang et al. (2023). *MedMNIST v2—A large-scale lightweight benchmark for 2D and 3D biomedical image classification.* https://doi.org/10.1038/s41597-022-01721-8
- He et al. (2016). *Deep Residual Learning for Image Recognition.* https://arxiv.org/abs/1512.03385
- Selvaraju et al. (2017). *Grad-CAM: Visual Explanations from Deep Networks via Gradient-Based Localization.* https://arxiv.org/abs/1610.02391
- MedMNIST data and package: https://medmnist.com/ and https://github.com/MedMNIST/MedMNIST
- torchvision ResNet18: https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html
- LecTrace: https://pypi.org/project/lectrace/1.1.30/

Course connections include Lecture 2, PDF pp.107–117 and 123–124 (transfer learning and independent samples), and Lecture 11, PDF pp.30–33 (Grad-CAM).

## AI assistance

AI assisted with code drafting, debugging, documentation, and presentation preparation. Experiments were executed in Colab. Exported predictions, labels, split counts, and classification metrics were cross-checked for consistency. The authors remain responsible for understanding and validating the submitted work.
