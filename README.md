# A Computationally Efficient Multi-Section Whole Slide Image Analysis for Capturing Prostate Cancer Multifocality in Biochemical Recurrence Prediction

This repository provides the code for a computationally efficient **multi-section whole-slide image (WSI)** analysis
framework that predicts **biochemical recurrence (BCR)** after radical prostatectomy using an
**attention-based multiple-instance learning (MIL)** model.

The MIL backbone is adapted from [MCAT](https://github.com/mahmoodlab/MCAT/tree/master).
Preprint: [arXiv:2603.20273](https://arxiv.org/abs/2603.20273)

---

## Pipeline overview

The full workflow consists of four stages:

1. **Feature extraction** — extract patch-level features from multi-section WSIs with the UNI foundation model.
2. **Training** — train the attention-based MIL model with 5-fold cross-validation on the development cohort.
3. **Risk-score generation** — apply the trained (ensembled) models to produce a per-patient AI risk score.
4. **Evaluation** — quantify discrimination, prognostic value, calibration, and clinical utility.

---

## Repository structure

```
Code_available/
├── feature_extract_PNU_UNI.py          # Stage 1: UNI patch-feature extraction
├── PNU_train_double.py                 # Stage 2: MIL training (5-fold CV)
├── train_test_risk_score_generation.ipynb   # Stage 3: per-patient risk-score generation
├── Final_evaluation.ipynb              # Stage 4: evaluation (AUC, C-index, KM, Cox, calibration, DCA)
├── models/
│   ├── model_double.py                 # Attention-based MIL (ABMIL) with class-specific branches
├── datasets/
│   └── dataset_pfs_pnu.py              # Dataset / dataloader definitions
├── utils/
│   └── focal_loss.py                   # Focal loss + class-weight utilities
├── folds/                              # Train/val/test patient splits (5-fold CV)
└── requirements.yml                    # Conda environment
```

---

## Environment

**Option A — Conda**
```bash
conda env create -f requirements.yml
conda activate MIL_recur
```

**Option B — uv** (PyTorch 2.0.1 + CUDA 11.7)
```bash
uv venv --python 3.11 .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
uv pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 \
    --index-url https://download.pytorch.org/whl/cu117
uv pip install numpy==1.26.4 pandas==2.0.3 h5py==3.13.0 scikit-learn==1.6.1 \
    scipy==1.13.1 tqdm==4.65.0 tensorboardX==1.9 protobuf==3.20.3
```

---

## Stage 1 — Feature extraction

After tiling each WSI, extract deep features from a pretrained pathology foundation model.
Here we use [UNI](https://www.nature.com/articles/s41591-024-02857-3). Coordinates are read from `.h5`
files and the corresponding slides, and one feature file is written per slide.

```bash
python feature_extract_PNU_UNI.py \
    --root_h5     [COORDINATES_DIR] \
    --root_wsi    [SLIDES_DIR] \
    --output_dir  [FEATURES_DIR] \
    --model_dir   [FEATURE_EXTRACTOR_DIR] \
    --patch_size  [PATCH_SIZE]
```

---

## Stage 2 — Training

Train the attention-based MIL model on the development cohort using the 5-fold splits in `folds/`.

```bash
python PNU_train_double.py \
    --dropout 0.25 \
    --writer_dir ./tensorboard_log \
    --result_df_path results_dataframe \
    --weights_dir weights_saving \
    --lr 5e-6 --wd 5e-7 --epoch 100 --gc 16
```

Key arguments: `--lr` learning rate, `--wd` weight decay, `--epoch` number of epochs,
`--gc` gradient-accumulation steps, `--dropout` dropout rate. Model weights are saved per fold to `--weights_dir`.

---

## Stage 3 — Risk-score generation

`train_test_risk_score_generation.ipynb` loads the trained fold models, runs inference on the
internal test and external (CHIMERA) cohorts, and **ensembles the five folds** to produce a single
per-patient AI risk score. The resulting scores are saved to a dataframe used by the evaluation notebook.

---

## Stage 4 — Evaluation

`Final_evaluation.ipynb` reproduces the paper's evaluation from the generated risk scores:

- **Discrimination** — ROC/AUC (with bootstrap 95% CIs) and Harrell's C-index; pairwise DeLong tests against the clinical benchmarks (CAPRA-S, XGBoost).
- **Risk stratification** — Kaplan–Meier recurrence-free survival with the log-rank test, using the median development-cohort risk score as a pre-specified threshold.
- **Independent prognostic value** — multivariable Cox proportional-hazards regression (hazard ratios with 95% CIs).
- **Calibration** — reliability curves with the Brier score and Brier skill score (95% bootstrap CIs).
- **Clinical utility** — decision curve analysis (net benefit vs. treat-all / treat-none).

---

## Model weights:
The trained model weights are available from the corresponding author upon reasonable request.

## Data availability

The internal dataset contains sensitive patient information and cannot be shared publicly, but de-identified data may be made available from the corresponding author upon reasonable request and subject to institutional/IRB approval. The external validation cohort (CHIMERA) is publicly available at <https://chimera.grand-challenge.org/>.

## Citation

If you use this code, please cite the paper above.

## License

This work is licensed under a
[Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International License (CC BY-NC-ND 4.0)](https://creativecommons.org/licenses/by-nc-nd/4.0/).

[![License: CC BY-NC-ND 4.0](https://img.shields.io/badge/License-CC%20BY--NC--ND%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc-nd/4.0/)

Under this license, you are free to **share** — copy and redistribute the material in any medium or format — under the following terms:

- **Attribution** — You must give appropriate credit, provide a link to the license, and indicate if changes were made.
- **NonCommercial** — You may not use the material for commercial purposes.
- **NoDerivatives** — If you remix, transform, or build upon the material, you may not distribute the modified material.

See the [full license text](https://creativecommons.org/licenses/by-nc-nd/4.0/legalcode) for details.
