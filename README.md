# AGE-DeepCox

AGE-DeepCox is an interpretable survival-analysis framework for estimating continuous age-dependent gene effects while retaining complementary nonlinear prognostic information.

The framework was developed for high-dimensional cancer transcriptomic data and evaluated on the TCGA kidney renal clear cell carcinoma (KIRC), lung adenocarcinoma (LUAD), and bladder urothelial carcinoma (BLCA) cohorts.

## Overview

Many survival models assume that the prognostic effect of a gene remains constant across patients. AGE-DeepCox instead represents the effect of gene *j* for patient *i* as

```text
theta_j(age_i) = theta_j^R + z_i * theta_j^M,
```

where `theta_j^R` is the reference-age effect, `theta_j^M` is the continuous age-dependent modification effect, and `z_i` is age standardized using the training-set mean and standard deviation.

AGE-DeepCox contains three main components:

1. **Continuous age-dependent gene-effect decomposition.** Each gene has a reference-age effect and an age-dependent modification effect. The model also includes a separate overall age effect.
2. **Hierarchical sparse selection.** The age-dependent modification component is nested within the corresponding reference component, so an age-modified gene can only be retained when its reference effect is also retained.
3. **Orthogonalized nonlinear residual learning.** Explicit patient-level gene contributions are passed to a multilayer perceptron. Its output is centered, orthogonalized with respect to the explicit risk score, and scaled by a learnable gate before being added to the final risk score.

## Repository structure

The public release is organized as follows:

```text
AGE-DeepCox/
|-- README.md
|-- requirements.txt
|-- train.py
|-- evaluate.py
|-- model.py
|-- optimizer.py
`-- configs.yaml
```

The repository contains the AGE-DeepCox implementation and the analysis code needed for repeated evaluation, gene-selection stability analysis, and the training-set age-permutation negative control.

## Requirements

The implementation requires Python 3.10 or later. The principal dependencies are:

- NumPy 1.26 or later
- pandas 2.1 or later
- PyTorch 2.2 or later
- scikit-learn 1.4 or later
- scikit-survival 0.23 or later
- PyYAML 6.0 or later

Install the dependencies with:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

On Windows PowerShell, activate the environment with:

```powershell
.venv\Scripts\Activate.ps1
```

The model can run on CPU. A CUDA-capable GPU may reduce training time but is not required.

## Data availability

The TCGA expression and clinical data used in the study are available through the [UCSC Xena platform](https://xena.ucsc.edu/). Patient-level data are not redistributed in this repository. Users are responsible for downloading the data and complying with the applicable data-use conditions.

RNA-seq expression values in the study were represented as `log2(TPM + 1)`. Lowly expressed genes were removed, after which differential expression between primary tumor and normal tissue was assessed using limma-trend with Benjamini-Hochberg correction. Genes satisfying adjusted `P < 0.05` and `|log2 fold change| > 1` were ranked by adjusted P value, and the top 1,500 genes were retained for each cohort.

After matching expression and clinical records and removing samples with incomplete survival or age information, the study included:

| Cohort | Patients | Events |
|---|---:|---:|
| TCGA-KIRC | 525 | 173 |
| TCGA-LUAD | 490 | 177 |
| TCGA-BLCA | 398 | 176 |

## Input data

AGE-DeepCox accepts two comma-separated files: a sample-by-gene expression matrix and a clinical survival table. Samples are aligned across the two files by `sampleID`.

### Gene-expression matrix

The expression file contains one sample per row and one gene per column:

```csv
sampleID,GeneA,GeneB,GeneC
S001,1.25,0.31,2.04
S002,0.72,1.18,1.76
```

`sampleID` must be unique. All gene columns must contain finite numeric values. Expression values should already have undergone the preprocessing and gene-selection procedure described above.

### Clinical survival table

The clinical file contains the survival outcome and age at diagnosis for each sample:

```csv
sampleID,time,status,age
S001,350,1,58
S002,420,0,71
```

- `sampleID`: unique identifier matching a row in the expression matrix
- `time`: observed overall-survival or censoring time in days; values must be positive
- `status`: event indicator, coded as `1` for death and `0` for censoring
- `age`: age at diagnosis in years, modeled as a continuous variable

The input files must contain no missing values in the model columns. Age is standardized using the training-set mean and standard deviation within each split; it is not dichotomized into younger and older subgroups.

## Quick Start

Prepare the expression matrix and clinical survival table in the formats shown above, then run:

```bash
python train.py single \
  --expression /path/to/expression.csv \
  --clinical /path/to/clinical.csv \
  --config configs.yaml \
  --output outputs/run_01 \
  --seed 2027
```

Here:

- `--expression` specifies the sample-by-gene expression matrix;
- `--clinical` specifies the survival table matched by `sampleID`;
- `--config` provides the model width and sparse penalties;
- `--output` specifies where the results will be saved; and
- `--seed` controls the data split and model initialization.

The output directory contains the fitted model, estimated reference and age-modification coefficients, risk predictions for the training and validation sets, and a fit summary.

For repeated training, gene-selection stability, age-permutation analysis, predictive evaluation, and the full command-line options, run:

```bash
python train.py -h
python evaluate.py -h
```

Implementation details are provided in `model.py`, `optimizer.py`, `train.py`, and `evaluate.py`.
