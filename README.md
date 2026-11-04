# AGE-DeepCox

AGE-DeepCox is an interpretable survival-analysis framework for estimating continuous age-dependent gene effects while retaining complementary nonlinear prognostic information.

The framework was developed for high-dimensional cancer transcriptomic data and evaluated on the TCGA kidney renal clear cell carcinoma (KIRC), lung adenocarcinoma (LUAD), and bladder urothelial carcinoma (BLCA) cohorts.

Gene-expression and clinical data were obtained and preprocessed as described in the manuscript. This repository expects a processed sample-by-gene expression matrix and a clinical survival table as input. Data acquisition and upstream transcriptomic preprocessing are not included in this repository.

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

No TCGA-derived example or test dataset is included in this repository.

## Training AGE-DeepCox

Train one cohort with:

```bash
python train.py single \
  --expression /path/to/KIRC_expression.csv \
  --clinical /path/to/KIRC_clinical.csv \
  --config configs.yaml \
  --output outputs/kirc/run_01 \
  --seed 2027
```

Replace the input paths and output directory for LUAD or BLCA. All three cohorts use the same model configuration.

The configuration file records the reported model width, sparse penalties, repeat count, and random seed. Fixed optimization details are defined in `optimizer.py`.

The training procedure:

1. creates a stratified 70:30 training-validation split;
2. standardizes gene expression and age using training-set statistics;
3. residualizes each standardized age-by-gene feature against its corresponding gene feature using the training set;
4. alternates proximal updates of the explicit coefficients with Adam updates of the nonlinear branch and learnable gate;
5. calculates risk scores for the training and held-out validation sets; and
6. saves the fitted model, coefficients, predictions, and fit summary.

Tied event times in the Cox partial likelihood are handled using the Breslow approximation.

## Repeated evaluation

The study used 50 repeated stratified random splits. Stratification jointly considered event status and follow-up time dichotomized at the cohort median. Run the repeated analysis with:

```bash
python train.py repeated \
  --expression /path/to/KIRC_expression.csv \
  --clinical /path/to/KIRC_clinical.csv \
  --config configs.yaml \
  --repeats 50 \
  --output outputs/kirc/repeated
```

Repeated training also creates `predictions.csv` and `coefficients.csv` by combining the corresponding outputs from all runs. Evaluate the combined predictions with:

```bash
python evaluate.py \
  --predictions outputs/kirc/repeated/predictions.csv \
  --output outputs/kirc/evaluation
```

Predictive performance is evaluated on the held-out validation sets using:

- Harrell's concordance index;
- cumulative/dynamic AUC at one, three, and five years; and
- survival-function-weighted mean AUC from one to five years.

AUC estimates use inverse probability of censoring weighting. Metrics are summarized as mean and standard deviation across the repeated splits.

## Gene-selection stability

Reference-age and age-dependent modification effects are summarized separately across the 50 runs. A gene is considered stable when it is selected in more than half of the runs.

```bash
python train.py stability \
  --coefficients outputs/kirc/repeated/coefficients.csv \
  --output outputs/kirc/selection_stability.csv
```

The resulting table reports reference-effect and modification-effect selection frequencies, mean coefficients, and indicators for frequencies above 0.5.

## Training-set age-permutation analysis

The age-permutation negative control preserves the expression values, survival outcomes, validation observations, and original split assignments. Age is permuted only among the training patients within each repeated split, after which all age-dependent preprocessing quantities and model parameters are re-estimated.

```bash
python train.py age-permutation \
  --expression /path/to/KIRC_expression.csv \
  --clinical /path/to/KIRC_clinical.csv \
  --config configs.yaml \
  --repeats 50 \
  --output outputs/kirc/age_permutation
```

The command produces fitted coefficients and risk predictions under permuted training ages. Run `evaluate.py` on the combined `predictions.csv` to calculate the corresponding performance metrics.

## Output files

A repeated-training directory contains:

```text
run_001.pt
run_001_coefficients.csv
run_001_predictions.csv
coefficients.csv
predictions.csv
runs.csv
```

- `run_001.pt`: fitted model state, preprocessing parameters, and fit summary
- `run_001_coefficients.csv`: reference-age and age-modification coefficients
- `run_001_predictions.csv`: training and held-out validation risk scores
- `coefficients.csv`: coefficients combined across repeated runs
- `predictions.csv`: predictions combined across repeated runs
- `runs.csv`: run seeds, convergence summaries, and output paths
