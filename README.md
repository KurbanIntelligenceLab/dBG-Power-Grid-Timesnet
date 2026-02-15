# Official Repository for An Extended Frequency-Improved Legendre Memory Model for Enhanced Long-Term Electricity Load Forecasting

## Setup

Download the cleaned dataset from: [[Google Drive]](https://drive.google.com/file/d/1OHLoUDPSramOB5hAx8BkW41BfABBPLPO/view?usp=sharing)

Unzip the csv files and make sure the files are placed under `dataset/MW` directory

### Create and activate a conda environment

```bash
conda create --name dBGTimesNet python=3.8.19
```

```bash
conda activate dBGTimesNet
```

## Install required packages
```bash
pip install -r requirements.txt
```
## Compile the C implementation for the Edit Distance algorithm:
```bash
gcc -shared -fPIC -o levenshtein.so levenshtein.c
```

## Preprocessing

To run preprocessing steps for all experiments run

```bash
python dBGPreprocess_MW.py
```

## Run all experiments
**Note:** Make sure the parameters in the preprocessing step and on the experiment scripts are matching.
```bash
./scripts/dBG_experiments/dbg_timesnet_testall_MW.sh
```
