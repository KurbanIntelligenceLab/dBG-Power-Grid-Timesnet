# Power Grid Project


## Setup

Make sure the data is placed under `dataset/MW` directory with a csv format.

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