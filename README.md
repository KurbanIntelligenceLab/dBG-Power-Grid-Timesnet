# A Novel Discrete Time Series Representation with De Bruijn Graphs for Enhanced Forecasting using TimesNet

This repository extends the [Time Series Library (TSlib)](https://github.com/thuml/Time-Series-Library) with our de Bruijn Graph (dBG)-based representation for time series forecasting. Our work integrates with the TimesNet model to enhance prediction accuracy. Additionally, we include the implementation of [struct2vec](https://github.com/leoribeiro/struc2vec) for structural feature extraction. Please check respective repositories for more information on how they work.

# How to Reproduce the Results

This section outlines the steps needed to recreate the experiments described in our paper and supplementary materials.

## Set up
Download the M4 dataset from the original repository and place it inside the  `dataset\` directory in the root of this repository.

Install the required Python packages with:
```sh
pip install -r requirements.txt
```

Compile the C implementation for the Edit Distance algorithm:

```
gcc -shared -fPIC -o levenshtein.so levenshtein.c
```

## Preprocessing
Before running the experiments, preprocess the data using our dBG implementation and struct2vec. This can be done with the following script, which will generate and store the dBG features in the `dataset\` directory:


```sh
python dBGPreprocess.py
```

## Running the Experiments
To run all experiments on a single GPU, use the following command:

```sh
./scripts/dBG_experiments/dbg_timesnet_testall.sh
```

For a multi-GPU environment, we provide a script optimized for a 4-GPU setup. Adjust the `CUDA_VISIBLE_DEVICES` variable within the scripts if you are using a different number of GPUs:

```sh
./scripts/dBG_experiments/split/run_all.sh
```

### ARIMA
To run and evaluate the arima baseline:
```sh
python arima.py
```
