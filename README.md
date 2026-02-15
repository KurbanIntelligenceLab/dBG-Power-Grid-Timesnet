# Official Repository for An Extended Frequency-Improved Legendre Memory Model for Enhanced Long-Term Electricity Load Forecasting

Link to the [[Publication]](https://ieeexplore.ieee.org/abstract/document/11184177)

## Cite our work:
```bib
@ARTICLE{11184177,
  author={Onur Cakiroglu, Mert and Bilge Altun, Idil and Rahman Fahim, Shahriar and Kurban, Hasan and Dalkilic, Mehmet M. and Atat, Rachad and Takiddin, Abdulrahman and Serpedin, Erchin},
  journal={IEEE Open Access Journal of Power and Energy}, 
  title={An Extended Frequency-Improved Legendre Memory Model for Enhanced Long-Term Electricity Load Forecasting}, 
  year={2025},
  volume={12},
  number={},
  pages={691-701},
  keywords={Power grids;Load forecasting;Time series analysis;Predictive models;Substations;Feature extraction;Transformers;Electricity supply industry;Power grids;Sequential analysis;Load forecasting;de Bruijn graphs;time series analysis;graph encoding;struct2vec;power grid;sequential data modeling;feature extraction},
  doi={10.1109/OAJPE.2025.3615513}}

```

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
