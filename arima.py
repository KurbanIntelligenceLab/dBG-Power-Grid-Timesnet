from statsmodels.tsa.arima.model import ARIMA
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import warnings

from data_provider.m4 import M4Dataset


from collections import OrderedDict

import numpy as np
import pandas as pd

from data_provider.m4 import M4Meta
import os


def group_values(values, groups, group_name):
    return np.array([v[~np.isnan(v)] for v in values[groups == group_name]])


def mase(forecast, insample, outsample, frequency):
    return np.mean(np.abs(forecast - outsample)) / np.mean(np.abs(insample[:-frequency] - insample[frequency:]))


def smape_2(forecast, target):
    denom = np.abs(target) + np.abs(forecast)
    # divide by 1.0 instead of 0.0, in case when denom is zero the enumerator will be 0.0 anyway.
    denom[denom == 0.0] = 1.0
    return 200 * np.abs(forecast - target) / denom


def mape(forecast, target):
    denom = np.abs(target)
    # divide by 1.0 instead of 0.0, in case when denom is zero the enumerator will be 0.0 anyway.
    denom[denom == 0.0] = 1.0
    return 100 * np.abs(forecast - target) / denom


def clean_nan_vals(arr, indexes):
    if len(indexes) == 0:
        return arr
    return np.delete(arr, indexes, axis=0)

class M4Summary:
    def __init__(self, file_path, root_path):
        self.file_path = file_path
        self.training_set = M4Dataset.load(training=True, dataset_file=root_path)
        self.test_set = M4Dataset.load(training=False, dataset_file=root_path)
        self.naive_path = os.path.join(root_path, 'submission-Naive2.csv')

    def evaluate(self):
        """
        Evaluate forecasts using M4 test dataset.

        :param forecast: Forecasts. Shape: timeseries, time.
        :return: sMAPE and OWA grouped by seasonal patterns.
        """
        grouped_owa = OrderedDict()

        naive2_forecasts = pd.read_csv(self.naive_path).values[:, 1:].astype(np.float32)
        naive2_forecasts = np.array([v[~np.isnan(v)] for v in naive2_forecasts])

        model_mases = {}
        naive2_smapes = {}
        naive2_mases = {}
        grouped_smapes = {}
        grouped_mapes = {}
        for group_name in M4Meta.seasonal_patterns:
            file_name = self.file_path + group_name + "_forecast.csv"
            if os.path.exists(file_name):
                model_forecast = pd.read_csv(file_name).values
                

            naive2_forecast = group_values(naive2_forecasts, self.test_set.groups, group_name)
            target = group_values(self.test_set.values, self.test_set.groups, group_name)
            # all timeseries within group have same frequency
            frequency = self.training_set.frequencies[self.test_set.groups == group_name][0]
            insample = group_values(self.training_set.values, self.test_set.groups, group_name)

            nan_rows = np.where(np.isnan(model_forecast).any(axis=1))[0]
            model_forecast = clean_nan_vals(model_forecast, nan_rows)
            naive2_forecast = clean_nan_vals(naive2_forecast, nan_rows)
            target = clean_nan_vals(target, nan_rows)

            model_mases[group_name] = np.mean([mase(forecast=model_forecast[i],
                                                    insample=insample[i],
                                                    outsample=target[i],
                                                    frequency=frequency) for i in range(len(model_forecast))])
            naive2_mases[group_name] = np.mean([mase(forecast=naive2_forecast[i],
                                                     insample=insample[i],
                                                     outsample=target[i],
                                                     frequency=frequency) for i in range(len(model_forecast))])

            naive2_smapes[group_name] = np.mean(smape_2(naive2_forecast, target))
            grouped_smapes[group_name] = np.mean(smape_2(forecast=model_forecast, target=target))
            grouped_mapes[group_name] = np.mean(mape(forecast=model_forecast, target=target))

        grouped_smapes = self.summarize_groups(grouped_smapes)
        grouped_mapes = self.summarize_groups(grouped_mapes)
        grouped_model_mases = self.summarize_groups(model_mases)
        grouped_naive2_smapes = self.summarize_groups(naive2_smapes)
        grouped_naive2_mases = self.summarize_groups(naive2_mases)
        for k in grouped_model_mases.keys():
            grouped_owa[k] = (grouped_model_mases[k] / grouped_naive2_mases[k] +
                              grouped_smapes[k] / grouped_naive2_smapes[k]) / 2

        def round_all(d):
            return dict(map(lambda kv: (kv[0], np.round(kv[1], 3)), d.items()))

        return round_all(grouped_smapes), round_all(grouped_owa), round_all(grouped_mapes), round_all(
            grouped_model_mases)

    def summarize_groups(self, scores):
        """
        Re-group scores respecting M4 rules.
        :param scores: Scores per group.
        :return: Grouped scores.
        """
        scores_summary = OrderedDict()

        def group_count(group_name):
            return len(np.where(self.test_set.groups == group_name)[0])

        weighted_score = {}
        for g in ['Yearly', 'Quarterly', 'Monthly']:
            weighted_score[g] = scores[g] * group_count(g)
            scores_summary[g] = scores[g]

        others_score = 0
        others_count = 0
        for g in ['Weekly', 'Daily', 'Hourly']:
            others_score += scores[g] * group_count(g)
            others_count += group_count(g)
        weighted_score['Others'] = others_score
        scores_summary['Others'] = others_score / others_count

        average = np.sum(list(weighted_score.values())) / len(self.test_set.groups)
        scores_summary['Average'] = average

        return scores_summary


# Seed for reproducibility
np.random.seed(2024)

# Settings
seasonal_patterns = ['Monthly', 'Quarterly', 'Yearly', 'Weekly', 'Daily', 'Hourly']
warnings.simplefilter("ignore")
horizons_map = {
    'Yearly': 6,
    'Quarterly': 8,
    'Monthly': 18,
    'Weekly': 13,
    'Daily': 14,
    'Hourly': 48
}

def get_data(pattern, training=True):
    m4 = M4Dataset.load(training=training, dataset_file='dataset/m4')
    values = [v[~np.isnan(v)] for v in m4.values[m4.groups == pattern]]
    data = [ts for ts in values]
    return data

def test(file_path):
    m4_summary = M4Summary(file_path, 'dataset/m4')
    smape_results, owa_results, mape, mase = m4_summary.evaluate()
    print(f'smape: {smape_results}\n')
    print(f'mase: {mase}\n')
    print(f'owa: {owa_results}\n')


def main():
    results_dir = 'm4_results/arima/'

    os.makedirs(results_dir, exist_ok=True)
    for pattern in seasonal_patterns:
        print(f'Now loading data {pattern}...')
        sequences = get_data(pattern)
                
        horizon = horizons_map[pattern]
        forecasts = []
        
        # Apply best parameters to all sequences
        for sequence in tqdm(sequences, desc=f"Forecasting {pattern}"):
            try:
                model = ARIMA(sequence)
                fitted_model = model.fit()
                forecast = fitted_model.forecast(steps=horizon)
                forecasts.append(forecast)
            except Exception as e:
                print(f"Error fitting model on sequence: {sequence}: {str(e)}")
                forecasts.append([np.nan] * horizon)

        forecast_df = pd.DataFrame(forecasts)
        forecast_df.columns = [f'V{i+1}' for i in range(horizon)]
        forecast_df.to_csv(os.path.join(results_dir, f'{pattern}_forecast.csv'), index=False)
    test(results_dir)
    
if __name__ == "__main__":
    main()