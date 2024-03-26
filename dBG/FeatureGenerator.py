import concurrent.futures
import ctypes
import time

import joblib
import numpy as np
import pandas as pd
import swalign
from tqdm import tqdm

from dBG.FeatureGraph import FeatureGraph
from dBG.utils.Substitute.LogOdds import LogOdds
from data_provider.m4 import M4Dataset
from sklearn.preprocessing import KBinsDiscretizer
from collections import Counter
import random

random.seed(2024)


class FeatureGenerator:


    def __init__(self, weight_threshold, **kwargs):
        self.dbg_params = kwargs
        self.sequence = self.dbg_params['sequences']
        self.weight_threshold = weight_threshold
        self.cuts = np.array([-np.inf, -1.64485362695147, -1.2815515655446,
                              -1.03643338949379, -0.841621233572914, -0.674489750196082,
                              -0.524400512708041, -0.385320466407568, -0.2533471031358,
                              -0.125661346855074, 0, 0.125661346855074, 0.2533471031358,
                              0.385320466407568, 0.524400512708041, 0.674489750196082,
                              0.841621233572914, 1.03643338949379, 1.2815515655446,
                              1.64485362695147])

    def generate_features(self):
        descretized_data = self.__bin_descretize()
        self.dbg_params['sequences'] = descretized_data
        sub_matrix = None
        if self.dbg_params['approximate']:
            sub_matrix = LogOdds(descretized_data)

        dbg = FeatureGraph(substitute=sub_matrix, **self.dbg_params)
        features = dbg.generate_features(self.weight_threshold)
        return features, descretized_data

    def __bin_descretize(self):
        flat_data = np.concatenate(self.sequence)
        discretizer = KBinsDiscretizer(n_bins=25, encode='ordinal', strategy='quantile')
        discrete_time_sequence = discretizer.fit_transform(flat_data.reshape(-1, 1))
        joblib.dump(discretizer, f'../dataset/Discretizer/{pattern}_discretizer_model.joblib')

        discrete_time_sequence = discrete_time_sequence.flatten().astype(int)
        # Count the frequency of each value in the discretized data
        distribution = Counter(discrete_time_sequence)

        # Convert the distribution to a more readable format
        print(dict(distribution))
        original_shapes = [array.shape for array in self.sequence]
        reshaped_data = []
        start = 0
        for shape in original_shapes:
            end = start + shape[0]
            reshaped_data.append(discrete_time_sequence[start:end].tolist())
            start = end
        return reshaped_data

    def __descretize(self, normalized_series):
        a_size = len(self.cuts)
        sax = list()
        for i in range(0, len(normalized_series)):
            num = normalized_series[i]
            if num >= 0:
                j = a_size - 1
                while (j > 0) and (self.cuts[j] >= num):
                    j = j - 1
                sax.append(j)
            else:
                j = 1
                while j < a_size and self.cuts[j] <= num:
                    j = j + 1
                sax.append(j - 1)
        return sax

    def __znorm(self, znorm_threshold=0.01):
        """Znorm implementation."""
        flat_data = np.concatenate(self.sequence)
        sd = np.std(flat_data)
        if sd < znorm_threshold:
            return flat_data
        mean = np.mean(flat_data)
        return (flat_data - mean) / sd

    def __transform_sequence(self):
        dat_znorm = self.__znorm()
        sax_flat_data = self.__descretize(dat_znorm)
        sax_flat_data = np.array(sax_flat_data)

        original_shapes = [array.shape for array in self.sequence]
        reshaped_data = []
        start = 0
        for shape in original_shapes:
            end = start + shape[0]
            reshaped_data.append(sax_flat_data[start:end].tolist())
            start = end
        return reshaped_data


def to_c_array(py_list):
    return (ctypes.c_int * len(py_list))(*py_list)


def tuple_to_string(t):
    return ''.join(chr(i) for i in t)


def process_row(row, features):
    row_str = to_c_array(row)
    csv_row = [row]
    for feature in features:
        feature_str = to_c_array(feature)
        score = levenshtein.levenshtein_distance(row_str, len(row_str), feature_str, len(feature_str))
        csv_row.append(score)
    return csv_row


def align(data, features):
    headers = ['Data'] + [feature for feature in features]

    # Selecting only 1% of the rows randomly
    sampled_data = random.sample(data, k=int(len(data) * 0.01))

    with concurrent.futures.ProcessPoolExecutor() as executor:
        # Note that [features] * len(sampled_data) is used to match the length of sampled_data
        results = list(tqdm(executor.map(process_row, sampled_data, [features] * len(sampled_data)), total=len(sampled_data)))

    df = pd.DataFrame(results, columns=headers)
    return df


def select_features(k, features):
    if len(features.columns) <= k:
        return features

    # Compute the correlation matrix
    correlation_matrix = features.corr().abs()

    # Initialize the list of selected features
    selected_features = []

    # Greedy algorithm to select features
    for _ in range(k):
        remaining_features = list(set(correlation_matrix.columns) - set(selected_features))
        avg_corr = pd.Series(index=remaining_features)

        for feature in remaining_features:
            if selected_features:
                avg_corr[feature] = correlation_matrix.loc[selected_features][feature].mean()
            else:
                avg_corr[feature] = 0

        # Select the feature with the least average correlation
        next_feature = avg_corr.idxmin()
        selected_features.append(next_feature)

    return selected_features

if __name__ == '__main__':
    feats_count = 15

    # Constants
    MATCH = 2
    MISMATCH = -1
    LEVENSHTEIN_SO_PATH = '/run/media/lumpus/HDD Storage/PycharmProjects/Time-Series-Library/levenshtein.so'

    # Initialize scoring matrix and alignment
    SCORING = swalign.NucleotideScoringMatrix(MATCH, MISMATCH)
    SW = swalign.LocalAlignment(SCORING)

    # Load the Levenshtein distance function from the shared library
    levenshtein = ctypes.CDLL(LEVENSHTEIN_SO_PATH)
    levenshtein.levenshtein_distance.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int,
                                                 ctypes.POINTER(ctypes.c_int), ctypes.c_int]
    levenshtein.levenshtein_distance.restype = ctypes.c_int

    seasonal_patterns = ['Yearly', 'Quarterly', 'Monthly', 'Weekly', 'Daily', 'Hourly']

    similarity_threshold = {
        'Yearly': 0.08,
        'Quarterly': 0.025,
        'Monthly': 0.003,
        'Weekly': 0,
        'Daily': -0.2,
        'Hourly': -0.4
    }

    weight_threshold = {
        'Yearly': 0.01,
        'Quarterly': 0.01,
        'Monthly': 0.01,
        'Weekly': 0,
        'Daily': 0.005,
        'Hourly': 0.01
    }

    approximate = {
        'Yearly': True,
        'Quarterly': True,
        'Monthly': True,
        'Weekly': False,
        'Daily': True,
        'Hourly': True
    }

    for pattern in seasonal_patterns:
        print(f'Reading {pattern} data...')
        m4 = M4Dataset.load(training=True, dataset_file='../dataset/m4')
        training_values = np.array([v[~np.isnan(v)] for v in m4.values[m4.groups == pattern]])
        data = [ts for ts in training_values]

        print('Dataset size: ', len(data))
        start = time.time()

        fg = FeatureGenerator(weight_threshold[pattern], k=3, sequences=data, approximate=approximate[pattern],
                              similarity_threshold=similarity_threshold[pattern])
        fg, disc = fg.generate_features()

        print('Picking best features...')


        print('FEATURES:', len(fg))

        reduced_features = set()

        """
        for feat in fg:
            if len(feat) >= 16 :
                reduced_features.add(feat)
        """
        if len(reduced_features) < feats_count:
            reduced_features = fg

        print('REDUCED FEATURES:', len(reduced_features))

        feats = align(disc, reduced_features)

        selected = select_features(48, feats)

        print(len(selected))

        end = time.time()
        print('EXECUTION TIME: ', end - start)

        with open(f'../dataset/features/{pattern}_features.txt', 'w') as file:
            for feature in selected:
                file.write(str(feature) + '\n')
