import concurrent.futures
import copy
import ctypes
import os
import random
import subprocess
import time
import warnings
from collections import Counter
import glob
import joblib
import networkx as nx
import numpy as np
import pandas as pd
import swalign
from data_provider.m4 import M4Dataset
from sklearn.preprocessing import KBinsDiscretizer
from tqdm import tqdm

from dBG.FeatureGraph import FeatureGraph
from dBG.utils.Substitute.LogOdds import LogOdds

warnings.filterwarnings('ignore')
# 543490


# Constants
MATCH = 2
MISMATCH = -1
LEVENSHTEIN_SO_PATH = './levenshtein.so'

# Initialize scoring matrix and alignment
SCORING = swalign.NucleotideScoringMatrix(MATCH, MISMATCH)
SW = swalign.LocalAlignment(SCORING)

# Load the Levenshtein distance function from the shared library
levenshtein = ctypes.CDLL(LEVENSHTEIN_SO_PATH)
levenshtein.levenshtein_distance.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int,
                                             ctypes.POINTER(ctypes.c_int), ctypes.c_int]
levenshtein.levenshtein_distance.restype = ctypes.c_int


def clear_files(path):
    files = glob.glob(os.path.join(path, '*'))
    print('Removing:', files)
    for file in files:
        if os.path.isfile(file):
            os.remove(file)


def bin_descretize(sequence, discretizer):
    flat_data = np.concatenate(sequence)
    discrete_time_sequence = discretizer.fit_transform(flat_data.reshape(-1, 1))

    discrete_time_sequence = discrete_time_sequence.flatten().astype(int)
    # Count the frequency of each value in the discretized data
    distribution = Counter(discrete_time_sequence)

    # Convert the distribution to a more readable format
    print(dict(distribution))
    original_shapes = [array.shape for array in sequence]
    reshaped_data = []
    start = 0
    for shape in original_shapes:
        end = start + shape[0]
        reshaped_data.append(discrete_time_sequence[start:end].tolist())
        start = end
    return reshaped_data


def extract_and_save_graph_emb(pattern, graph_path, emb_path, graph_dim_size):
    clear_files('./struc2vec-master/pickles')
    input_path = os.path.join(graph_path, f'{pattern}_edges.txt')
    output_path = os.path.join(emb_path, f'{pattern}_emb.txt')
    if os.path.exists(output_path):
        return  # return if already generated

    # Define the command and parameters
    command = 'python'
    script_path = 'struc2vec-master/src/main.py'
    params = [
        '--input',
        input_path,
        '--output',
        output_path,
        '--weighted',
        '--directed',
        '--workers', '32',
        '--dimensions', f'{graph_dim_size}',
        '--OPT1', 'true',
        '--OPT2', 'true',
        '--OPT3', 'true'
    ]

    # Combine command and parameters
    full_command = [command, script_path] + params

    start = time.time()
    try:
        # Run the command
        subprocess.run(full_command, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Command '{' '.join(full_command)}' failed with error: {e}")
        raise
    end = time.time()

    print(f'Time Cost {(end - start):.2f}s')


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


def to_c_array(py_list):
    return (ctypes.c_int * len(py_list))(*py_list)


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
        results = list(tqdm(executor.map(process_row, sampled_data, [features] * len(sampled_data)),
                            total=len(sampled_data)))

    df = pd.DataFrame(results, columns=headers)
    return df


def generate_and_save_graph(pattern, disc_path, graph_path, data, discretizer, k, approximate):
    joblib.dump(discretizer, os.path.join(disc_path, f'{pattern}_discretizer_model.joblib'))
    print(len(data))
    print('Dataset size: ', len(data))

    sub_matrix = LogOdds(data)
    dbg = FeatureGraph(k=k, sequences=data, approximate=approximate, substitute=sub_matrix,
                       similarity_threshold=0.2)
    G = dbg.graph
    print(G)
    node_label_mapping = {node: i for i, node in enumerate(G.nodes)}
    G = nx.relabel_nodes(G, node_label_mapping)

    print('Saving graph...')
    with open(os.path.join(graph_path, f'{pattern}_edges.txt'), 'w') as f:
        for u, v in tqdm(G.edges()):
            for _ in range(int(G[u][v].get('weight', 1))):
                f.write(f"{u} {v}\n")

    # Save the node label mapping to a separate file
    mapping_file_path = os.path.join(graph_path, f'{pattern}_nodes.joblib')
    joblib.dump(node_label_mapping, mapping_file_path)
    print(f'Node label mapping saved to {mapping_file_path}')
    return dbg


def generate_and_save_proto_features(pattern, motif_path, motifs, data, motif_count):
    save_path = os.path.join(motif_path, f'{pattern}_features.txt')
    if os.path.exists(save_path):
        return  # return if already generated

    print('Total motifs:', len(motifs))
    if len(motifs) < motif_count:
        raise Exception('Unable generate enough proto-features')

    proto_features = align(data, motifs)

    proto_features = select_features(motif_count, proto_features)

    print(f'Selected motifs: {len(proto_features)}')
    with open(save_path, 'w') as file:
        for feature in proto_features:
            file.write(str(feature) + '\n')


def parse_data(raw_file_path, test_size=1440):
    sequences = pd.read_csv(raw_file_path)
    sequences = sequences.loc[:, ~(sequences.isna().any() | (sequences == 0).all())]

    busses = list(sequences.columns)
    sequences = sequences.to_numpy()
    train_sequences = sequences[:-test_size].T
    test_sequences = sequences[-test_size:].T

    train_filepath = os.path.join(os.path.dirname(raw_file_path), 'training.npy')
    test_filepath = os.path.join(os.path.dirname(raw_file_path), 'test.npy')
    info_filepath = os.path.join(os.path.dirname(raw_file_path), 'M4-info.csv')

    info_data = {
        "M4id": busses,
        "category": ["Other"] * len(busses),
        "Frequency": [1] * len(busses),
        "Horizon": [10] * len(busses),
        "SP": ["MW"] * len(busses)
    }
    info_df = pd.DataFrame(info_data)
    info_df.to_csv(info_filepath, index=False)

    np.save(train_filepath, train_sequences)
    np.save(test_filepath, test_sequences)


def main():
    pattern='MW'
    k_params = [4, 5, 6]
    alphabet_params = [15, 20, 25]
    graph_dim_params = [16, 32, 64, 128]
    approximate = True
    parse_data('dataset/MW/Cleaned_ACTIVISg2000_load_time_series_MW.csv')
    for alphabet in alphabet_params:
        disc_path = f'dataset/MW/Discretizer/{alphabet}Disc'
        os.makedirs(disc_path, exist_ok=True)
        for k in k_params:
            graph_path = f'dataset/MW/Graphs/k{k}_disc{alphabet}_ap{approximate}'
            os.makedirs(graph_path, exist_ok=True)
            m4 = M4Dataset.load(training=True,
                                dataset_file='dataset/MW',
                                training_file='training.npy',
                                test_file='test.npy')
            training_values = [v[~np.isnan(v)] for v in m4.values[m4.groups == pattern]]
            data = [ts for ts in training_values]
            discretizer = KBinsDiscretizer(n_bins=alphabet, encode='ordinal', strategy='quantile')
            data = bin_descretize(data, discretizer)
            dbg = generate_and_save_graph(pattern=pattern,
                                          disc_path=disc_path,
                                          graph_path=graph_path,
                                          data=data,
                                          discretizer=discretizer,
                                          k=k,
                                          approximate=approximate)
            for graph_dim in graph_dim_params:
                emb_path = os.path.join(graph_path, f'graph_emb_{graph_dim}')
                os.makedirs(emb_path, exist_ok=True)
                extract_and_save_graph_emb(pattern=pattern,
                                           graph_path=graph_path,
                                           emb_path=emb_path,
                                           graph_dim_size=graph_dim)


if __name__ == "__main__":
    main()
