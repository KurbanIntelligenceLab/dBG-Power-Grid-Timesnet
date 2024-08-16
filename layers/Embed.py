import ctypes
import math

import joblib
import numpy as np
import torch
import torch.nn as nn
from scipy.interpolate import interp1d
from scipy.spatial import distance
from torch import dropout

import dBG.utils.IterationUtils as Iter


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float()
                    * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]


class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model,
                                   kernel_size=3, padding=padding, padding_mode='circular', bias=False)
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
        return x


class dBGraphEmbedding(nn.Module):
    def __init__(self, c_in, d_model, season, seq_len, pred_len, k, ap, disc, dBGEmb):
        super(dBGraphEmbedding, self).__init__()
        self.graph_embedding = dict()
        self.k = k
        self.window_count = seq_len - self.k + 2
        self.dBGEmb = dBGEmb
        graph_id = f"k{k}_disc{disc}_ap{ap}"
        with open(f'dataset/Graphs/{graph_id}/graph_emb_{dBGEmb}/{season}_emb.txt','r') as file:
            # Read the first line to get dimensions
            first_line = file.readline()
            dimensions = [int(dim) for dim in first_line.split()]

            # Read the rest of the data line by line
            for line in file:
                data = [float(num) for num in line.split()]
                key = data[0]  # First column is the key
                value = data[1:dimensions[1] + 1]  # Rest of the columns are the values
                self.graph_embedding[key] = torch.tensor(value)
        self.node_mapping = joblib.load(f'dataset/Graphs/{graph_id}/{season}_nodes.joblib')

        if not all(len(key) == self.k - 1 for key in self.node_mapping.keys()):
            raise Exception("Unmatching tuple size")

        self.desc = joblib.load(f'dataset/Discretizer/{disc}Disc/{season}_discretizer_model.joblib')

    def forward(self, x):
        shape = x.shape
        discretized_data = self.desc.transform(x.reshape(-1, 1).cpu()).astype(int)
        discretized_data = torch.tensor(discretized_data.reshape(shape), dtype=torch.int32)
        x_enc_out = torch.zeros(x.shape[0], x.shape[1] - self.k + 2, self.dBGEmb)
        for i, sequence in enumerate(discretized_data.squeeze(-1)):
            for j, current in enumerate(Iter.sliding_window(sequence, self.k - 1)):
                node_key = tuple([int(k) for k in current])
                if node_key in self.node_mapping:
                    node_id = self.node_mapping[node_key]
                else:
                    closest_node_key = min(self.node_mapping.keys(), key=lambda x: distance.euclidean(x, node_key))
                    self.node_mapping[node_key] = self.node_mapping[closest_node_key]
                    node_id = self.node_mapping[node_key]
                x_enc_out[i, j] = self.graph_embedding[node_id]
        x_enc_out = torch.nn.functional.pad(x_enc_out, (0, 0, 0, x.shape[1] - self.window_count))
        return x_enc_out.to(x.device)

def to_c_array(py_list):
    return (ctypes.c_int * len(py_list))(*py_list)


def load_features(season, graph_id, feat_count):
    features = set()
    with open(f'dataset/Graphs/{graph_id}/proto_features_{feat_count}/{season}_features.txt', 'r') as file:
        for line in file:
            tuple_elements = tuple(map(int, line.strip("()\n").split(", ")))
            features.add(tuple_elements)
    return features

def interpolate(list_a, list_b):
    max_length = max(len(list_a), len(list_b))
    x_new = np.linspace(0, 1, max_length)

    # Interpolate list_a if it is shorter, else interpolate list_b
    if len(list_a) < len(list_b):
        x = np.linspace(0, 1, len(list_a))
        f = interp1d(x, list_a, kind='linear')
        return f(x_new).astype(int), list_b
    elif len(list_b) < len(list_a):
        x = np.linspace(0, 1, len(list_b))
        f = interp1d(x, list_b, kind='linear')
        return list_a, f(x_new).astype(int)
    else:
        # If both lists are of equal length, return them as is
        return list_a, list_b


class dBGEmbedding(nn.Module):
    def __init__(self, c_in, d_model, season, seq_len, k, ap, disc, feat_cnt, dropout=0.1, include_corr=False, lin_layer=False):
        super(dBGEmbedding, self).__init__()
        graph_id = f"k{k}_disc{disc}_ap{ap}"

        self.features = load_features(season, graph_id, feat_cnt)
        self.include_corr = include_corr
        # Loading external C library for Levenshtein distance
        self.levenshtein = ctypes.CDLL('./levenshtein.so')
        self.levenshtein.levenshtein_distance.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int,
                                                          ctypes.POINTER(ctypes.c_int), ctypes.c_int]
        self.levenshtein.levenshtein_distance.restype = ctypes.c_int

        padding = 1 if torch.__version__ >= '1.5.0' else 2

        # Define your network layers
        self.conv = nn.Conv1d(in_channels=c_in + include_corr, out_channels=d_model,
                              kernel_size=3, padding=padding, padding_mode='circular', bias=False)

        self.dropout = nn.Dropout(p=dropout)

        if lin_layer:
            self.lin_layer = nn.Linear(d_model, d_model)
            self.dropout_lin = nn.Dropout(p=dropout)
        else:
            self.lin_layer = None
            
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_in', nonlinearity='leaky_relu')

        self.desc = joblib.load(f'dataset/Discretizer/{disc}Disc/{season}_discretizer_model.joblib')

    def forward(self, x):
        shape = x.shape
        discretized_data = self.desc.transform(x.cpu().reshape(-1, 1)).astype(int)
        discretized_data = torch.tensor(discretized_data.reshape(shape), dtype=torch.int32)

        # Vectorizing the computation of embeddings
        dbg_feats, corr_feats = self.compute_embeddings(discretized_data, self.include_corr)

        # Normalization
        means = dbg_feats.mean(1, keepdim=True)
        stdev = torch.sqrt(torch.var(dbg_feats, dim=1, keepdim=True, unbiased=False) + 1e-5)
        dbg_feats = (dbg_feats - means) / stdev

        if self.include_corr:
            dbg_feats = torch.cat((corr_feats, dbg_feats), dim=2)
            
        x = self.conv(dbg_feats.permute(0, 2, 1).to(x.device))

        if self.lin_layer is not None:
            x = self.lin_layer(x.permute(0, 2, 1))
            x = self.dropout_lin(x).permute(0, 2, 1)
        
        return self.dropout(x).permute(0, 2, 1)

    def compute_embeddings(self, discretized_data, include_corr=False):
        corr_embeds = [] if include_corr else None
        dist_embeds = []
        for row in discretized_data.permute(0, 2, 1):
            if include_corr:
                corr_embed = self.calculate_row_corr_embeddings(row[0])
                corr_embeds.append(corr_embed)
            dist_embed = self.calculate_row_dist_embeddings(row[0])
            dist_embeds.append(dist_embed)
        return torch.tensor(dist_embeds).unsqueeze(-1), torch.tensor(corr_embeds).unsqueeze(-1) if include_corr else None


    def calculate_row_corr_embeddings(self, row):
        row_embeds = list()
        for feature in self.features:
            feat_i, row_i = interpolate(feature, row)
            corr = np.corrcoef(feat_i, row_i)[0, 1]
            row_embeds.append(float(0 if np.isnan(corr) else corr))
        return row_embeds

    def calculate_row_dist_embeddings(self, row):
        row_str = to_c_array(row.tolist())
        row_embeds = [
            float(self.levenshtein.levenshtein_distance(row_str, len(row_str), to_c_array(list(feature)), len(feature)))
            for feature in self.features]
        return row_embeds


class FixedEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(FixedEmbedding, self).__init__()

        w = torch.zeros(c_in, d_model).float()
        w.require_grad = False

        position = torch.arange(0, c_in).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float()
                    * -(math.log(10000.0) / d_model)).exp()

        w[:, 0::2] = torch.sin(position * div_term)
        w[:, 1::2] = torch.cos(position * div_term)

        self.emb = nn.Embedding(c_in, d_model)
        self.emb.weight = nn.Parameter(w, requires_grad=False)

    def forward(self, x):
        return self.emb(x).detach()


class TemporalEmbedding(nn.Module):
    def __init__(self, d_model, embed_type='fixed', freq='h'):
        super(TemporalEmbedding, self).__init__()

        minute_size = 4
        hour_size = 24
        weekday_size = 7
        day_size = 32
        month_size = 13

        Embed = FixedEmbedding if embed_type == 'fixed' else nn.Embedding
        if freq == 't':
            self.minute_embed = Embed(minute_size, d_model)
        self.hour_embed = Embed(hour_size, d_model)
        self.weekday_embed = Embed(weekday_size, d_model)
        self.day_embed = Embed(day_size, d_model)
        self.month_embed = Embed(month_size, d_model)

    def forward(self, x):
        x = x.long()
        minute_x = self.minute_embed(x[:, :, 4]) if hasattr(
            self, 'minute_embed') else 0.
        hour_x = self.hour_embed(x[:, :, 3])
        weekday_x = self.weekday_embed(x[:, :, 2])
        day_x = self.day_embed(x[:, :, 1])
        month_x = self.month_embed(x[:, :, 0])

        return hour_x + weekday_x + day_x + month_x + minute_x


class TimeFeatureEmbedding(nn.Module):
    def __init__(self, d_model, embed_type='timeF', freq='h'):
        super(TimeFeatureEmbedding, self).__init__()

        freq_map = {'h': 4, 't': 5, 's': 6,
                    'm': 1, 'a': 1, 'w': 2, 'd': 3, 'b': 3}
        d_inp = freq_map[freq]
        self.embed = nn.Linear(d_inp, d_model, bias=False)

    def forward(self, x):
        return self.embed(x)


class DataEmbedding(nn.Module):
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        # self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.temporal_embedding = TemporalEmbedding(d_model=d_model, embed_type=embed_type,
                                                    freq=freq) if embed_type != 'timeF' else TimeFeatureEmbedding(
            d_model=d_model, embed_type=embed_type, freq=freq)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        if x_mark is None:
            x = self.value_embedding(x)
        else:
            x = self.value_embedding(
                x) + self.temporal_embedding(x_mark) + self.position_embedding(x)
        return self.dropout(x)


class DataEmbedding_inverted(nn.Module):
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding_inverted, self).__init__()
        self.value_embedding = nn.Linear(c_in, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        x = x.permute(0, 2, 1)
        # x: [Batch Variate Time]
        if x_mark is None:
            x = self.value_embedding(x)
        else:
            x = self.value_embedding(torch.cat([x, x_mark.permute(0, 2, 1)], 1))
        # x: [Batch Variate d_model]
        return self.dropout(x)


class DataEmbedding_wo_pos(nn.Module):
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding_wo_pos, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.temporal_embedding = TemporalEmbedding(d_model=d_model, embed_type=embed_type,
                                                    freq=freq) if embed_type != 'timeF' else TimeFeatureEmbedding(
            d_model=d_model, embed_type=embed_type, freq=freq)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        if x_mark is None:
            x = self.value_embedding(x)
        else:
            x = self.value_embedding(x) + self.temporal_embedding(x_mark)
        return self.dropout(x)


class PatchEmbedding(nn.Module):
    def __init__(self, d_model, patch_len, stride, padding, dropout):
        super(PatchEmbedding, self).__init__()
        # Patching
        self.patch_len = patch_len
        self.stride = stride
        self.padding_patch_layer = nn.ReplicationPad1d((0, padding))

        # Backbone, Input encoding: projection of feature vectors onto a d-dim vector space
        self.value_embedding = nn.Linear(patch_len, d_model, bias=False)

        # Positional embedding
        self.position_embedding = PositionalEmbedding(d_model)

        # Residual dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # do patching
        n_vars = x.shape[1]
        x = self.padding_patch_layer(x)
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        x = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))
        # Input encoding
        x = self.value_embedding(x) + self.position_embedding(x)
        return self.dropout(x), n_vars
