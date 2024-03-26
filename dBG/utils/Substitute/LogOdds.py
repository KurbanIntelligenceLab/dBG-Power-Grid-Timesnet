import math

import numpy as np
from future.moves import itertools


class LogOdds:

    def __init__(self, sequences, scale=0.5, gap_char='.'):
        self.sequences = sequences
        self.scale = scale
        self.gap_char = gap_char
        self.alphabet_char_to_index = None
        self.similarity_matrix = None
        self.alphabet = None
        self.total_pairs = None
        self.char_freq = dict()
        self.__parse_all_sequences()

    def __parse_all_sequences(self):
        longest_seq_len = max(len(lst) for lst in self.sequences)
        # -1 reserved for short sequences
        self.sequences = [lst + [-1] * (longest_seq_len - len(lst)) for lst in self.sequences]
        self.sequences = np.array(self.sequences, dtype=int).T
        self.alphabet = np.unique(self.sequences)
        self.alphabet = self.alphabet[self.alphabet != -1]
        self.alphabet_char_to_index = {k: v for v, k in enumerate(self.alphabet)}
        self.similarity_matrix = np.zeros((len(self.alphabet), len(self.alphabet)))
        self.__calculate_matrix()

    def __calculate_matrix(self):
        total_combinations = 0
        for row in self.sequences:
            row = row[row != -1]
            unique, counts = np.unique(row, return_counts=True)
            char_counts = np.asarray((unique, counts)).T
            for pair in itertools.combinations_with_replacement(char_counts, 2):
                if pair[0][0] == pair[1][0]:
                    index = self.alphabet_char_to_index[pair[0][0]]
                    count = pair[0][1]
                    self.similarity_matrix[index][index] += math.comb(count, 2)
                else:
                    index1 = self.alphabet_char_to_index[pair[0][0]]
                    index2 = self.alphabet_char_to_index[pair[1][0]]
                    count1 = pair[0][1]
                    count2 = pair[1][1]
                    self.similarity_matrix[index1][index2] += count1 * count2
                    self.similarity_matrix[index2][index1] += count1 * count2
            total_combinations += math.comb(len(row), 2)
        self.similarity_matrix /= total_combinations
        diagonal_matrix = np.diag(self.similarity_matrix)
        expected_probabilities = diagonal_matrix + (
                np.sum(self.similarity_matrix - np.diag(diagonal_matrix), axis=0) / 2)
        expected_matrix = np.tile(expected_probabilities, (len(expected_probabilities), 1)) * \
                          np.tile(np.reshape(expected_probabilities, (len(expected_probabilities), 1)),
                                  (1, len(expected_probabilities)))
        mult = np.full(self.similarity_matrix.shape, 2)
        np.fill_diagonal(mult, 1)
        expected_matrix *= mult
        self.similarity_matrix[np.where(self.similarity_matrix == 0)] = np.nan
        self.similarity_matrix = np.log2(self.similarity_matrix / expected_matrix) / self.scale
        self.min = self.similarity_matrix.min()
        del self.sequences
        print(self.similarity_matrix)

    def get_score(self, letter1, letter2):
        if letter1 == self.gap_char or letter2 == self.gap_char:
            return None
        else:
            score = self.similarity_matrix[self.alphabet_char_to_index[letter1]][self.alphabet_char_to_index[letter2]]
            if np.isnan(score):
                return self.similarity_matrix.min() - 1
            else:
                return score
