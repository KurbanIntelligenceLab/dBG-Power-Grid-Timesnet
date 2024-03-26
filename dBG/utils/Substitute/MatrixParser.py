import os


class SubstitutionMatrix:
    def __init__(self, filename, root_path='/home/lumpus/Documents/deBruijnData/substitution_matrices/', gap_char='.'):
        self.root_path = root_path
        self.gap_char = gap_char
        self.matrix = {}
        self.alphabet = []
        self.name = os.path.basename(filename)
        self._parse_file(filename)

    def _parse_file(self, file_path):
        with open(self.root_path + file_path, 'r') as file:
            for line in file:
                # Ignore commented lines
                if line.startswith('#'):
                    continue
                # Parse the alphabet from the first non-commented line
                if not self.alphabet:
                    self.alphabet = line.split()
                    continue
                # Parse the matrix lines
                parts = line.split()
                letter = parts[0]
                scores = [int(score) for score in parts[1:]]
                self.matrix[letter] = dict(zip(self.alphabet, scores))

    def get_score(self, letter1, letter2):
        if letter1 == self.gap_char or letter2 == self.gap_char:
            return None
        else:
            return self.matrix[letter1][letter2]
