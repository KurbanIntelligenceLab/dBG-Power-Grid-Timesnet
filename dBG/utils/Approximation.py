from dBG.utils.PropertyNames import GraphProperties as Props
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
from tqdm.contrib.concurrent import process_map  # Import process_map from tqdm.contrib.concurrent

class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_end_of_tuple = False
        self.weight = None


class Trie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, edge_attributes):
        node = self.root
        for element in edge_attributes[Props.tuple]:
            if element not in node.children:
                node.children[element] = TrieNode()
            node = node.children[element]
        node.is_end_of_tuple = True
        node.weight = edge_attributes[Props.weight]

    def find_similar_tuples(self, tuple_key, sub, t, current_node=None, index=0):
        if current_node is None:
            current_node = self.root

        # Base case: if we have traversed the entire tuple
        if index == len(tuple_key):
            if current_node.is_end_of_tuple:
                return {tuple(tuple_key): current_node.weight}
            return {}

        similar_tuples = {}
        current_element = tuple_key[index]

        # Check each child node for similarity
        for child_element, child_node in current_node.children.items():
            score = sub.get_score(current_element, child_element)
            if score is not None and score < t:
                continue  # Skip non-similar branches
            child_tuples = self.find_similar_tuples(
                tuple_key[:index] + [child_element] + tuple_key[index + 1:],
                sub, t, child_node, index + 1
            )
            similar_tuples.update(child_tuples)

        return similar_tuples


def calculate_new_weights(edge_attributes, substitution_matrix, similarity_constant, similarity_threshold):
    trie = Trie()

    # Insert all tuples into the trie
    for edge_attribute in edge_attributes:
        trie.insert(edge_attribute[-1])

    tuple_key_dict = {tuple(edge_data[-1][Props.tuple]): edge_data[:2] for edge_data in edge_attributes}
    new_weight_attribute = {}
    substitution_attribute = {}

    # Find similar tuples and calculate new weights and similarity scores
    for edge_attribute in tqdm(edge_attributes):
        tuple_key = edge_attribute[-1][Props.tuple]
        old_weight = edge_attribute[-1][Props.weight]

        similar_tuples = trie.find_similar_tuples(tuple_key, substitution_matrix, similarity_threshold)

        if tuple(tuple_key) in similar_tuples.keys():
            del similar_tuples[tuple(tuple_key)]

        # Dictionary to store similar tuples and their similarity scores
        similar_tuples_with_scores = {}

        # Calculate similarity score and new weight
        new_weight = old_weight
        for similar_tuple in similar_tuples.keys():
            similarity_score = 0
            for i in range(len(tuple_key)):
                score1 = substitution_matrix.get_score(tuple_key[i], similar_tuple[i])
                score2 = substitution_matrix.get_score(tuple_key[i], tuple_key[i])

                if score1 is not None and score2 is not None:
                    similarity_score += score1 / score2
            new_weight += similarity_constant * (similar_tuples[similar_tuple] * similarity_score)

            # Store the similarity score
            similar_tuples_with_scores[similar_tuple] = similarity_score
        similar_tuples_with_scores = {tuple_key_dict[key]: value for key, value in similar_tuples_with_scores.items()}
        new_weight_attribute[edge_attribute[:2]] = new_weight
        substitution_attribute[edge_attribute[:2]] = similar_tuples_with_scores
    return new_weight_attribute, substitution_attribute

"""
# Example usage
sub = SubstitutionMatrix('BLOSUM62')
k = 1  # Assume a constant K
t = 0  # Assume a threshold
data = {('A', 'A'): 10, ('S', 'S'): 20, ('D', 'B'): 15}

processed_data = calculate_new_weights(data, sub, k, t)
print(processed_data)
"""
