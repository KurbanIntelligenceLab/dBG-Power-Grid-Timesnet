import networkx as nx
from tqdm import tqdm

from dBG.DeBruijn import DeBruijnGraph
from dBG.utils.Approximation import calculate_new_weights
from dBG.utils.PropertyNames import GraphProperties as Props
from dBG.utils.strUtils import generate_tuple, match_strings


class FeatureGraph(DeBruijnGraph):
    def __init__(self, k: int, sequences: list, c: float = 1, masks: list = None, gap_char='.',
                 approximate=True, substitute=None, similarity_constant: float = 0.5,
                 similarity_threshold: float = 0):
        super().__init__(k, sequences, c, gap_char)
        self.approximate = approximate
        self.similarity_constant = similarity_constant
        self.similarity_threshold = similarity_threshold

        if masks is not None:
            self.masks = list()
            for mask in masks:
                mask_binary = bin(mask)[2:]
                if len(mask_binary) < k:
                    mask_binary = mask_binary.zfill(k)
                elif len(mask_binary) > k:
                    mask_binary = mask_binary[-k:]
                self.masks.append(mask_binary)
            print(f'Masks :{self.masks}')
            self.__gapify()

        if self.approximate:
            self.substitution_matrix = substitute
            print('Approximating...')
            self.approximate_graph()
            self.weight_attribute_name = Props.approx_weight
        else:
            self.weight_attribute_name = Props.weight

    def approximate_graph(self):
        edge_tuple_attributes = self.graph.edges(data=True)
        new_weight_attribute, substitution_attribute = calculate_new_weights(edge_tuple_attributes,
                                                                             self.substitution_matrix,
                                                                             self.similarity_constant,
                                                                             self.similarity_threshold)

        nx.set_edge_attributes(self.graph, new_weight_attribute, Props.approx_weight)
        nx.set_edge_attributes(self.graph, substitution_attribute, Props.substitute)

    def generate_features(self, weight_threshold: float):
        print('Generating features...')
        init_max = max(dict(self.graph.edges).items(), key=lambda x: x[1][self.weight_attribute_name])[-1][
            self.weight_attribute_name]
        threshold = init_max * weight_threshold

        progress_bar = tqdm(total=int(init_max - threshold))
        step = 0
        seeds = set()
        counter = 1
        while True:
            max_edge = max(self.graph.edges.items(), key=lambda x: x[1][self.weight_attribute_name])
            # print(f'Max edge: {max_edge[1][self.weight_attribute_name]} Thr: {threshold}')
            progress_bar.n = int(init_max - max_edge[1][self.weight_attribute_name])
            progress_bar.refresh()
            if max_edge[-1][self.weight_attribute_name] < threshold:
                break
            step += 1
            seed = tuple(self.__traverse(threshold, self.similarity_constant, max_edge))
            # print(counter, ','.join(map(str, seed)))
            seeds.add(seed)
        progress_bar.close()
        return seeds

    def __traverse(self, threshold, similarity_const, max_edge):
        if not max_edge:
            return None

        forward_current_node = max_edge[0][1]
        back_current_node = max_edge[0][0]
        seed = self.graph.edges[max_edge[0]][Props.tuple].copy()
        visited = {max_edge[0]}

        all_effected_edges = {max_edge[0]: -1}
        all_branches = {}

        # Forward
        while True:
            forward_node_list = [(n[0], n[1][self.weight_attribute_name]) for n in
                                 self.graph[forward_current_node].items() if
                                 (forward_current_node, n[0]) not in visited and n[1][
                                     self.weight_attribute_name] >= threshold]
            if not forward_node_list:
                break

            next_node = max(forward_node_list, key=lambda node: node[1])
            current_edge = (forward_current_node, next_node[0])

            substitutes = self.graph.edges[current_edge][Props.substitute]
            all_branches[current_edge] = forward_node_list

            all_effected_edges[current_edge] = all_effected_edges.get(next_node[0], 0) - 1
            for substitute in substitutes:
                all_effected_edges[substitute] = all_effected_edges.get(substitute, 0) - similarity_const * substitutes[
                    substitute]

            visited.add(current_edge)
            seed += [next_node[0][-1]]
            forward_current_node = next_node[0]

        # Backwards
        while True:
            back_node_list = [(n[0], n[2][self.weight_attribute_name]) for n in
                              self.graph.in_edges(back_current_node, data=True)
                              if (n[0], n[1]) not in visited and n[2][self.weight_attribute_name] >= threshold]
            if not back_node_list:
                break

            next_node = max(back_node_list, key=lambda node: node[1])

            current_edge = (next_node[0], back_current_node)
            substitutes = self.graph.edges[current_edge][Props.substitute]
            all_branches[current_edge] = back_node_list

            all_effected_edges[current_edge] = all_effected_edges.get(next_node[0], 0) - 1
            for substitute in substitutes:
                all_effected_edges[substitute] = all_effected_edges.get(substitute, 0) - similarity_const * substitutes[
                    substitute]

            visited.add(current_edge)
            seed = [next_node[0][0]] + seed
            back_current_node = next_node[0]

        turns = float('inf')
        for edge in all_effected_edges.keys():
            max_edge_weight = self.graph.edges[max_edge[0]][self.weight_attribute_name]
            max_edge_diff = all_effected_edges[max_edge[0]]
            edge_weight = self.graph.edges[edge][self.weight_attribute_name]
            edge_diff = all_effected_edges[edge]
            if max_edge_diff < edge_diff:
                turns_to_catch_up = -((max_edge_weight - edge_weight) // (max_edge_diff - edge_diff))
                turns = min(turns, turns_to_catch_up)
            if turns <= 1:
                break
            if edge in visited:
                turns_to_crop = -((edge_weight - threshold) // edge_diff)
                turns = min(turns, turns_to_crop)
                if turns <= 1:
                    break
                if edge != max_edge[0]:
                    possible_branches = all_branches[edge]
                    for branch in possible_branches:
                        branch_edge = (edge[0], branch[0])
                        if branch_edge in all_effected_edges.keys():
                            max_edge_weight = self.graph.edges[edge][self.weight_attribute_name]
                            max_edge_diff = all_effected_edges[edge]
                            branch_weight = branch[1]
                            branch_diff = all_effected_edges[branch_edge]
                            if max_edge_diff < edge_diff:
                                turns_to_catch_up = -(
                                        (max_edge_weight - branch_weight) // (max_edge_diff - branch_diff))
                                turns = min(turns, turns_to_catch_up)
        if turns < 1:
            turns = 1
        # print(f'Turns: {turns}')
        for edge in visited:
            self.graph.edges[edge][self.weight_attribute_name] -= abs(1 * turns)
            for sim_edge in self.graph.edges[edge][Props.substitute]:
                self.graph.edges[sim_edge][self.weight_attribute_name] -= abs((similarity_const *
                                                                               self.graph.edges[edge][Props.substitute][
                                                                                   sim_edge]) * turns)
        return seed

    def __gapify(self):
        old_edges = list(
            (generate_tuple(s[0], s[1], self.gap_char), s[2][Props.weight]) for s in
            self.graph.edges(data=True))
        self.__apply_masks()
        with tqdm(total=self.graph.number_of_nodes(), desc="Adding gaps...") as pbar:
            for node1 in self.graph.nodes:
                for node2 in self.graph.nodes:
                    if not self.graph.has_edge(node1, node2):
                        edge = True
                        for l1, l2 in zip(node1[1:], node2[:-1]):
                            if l1 != self.gap_char and l2 != self.gap_char and l1 != l2:
                                edge = False
                                break
                        if edge:
                            weight = match_strings(old_edges, generate_tuple(node1, node2, self.gap_char),
                                                   self.gap_char)
                            if weight > 0:
                                edge_attributes = {
                                    Props.weight: weight,
                                    Props.tuple: generate_tuple(node1, node2, self.gap_char),
                                    Props.substitute: dict(),
                                }
                                self.graph.add_edge(node1, node2, **edge_attributes)
                pbar.update(1)

    def __apply_masks(self):
        for node in list(self.graph.nodes):
            for mask in self.masks:
                masked = ''
                for letter, binary in zip(node, mask):
                    if binary == 0:
                        masked += self.gap_char
                    else:
                        masked += letter
                if masked not in self.graph:
                    self.graph.add_node(masked)
