from dataclasses import dataclass


@dataclass
class GraphProperties:
    """
    A simple data class that stores the edge and node attribute names used in the de Bruijn graph

    :ivar weight: Weight of the edges
    :ivar tuple: The tuple subsequence that an edge represents
    """

    weight: str = 'weight'
    tuple: str = 'tuple'
    substitute: str = 'substitute_edges'
    approx_weight: str = 'approximate_weight'
