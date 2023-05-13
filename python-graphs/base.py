from typing import Optional
from pydantic import BaseModel
from numpy.typing as nptype


class Node(BaseModel):
    number: int
    node_degree: Optional[int]


class Edge(BaseModel):
    start_node: Node
    end_node: Node
    weight: int


class StaticGraph(BaseModel):
    pass

class StaticGraph(BaseModel):
    adjacency_matrix: nptype.DTypeLike # Матрица смежности
    edge_sets: set[Edge] # для полных
    adjacency_lists: dict[Node: list[Node]] # для разреженных
    largest_connected_component: Optional[StaticGraph]
    count_vertices: Optional[int]
    count_edges: Optional[int]
    density: Optional[float]

    def add_node(node: Node):
        pass

    def count_vertices():
        pass
    def count_edges():
        pass
    def density():
        pass
    def number_of_connected_components():
        pass
    def share_of_vertices(): # in the connected component with the maximum cardinality
        pass
