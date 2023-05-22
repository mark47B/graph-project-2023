from typing import Optional
from pydantic import BaseModel
# from numpy.typing import nptype


class Node(BaseModel):
    number: int
    node_degree: Optional[int]


class Edge(BaseModel):
    start_node: Node
    end_node: Node
    weight: int


class TemporalGraph:
    edge_list: list[tuple[int, Edge]]

    def __init__(self, path: str = 'out.radoslaw_email_email'):
        self.edge_list = list()
        with open(path) as raw_data:
            raw_data.readline()
            raw_data.readline()
            list_of_items = raw_data.read().split('\n')
            list_of_items.pop(-1)
            for item in list_of_items:
                item = item.split(' ')
                item.pop(2)
                self.edge_list.append(
                                    (int(item[-1]), 
                                    Edge(
                                        start_node=Node(number=int(item[0])),
                                        end_node=Node(number=int(item[1])),
                                        weight=int(item[2]),
                                        )
                                    ))

    def get_static_graph(t_1: int, t_2: int) -> StaticGraph:
        pass            



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
