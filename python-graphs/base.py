from typing import Optional, Dict, List
from pydantic import BaseModel, Field
import numpy as np
import pydantic_numpy.dtype as pnd
from pydantic_numpy import NDArray



class Node(BaseModel):
    number: int
    node_degree: Optional[int]

    def __lt__(self, other):
        return self.number < other.number

    def __eq__(self, other):
        return self.number == other.number


class Edge(BaseModel):
    start_node: Node
    end_node: Node
    weight: int

    def __lt__(self, other):
        return max(self.end_node, self.start_node) < max(other.end_node, other.start_node)
    
    def __eq__(self, other):
        return self.start_node == other.start_node and self.end_node == other.end_node and self.weight == other.weight
    
    def get_max_node(self):
        return max(self.end_node, self.start_node)


class StaticGraph(BaseModel):
    adjacency_matrix: NDArray[int, pnd.float32]  # Матрица смежности
    time: tuple[int, int]
    edge_list: List[Edge] # для полных
    adjacency_lists: Optional[Dict[Node, list[Node]]] # для разреженных dict[Node: list[Node]]
    largest_connected_component: Optional['StaticGraph']
    count_vertices: Optional[int]
    count_edges: Optional[int]
    density: Optional[float]

    class Config:
        arbitrary_types_allowed = True

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
                                        start_node=Node(number=int(item[0])-1),
                                        end_node=Node(number=int(item[1])-1),
                                        weight=int(item[2]),
                                        )
                                    ))

    def get_static_graph(self, t_1: int, t_2: int) -> StaticGraph:
#         # Выделяем нужные рёбра
#         edge_in_static = [x[1] for x in self.edge_list if t_1 <= x[0] <= t_2]
#         # Ищем диапазон для матрицы смежностей
#         max_number_of_node = max(edge_in_static).get_max_node().number
#         print('----------------------', max_number_of_node, '------------------')
#         # Создаём матрицу смежностей
#         aj_matr = np.zeros((max_number_of_node, max_number_of_node))
#         for e in edge_in_static:
#             aj_matr[e.end_node.number, e.start_node.number] = e.weight
            
#         return StaticGraph(time=(t_1, t_2), edge_list=edge_in_static.copy(), adjacency_matrix=aj_matr)
        SG = StaticGraph()
        for x in self.edge_list:
            if t_1 <= x[0] <= t_2:
                SG.add_edge(x[1])
        return SG
