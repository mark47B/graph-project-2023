from typing import Optional
import time
from pydantic import BaseModel
from dataclasses import dataclass
import numpy as np
import pydantic_numpy.dtype as pnd
import pandas as pd


class Node(BaseModel):
    number: int

    def __lt__(self, other: 'Node'):
        return self.number < other.number

    def __eq__(self, other: 'Node'):
        return self.number == other.number


class Edge(BaseModel):
    number: int
    start_node: Node
    end_node: Node
    timestamp: int

    def __lt__(self, other: 'Edge'):
        return max(self.end_node, self.start_node) < max(other.end_node, other.start_node)
    
    def __eq__(self, other: 'Edge'):
        return self.start_node == other.start_node and self.end_node == other.end_node and self.timestamp == other.timestamp
    
    def get_max_node(self):
        return max(self.end_node, self.start_node)


class TemporalGraph:
    edge_list: list[Edge]

    def __init__(self, path: str = './datasets/radoslaw_email/out.radoslaw_email_email'):
        self.edge_list = list()
        with open(path) as raw_data:
            raw_data.readline()
            raw_data.readline()
            edge_number = 0
            list_of_items = raw_data.read().split('\n')
            list_of_items.pop(-1)
            for item in list_of_items:
                item = item.split(" ")
                if int(item[0]) == int(item[1]):
                    continue
                self.edge_list.append(
                                    Edge(
                                        number=edge_number,
                                        start_node=Node(number=int(item[0])-1),
                                        end_node=Node(number=int(item[1])-1),
                                        timestamp=int(item[2]),
                                        )
                                    )
                edge_number += 1
        self.edge_list.sort(key=lambda x: x.timestamp)


    def get_static_graph(self, l: float, r: float, prediction: bool = False) -> 'StaticGraph':
        t_1 = self.edge_list[int(l * (len(self.edge_list) - 1))].timestamp
        t_2 = self.edge_list[int(r * (len(self.edge_list) - 1))].timestamp
        st = set()
        for i in self.edge_list:
            st.add(i.start_node.number)
            st.add(i.end_node.number)
        sg = StaticGraph(t_1, t_2, len(st), prediction)
        for x in self.edge_list:
            if t_1 <= x.timestamp <= t_2:
                sg.add_edge(x)
        return sg
    
    def get_max_timestamp(self):
        return max(self.edge_list, key=lambda x: x.timestamp).timestamp
    
    def get_min_timestamp(self):
        return min(self.edge_list, key=lambda x: x.timestamp).timestamp


@dataclass
class StaticGraph:
    t_min: int
    t_max: int
    prediction: bool = False
    node_set: pd.DataFrame = None
    edge_set: pd.DataFrame = None
    adjacency_matrix: pnd.NDArrayBool = None  # Матрица смежности
    largest_connected_component: Optional['StaticGraph'] = None
    number_of_connected_components: Optional[int] = None
    
    def __init__(self, t_1: int = 0, t_2: int = 10000000000, size: int = 10000, prediction=False):
        self.t_min = t_1
        self.t_max = t_2
        self.prediction = prediction
        if not prediction:
            self.adjacency_matrix = np.full((size, size), False, dtype=bool)
        else:
            self.adjacency_matrix = None
        self.largest_connected_component = None
        self.number_of_connected_components = None

    def get_node_set(self) -> pd.DataFrame:
        # создадим датафрейм для вершин, если такового нет
        if self.node_set is None:
            self.node_set = pd.DataFrame({
                "number": pd.Series(dtype='int'),
                "number_in_temporal_graph": pd.Series(dtype='int'),
                "node_degree": pd.Series(dtype='int'),
                "node_activity_zeroth_quantile_wl": pd.Series(dtype='float'),
                "node_activity_first_quantile_wl": pd.Series(dtype='float'),
                "node_activity_second_quantile_wl": pd.Series(dtype='float'),
                "node_activity_third_quantile_wl": pd.Series(dtype='float'),
                "node_activity_fourth_quantile_wl": pd.Series(dtype='float'),
                "node_activity_sum_wl": pd.Series(dtype='float'),
                "node_activity_mean_wl": pd.Series(dtype='float'),
                "node_activity_zeroth_quantile_we": pd.Series(dtype='float'),
                "node_activity_first_quantile_we": pd.Series(dtype='float'),
                "node_activity_second_quantile_we": pd.Series(dtype='float'),
                "node_activity_third_quantile_we": pd.Series(dtype='float'),
                "node_activity_fourth_quantile_we": pd.Series(dtype='float'),
                "node_activity_sum_we": pd.Series(dtype='float'),
                "node_activity_mean_we": pd.Series(dtype='float'),
                "node_activity_zeroth_quantile_wsr": pd.Series(dtype='float'),
                "node_activity_first_quantile_wsr": pd.Series(dtype='float'),
                "node_activity_second_quantile_wsr": pd.Series(dtype='float'),
                "node_activity_third_quantile_wsr": pd.Series(dtype='float'),
                "node_activity_fourth_quantile_wsr": pd.Series(dtype='float'),
                "node_activity_sum_wsr": pd.Series(dtype='float'),
                "node_activity_mean_wsr": pd.Series(dtype='float'),
            })
        return self.node_set
    
    def get_edge_set(self) -> pd.DataFrame:
        # создадим датафрейм для рёбер, если такового нет
        if self.edge_set is None:
            self.edge_set = pd.DataFrame({
                "number": pd.Series(dtype='int'),
                "start_node": pd.Series(dtype='int'),
                "end_node": pd.Series(dtype='int'),
                "timestamp": pd.Series(dtype='int'),
                "weight_linear": pd.Series(dtype='float'),
                "weight_exponential": pd.Series(dtype='float'),
                "weight_square_root": pd.Series(dtype='float'),
            })
        return self.edge_set


    def add_node(self, node: Node) -> int:
        # добавим вершину
        self.get_node_set().loc[self.count_vertices()] = {
            "number_in_temporal_graph": node.number,
            "node_degree": 0,
            "number": self.count_vertices()
        }
        return self.count_vertices() - 1

    def add_edge_non_multiedge(self, edge: Edge) -> int:
        start_node_index: int = -1
        end_node_index: int = -1
        # если вершин не существует, добавим их, и сохраним их индексы
        if self.get_node_set().loc[self.get_node_set()["number_in_temporal_graph"] == edge.start_node.number].empty:
            start_node_index = self.add_node(edge.start_node)
        else:
            start_node_index = self.get_node_set().loc[
                self.get_node_set()["number_in_temporal_graph"] == edge.start_node.number, 
            "number"].to_list()[0]

        if self.get_node_set().loc[self.get_node_set()["number_in_temporal_graph"] == edge.end_node.number].empty:
            end_node_index = self.add_node(edge.end_node)
        else:
            end_node_index = self.get_node_set().loc[
                self.get_node_set()["number_in_temporal_graph"] == edge.end_node.number, 
            "number"].to_list()[0]

        # свапнем вершины, если start_node_index > end_node_index
        if start_node_index > end_node_index:
            start_node_index, end_node_index = end_node_index, start_node_index

        if ((self.get_edge_set()["start_node"] == start_node_index) & 
            (self.get_edge_set()["end_node"] == end_node_index)).any(): # исправить таймстемп
            self.get_edge_set().loc[
                (self.get_edge_set()["start_node"] == start_node_index) &
                (self.get_edge_set()["end_node"] == end_node_index), "timestamp"] = edge.timestamp
        else:
            # добавим ребро
            self.get_edge_set().loc[self.count_edges()] = {
                "number": edge.number,
                "start_node": start_node_index,
                "end_node": end_node_index,
                "timestamp": edge.timestamp,
            }
            # увеличим степень вершин
            self.get_node_set().at[start_node_index, "node_degree"] += 1
            self.get_node_set().at[end_node_index, "node_degree"] += 1
            
            if not self.prediction: 
                # обозначим, что вершины смежны
                self.adjacency_matrix[start_node_index][end_node_index] = True
                self.adjacency_matrix[end_node_index][start_node_index] = True

        return self.count_edges() - 1


    def add_edge(self, edge: Edge) -> int:
        start_node_index: int = -1
        end_node_index: int = -1
        # если вершин не существует, добавим их, и сохраним их индексы
        if self.get_node_set().loc[self.get_node_set()["number_in_temporal_graph"] == edge.start_node.number].empty:
            start_node_index = self.add_node(edge.start_node)
        else:
            start_node_index = self.get_node_set().loc[
                self.get_node_set()["number_in_temporal_graph"] == edge.start_node.number, 
            "number"].to_list()[0]

        if self.get_node_set().loc[self.get_node_set()["number_in_temporal_graph"] == edge.end_node.number].empty:
            end_node_index = self.add_node(edge.end_node)
        else:
            end_node_index = self.get_node_set().loc[
                self.get_node_set()["number_in_temporal_graph"] == edge.end_node.number, 
            "number"].to_list()[0]

        # свапнем вершины, если start_node_index > end_node_index
        if start_node_index > end_node_index:
            start_node_index, end_node_index = end_node_index, start_node_index

        if not ((self.get_edge_set()["start_node"] == start_node_index) & 
            (self.get_edge_set()["end_node"] == end_node_index)).any():  # если ребро пришло первый раз
            # увеличим степень вершин
            self.get_node_set().at[start_node_index, "node_degree"] += 1
            self.get_node_set().at[end_node_index, "node_degree"] += 1

            if not self.prediction:
                # обозначим, что вершины смежны
                self.adjacency_matrix[start_node_index][end_node_index] = True
                self.adjacency_matrix[end_node_index][start_node_index] = True

            # добавим ребро
            self.get_edge_set().loc[self.count_edges()] = {
                "number": edge.number,
                "start_node": start_node_index,
                "end_node": end_node_index,
                "timestamp": edge.timestamp,
            }
        elif not ((self.get_edge_set()["number"] == edge.number)).any():  # проверка на полный дубликат
            # добавим ребро
            self.get_edge_set().loc[self.count_edges()] = {
                "number": edge.number,
                "start_node": start_node_index,
                "end_node": end_node_index,
                "timestamp": edge.timestamp,
            }
        return self.count_edges() - 1

    def count_vertices(self) -> int:
        return len(self.get_node_set())

    def count_edges(self) -> int:
        return len(self.get_edge_set())

    def density(self) -> float:
        cnt_vert: int = self.count_vertices()
        return 2 * self.count_edges() / (cnt_vert * (cnt_vert - 1))


    def __find_size_of_connected_component(self, used, v) -> int:
        used[v] = True
        sz = 1
        cnt_vert = self.count_vertices()
        for to in range(cnt_vert):
            if not self.adjacency_matrix[v][to]:
                continue
            if not used[to]:
                sz += self.__find_size_of_connected_component(used, to)
        return sz
    
    def __find_largest_connected_component(self, used, v):
        # Обойдём всю компоненту слабой связности и запишем её как отдельный граф
        used[v] = True
        cnt_vert = self.count_vertices()
        for to in range(cnt_vert):
            if not self.adjacency_matrix[v][to]:
                continue
            if not used[to]:
                self.__find_largest_connected_component(used, to)
            
            v_number = self.get_node_set().at[v, "number_in_temporal_graph"]
            to_number = self.get_node_set().at[to, "number_in_temporal_graph"]

            edge_df = self.get_edge_set().loc[
                (self.get_edge_set()["start_node"] == min(v, to)) & 
                (self.get_edge_set()["end_node"] == max(v, to)), 
            ["number", "timestamp"]]
            for _, row in edge_df.iterrows():
                new_edge = Edge(
                    number=row["number"], 
                    start_node=Node(number=v_number),
                    end_node=Node(number=to_number),
                    timestamp=row["timestamp"])
                self.largest_connected_component.add_edge(new_edge)

    def __update_number_of_connected_components_and_largest_connected_component(self):
        # Запустим DFS от каждой ещё не посещённой вершины, получая компоненты слабой связности
        # Заодно считаем количество этих компонент и максимальную по мощности компоненту слабой связности
        cnt_vert: int = self.count_vertices()
        used = [False for _ in range(cnt_vert)]
        vertice: int = 0
        self.number_of_connected_components = 0
        max_component_size: int = 0
        for i in range(cnt_vert):
            if not used[i]:
                self.number_of_connected_components += 1
                component_size = self.__find_size_of_connected_component(used, i)
                if component_size > max_component_size:
                    max_component_size = component_size
                    vertice = i

        # Обновляем посещенность вершин для обработки максимальной по мощности компоненты
        # слабой связности
        used = [False for _ in range(cnt_vert)]

        # Нашли максимальную по мощности компоненту слабой связности, запишем её в поле
        self.largest_connected_component = StaticGraph(self.t_min, self.t_max, cnt_vert)
        self.__find_largest_connected_component(used, vertice)

    def get_largest_connected_component(self) -> 'StaticGraph':
        # если максимальную по мощность компоненту слабой связности не нашли, найдём
        if self.largest_connected_component is None:
            self.__update_number_of_connected_components_and_largest_connected_component()

        return self.largest_connected_component

    def get_number_of_connected_components(self) -> int:
        # если число компонент слабой связности не нашли, найдём
        if self.largest_connected_component is None:
            self.__update_number_of_connected_components_and_largest_connected_component()

        return self.number_of_connected_components

    def share_of_vertices(self) -> float: 
        return self.get_largest_connected_component().count_vertices() / self.count_vertices()

    def get_radius(self, method: 'SelectApproach') -> int:
        # Находим подграф с помощью выбранного метода
        sample_graph: StaticGraph = method

        # Алгоритм Флойда-Уоршелла
        cnt_verts = sample_graph.count_vertices()
        shortest_paths = np.zeros((cnt_verts, cnt_verts))
        for i in range(cnt_verts):
            for j in range(cnt_verts):
                if sample_graph.adjacency_matrix[i][j]:
                    shortest_paths[i][j] = 1
                else:
                    shortest_paths[i][j] = 1000000000
        for k in range(cnt_verts):
            for i in range(cnt_verts):
                for j in range(cnt_verts):
                    shortest_paths[i][j] = min(shortest_paths[i][j], shortest_paths[i][k] + shortest_paths[k][j])

        radius = 1000000000
        for i in range(cnt_verts):
            eccentricity = 0
            for j in range(cnt_verts):
                if shortest_paths[i][j] != 1000000000:
                    eccentricity = max(eccentricity, shortest_paths[i][j])
            if eccentricity > 0:
                radius = min(radius, eccentricity)
        return radius
        

    def get_diameter(self, method: 'SelectApproach') -> int:
        # Находим подграф с помощью выбранного метода
        sample_graph: StaticGraph = method

        # Алгоритм Флойда-Уоршелла
        cnt_verts = sample_graph.count_vertices()
        shortest_paths = np.zeros((cnt_verts, cnt_verts))
        for i in range(cnt_verts):
            for j in range(cnt_verts):
                if sample_graph.adjacency_matrix[i][j]:
                    shortest_paths[i][j] = 1
                else:
                    shortest_paths[i][j] = 1000000000
        for k in range(cnt_verts):
            for i in range(cnt_verts):
                for j in range(cnt_verts):
                    shortest_paths[i][j] = min(shortest_paths[i][j], shortest_paths[i][k] + shortest_paths[k][j])

        diameter = 0
        for i in range(cnt_verts):
            for j in range(cnt_verts):
                if shortest_paths[i][j] != 1000000000:
                    diameter = max(diameter, shortest_paths[i][j])
        return diameter
        

    def percentile_distance(self, method: 'SelectApproach', percentile: int = 90) -> int:
        # Находим подграф с помощью выбранного метода
        sample_graph: StaticGraph = method

        # Алгоритм Флойда-Уоршелла
        cnt_verts = sample_graph.count_vertices()
        shortest_paths = np.zeros((cnt_verts, cnt_verts))
        for i in range(cnt_verts):
            for j in range(cnt_verts):
                if sample_graph.adjacency_matrix[i][j]:
                    shortest_paths[i][j] = 1
                else:
                    shortest_paths[i][j] = 1000000000
        for k in range(cnt_verts):
            for i in range(cnt_verts):
                for j in range(cnt_verts):
                    shortest_paths[i][j] = min(shortest_paths[i][j], shortest_paths[i][k] + shortest_paths[k][j])
        dists = []
        for i in range(cnt_verts):
            for j in range(cnt_verts):
                if shortest_paths[i][j] != 1000000000:
                    dists.append(shortest_paths[i][j])
        dists.sort()
        return dists[int(percentile / 100 * (len(dists) - 1))]
    
    def average_cluster_factor(self) -> float:
        cnt_verts = self.get_largest_connected_component().count_vertices()
        result = 0
        for i in range(cnt_verts):
            i_degree = self.get_largest_connected_component().get_node_set().at[i, "node_degree"]
            if i_degree < 2: 
                continue
            l_u = 0
            for j in range(cnt_verts):
                if i == j or (not self.get_largest_connected_component().adjacency_matrix[i][j]):
                    continue
                for k in range(cnt_verts):
                    if k == j or k == i or (not self.get_largest_connected_component().adjacency_matrix[i][k]):
                        continue
                    if self.get_largest_connected_component().adjacency_matrix[j][k]:
                        l_u += 1

            result += l_u / (i_degree * (i_degree - 1))
        return result / cnt_verts

    def assortative_factor(self) -> float:
        # считаем по формулке из статьи Ньюмана 2002 года
        m = sum(self.get_largest_connected_component().get_node_set()["node_degree"].to_list()) / 2
        cnt_vert = self.get_largest_connected_component().count_vertices()
        r1 = 0
        r2 = 0
        r3 = 0
        for u in range(cnt_vert):
            for v in range(u + 1, cnt_vert):
                if not self.get_largest_connected_component().adjacency_matrix[u][v]:
                    continue
                v_degree = self.get_largest_connected_component().get_node_set().at[u, "node_degree"]
                u_degree = self.get_largest_connected_component().get_node_set().at[v, "node_degree"]

                r1 += u_degree * v_degree
                r2 += (u_degree + v_degree) / 2
                r3 += (u_degree * u_degree + v_degree * v_degree) / 2

        return (r1 - (r2 * r2) / m) / (r3 - (r2 * r2) / m)


@dataclass
class SelectApproach:
    start_node1_index: Optional[int]
    start_node2_index: Optional[int]

    def __init__(self, s_node1_index: int = None, s_node2_index: int = None):
        self.start_node1_index = s_node1_index
        self.start_node2_index = s_node2_index

    
    def snowball_sample(self, graph: StaticGraph) -> StaticGraph:
        queue = list()
        start_node_index1 = self.start_node1_index
        start_node_index2 = self.start_node2_index

        # добавляем две вершины в очередь для BFS
        queue.append(start_node_index1)
        queue.append(start_node_index2)
        cnt_verts = graph.count_vertices()


        size = min(500, cnt_verts)

        sample_graph = StaticGraph(graph.t_min, graph.t_max, size)  # новый граф, который должны получить в результате

        used = [False for _ in range(cnt_verts)]
        used[start_node_index1] = True
        used[start_node_index2] = True

        size -= 2

        while len(queue) > 0:  # BFS
            v = queue.pop(0)
            for i in range(cnt_verts):
                if not graph.adjacency_matrix[v][i]:
                    continue
                if not used[i]:
                    if size > 0:
                        used[i] = True
                        size -= 1
                        queue.append(i)
                    else:
                        continue
                # добавляем рёбра
                v_number = graph.get_node_set().at[v, "number"]
                i_number = graph.get_node_set().at[i, "number"]
                edge_df = graph.get_edge_set().loc[
                    (graph.get_edge_set()["start_node"] == min(v, i)) & 
                    (graph.get_edge_set()["end_node"] ==  max(v, i)),
                ["number", "timestamp"]]
                for _, row in edge_df.iterrows():
                    new_edge = Edge(
                        number=row["number"], 
                        start_node=Node(number=v_number),
                        end_node=Node(number=i_number),
                        timestamp=row["timestamp"]
                    )
                    sample_graph.add_edge(new_edge)
        return sample_graph

    def random_selected_vertices(self, graph: StaticGraph) -> StaticGraph:
        remaining_vertices = [int(i) for i in range(graph.count_vertices())]  # множество оставшихся вершин
        size = min(500, graph.count_vertices())
        sample_graph = StaticGraph(graph.t_min, graph.t_max, size)  # новый граф, который должны получить в результате

        for _ in range(size):
            # выберем новую вершину для добавления в граф
            new_vertice_index = remaining_vertices[np.random.randint(0, len(remaining_vertices))]
            new_vertice_number = graph.get_node_set().loc[
                graph.get_node_set()["number"] == new_vertice_index, 
            "number_in_temporal_graph"].to_list()[0]

            remaining_vertices.remove(new_vertice_index)
            sample_graph.add_node(Node(number=new_vertice_number))
            for vertice_index_sample in range(sample_graph.count_vertices()):
                vertice_index = graph.get_node_set().loc[graph.get_node_set()["number"] == vertice_index_sample, "number"].to_list()[0]
                vertice_number = sample_graph.get_node_set().loc[
                    sample_graph.get_node_set()["number"] == vertice_index_sample, 
                "number_in_temporal_graph"].to_list()[0]
                # если вершины смежны в исходном графе, то добавим рёбра
                if graph.adjacency_matrix[vertice_index][new_vertice_index]:
                    edge_df = graph.get_edge_set().loc[
                        (graph.get_edge_set()["start_node"] == min(vertice_index, new_vertice_index)) &  
                        (graph.get_edge_set()["end_node"] == max(vertice_index, new_vertice_index)),
                    ["number", "timestamp"]]
                    for _, row in edge_df.iterrows():
                        new_edge = Edge(
                            number=row["number"], 
                            start_node=Node(number=new_vertice_number),
                            end_node=Node(number=vertice_number),
                            timestamp=row["timestamp"])
                        sample_graph.add_edge(new_edge)

        return sample_graph
            

    def __call__(self, graph: StaticGraph):
        if self.start_node1_index is None:
            return self.random_selected_vertices(graph)
        return self.snowball_sample(graph)