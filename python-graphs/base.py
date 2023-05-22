from typing import Optional
from pydantic import BaseModel
import numpy as np
import pandas as pd


class Node(BaseModel):
    number: int


class Edge(BaseModel):
    number: int
    start_node: int
    end_node: int
    timestamp: int


class StaticGraph(BaseModel):
    node_set: Optional[pd.DataFrame]
    edge_set: Optional[pd.DataFrame]
    adjacency_matrix: np.matrix
    number_of_connected_components: Optional[int]

    def add_node(self, node: Node) -> int:
        pass

    def add_edge(self, edge: Edge) -> int:
        pass

    def count_vertices(self) -> int:
        pass

    def count_edges(self) -> int:
        pass

    def density(self) -> float:
        pass

    def get_largest_connected_component(self):
        pass

    def get_number_of_connected_components(self) -> int:
        pass

    def share_of_vertices(self) -> float: 
        pass

    def get_radius(self, method) -> int:
        pass 

    def get_diameter(self, method) -> int:
        pass

    def percentile_distance(method, percentile = 90) -> float:
        pass
    
    def average_cluster_factor(self) -> float:
        pass

    def assortative_factor(self) -> float:
        pass

    
    def common_neighbours(u: Node, v: Node) -> int:
        pass

    def adamic_adar(u: Node, v: Node) -> float:
        pass

    def jaccard_coefficient(u: Node, v: Node) -> float:
        pass

    def preferential_attachment(u: Node, v: Node) -> int:
        pass



class SelectApproach(BaseModel):

    def snowball_sample(self, graph: StaticGraph, start_node: Node) -> StaticGraph:
        pass

    def random_selected_vertices(self, graph: StaticGraph) -> StaticGraph:
        pass
            
    def __call__(self, graph: StaticGraph, *args) -> StaticGraph:
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
                                        start_node=Node(number=int(item[0])),
                                        end_node=Node(number=int(item[1])),
                                        weight=int(item[2]),
                                        )
                                    ))

    def get_static_graph(t_1: int, t_2: int) -> StaticGraph:
        pass            


class StaticGraph(BaseModel):
    node_set: Optional[pd.DataFrame]
    edge_set: Optional[pd.DataFrame]
    adjacency_matrix: np.matrix
    largest_connected_component: Optional[StaticGraph]
    number_of_connected_components: Optional[int]
    

    def add_node(self, node: Node) -> int:
        # создадим датафрейм для вершин, если такового нет
        if self.node_set is None:
            self.node_set = pd.DataFrame({
                "number": pd.Series(dtype='int'),
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
        # добавим вершину
        self.node_set.loc[len(self.count_vertices)] = {"number": node.number, "node_degree": 0}
        return len(self.count_vertices) - 1

    def add_edge(self, edge: Edge) -> int:
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

        # свапнем вершины, если start_node > end_node
        if edge.start_node > edge.end_node:
            edge.start_node, edge.end_node = edge.end_node, edge.start_node

        # добавим ребро
        self.edge_set.loc[len(self.count_edges)] = {
            "number": edge.number,
            "start_node": edge.start_node,
            "end_node": edge.end_node,
            "timestamp": edge.timestamp,
        }

        start_node_index = -1
        end_node_index = -1

        # если вершин не существует, добавим их, и сохраним их индексы
        if self.node_set.loc[self.node_set["number"] == edge.start_node].empty:
            start_node_index = self.add_node(Node(number=edge.start_node))
        else:
            start_node_index = self.node_set.loc[self.node_set["number"] == edge.start_node].index

        if self.node_set.loc[self.node_set["number"] == edge.end_node].empty:
            end_node_index = self.add_node(Node(number=edge.end_node))
        else:
            end_node_index = self.node_set.loc[self.node_set["number"] == edge.end_node].index
        
        # увеличим степень вершин
        self.node_set.at[start_node_index, "node_degree"] += 1
        self.node_set.at[end_node_index, "node_degree"] += 1

        # обозначим, что вершины смежны
        self.adjacency_matrix[start_node_index][end_node_index] = True
        self.adjacency_matrix[end_node_index][start_node_index] = True
        return len(self.count_edges) - 1

    def count_vertices(self) -> int:
        return len(self.node_set)

    def count_edges(self) -> int:
        return len(self.edge_set)

    def density(self) -> float:
        cnt_vert: int = self.count_vertices()
        return 2 * self.count_edges / (cnt_vert * (cnt_vert - 1))


    def __find_size_of_connected_component(self, used, v) -> int:
        used[v] = True
        sz = 1
        cnt_vert = self.count_vertices()
        for to in range(cnt_vert):
            if self.adjacency_matrix[v][to] is False:
                continue
            if used[to] is False:
                sz += self.__find_size_of_connected_component(used, to)
        return sz
    
    def __find_largest_connected_component(self, used, v):
        # Обойдём всю компоненту слабой связности и запишем её как отдельный граф
        used[v] = True
        cnt_vert = self.count_vertices()
        for to in range(cnt_vert):
            if self.adjacency_matrix[v][to] is False:
                continue
            if used[to] is False:
                self.__find_largest_connected_component(used, to)
            edge_df = self.edge_set.loc[(self.edge_set["start_node"] == v) & (self.edge_set["end_node"] == to), ["number", "start_node", "end_node", "timestamp"]]
            pd.concat(self.largest_connected_component.edge_set, edge_df)

    def __update_number_of_connected_components_and_largest_connected_component(self):
        # Запустим DFS от каждой ещё не посещённой вершины, получая компоненты слабой связности
        # Заодно считаем количество этих компонент и максимальную по мощности компоненту слабой связности
        cnt_vert: int = self.count_vertices()
        used = np.array([False for _ in range(cnt_vert)])
        vertice: int = 0
        self.number_of_components: int = 0
        max_component_size: int = 0
        for i in range(cnt_vert):
            if used[i] is False:
                self.number_of_components += 1
                component_size = self.__find_size_of_connected_component(used, i)
                if component_size > max_component_size:
                    max_component_size = component_size
                    vertice = i

        # Обновляем посещенность вершин для обработки максимальной по мощности компоненты
        # слабой связности
        for i in range(cnt_vert):
            used[i] = False

        # Нашли максимальную по мощности компоненту слабой связности, запишем её в поле
        self.largest_connected_component = StaticGraph()
        self.__find_largest_connected_component(used, vertice)

    def get_largest_connected_component(self) -> StaticGraph:
        # если максимальную по мощность компоненту слабой связности не нашли, найдём
        if self.largest_connected_component is None:
            self.__update_number_of_connected_components_and_largest_connected_component()

        return self.largest_connected_component

    def get_number_of_connected_components(self) -> int:
        # если число компонент слабой связности не нашли, найдём
        if self.number_of_connected_components is None:
            self.__update_number_of_connected_components_and_largest_connected_component()

        return self.number_of_connected_components

    def share_of_vertices(self) -> float: 
        # если максимальную по мощность компоненту слабой связности не нашли, найдём
        if self.largest_connected_component is None:
            self.__update_number_of_connected_components_and_largest_connected_component()

        return self.largest_connected_component.count_vertices() / self.count_vertices()

    def get_radius(self, method: SelectApproach) -> int:
        sample_graph = method(self.largest_connected_component)
        shortest_paths: np.matrix = np.power(sample_graph.adjacency_matrix, sample_graph.count_vertices())
        return np.min([np.max(shortest_paths[i]) for i in range(sample_graph.count_vertices())])
        

    def get_diameter(self, method: SelectApproach) -> int:
        sample_graph = method(self.largest_connected_component)
        shortest_paths: np.matrix = np.power(sample_graph.adjacency_matrix, sample_graph.count_vertices())
        return np.max(shortest_paths)

    def percentile_distance(self, method: SelectApproach, percentile: int = 90) -> float:
        sample_graph = method(self.largest_connected_component)
        cnt_verts = sample_graph.count_vertices()
        shortest_paths: np.matrix = np.power(sample_graph.adjacency_matrix, cnt_verts)
        dists = []
        for i in range(cnt_verts):
            for j in range(cnt_verts):
                if shortest_paths[i][j] > 0:
                    dists.append(shortest_paths[i][j])
        dists.sort()
        return dists[np.ceil(percentile / 100 * len(dists))]
    
    def average_cluster_factor(self) -> float:
        cnt_verts = self.largest_connected_component.count_vertices()
        result = 0
        for i in range(cnt_verts):
            i_degree = self.largest_connected_component.node_set.at[i, "node_degree"]
            if i_degree < 2: 
                continue
            l_u = 0  # будем учитывать каждое ребро два раза, поэтому фактически это 2 * L_u
            for j in range(cnt_verts):
                if i == j or self.largest_connected_component.adjacency_matrix[i][j] is False:
                    continue
                for k in range(cnt_verts):
                    if k == j or k == i or self.largest_connected_component.adjacency_matrix[i][k] is False:
                        continue
                    if self.largest_connected_component.adjacency_matrix[j][k] is True:
                        j_number = self.largest_connected_component.node_set.at[j, "number"]
                        k_number = self.largest_connected_component.node_set.at[k, "number"]
                        l_u += len(self.largest_connected_component.edge_set.loc[
                            ("start_node" == min(j_number, k_number)) & ("end_node" == max(j_number, k_number))
                        ])

            result += l_u / (i_degree * (i_degree - 1))
        return result / cnt_verts

    def assortative_factor(self) -> float:
        # считаем по формулке из статьи Ньюмана 2002 года
        m = self.largest_connected_component.count_edges()
        r1 = 0
        r2 = 0
        r3 = 0
        for i in range(m):
            v = self.largest_connected_component.edge_set.at[i, "start_node"]
            v_degree = self.largest_connected_component.node_set.loc["number" == v, "node_degree"]
            u = self.largest_connected_component.edge_set.at[i, "end_node"]
            u_degree = self.largest_connected_component.node_set.loc["number" == u, "node_degree"]
            
            r1 += u_degree * v_degree
            r2 += (u_degree + v_degree) / 2
            r3 += (u_degree * u_degree + v_degree * v_degree) / 2
        return (r1 - (r2 * r2) / m) / (r3 - (r2 * r2) / m)

    
    def common_neighbours(u: Node, v: Node) -> int:
        pass

    def adamic_adar(u: Node, v: Node) -> float:
        pass

    def jaccard_coefficient(u: Node, v: Node) -> float:
        pass

    def preferential_attachment(u: Node, v: Node) -> int:
        pass


class SelectApproach(BaseModel):
    
    def snowball_sample(self, graph: StaticGraph, start_node1: Node, start_node2: Node):
        queue = list()
        start_node_index1 = graph.node_set.loc["number" == start_node1.number].index
        start_node_index2 = graph.node_set.loc["number" == start_node2.number].index

        # добавляем две вершины в очередь для BFS
        queue.append(start_node_index1)
        queue.append(start_node_index2)
        cnt_verts = graph.count_vertices()

        sample_graph = StaticGraph()  # новый граф, который должны получить в результате
        
        size = 500
        if cnt_verts > 1000:
            size = 1000

        used = np.array([False for _ in range(cnt_verts)])
        used[start_node1] = used[start_node2] = True

        while len(queue) > 0 and size > 0:  # BFS
            v = queue.pop(0)
            for i in range(cnt_verts):
                if graph.adjacency_matrix[v][i] is True and used[v][i] is False:
                    used[v][i] = True
                    # добавляем рёбра
                    sample_graph.adjacency_matrix[v][i] = True
                    sample_graph.adjacency_matrix[i][v] = True
                    v_number = graph.node_set.at[v, "number"]
                    i_number = graph.node_set.at[i, "number"]
                    pd.concat([
                        sample_graph.edge_set, 
                        graph.edge_set.loc[("start_node" == min(v_number, i_number)) & ("end_node" ==  max(v_number, i_number))]
                    ])
                    queue.append(i)
            size -= 1
        queue.clear()
        return sample_graph

    def random_selected_vertices(self, graph: StaticGraph) -> StaticGraph:
        remaining_vertices = graph.node_set["number"].to_list()  # множество оставшихся вершин
        size = 500
        if graph.count_vertices() > 1000:
            size = 1000
        sample_graph = StaticGraph()  # новый граф, который должны получить в результате

        for _ in range(size):
            # выберем новую вершину для добавления в граф
            new_vertice_number = remaining_vertices[np.random.randint(0, len(remaining_vertices))]
            remaining_vertices.remove(new_vertice_number)
            new_vertice_index = sample_graph.add_node(Node(number=new_vertice_number))
            for ind in range(len(sample_graph.node_set)):
                vertice_number = sample_graph.node_set.at[ind, "number"]
                # если вершины смежны в исходном графе, то добавим рёбра
                if graph.adjacency_matrix[ind][new_vertice_index] is True:
                    sample_graph.adjacency_matrix[ind][new_vertice_index] = True
                    sample_graph.adjacency_matrix[new_vertice_index][ind] = True
                    pd.concat([
                        sample_graph.edge_set,
                        graph.edge_set.loc[
                            ("start_node" == min(new_vertice_number, vertice_number)) & 
                            ("end_node" == max(new_vertice_number, vertice_number))
                        ]
                    ])
        return sample_graph
            

    def __call__(self, graph: StaticGraph, *args):
        if args is None:
            return self.random_selected_vertices(graph)
        return self.snowball_sample(graph, args[0])