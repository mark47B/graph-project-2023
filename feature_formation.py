# Gamma = Г(u) - число соседей вершины u (степень вершины u)
# n = |V| - число вершин
# m = |E_G| - число ребер
# l=0.2
# t_min - самая ранняя наблюдаемая отметка по всем ребрам сети
# t_max - самая последняя наблюдаемая отметка по всем ребрам сети
# t - временная метка ребра
# 
# Вектор признаков описывает ребро черех активности узлов, которое оно соединяет
# Вектор признаков длиной 3*7*4 = 84

import numpy as np
import pandas as pd

def node_degree(numbers_of_nodes:np.array, adjacency_matrix:np.array):
    return np.sum(adjacency_matrix[numbers_of_nodes,:],axis=1)
def common_neighbours(u:int, v:int, adjacency_matrix:np.array) -> int:
    return np.sum(np.where(adjacency_matrix[u,:]+adjacency_matrix[v,:] != 0, 1, 0))
def adamic_adar(u:int, v:int, adjacency_matrix:np.array) -> float:
    return np.sum(1/node_degree(np.nonzero(adjacency_matrix[u,:]+adjacency_matrix[v,:]),adjacency_matrix))
def jaccard_coefficient(u:int, v:int, adjacency_matrix:np.array) -> float:
    return np.sum(np.where(adjacency_matrix[u,:]-adjacency_matrix[v,:] == 0, 1, 0))/np.sum(
        np.where(adjacency_matrix[u,:]+adjacency_matrix[v,:] != 0, 1, 0))
def preferential_attachment(u:int, v:int, adjacency_matrix:np.array) -> int:
    return np.prod(node_degree(np.array([u,v]),adjacency_matrix))

# 3 функции для вычисления весов
# Во все функции в качестве t можно передавать np массив временных меток всех ребер
def temporal_weighting_w_linear(t:np.array,t_min:int,t_max:int,l:float=0.2) -> float:
    return l+(1-l)*(t-t_min)/(t_max-t_min) # w_linear
def temporal_weighting_w_exponential(t:np.array,t_min:float,t_max:float,l:float=0.2) -> float:
    return l+(1-l)*(np.exp(3*(t-t_min)/(t_max-t_min))-1)/(np.exp(3)-1) # w_exponential 
def temporal_weighting_w_square_root(t:np.array,t_min:float,t_max:float,l:float=0.2) -> float:
    return l+(1-l)*np.sqrt((t-t_min)/(t_max-t_min)) #w_square_root

# 7 функций для аггрегации весов ребер, прилегающих к узлу
# edge_weights - np array размерности 2; i-ая строка - веса ребер, прилегающих к i-ой вершине
def aggregation_of_node_activity_zeroth_quantile(edges_weights:list)-> float:
    return [np.quantile(weights_set,0) for weights_set in edges_weights]
def aggregation_of_node_activity_first_quantile(edges_weights:list)-> float:
    return [np.quantile(weights_set,0.25) for weights_set in edges_weights]
def aggregation_of_node_activity_second_quantile(edges_weights:list)-> float:
    return [np.quantile(weights_set,0.5) for weights_set in edges_weights]
def aggregation_of_node_activity_third_quantile(edges_weights:list)-> float:
    return [np.quantile(weights_set,0.75) for weights_set in edges_weights]
def aggregation_of_node_activity_fourth_quantile(edges_weights:list)-> float:
    return [np.quantile(weights_set,1) for weights_set in edges_weights]
def aggregation_of_node_activity_sum(edges_weights:list)-> float:
    return [np.sum(weights_set,0) for weights_set in edges_weights]
def aggregation_of_node_activity_mean(edges_weights:list)-> float:
    return [np.mean(weights_set,0) for weights_set in edges_weights]
    
# 4 функции для объединения статистик по парам узлов
def combining_node_activity_sum(aggregarion_node_one:np.array, aggregation_node_two:np.array)-> float:
    return aggregarion_node_one + aggregation_node_two
def combining_node_activity_absolute_diference(aggregarion_node_one:np.array, aggregation_node_two:np.array)-> float:
    return np.abs(aggregarion_node_one - aggregation_node_two)
def combining_node_activity_minimum(aggregarion_node_one:np.array, aggregation_node_two:np.array)-> float:
    return np.min(np.concatenate([aggregarion_node_one[:, np.newaxis], aggregation_node_two[:, np.newaxis]], axis=1), axis=1)
def combining_node_activity_maximum(aggregarion_node_one:np.array, aggregation_node_two:np.array)-> float:
    return np.max(np.concatenate([aggregarion_node_one[:, np.newaxis], aggregation_node_two[:, np.newaxis]], axis=1), axis=1)

def temporal_weighting(edge: pd.DataFrame, t_min: int, t_max: int):
    '''
    Взвешивание во времени: расчет трех весов для каждого ребра по их временным меткам
    '''
    edge['weight_linear'] = temporal_weighting_w_linear(edge['last_timestamp'],t_min,t_max)
    edge['weight_exponential'] = temporal_weighting_w_exponential(edge['last_timestamp'],t_min,t_max)
    edge['weight_square_root'] = temporal_weighting_w_square_root(edge['last_timestamp'],t_min,t_max)

def replace_nan(df, columns, start_node_column, end_node_column):
    '''
    Замена NaN на [] в числовых ячейках и сопоставление номеров для вершин,
    которые были только либо в end_node, либо в start_node
    '''
    for column in columns:
        df[column] = df[column].apply(lambda x: [] if not isinstance(x, list) else x)
    df[start_node_column] = np.where(not isinstance(df[start_node_column], int), df[end_node_column], df[start_node_column])
    df[end_node_column] = np.where(not isinstance(df[end_node_column], int), df[start_node_column], df[end_node_column])
    return df

def aggregation_of_node_activity(node: pd.DataFrame, edges_weights_for_node: pd.DataFrame):
    '''
    Агрегация активности узлов на основе 7 функций: 
    нулевой, первый,второй,третий,чертвертый квантили; сумма и среднее
    по весам ребер смежных с вершиной
    '''
    node['node_activity_zeroth_quantile_wl'] = aggregation_of_node_activity_zeroth_quantile(edges_weights_for_node["weight_linear"])
    node['node_activity_first_quantile_wl'] = aggregation_of_node_activity_first_quantile(edges_weights_for_node["weight_linear"])
    node['node_activity_second_quantile_wl'] = aggregation_of_node_activity_second_quantile(edges_weights_for_node["weight_linear"])
    node['node_activity_third_quantile_wl'] = aggregation_of_node_activity_third_quantile(edges_weights_for_node["weight_linear"])
    node['node_activity_fourth_quantile_wl'] = aggregation_of_node_activity_fourth_quantile(edges_weights_for_node["weight_linear"])
    node['node_activity_sum_wl'] = aggregation_of_node_activity_sum(edges_weights_for_node["weight_linear"])
    node['node_activity_mean_wl'] = aggregation_of_node_activity_mean(edges_weights_for_node["weight_linear"])
    
    node['node_activity_zeroth_quantile_we'] = aggregation_of_node_activity_zeroth_quantile(edges_weights_for_node["weight_exponential"])
    node['node_activity_first_quantile_we'] = aggregation_of_node_activity_first_quantile(edges_weights_for_node["weight_exponential"])
    node['node_activity_second_quantile_we'] = aggregation_of_node_activity_second_quantile(edges_weights_for_node["weight_exponential"])
    node['node_activity_third_quantile_we'] = aggregation_of_node_activity_third_quantile(edges_weights_for_node["weight_exponential"])
    node['node_activity_fourth_quantile_we'] = aggregation_of_node_activity_fourth_quantile(edges_weights_for_node["weight_exponential"])
    node['node_activity_sum_we'] = aggregation_of_node_activity_sum(edges_weights_for_node["weight_exponential"])
    node['node_activity_mean_we'] = aggregation_of_node_activity_mean(edges_weights_for_node["weight_exponential"])
    
    node['node_activity_zeroth_quantile_wsr'] = aggregation_of_node_activity_zeroth_quantile(edges_weights_for_node["weight_square_root"])
    node['node_activity_first_quantile_wsr'] = aggregation_of_node_activity_first_quantile(edges_weights_for_node["weight_square_root"])
    node['node_activity_second_quantile_wsr'] = aggregation_of_node_activity_second_quantile(edges_weights_for_node["weight_square_root"])
    node['node_activity_third_quantile_wsr'] = aggregation_of_node_activity_third_quantile(edges_weights_for_node["weight_square_root"])
    node['node_activity_fourth_quantile_wsr'] = aggregation_of_node_activity_fourth_quantile(edges_weights_for_node["weight_square_root"])
    node['node_activity_sum_wsr'] = aggregation_of_node_activity_sum(edges_weights_for_node["weight_square_root"])
    node['node_activity_mean_wsr'] = aggregation_of_node_activity_mean(edges_weights_for_node["weight_square_root"])

def combining_node_activity(node: pd.DataFrame)->pd.DataFrame:
    '''
    Объединение активности узлов для формирования векторного описания
    ребра на основе 4 функций:
    сумма, абсолютная разность, мин, макс 
    по парным активностям инцидентных вершин
    '''
    values = node[['node_activity_zeroth_quantile_wl',
    'node_activity_first_quantile_wl',
    'node_activity_second_quantile_wl',
    'node_activity_third_quantile_wl',
    'node_activity_fourth_quantile_wl',
    'node_activity_sum_wl',
    'node_activity_mean_wl',
    
    'node_activity_zeroth_quantile_we',
    'node_activity_first_quantile_we',
    'node_activity_second_quantile_we',
    'node_activity_third_quantile_we',
    'node_activity_fourth_quantile_we',
    'node_activity_sum_we',
    'node_activity_mean_we',
    
    'node_activity_zeroth_quantile_wsr',
    'node_activity_first_quantile_wsr',
    'node_activity_second_quantile_wsr',
    'node_activity_third_quantile_wsr',
    'node_activity_fourth_quantile_wsr',
    'node_activity_sum_wsr',
    'node_activity_mean_wsr']].values

    num_of_nodes = node.shape[0]
    num_of_feature = 21

    combined_values = []
    start_nodes_vector = []

    for i, row in enumerate(values):
        repeated_row = np.tile(row, num_of_nodes - i - 1)
        repeated_node = np.tile(i, num_of_nodes - i - 1)
        
        combined_values.extend(repeated_row)
        start_nodes_vector.extend(repeated_node)

    # Преобразование объединенных значений в массив NumPy
    first_combined_array = np.array(combined_values)
    
    end_nodes_vector = list(np.concatenate([np.arange(i, num_of_nodes) for i in range(1, num_of_nodes)]))
    
    # Замена номеров строк на значения из DataFrame и объединение в один вектор
    second_combined_array = np.concatenate([values[end_nodes_vector[i]] for i in range(len(end_nodes_vector))])
    
    feature_by_sym = combining_node_activity_sum(first_combined_array,second_combined_array)
    feature_by_abs_dif = combining_node_activity_absolute_diference(first_combined_array,second_combined_array)
    feature_by_min = combining_node_activity_minimum(first_combined_array,second_combined_array)
    feature_by_max = combining_node_activity_maximum(first_combined_array,second_combined_array)

    
    all_feature = [np.concatenate([feature_by_sym[i:i+num_of_feature], 
                                   feature_by_abs_dif[i:i+num_of_feature], 
                                   feature_by_min[i:i+num_of_feature], 
                                   feature_by_max[i:i+num_of_feature]]) 
                   for i in range(0, feature_by_sym.shape[0], num_of_feature)]
    
    feature_column_name = "feature_vector"
    Edge_feature = pd.DataFrame({"start_node":start_nodes_vector, "end_node":end_nodes_vector, feature_column_name: all_feature})
    return (Edge_feature,feature_column_name)

def make_edges_weights_adjacent_to_node(edge: pd.DataFrame):
    '''
    Формирование датафрейма с весами примыкающих к вершине ребер.
    Строка соответствует определенной вершине.
    '''
    grouped_by_start_node = edge.drop(['end_node',"number","last_timestamp"],
                                      axis=1).groupby("start_node").agg(
    {'weight_linear': list, 'weight_exponential': list,
     'weight_square_root': list}).reset_index()
    grouped_by_end_node = edge.drop(['start_node',"number","last_timestamp"],
                                    axis=1).groupby("end_node").agg(
    {'weight_linear': list, 'weight_exponential': list,
     'weight_square_root': list}).reset_index()

    edges_weights_for_node = grouped_by_start_node.merge(
        grouped_by_end_node, left_on='start_node', right_on='end_node', how='outer')


    edges_weights_for_node = replace_nan(edges_weights_for_node,
                                         ['weight_linear_x', 'weight_exponential_x',
                                          "weight_square_root_x","weight_linear_y",
                                          'weight_exponential_y', "weight_square_root_y"],
                                         'start_node',"end_node")

    edges_weights_for_node["weight_linear"]=edges_weights_for_node[
        "weight_linear_x"] + edges_weights_for_node["weight_linear_y"]
    edges_weights_for_node["weight_exponential"]=edges_weights_for_node[
        "weight_exponential_x"]+edges_weights_for_node["weight_exponential_y"]
    edges_weights_for_node["weight_square_root"]=edges_weights_for_node[
        "weight_square_root_x"]+edges_weights_for_node["weight_square_root_y"]
    edges_weights_for_node = edges_weights_for_node.drop(
        ["end_node",'weight_linear_x', 'weight_exponential_x',
         "weight_square_root_x","weight_linear_y", 'weight_exponential_y',
         "weight_square_root_y"], axis=1)

    edges_weights_for_node['start_node'] = edges_weights_for_node['start_node'].astype(int)
    edges_weights_for_node[[
        "weight_linear","weight_exponential","weight_square_root"]] = edges_weights_for_node[[
        "weight_linear","weight_exponential","weight_square_root"]].apply(lambda x: np.array(x))

    edges_weights_for_node=edges_weights_for_node.sort_values(by='start_node')

    return edges_weights_for_node

def split_list_cell(df: pd.DataFrame, column_name:str):
    '''
    Разбиение списка на отдельные столбцы с автоматической генерацией имен
    '''
    return pd.concat([df.drop(column_name, axis=1),
                df[column_name].apply(lambda x: pd.Series(x))],
               axis=1)

def merge_with_temporal_graph_number(df: pd.DataFrame,node: pd.DataFrame):
    '''
    Сопоставление номера в статичном графе номеру из временного графа
    '''
    df = pd.merge(df, node[['number',"number_in_temporal_graph"]],
                            left_on='start_node', right_on='number')
    
    df = pd.merge(df, node[['number',"number_in_temporal_graph"]],
                            left_on='end_node', right_on='number')
    
    df = df.drop(["number_x","number_y"], axis=1)
    df = df.rename(columns={'number_in_temporal_graph_x': 'number_in_temporal_graph_start_node'})
    df = df.rename(columns={'number_in_temporal_graph_y': 'number_in_temporal_graph_end_node'})
    
    return df

def count_static_topological_features(df: pd.DataFrame):
    '''
    Рассчет статичных топологических признаков
    '''
    df["common_neighbours"] = df.apply(lambda row: common_neighbours(row["start_node"],row["end_node"],adjacency_matrix), axis=1)
    df["adamic_adar"] = df.apply(lambda row: adamic_adar(row["start_node"],row["end_node"],adjacency_matrix), axis=1)
    df["jaccard_coefficient"] = df.apply(lambda row: jaccard_coefficient(row["start_node"],row["end_node"],adjacency_matrix), axis=1)
    df["preferential_attachment"] = df.apply(lambda row: preferential_attachment(row["start_node"],row["end_node"],adjacency_matrix), axis=1)

def feature_for_edges(edge: pd.DataFrame, node: pd.DataFrame, adjacency_matrix: np.array, t_min:int, t_max:int):
    '''
    Получение датафрейма с признаками для ребер
    '''
    temporal_weighting(edge, t_min, t_max)
    
    edges_weights_adjacent_to_node = make_edges_weights_adjacent_to_node(edge)
    
    aggregation_of_node_activity(node,edges_weights_adjacent_to_node)
    
    Edge_feature,feature_column_name = combining_node_activity(node)

    Edge_feature = merge_with_temporal_graph_number(Edge_feature, node)
    
    count_static_topological_features(Edge_feature)
            
    Edge_feature = split_list_cell(Edge_feature, feature_column_name)

    return Edge_feature

# Example:
#       Edge_feature = feature_for_edges(edge, node, adjacency_matrix, 0, 20000000000)





