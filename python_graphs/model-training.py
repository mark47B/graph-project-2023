from .base import TemporalGraph
from .feature_formation import feature_for_edges
import sklearn
import numpy as np


def train_test_split_temporal_graph(edge_list:list, split_ratio: float):
    '''
    Разделение выборки на части формирования признаков и на части предсказания
    '''
    edge_list_feature_build_part = edge_list[:int(len(edge_list)*split_ratio)]
    edge_list_prediction_part = edge_list[len(edge_list_feature_build_part):]
    return (edge_list_feature_build_part,edge_list_prediction_part)




def get_performance(temporalG: TemporalGraph, split_ratio: float):

    t_min = temporalG.get_min_timestamp()
    t_max = temporalG.get_max_timestamp()
    
    edge_list_feature_build_part, edge_list_prediction_part = train_test_split_temporal_graph(temporalG.edge_list, split_ratio)
    
    'формирование статик графа и получение для него edge,node,adjacency_matrix'
    build_static_graph = temporalG.get_static_graph(0, split_ratio, False)

    edge_feature_build_part = build_static_graph.get_edge_set()
    node_feature_build_part = build_static_graph.get_node_set()

    prediction_static_graph = temporalG.get_static_graph(split_ratio, 1, True)
    
    Edge_feature = feature_for_edges(edge_feature_build_part, node_feature_build_part, build_static_graph.adjacency_matrix, t_min, t_max)
    
    node_prediction_part = prediction_static_graph.get_node_set()
    edge_prediction_part = prediction_static_graph.get_edge_set()

    # выделить в отдельную функцию по присоединению 
    # номеров вершин из темпорального графа к edge_list
    edge_prediction_part = edge_prediction_part.merge(
        node_prediction_part['number','number_in_temporal_graph'], left_on='start_node', right_on='number', how='left')
    
    edge_prediction_part = edge_prediction_part.rename(columns={'number_in_temporal_graph': 'number_in_temporal_graph_start_node'})

    
    edge_prediction_part = edge_prediction_part.merge(
        node_prediction_part['number','number_in_temporal_graph'], left_on='end_node', right_on='number', how='left')
    
    edge_prediction_part = edge_prediction_part.rename(columns={'number_in_temporal_graph': 'number_in_temporal_graph_end_node'})

    
    
    
    Edge_feature = Edge_feature.merge(
        edge_prediction_part['number_in_temporal_graph_start_node','number_in_temporal_graph_end_node','number'], 
        left_on=['number_in_temporal_graph_start_node','number_in_temporal_graph_end_node'], 
        right_on=['number_in_temporal_graph_start_node','number_in_temporal_graph_end_node'], 
        how='left')

    
    Edge_feature['number'] = Edge_feature['number'].apply(lambda x: 0 if np.isnan(x) else 0)

    X = Edge_feature.drop(['number'])
    
    y = Edge_feature['number']
    
    X_train, X_test, y_train, y_test = (
        sklearn.model_selection.train_test_split(X, y, random_state=random_state))
    
    pipe = sklearn.pipeline.make_pipeline(
            sklearn.preprocessing.StandardScaler(),
            sklearn.linear_model.LogisticRegression(max_iter=10000, n_jobs=-1, 
                                                    random_state=42)
        )
    pipe.fit(X_train, y_train)
    
    auc = sklearn.metrics.roc_auc_score(
        y_true=y_test, y_score=pipe.predict_proba(X_test)[:,1])
    
    return auc