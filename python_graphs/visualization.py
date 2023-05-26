import pandas as pd
import numpy as np
import python_graphs.base as graphs
import python_graphs.model_training as mdtr

import importlib
importlib.reload(graphs)

def get_stats(network_info):
    
    tmpGraph = graphs.TemporalGraph(network_info['Path'])
    print('загрузили граф')
    staticGraph = tmpGraph.get_static_graph(0.2, 0.6)
    print('создали статичный')
    #snowball_sample_approach = graphs.SelectApproach(0, 5)
    #random_selected_vertices_approach = graphs.SelectApproach()
    print('snowball START')
    #sg_sb = snowball_sample_approach(staticGraph.get_largest_connected_component())
    print('random select START')
    #sg_rsv = random_selected_vertices_approach(staticGraph.get_largest_connected_component())
    print('Получили статистику графа')
    # ск - снежный ком
    # свв - случайный выбор вершин
    # return {
    #     'Сеть': network_info['Label'],
    #     'Категория': network_info['Category'],
    #     'Вершины': staticGraph.count_vertices(),
    #     'Тип ребер': network_info['Edge type'],
    #     'Ребра':staticGraph.count_edges(),
    #     'Плотность графа':staticGraph.density(),
    #     'Доля вершин':staticGraph.share_of_vertices(),
    #     'Компоненты с/с':staticGraph.get_number_of_connected_components(),
    #     'Вершины в наибольшей компоненте с/с':staticGraph.get_largest_connected_component().count_vertices(),
    #     'Ребра в наибольшей компоненте с/с':staticGraph.get_largest_connected_component().count_edges(),
    #     'Радиус графа(ск)': staticGraph.get_radius(sg_sb),
    #     'Диаметр графа(ск)': staticGraph.get_diameter(sg_sb),
    #     '90 проц. расстояния(ск)': staticGraph.percentile_distance(sg_sb),
    #     'Радиус графа(свв)': staticGraph.get_radius(sg_rsv),
    #     'Диаметр графа(свв)': staticGraph.get_diameter(sg_rsv),
    #     '90 проц.расстояния(свв)': staticGraph.percentile_distance(sg_rsv),
    #     'Коэф.ассортативности': staticGraph.assortative_factor(),
    #     'Сред.класт.коэф.сети': staticGraph.average_cluster_factor(),
    #     'AUC': mdtr.get_performance(tmpGraph, 0.67),
    # }
    return {
        'Сеть': network_info['Label'],
        'Категория': network_info['Category'],
        'Вершины': 0,
        'Тип ребер': network_info['Edge type'],
        'Ребра': 0,
        'Плотность графа': 0,
        'Доля вершин': 0,
        'Компоненты с/с': 0,
        'Вершины в наибольшей компоненте с/с': 0,
        'Ребра в наибольшей компоненте с/с': 0,
        'Радиус графа(ск)': 0,
        'Диаметр графа(ск)': 0,
        '90 проц. расстояния(ск)': 0,
        'Радиус графа(свв)': 0,
        'Диаметр графа(свв)': 0,
        '90 проц.расстояния(свв)': 0,
        'Коэф.ассортативности': 0,
        'Сред.класт.коэф.сети': 0,
        'AUC': mdtr.get_performance(tmpGraph, 0.67),
    }


def graph_features_tables(datasets_info: pd.DataFrame):

    table = pd.DataFrame([get_stats(network_info) for index, network_info in datasets_info.iterrows()]).sort_values('Вершины')
    print(table)
    columns_to_include_to_feature_network_table = [
        'Сеть',
        'Категория',
        'Вершины', 
        'Тип ребер',
        'Ребра',
        'Плотность графа',
        'Доля вершин',
        'Компоненты с/с',
        'Вершины в наибольшей компоненте с/с',
        'Ребра в наибольшей компоненте с/с',
        'Радиус графа(ск)',
        'Диаметр графа(ск)',
        '90 проц. расстояния(ск)',
        'Радиус графа(свв)',
        'Диаметр графа(свв)',
        '90 проц.расстояния(свв)',
        'Коэф.ассортативности',
        'Сред.класт.коэф.сети',
    ]
    columns_to_include_to_auc_table = [
        'Сеть',
        'AUC',
    ]
    latex_feature_network_table = table.to_latex(
        formatters={
            'Вершины': lambda x: f'{x:,}', 
            'Ребра': lambda x: f'{x:,}',
            'Плотность графа': lambda x: f'{x:.2f}',
            'Доля вершин': lambda x: f'{x:.2f}',
            'Компоненты с/с': lambda x: f'{x:,}',
            'Вершины в наибольшей компоненте с/с': lambda x: f'{x:,}',
            'Ребра в наибольшей компоненте с/с': lambda x: f'{x:,}',
            'Радиус графа(ск)': lambda x: f'{x:.2f}',
            'Диаметр графа(ск)': lambda x: f'{x:.2f}',
            '90 проц. расстояния(ск)': lambda x: f'{x:.2f}',
            'Радиус графа(свв)': lambda x: f'{x:.2f}',
            'Диаметр графа(свв)': lambda x: f'{x:.2f}',
            '90 проц.расстояния(свв)': lambda x: f'{x:.2f}',
            'Коэф.ассортативности': lambda x: f'{x:.2f}',
            'Сред.класт.коэф.сети': lambda x: f'{x:.2f}',
        },
        column_format='l@{\hspace{1em}}c@{\hspace{1em}}c@{\hspace{0.5em}}r@{\hspace{1em}}r@{\hspace{1em}}c@{\hspace{1em}}c@{\hspace{1em}}c@{\hspace{1em}}c@{\hspace{0.5em}}c',
        index=False,
        caption=(
            "Признаки для сетей, рассмотренных в ходе работы "
        ),
        label='Таблица: Признаки сетей',
        escape=False,
        multicolumn=False,
        columns=columns_to_include_to_feature_network_table
    )
    latex_auc_table = table.to_latex(
        formatters={
            'AUC': lambda x: f'{x:.2f}',
        },
        column_format='l@{\hspace{1em}}c@{\hspace{1em}}c@{\hspace{0.5em}}r@{\hspace{1em}}r@{\hspace{1em}}c@{\hspace{1em}}c@{\hspace{1em}}c@{\hspace{1em}}c@{\hspace{0.5em}}c',
        index=False,
        caption=(
            "Точность пердсказания появления ребер"
        ),
        label='Таблица: AUC',
        escape=False,
        multicolumn=False,
        columns=columns_to_include_to_auc_table
    )
    return (latex_feature_network_table,latex_auc_table)