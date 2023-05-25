Networks = ['email-Eu-core-temporal', 'munmun_digg_reply', 'opsahi-ucsocial','radoslaw_email','soc-sign-bitcoinalpha', 'sx-mathoverflow']

networks_files_names = [ f'datasets/{i}/out.{i}' for i in Networks]
number_of_datasets = 6
datasets_info = {'Network': ['email-Eu-core-temporal', 'munmun_digg_reply', 'opsahi-ucsocial','radoslaw_email','soc-sign-bitcoinalpha', 'sx-mathoverflow'],
'Label': ['EU','D-rep','UC','Rado','bitA','SX-MO'],
'Category': ['Social',"Social","Information","Social","Social","Social"],
'Edge type': ['Multi','Simple','Multi','Multi','Simple','Multi'],
'Path': networks_files_names}

datasets_info = pd.DataFrame(datasets_info)

# Таблица признаков для графа
def get_stats(network_file_name: str):
    
    tmpGraph = graphs.TemporalGraph(network_file_name)
    staticGraph = tmpGraph.get_static_graph(0.2, 0.6)
    snowball_sample_approach = graphs.SelectApproach(0, 5)
    random_selected_vertices_approach = graphs.SelectApproach()
    sg_sb = snowball_sample_approach(staticGraph.get_largest_connected_component())
    sg_rsv = random_selected_vertices_approach(staticGraph.get_largest_connected_component())
    
    # ск - снежный ком
    # свв - случайный выбор вершин
    return {
        'Сеть': network_file_name['Label'],
        'Категория': network_file_name['Category'],
        'Вершины': staticGraph.count_vertices(), 
        'Тип ребер': network_file_name['Edge type'],
        'Ребра':staticGraph.count_edges(),
        'Плотность графа':staticGraph.density(),
        'Доля вершин':staticGraph.share_of_vertices(),
        'Компоненты с/с':staticGraph.get_number_of_connected_components(),
        'Вершины в наибольшей компоненте с/с':staticGraph.get_largest_connected_component().count_vertices(),
        'Ребра в наибольшей компоненте с/с':staticGraph.get_largest_connected_component().count_edges(),
        'Радиус графа(ск)': staticGraph.get_radius(sg_sb),
        'Диаметр графа(ск)': staticGraph.get_diameter(sg_sb),
        '90 проц. расстояния(ск)': staticGraph.percentile_distance(sg_sb),
        'Радиус графа(свв)': staticGraph.get_radius(sg_rsv),
        'Диаметр графа(свв)': staticGraph.get_diameter(sg_rsv),
        '90 проц.расстояния(свв)': staticGraph.percentile_distance(sg_rsv),
        'Коэф.ассортативности': staticGraph.assortative_factor(),
        'Сред.класт.коэф.сети': staticGraph.average_cluster_factor(),
        'AUC': get_performance(tmpGraph),
    }


def graph_features_tables(datasets_info: pd.DataFrame):

    table = pd.DataFrame([get_stats(network) for index, network in datasets_info.iterrows()]).sort_values('Вершины')
    
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