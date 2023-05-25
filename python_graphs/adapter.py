from .base import TemporalGraph, StaticGraph


def adapter(tmpG: TemporalGraph, staticG: StaticGraph) -> (pd.DataFrame, pd.DataFrame, pnd.NDArrayBool, int, int):
    '''
    return: tuple(edge, node, adjacency_matrix, min_timestamp, max_timestamp)
    '''
    return (staticG.get_edge_set(),
            staticG.get_node_set(),
            staticG.adjacency_matrix.copy(),
            tmpG.get_min_timestamp(),
            tmpG.get_max_timestamp())
