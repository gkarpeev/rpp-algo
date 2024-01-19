import itertools
import networkx as nx

import pandas as pd

def read_edgelist(edgelist_filename, keep_optional=False):
    """
    Читаем список ребер
    """
    el = pd.read_csv(edgelist_filename, dtype={0: str, 1: str})
    el = el.dropna(how='all')

    if (not keep_optional) & ('required' in el.columns):
        el = el[el['required'] == 1]

    return el


def create_networkx_graph_from_edgelist(edgelist, edge_id='id'):
    """
    Создаем граф из списка ребер
    """
    g = nx.MultiGraph()
    for i, row in enumerate(edgelist.iterrows()):
        edge_attr_dict = row[1][2:].to_dict()
        if edge_id not in edge_attr_dict:
            edge_attr_dict[edge_id] = i
        g.add_edge(row[1][0], row[1][1], **edge_attr_dict)
    return g


def get_odd_nodes(graph):
    """
    Вершины с нечетной степенью
    """
    degree_nodes = []
    for v, d in graph.degree():
        if d % 2 == 1:
            degree_nodes.append(v)
    return degree_nodes


def get_shortest_paths_distances(graph, pairs, edge_weight_name='distance'):
    """
    Считаем кратчайшии расстояние
    """
    distances = {}
    for pair in pairs:
        distances[pair] = nx.dijkstra_path_length(graph, pair[0], pair[1], weight=edge_weight_name)
    return distances


def create_complete_graph(pair_weights, flip_weights=True):
    """
    Создаем полный граф
    """
    g = nx.Graph()
    for k, v in pair_weights.items():
        wt_i = -v if flip_weights else v
        g.add_edge(k[0], k[1], **{'distance': v, 'weight': wt_i})
    return g


def dedupe_matching(matching):
    """
    Удаляем дубликаты
    """
    matched_pairs_w_dupes = [tuple(sorted([k, v])) for k, v in matching.items()]
    return list(set(matched_pairs_w_dupes))


def add_augmenting_path_to_graph(graph, min_weight_pairs, edge_weight_name='weight'):
    """
    Добавляем ребра
    """
    graph_aug = graph.copy()
    for pair in min_weight_pairs:
        graph_aug.add_edge(pair[0],
                           pair[1],
                           **{'distance': nx.dijkstra_path_length(graph, pair[0], pair[1], weight=edge_weight_name),
                              'augmented': True}
                           )
    return graph_aug


def create_eulerian_circuit(graph_augmented, graph_original, start_node=None):
    """
    Эйлеров цикл
    """

    euler_circuit = list(nx.eulerian_circuit(graph_augmented, source=start_node, keys=True))
    assert len(graph_augmented.edges()) == len(euler_circuit), 'graph and euler_circuit do not have equal number of edges.'

    for edge in euler_circuit:
        aug_path = nx.shortest_path(graph_original, edge[0], edge[1], weight='distance')
        edge_attr = graph_augmented[edge[0]][edge[1]][edge[2]]
        if not edge_attr.get('augmented'):
            yield edge + (edge_attr,)
        else:
            for edge_aug in list(zip(aug_path[:-1], aug_path[1:])):
                edge_aug_dict = graph_original[edge_aug[0]][edge_aug[1]]
                edge_key = min(edge_aug_dict.keys(), key=(lambda k: edge_aug_dict[k]['distance']))
                edge_aug_shortest = edge_aug_dict[edge_key]
                edge_aug_shortest['augmented'] = True
                edge_aug_shortest['id'] = edge_aug_dict[edge_key]['id']
                yield edge_aug + (edge_key, edge_aug_shortest, )


def create_required_graph(graph):
    graph_req = graph.copy()

    for e in list(graph_req.edges(data=True, keys=True)):
        if not e[3]['required']:
            graph_req.remove_edge(e[0], e[1], key=e[2])

    for n in list(nx.isolates(graph_req)):
        graph_req.remove_node(n)

    return graph_req


def assert_graph_is_connected(graph):
    """
    Проверяем граф на связность
    """

    assert nx.algorithms.connected.is_connected(graph), "Граф не является связным"

    return True

def RPP(edgelist_filename, start_node=None, edge_weight='distance', verbose=False):
    """
    Решение задачи RPP
    """
    el = read_edgelist(edgelist_filename, keep_optional=True)

    g_full = create_networkx_graph_from_edgelist(el)
    g_req = create_required_graph(g_full)
    assert_graph_is_connected(g_req)

    odd_nodes = get_odd_nodes(g_req)
    odd_node_pairs = list(itertools.combinations(odd_nodes, 2))

    odd_node_pairs_shortest_paths = get_shortest_paths_distances(g_full, odd_node_pairs, edge_weight)

    g_odd_complete = create_complete_graph(odd_node_pairs_shortest_paths, flip_weights=True)
    odd_matching = nx.algorithms.max_weight_matching(g_odd_complete, True)

    g_aug = add_augmenting_path_to_graph(g_req, odd_matching)

    circuit = list(create_eulerian_circuit(g_aug, g_full, start_node))

    return circuit, g_full
