import os
import glob
import tqdm
import numpy as np
import networkx as nx
import graphviz as gv
from collections import defaultdict


def add_node_attributes(graph, nodelist):
    """
    Добавляем аттрибуты
    """
    for i, row in nodelist.iterrows():
        nx.set_node_attributes(graph, {row['id']: row.to_dict()})
    return graph


def add_pos_node_attribute(graph, origin='bottomleft'):
    """
    Определяем позицию
    """

    ori = {
        'bottomleft': {'X': 1, 'Y': 1},
        'topleft': {'X': 1, 'Y': -1},
        'topright': {'X': -1, 'Y': -1},
        'bottomright': {'X': -1, 'Y': 1}
    }[origin]

    for node_id in graph.nodes():
        try:
            graph.nodes[node_id]['pos'] = "{},{}!".format(ori['X']*graph.nodes[node_id]['X']/100,
                                                          ori['Y']*graph.nodes[node_id]['Y']/100)
        except KeyError as e:
            print(e)
            print('No X, Y coordinates found for node: {}'.format(node_id))
    return graph


def prepare_networkx_graph_circuit_for_transformation_to_graphviz(circuit, graph, edge_label_attr=None):
    """
    Преобразование данных
    """
    edge_cnter = defaultdict(lambda: 0)
    for i, e in enumerate(circuit):
        eid = e[3]['id']
        key = e[2]

        if eid not in edge_cnter:
            graph[e[0]][e[1]][key]['label'] = str(graph[e[0]][e[1]][key][edge_label_attr]) if edge_label_attr else str(i)
            graph[e[0]][e[1]][key]['penwidth'] = 1
            graph[e[0]][e[1]][key]['decorate'] = 'true'
        else:
            if edge_label_attr is None:
                graph[e[0]][e[1]][key]['label'] += ', ' + str(i)
            graph[e[0]][e[1]][key]['penwidth'] += 3
        edge_cnter[eid] += 1

    return graph


def convert_networkx_graph_to_graphiz(graph, directed=False):
    """
    Преобразуем в нужный формат
    """
    if directed:
        G = gv.Digraph()
    else:
        G = gv.Graph()

    for n in graph.nodes():
        n_attr = {k: str(v) for k, v in graph.nodes[n].items()}
        G.attr('node', n_attr)
        G.node(str(n), str(n))

    for e in graph.edges(keys=True):
        e_attr = {k: str(v) for k, v in graph[e[0]][e[1]][e[2]].items()}
        G.edge(str(e[0]), str(e[1]), **e_attr)

    return G


def plot_graphviz(graph, filename=None, format='svg', engine='dot', edge_label_attr=None,
                  graph_attr={'strict': 'false', 'forcelabels': 'true'}, node_attr=None, edge_attr=None):
    """
    Визуализация
    """

    if edge_label_attr:
        for i, e in enumerate(graph.edges(data=True, keys=True)):
            key = e[2]
            graph[e[0]][e[1]][key]['label'] = str(graph[e[0]][e[1]][key][edge_label_attr])

    graph_gv = convert_networkx_graph_to_graphiz(graph, directed=False)
    graph_gv.engine = engine
    graph_gv.format = format

    if graph_attr:
        for k, v in graph_attr.items():
            graph_gv.graph_attr[k] = v

    if node_attr:
        for k, v in node_attr.items():
            graph_gv.node_attr[k] = v

    if edge_attr:
        for k, v in edge_attr.items():
            graph_gv.edge_attr[k] = v

    if filename:
        graph_gv.render(filename=filename, view=False)

    return "Plot written to {}".format(filename)


def plot_circuit_graphviz(circuit, graph, filename=None, format='svg', engine='dot', edge_label_attr=None,
                          graph_attr={'strict': 'false', 'forcelabels': 'true'}, node_attr=None, edge_attr=None):
    graph_gv = prepare_networkx_graph_circuit_for_transformation_to_graphviz(circuit, graph, edge_label_attr)
    return plot_graphviz(graph_gv, filename, format, engine, edge_label_attr, graph_attr, node_attr, edge_attr)

import csv
from io import StringIO

def create_csv_from_dataframe(df):
    """
    Создаем csv файл
    """

    csvfile = StringIO()
    csvfile.seek(0)
    fieldnames = df.columns
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for row in df.iterrows():
        writer.writerow(row[1].to_dict())
    csvfile.seek(0)

    return csvfile
