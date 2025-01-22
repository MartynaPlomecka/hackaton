from langchain.graphs.networkx_graph import NetworkxEntityGraph
import networkx as nx
from graph_plt import plot_graph
import webbrowser
from langchain.graphs.networkx_graph import KnowledgeTriple
import neonx
import json


class Graph:
    def __init__(self):
        self.graph = NetworkxEntityGraph()
    def triple_add(self,source_node,target_node, edge_relation):
        triple = KnowledgeTriple(source_node, edge_relation, target_node)
        self.graph.add_triple(triple)
    def get_neo4j_graph(self):
        nx_graph = nx.DiGraph()  # Add nodes and edges

        for triple in self.graph.get_triples():
            source_node, target_node, edge_relation = triple
            source_node = json.JSONEncoder(source_node)
            target_node = json.JSONEncoder(target_node)
            edge_relation = json.JSONEncoder(json.JSONEncoder)
            nx_graph.add_node(source_node)
            nx_graph.add_node(target_node)
            nx_graph.add_edge(source_node, target_node, relation=edge_relation)



        data = neonx.get_geoff(nx_graph, "LINKS_TO")
        return data

    def plot(self):
        nx_graph = nx.DiGraph()  # Add nodes and edges
        for triple in self.graph.get_triples():
            source_node, target_node, edge_relation = triple
            nx_graph.add_node(source_node)
            nx_graph.add_node(target_node)
            nx_graph.add_edge(source_node, target_node, relation=edge_relation)
        plot_graph(nx_graph)
        webbrowser.open(f'KG.html')
