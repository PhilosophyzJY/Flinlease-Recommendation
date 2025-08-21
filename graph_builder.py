import pandas as pd
import networkx as nx

def create_province_subgraph(G, df, target_province):
    """
    Creates a subgraph containing only lessees from a target province and their connected lessors.
    """
    lessees_in_province = df[df['承租人所属省份'] == target_province]['承租人'].unique()
    connected_lessors = set()
    for lessee in lessees_in_province:
        if G.has_node(lessee):
            connected_lessors.update(G.successors(lessee))
    subgraph_nodes = set(lessees_in_province).union(connected_lessors)
    return G.subgraph(subgraph_nodes)

def create_industry_subgraph(G, df, target_industry):
    """
    Creates a subgraph containing only lessees from a target industry and their connected lessors.
    """
    lessees_in_industry = df[df['申万行业一级'] == target_industry]['承租人'].unique()
    connected_lessors = set()
    for lessee in lessees_in_industry:
        if G.has_node(lessee):
            connected_lessors.update(G.successors(lessee))
    subgraph_nodes = set(lessees_in_industry).union(connected_lessors)
    return G.subgraph(subgraph_nodes)
