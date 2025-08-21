import networkx as nx

def run_pagerank_on_subgraph(subgraph, weight_key, top_n=10):
    """
    Runs PageRank on a given subgraph using a specified edge weight.
    """
    if subgraph.number_of_edges() == 0:
        return []
    try:
        pagerank_scores = nx.pagerank(subgraph, alpha=0.85, weight=weight_key)
    except nx.PowerIterationFailedConvergence:
        print(f"Warning: PageRank failed to converge for weight_key='{weight_key}'.")
        return []
    lessor_scores = {n: s for n, s in pagerank_scores.items() if subgraph.nodes[n].get('type') == 'lessor'}
    sorted_lessors = sorted(lessor_scores.items(), key=lambda item: item[1], reverse=True)
    return [lessor for lessor, score in sorted_lessors[:top_n]]
