import pandas as pd
import numpy as np
import networkx as nx
from scipy.stats import chi2_contingency

def chimerge_binning(df: pd.DataFrame, feature: str, target: str, max_bins: int = 12, initial_bins: int = 50):
    """
    Performs a hybrid Chi-Merge binning. First, it creates coarse bins using quantiles,
    then merges them using Chi-Square statistic for optimal performance and statistical significance.
    """
    df_clean = df[[feature, target]].dropna()
    try:
        df_clean['coarse_bin'] = pd.qcut(df_clean[feature], q=initial_bins, labels=False, duplicates='drop')
    except ValueError:
        df_clean['coarse_bin'] = pd.cut(df_clean[feature], bins=initial_bins, labels=False, duplicates='drop')

    contingency_df = pd.crosstab(df_clean['coarse_bin'], df_clean[target])
    bin_boundaries = df_clean.groupby('coarse_bin', observed=False)[feature].agg(['min', 'max'])
    contingency_tables = [row.values for _, row in contingency_df.iterrows()]
    intervals = [pd.Interval(row['min'], row['max']) for _, row in bin_boundaries.iterrows()]

    while len(contingency_tables) > max_bins:
        min_chi2 = np.inf
        merge_idx = -1
        for i in range(len(contingency_tables) - 1):
            combined_table = contingency_tables[i] + contingency_tables[i+1]
            if np.any(combined_table.sum(axis=0) == 0): continue
            if combined_table.ndim > 1 and np.any(combined_table.sum(axis=1) == 0): continue
            try:
                chi2, _, _, _ = chi2_contingency(combined_table)
                if chi2 < min_chi2:
                    min_chi2 = chi2
                    merge_idx = i
            except (ValueError, ZeroDivisionError):
                continue
        if merge_idx == -1: break
        contingency_tables[merge_idx] += contingency_tables.pop(merge_idx + 1)
        intervals[merge_idx] = pd.Interval(intervals[merge_idx].left, intervals.pop(merge_idx + 1).right)

    final_boundaries = sorted(list(set([i.left for i in intervals] + [intervals[-1].right])))
    return final_boundaries

def load_data(filepath):
    df = pd.read_csv(filepath, encoding='utf-8-sig')
    required_columns = ['承租人', '出租人', '承租人所属地区', '出租人所属地区', '申万行业一级', '财产价值（万元）', '出租人企业类型']
    df.dropna(subset=required_columns, inplace=True)
    df.columns = df.columns.str.strip()
    str_cols = ['承租人', '出租人', '承租人所属地区', '出租人所属地区', '申万行业一级', '出租人企业类型']
    for col in str_cols:
        df[col] = df[col].astype(str).str.strip()
    df['出租人企业类型'] = df['出租人企业类型'].apply(lambda x: x.split(',')[0])
    df['财产价值（万元）'] = df['财产价值（万元）'].astype(str).str.replace(',', '')
    df['财产价值（万元）'] = pd.to_numeric(df['财产价值（万元）'], errors='coerce').fillna(0)
    df['承租人所属省份'] = df['承租人所属地区'].apply(lambda x: x.split('-')[0] if isinstance(x, str) and '-' in x else x)
    df['出租人所属省份'] = df['出租人所属地区'].apply(lambda x: x.split('-')[0] if isinstance(x, str) and '-' in x else x)
    return df

def create_province_subgraph(G, df, target_province):
    lessees_in_province = df[df['承租人所属省份'] == target_province]['承租人'].unique()
    connected_lessors = set()
    for lessee in lessees_in_province:
        if G.has_node(lessee):
            connected_lessors.update(G.successors(lessee))
    subgraph_nodes = set(lessees_in_province).union(connected_lessors)
    return G.subgraph(subgraph_nodes)

def create_industry_subgraph(G, df, target_industry):
    lessees_in_industry = df[df['申万行业一级'] == target_industry]['承租人'].unique()
    connected_lessors = set()
    for lessee in lessees_in_industry:
        if G.has_node(lessee):
            connected_lessors.update(G.successors(lessee))
    subgraph_nodes = set(lessees_in_industry).union(connected_lessors)
    return G.subgraph(subgraph_nodes)

def build_graph(df):
    G = nx.DiGraph()
    lessee_attrs = df.drop_duplicates(subset=['承租人'])[['承租人', '承租人所属省份', '申万行业一级']].set_index('承租人')
    lessor_attrs = df.drop_duplicates(subset=['出租人'])[['出租人', '出租人所属省份']].set_index('出租人')
    edge_data = df.groupby(['承租人', '出租人']).agg(
        count=('承租人', 'size'),
        total_value=('财产价值（万元）', 'sum')
    ).reset_index()
    for lessee, attrs in lessee_attrs.iterrows():
        G.add_node(lessee, type='lessee', region=attrs['承租人所属省份'], industry=attrs['申万行业一级'])
    for lessor, attrs in lessor_attrs.iterrows():
        G.add_node(lessor, type='lessor', region=attrs['出租人所属省份'])
    for _, row in edge_data.iterrows():
        if G.has_node(row['承租人']) and G.has_node(row['出租人']):
            G.add_edge(row['承租人'], row['出租人'], count=row['count'], total_value=row['total_value'])
    return G

def run_pagerank_on_subgraph(subgraph, weight_key, top_n=10):
    if subgraph.number_of_edges() == 0: return []
    try:
        pagerank_scores = nx.pagerank(subgraph, alpha=0.85, weight=weight_key)
    except nx.PowerIterationFailedConvergence:
        print(f"Warning: PageRank failed to converge for weight_key='{weight_key}'.")
        return []
    lessor_scores = {n: s for n, s in pagerank_scores.items() if subgraph.nodes[n].get('type') == 'lessor'}
    sorted_lessors = sorted(lessor_scores.items(), key=lambda item: item[1], reverse=True)
    return [lessor for lessor, score in sorted_lessors[:top_n]]

if __name__ == '__main__':
    data = load_data('finlease_train.csv')
    if data is not None:
        print("Data loaded successfully.")
        bin_boundaries = chimerge_binning(data, '财产价值（万元）', '出租人企业类型', max_bins=12)
        bin_boundaries[0] -= 0.001
        data['价值分箱'] = pd.cut(data['财产价值（万元）'], bins=bin_boundaries, labels=range(1, len(bin_boundaries)), include_lowest=True)
        print("Binning complete.")
        G_global = build_graph(data)
        print("Global graph built successfully.")
        target_province = '山东'
        target_industry = '农林牧渔'
        print(f"\n--- Generating 4-dimensional recommendations for Province='{target_province}' and Industry='{target_industry}' ---")
        G_province = create_province_subgraph(G_global, data, target_province)
        G_industry = create_industry_subgraph(G_global, data, target_industry)
        print(f"Province subgraph created with {G_province.number_of_nodes()} nodes and {G_province.number_of_edges()} edges.")
        print(f"Industry subgraph created with {G_industry.number_of_nodes()} nodes and {G_industry.number_of_edges()} edges.")
        rec_prov_count = run_pagerank_on_subgraph(G_province, 'count')
        rec_prov_value = run_pagerank_on_subgraph(G_province, 'total_value')
        rec_ind_count = run_pagerank_on_subgraph(G_industry, 'count')
        rec_ind_value = run_pagerank_on_subgraph(G_industry, 'total_value')
        print("\n--- Report 1: Province-Based, Ranked by Transaction Count ---")
        for i, r in enumerate(rec_prov_count, 1): print(f"{i}. {r}")
        print("\n--- Report 2: Province-Based, Ranked by Transaction Value ---")
        for i, r in enumerate(rec_prov_value, 1): print(f"{i}. {r}")
        print("\n--- Report 3: Industry-Based, Ranked by Transaction Count ---")
        for i, r in enumerate(rec_ind_count, 1): print(f"{i}. {r}")
        print("\n--- Report 4: Industry-Based, Ranked by Transaction Value ---")
        for i, r in enumerate(rec_ind_value, 1): print(f"{i}. {r}")
