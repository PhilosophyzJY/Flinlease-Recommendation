import pandas as pd
import numpy as np
import networkx as nx
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
import re
from itertools import combinations

# --- Step 1: Data Loading and Basic Cleaning ---
def load_data(filepath):
    print("  - Loading and cleaning data...")
    try:
        df = pd.read_csv(filepath, encoding='utf-8-sig')
    except Exception as e:
        raise ValueError(f"Error loading file: {e}")

    required_columns = ['承租人', '出租人', '承租人所属地区', '申万行业一级', '财产价值（万元）', '期限', '披露日期']
    df.dropna(subset=required_columns, inplace=True)
    df.columns = df.columns.str.strip()

    str_cols = ['承租人', '出租人', '承租人所属地区', '申万行业一级']
    for col in str_cols:
        df[col] = df[col].astype(str).str.strip()

    df['财产价值（万元）'] = pd.to_numeric(df['财产价值（万元）'].astype(str).str.replace(',', ''), errors='coerce').fillna(0)

    def parse_term(term_str):
        term_str = str(term_str)
        numbers = re.findall(r'\d+\.?\d*', term_str)
        if not numbers: return np.nan
        val = float(numbers[0])
        return val / 12 if '月' in term_str else val

    df['期限（年）'] = df['期限'].apply(parse_term)
    df['披露日期'] = pd.to_datetime(df['披露日期'], errors='coerce')
    df.dropna(subset=['期限（年）', '披露日期'], inplace=True)
    df['省份'] = df['承租人所属地区'].apply(lambda x: x.split('-')[0])
    return df

# --- Step 2: Feature Engineering (Fitting the Binners) ---
def fit_binners(df, n_value_bins=12, n_term_bins=5, n_province_clusters=5):
    print("  - Fitting binning rules on training data...")
    binning_artifacts = {}

    # Fit Value Binner (Quantile)
    _, bin_boundaries = pd.qcut(df['财产价值（万元）'], q=n_value_bins, labels=False, duplicates='drop', retbins=True)
    binning_artifacts['value_bins'] = bin_boundaries

    # Fit Term Binner (KMeans)
    kmeans_term = KMeans(n_clusters=n_term_bins, random_state=42, n_init=10)
    term_data = df[['期限（年）']].values
    kmeans_term.fit(term_data)

    df_temp = df.copy()
    df_temp['期限分箱_raw'] = kmeans_term.predict(term_data)
    cluster_order = df_temp.groupby('期限分箱_raw')['期限（年）'].mean().sort_values().index
    label_mapping = {old_label: f'期限{i+1}' for i, old_label in enumerate(cluster_order)}

    binning_artifacts['term_binner_model'] = kmeans_term
    binning_artifacts['term_binner_labels'] = label_mapping

    # Fit Province Binner (KMeans on industry profiles)
    province_industry_matrix = pd.crosstab(df['省份'], df['申万行业一级'])
    province_profiles = normalize(province_industry_matrix, norm='l1', axis=1)
    kmeans_prov = KMeans(n_clusters=n_province_clusters, random_state=42, n_init=10)
    kmeans_prov.fit(province_profiles)

    binning_artifacts['province_binner'] = kmeans_prov
    binning_artifacts['province_pivot'] = province_industry_matrix

    print("  - Binning rules fitted and saved.")
    return binning_artifacts

# --- Step 2b: Feature Engineering (Applying the Bins) ---
def transform_data(df, binning_artifacts):
    print("  - Applying pre-fitted binning rules...")
    df = df.copy()

    # Transform Value
    value_bins = binning_artifacts['value_bins']
    value_bins[0] -= 0.001 # Ensure the lowest value is included
    df['价值分箱'] = pd.cut(df['财产价值（万元）'], bins=value_bins, labels=[f'价值{i}' for i in range(len(value_bins)-1)], include_lowest=True)

    # Transform Term
    kmeans = binning_artifacts['term_binner_model']
    label_mapping = binning_artifacts['term_binner_labels']
    term_data = df[['期限（年）']].values
    df['期限分箱_raw'] = kmeans.predict(term_data)
    df['期限分箱'] = df['期限分箱_raw'].map(label_mapping)
    df = df.drop(columns=['期限分箱_raw'])

    return df

# --- Step 3: Graph Building and Recommendation Logic (remains mostly the same) ---
def build_heterogeneous_graph(df, lessee_lessor_weights, province_binner, province_pivot):
    """
    Builds the final heterogeneous graph.
    Accepts pre-computed lessee-lessor weights for optimization efficiency.
    """
    G = nx.DiGraph()

    # Province Similarity
    province_pivot_local = province_pivot.copy()
    province_profiles = normalize(province_pivot_local, norm='l1', axis=1)
    clusters = province_binner.predict(province_profiles)
    province_pivot_local['cluster'] = clusters
    similarity_matrix = cosine_similarity(province_profiles)
    similarity_df = pd.DataFrame(similarity_matrix, index=province_pivot_local.index, columns=province_pivot_local.index)

    # Pre-calculation for Attribute -> Lessee edges
    prov_totals = df.groupby('省份').size()
    ind_totals = df.groupby('申万行业一级').size()
    val_totals = df.groupby('价值分箱').size()
    term_totals = df.groupby('期限分箱').size()
    prov_lessee_counts = df.groupby(['省份', '承租人']).size().reset_index(name='count')
    ind_lessee_counts = df.groupby(['申万行业一级', '承租人']).size().reset_index(name='count')
    val_lessee_counts = df.groupby(['价值分箱', '承租人']).size().reset_index(name='count')
    term_lessee_counts = df.groupby(['期限分箱', '承租人']).size().reset_index(name='count')

    # Add Nodes
    node_types = {
        'province': set(df['省份']), 'industry': set(df['申万行业一级']),
        'value_bin': set(df['价值分箱']), 'term_bin': set(df['期限分箱']),
        'lessee': set(df['承租人']), 'lessor': set(df['出租人'])
    }
    for n_type, nodes in node_types.items():
        for node in nodes: G.add_node(node, node_type=n_type)

    # Add Attribute -> Lessee Edges
    edge_builders = [
        (prov_lessee_counts, '省份', '承租人', prov_totals),
        (ind_lessee_counts, '申万行业一级', '承租人', ind_totals),
        (val_lessee_counts, '价值分箱', '承租人', val_totals),
        (term_lessee_counts, '期限分箱', '承租人', term_totals),
    ]
    for counts_df, source_col, target_col, totals_map in edge_builders:
        for _, row in counts_df.iterrows():
            weight = row['count'] / totals_map.get(row[source_col], 1)
            G.add_edge(row[source_col], row[target_col], weight=weight)

    # Add Lessee -> Lessor Edges
    if lessee_lessor_weights is None: # Default behavior for recommender.py
        t_max = df['披露日期'].max()
        lambda_decay = 0.005
        df['time_decayed_value'] = df['财产价值（万元）'] * np.exp(-lambda_decay * (t_max - df['披露日期']).dt.days)
        global_total_weighted_value = df['time_decayed_value'].sum()
        if global_total_weighted_value > 0:
            lessee_lessor_weights = df.groupby(['承租人', '出租人'])['time_decayed_value'].sum().reset_index()
            lessee_lessor_weights['weight'] = lessee_lessor_weights['time_decayed_value'] / global_total_weighted_value
        else:
            lessee_lessor_weights = pd.DataFrame(columns=['承租人', '出租人', 'weight'])

    for _, row in lessee_lessor_weights.iterrows():
        if row['weight'] > 0:
            G.add_edge(row['承租人'], row['出租人'], weight=row['weight'])

    # Add Province -> Province Edges
    n_province_clusters = province_binner.n_clusters
    similarity_df_normalized = similarity_df.div(similarity_df.sum(axis=1), axis=0).fillna(0)
    for i in range(n_province_clusters):
        cluster_provinces = province_pivot_local[province_pivot_local['cluster'] == i].index.tolist()
        for p1, p2 in combinations(cluster_provinces, 2):
            similarity_p1_p2 = similarity_df_normalized.loc[p1, p2]
            if similarity_p1_p2 > 0: G.add_edge(p1, p2, weight=similarity_p1_p2)
            similarity_p2_p1 = similarity_df_normalized.loc[p2, p1]
            if similarity_p2_p1 > 0: G.add_edge(p2, p1, weight=similarity_p2_p1)

    return G

def get_recommendations(G, query, top_n=10):
    """
    Runs Personalized PageRank and scales the scores.
    """
    personalization = {
        query['province']: 0.25,
        query['industry']: 0.25,
        query['value_bin']: 0.25,
        query['term_bin']: 0.25,
    }

    pagerank_scores = nx.pagerank(G, alpha=0.85, personalization=personalization, weight='weight')

    lessor_scores = {n: s for n, s in pagerank_scores.items() if G.nodes[n].get('node_type') == 'lessor'}
    if not lessor_scores: return []

    min_score, max_score = min(lessor_scores.values()), max(lessor_scores.values())
    scaled_scores = {}
    for lessor, score in lessor_scores.items():
        if max_score == min_score: scaled_score = 300
        else:
            normalized = (score - min_score) / (max_score - min_score)
            scaled_score = 300 + normalized * (850 - 300)
        scaled_scores[lessor] = int(round(scaled_score))

    sorted_lessors = sorted(scaled_scores.items(), key=lambda item: item[1], reverse=True)
    return sorted_lessors[:top_n]
