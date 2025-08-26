import pandas as pd
import numpy as np
import networkx as nx
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
import re
from itertools import combinations

# --- Step 1: Data Loading and Cleaning ---
def load_and_clean_data(filepath):
    print("Step 1: Loading and cleaning data...")
    try:
        df_raw = pd.read_csv(filepath, encoding='utf-8-sig')
        # Keep a copy for the "before" view
        df_before = df_raw.head().to_dict(orient='records')
        df = df_raw.copy()
    except Exception as e:
        raise ValueError(f"Error loading file: {e}")

    required_columns = ['承租人', '出租人', '承租人所属地区', '申万行业一级', '财产价值（万元）', '期限']
    df.dropna(subset=required_columns, inplace=True)
    df.columns = df.columns.str.strip()

    str_cols = ['承租人', '出租人', '承租人所属地区', '申万行业一级']
    for col in str_cols:
        df[col] = df[col].astype(str).str.strip()

    df['省份'] = df['承租人所属地区'].apply(lambda x: x.split('-')[0])
    df['财产价值（万元）'] = pd.to_numeric(df['财产价值（万元）'].astype(str).str.replace(',', ''), errors='coerce').fillna(0)

    def parse_term(term_str):
        term_str = str(term_str)
        numbers = re.findall(r'\d+\.?\d*', term_str)
        if not numbers: return np.nan
        val = float(numbers[0])
        return val / 12 if '月' in term_str else val

    df['期限（年）'] = df['期限'].apply(parse_term)
    df.dropna(subset=['期限（年）'], inplace=True)

    df_after = df[['承租人', '出租人', '省份', '申万行业一级', '财产价值（万元）', '期限（年）']].head().to_dict(orient='records')

    return df, df_before, df_after

# --- Step 2: Feature Engineering (Binning) ---
def perform_binning(df, n_value_bins=12, n_term_bins=5):
    print("Step 2: Performing feature binning...")
    # Value Binning
    df['价值分箱'] = pd.qcut(df['财产价值（万元）'], q=n_value_bins, labels=False, duplicates='drop')
    df['价值分箱'] = '价值' + df['价值分箱'].astype(str)

    # Term Binning
    kmeans = KMeans(n_clusters=n_term_bins, random_state=42, n_init=10)
    term_data = df[['期限（年）']].values
    df['期限分箱_raw'] = kmeans.fit_predict(term_data)
    cluster_order = df.groupby('期限分箱_raw')['期限（年）'].mean().sort_values().index
    label_mapping = {old_label: f'期限{i+1}' for i, old_label in enumerate(cluster_order)}
    df['期限分箱'] = df['期限分箱_raw'].map(label_mapping)
    df = df.drop(columns=['期限分箱_raw'])

    value_bin_counts = df['价值分箱'].value_counts().to_dict()
    term_bin_counts = df['期限分箱'].value_counts().to_dict()

    return df, {"value_bins": value_bin_counts, "term_bins": term_bin_counts}

# --- Step 3: Graph Building and Recommendation ---
def build_and_recommend(df, query, n_province_clusters=5, top_n=10):
    print("Step 3: Building graph and running recommendation...")

    # --- Graph Building ---
    G = nx.DiGraph()

    # Province Similarity
    province_industry_matrix = pd.crosstab(df['省份'], df['申万行业一级'])
    province_profiles = normalize(province_industry_matrix, norm='l1', axis=1)
    kmeans = KMeans(n_clusters=n_province_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(province_profiles)
    province_industry_matrix['cluster'] = clusters
    similarity_matrix = cosine_similarity(province_profiles)
    similarity_df = pd.DataFrame(similarity_matrix, index=province_industry_matrix.index, columns=province_industry_matrix.index)

    # Node and Edge Weight Pre-calculation
    prov_totals = df.groupby('省份').size()
    ind_totals = df.groupby('申万行业一级').size()
    val_totals = df.groupby('价值分箱').size()
    term_totals = df.groupby('期限分箱').size()
    prov_lessee_counts = df.groupby(['省份', '承租人']).size().reset_index(name='count')
    ind_lessee_counts = df.groupby(['申万行业一级', '承租人']).size().reset_index(name='count')
    val_lessee_counts = df.groupby(['价值分箱', '承租人']).size().reset_index(name='count')
    term_lessee_counts = df.groupby(['期限分箱', '承租人']).size().reset_index(name='count')
    lessee_lessor_counts = df.groupby(['承租人', '出租人']).size().reset_index(name='count')

    # Add Nodes
    node_types = {
        'province': set(df['省份']), 'industry': set(df['申万行业一级']),
        'value_bin': set(df['价值分箱']), 'term_bin': set(df['期限分箱']),
        'lessee': set(df['承租人']), 'lessor': set(df['出租人'])
    }
    for n_type, nodes in node_types.items():
        for node in nodes: G.add_node(node, node_type=n_type)

    # Add Edges
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

    for _, row in lessee_lessor_counts.iterrows():
        G.add_edge(row['承租人'], row['出租人'], weight=row['count'])

    for i in range(n_province_clusters):
        cluster_provinces = province_industry_matrix[province_industry_matrix['cluster'] == i].index.tolist()
        for p1, p2 in combinations(cluster_provinces, 2):
            similarity = similarity_df.loc[p1, p2]
            if similarity > 0:
                G.add_edge(p1, p2, weight=similarity)
                G.add_edge(p2, p1, weight=similarity)

    print(f"Graph built with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")

    # --- Personalized PageRank ---
    personalization = {
        query['province']: 0.25,
        query['industry']: 0.25,
        query['value_bin']: 0.25,
        query['term_bin']: 0.25,
    }

    pagerank_scores = nx.pagerank(G, alpha=0.85, personalization=personalization, weight='weight')

    lessor_scores = {n: s for n, s in pagerank_scores.items() if G.nodes[n].get('node_type') == 'lessor'}
    sorted_lessors = sorted(lessor_scores.items(), key=lambda item: item[1], reverse=True)

    # Format for response
    response_data = [
        {"rank": i + 1, "lessor": lessor, "score": round(score * 1000, 4)}
        for i, (lessor, score) in enumerate(sorted_lessors[:top_n])
    ]

    return response_data
