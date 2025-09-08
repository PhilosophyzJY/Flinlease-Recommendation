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
    """
    Loads and cleans data, now using '起始日期' as the primary date column.
    Drops rows where '起始日期' is missing.
    """
    print("  - Loading and cleaning data...")
    try:
        df = pd.read_csv(filepath, encoding='utf-8-sig')
    except Exception as e:
        raise ValueError(f"Error loading file: {e}")

    # Define required columns, now including '起始日期'
    required_columns = ['承租人', '出租人', '承租人所属地区', '申万行业一级', '财产价值（万元）', '期限', '起始日期']
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

    # Use '起始日期' as the primary date, parse it, and drop rows if it's invalid
    df['transaction_date'] = pd.to_datetime(df['起始日期'], errors='coerce')
    df.dropna(subset=['期限（年）', 'transaction_date'], inplace=True)

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

# --- Step 3: Core Algorithm Logic ---

def calculate_advanced_scores(df, lambda_decay, attribute_weights):
    """
    Calculates the advanced, weighted composite score for each transaction.
    """
    df = df.copy()
    if df.empty or 'transaction_date' not in df.columns or df['transaction_date'].isnull().all():
        return df.assign(final_score=pd.Series(dtype='float64'))

    # --- Calculate context-dependent average terms ---
    # For a given attribute, the avg_term is based on that attribute AND the value bin
    df['avg_term_province'] = df.groupby(['省份', '价值分箱'])['期限（年）'].transform('mean')
    df['avg_term_industry'] = df.groupby(['申万行业一级', '价值分箱'])['期限（年）'].transform('mean')
    df['avg_term_term'] = df.groupby(['期限分箱', '价值分箱'])['期限（年）'].transform('mean')
    # For the value attribute itself, the context is just the value bin
    df['avg_term_value'] = df.groupby('价值分箱')['期限（年）'].transform('mean')

    # Fill any NaNs that might result from rare combinations
    for col in ['avg_term_province', 'avg_term_industry', 'avg_term_term', 'avg_term_value']:
        df[col].fillna(df['期限（年）'].mean(), inplace=True)

    # --- Calculate time decay component ---
    t_max = df['transaction_date'].max()
    today = t_max + pd.offsets.MonthEnd(0)
    days_ago = (today - df['transaction_date']).dt.days

    # --- Calculate context-dependent composite scores ---
    base_value = df['财产价值（万元）']
    df['score_province'] = base_value * np.exp(df['avg_term_province'] - lambda_decay * days_ago)
    df['score_industry'] = base_value * np.exp(df['avg_term_industry'] - lambda_decay * days_ago)
    df['score_value'] = base_value * np.exp(df['avg_term_value'] - lambda_decay * days_ago)
    df['score_term'] = base_value * np.exp(df['avg_term_term'] - lambda_decay * days_ago)

    # --- Calculate final weighted score ---
    w = attribute_weights
    df['final_score'] = (w['province'] * df['score_province'] +
                         w['industry'] * df['score_industry'] +
                         w['value'] * df['score_value'] +
                         w['term'] * df['score_term'])

    return df


def build_heterogeneous_graph(df, province_binner, province_pivot):
    """
    Builds the final heterogeneous graph.
    Assumes that the input DataFrame `df` already contains the 'final_score' column.
    """
    G = nx.DiGraph()

    # Province Similarity (unchanged)
    province_pivot_local = province_pivot.copy()
    province_profiles = normalize(province_pivot_local, norm='l1', axis=1)
    clusters = province_binner.predict(province_profiles)
    province_pivot_local['cluster'] = clusters
    similarity_matrix = cosine_similarity(province_profiles)
    similarity_df = pd.DataFrame(similarity_matrix, index=province_pivot_local.index, columns=province_pivot_local.index)

    # Add Nodes
    node_types = {
        'province': set(df['省份']), 'industry': set(df['申万行业一级']),
        'value_bin': set(df['价值分箱']), 'term_bin': set(df['期限分箱']),
        'lessee': set(df['承租人']), 'lessor': set(df['出租人'])
    }
    for n_type, nodes in node_types.items():
        for node in nodes:
            if pd.notna(node): G.add_node(node, node_type=n_type)

    # --- Edge Calculation based on final_score ---

    # Add Attribute -> Lessee Edges
    attribute_cols = ['省份', '申万行业一级', '价值分箱', '期限分箱']
    for attr_col in attribute_cols:
        attr_totals = df.groupby(attr_col)['final_score'].sum()
        attr_lessee_sums = df.groupby([attr_col, '承租人'])['final_score'].sum().reset_index()

        for _, row in attr_lessee_sums.iterrows():
            attr_node, lessee_node, score_sum = row[attr_col], row['承租人'], row['final_score']
            total_score = attr_totals.get(attr_node, 1)
            if total_score > 0:
                weight = score_sum / total_score
                if weight > 0 and pd.notna(attr_node) and pd.notna(lessee_node):
                    G.add_edge(attr_node, lessee_node, weight=weight)

    # Add Lessee -> Lessor Edges (Normalized per Lessee)
    lessee_totals = df.groupby('承租人')['final_score'].sum()
    lessee_lessor_sums = df.groupby(['承租人', '出租人'])['final_score'].sum()

    lessee_lessor_weights = (lessee_lessor_sums / lessee_totals).reset_index(name='weight')

    for _, row in lessee_lessor_weights.iterrows():
        lessee, lessor, weight = row['承租人'], row['出租人'], row['weight']
        if weight > 0 and pd.notna(lessee) and pd.notna(lessor):
            G.add_edge(lessee, lessor, weight=weight)

    # Add Province -> Province Edges (unchanged)
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

def get_recommendations(G, query, attribute_weights, top_n=10):
    """
    Runs Personalized PageRank and scales the scores.
    The personalization vector is now weighted by the optimized attribute weights.
    """
    # Create the personalization vector based on the query and optimized weights
    personalization = {
        query['province']: attribute_weights.get('province', 0.25),
        query['industry']: attribute_weights.get('industry', 0.25),
        query['value_bin']: attribute_weights.get('value', 0.25),
        query['term_bin']: attribute_weights.get('term', 0.25),
    }

    # Filter out any query nodes that might not be in the graph
    personalization = {k: v for k, v in personalization.items() if G.has_node(k)}

    # Normalize the personalization vector to ensure its values sum to 1
    total_weight = sum(personalization.values())
    if total_weight > 0:
        personalization = {k: v / total_weight for k, v in personalization.items()}
    else:
        # If no query nodes are in the graph, we cannot proceed.
        print("Warning: None of the query nodes are in the graph. Returning empty list.")
        return []

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
