import pandas as pd
import numpy as np
import networkx as nx
from scipy.stats import chi2_contingency
import pickle

# --- Configuration ---
INPUT_DATA_FILE = 'finlease_train.csv'
OUTPUT_DATA_PKL = 'data.pkl'
OUTPUT_GRAPH_PKL = 'graph.gpickle'

# --- All data processing and graph building functions ---

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

# --- Main Execution Block ---
if __name__ == '__main__':
    print(f"Starting training process using data from '{INPUT_DATA_FILE}'...")

    # 1. Load Data
    data = load_data(INPUT_DATA_FILE)
    print("Data loaded and cleaned.")

    # 2. Feature Engineering (Binning)
    # NOTE: Chi-Merge binning was too slow for the environment.
    # Reverting to faster, unsupervised quantile-based binning.
    data['价值分箱'] = pd.qcut(data['财产价值（万元）'], q=12, labels=False, duplicates='drop')
    print("Value binning complete (using quantiles).")

    # 3. Build Graph
    G_global = build_graph(data)
    print("Global graph built.")

    # 4. Save Artifacts
    with open(OUTPUT_DATA_PKL, 'wb') as f:
        pickle.dump(data, f)
    print(f"Processed data saved to '{OUTPUT_DATA_PKL}'.")

    with open(OUTPUT_GRAPH_PKL, 'wb') as f:
        pickle.dump(G_global, f)
    print(f"Graph object saved to '{OUTPUT_GRAPH_PKL}'.")

    print("\nTraining process complete.")
