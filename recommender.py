import configparser
import pandas as pd
import numpy as np
from utils import load_and_engineer_features, build_heterogeneous_graph, get_recommendations

def run_recommender():
    """
    Runs the final recommendation pipeline using optimized parameters from config.
    """
    # 1. Read Config
    config = configparser.ConfigParser()
    config.read('config.ini')

    data_file = config['Paths']['data_file']

    # Load optimized parameters
    best_lambda = config.getfloat('Optimized_Parameters', 'best_lambda')
    best_w_freq = config.getfloat('Optimized_Parameters', 'best_w_freq')
    best_w_val = config.getfloat('Optimized_Parameters', 'best_w_val')

    print("--- Running Recommender with Optimized Parameters ---")
    print(f"  - Data file: {data_file}")
    print(f"  - Using lambda: {best_lambda:.4f}")
    print(f"  - Using w_freq: {best_w_freq:.2f}")
    print(f"  - Using w_val: {best_w_val:.2f}")

    # 2. Load full dataset
    df = load_and_engineer_features(data_file)
    if df is None:
        return

    # 3. Calculate final edge weights using optimized params
    # This logic is duplicated from the optimizer, but is necessary here for the final build
    lessee_total_counts = df.groupby('承租人')['承租人'].transform('size')
    pair_counts = df.groupby(['承租人', '出租人'])['承租人'].transform('size')
    freq_prop = pair_counts / lessee_total_counts
    lessee_total_value = df.groupby('承租人')['财产价值（万元）'].transform('sum')
    pair_values = df.groupby(['承租人', '出租人'])['财产价值（万元）'].transform('sum')
    val_prop = pair_values / lessee_total_value
    t_max = df['披露日期'].max()
    time_decay = np.exp(-best_lambda * (t_max - df['披露日期']).dt.days)
    df['composite_score'] = (freq_prop * best_w_freq + val_prop * best_w_val) * time_decay
    lessee_lessor_weights = df.groupby(['承租人', '出租人'])['composite_score'].sum().reset_index(name='weight')

    # 4. Build graph with final weights
    G = build_heterogeneous_graph(df, lessee_lessor_weights=lessee_lessor_weights)

    # 5. Run a sample recommendation
    prov = '山东'
    ind = '农林牧渔'
    sample_query_df = df[(df['省份'] == prov) & (df['申万行业一级'] == ind)]

    if not sample_query_df.empty:
        query = {
            "province": prov,
            "industry": ind,
            "value_bin": sample_query_df['价值分箱'].mode()[0],
            "term_bin": sample_query_df['期限分箱'].mode()[0],
        }
        print(f"\n--- Running Sample Recommendation for: {query} ---")
        recommendations = get_recommendations(G, query)

        print("\n--- Top 10 Recommended Lessors ---")
        for i, (lessor, score) in enumerate(recommendations, 1):
            print(f"{i}. {lessor} (Score: {score})")
    else:
        print(f"\nCould not find sample data for query: Province='{prov}', Industry='{ind}'")

if __name__ == '__main__':
    run_recommender()
