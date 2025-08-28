import configparser
import pandas as pd
import numpy as np
import pickle
from utils import load_data, transform_data, build_heterogeneous_graph, get_recommendations

def run_recommender():
    """
    Runs the final recommendation pipeline using pre-trained artifacts and optimized parameters.
    """
    # 1. Read Config
    config = configparser.ConfigParser()
    config.read('config.ini')

    # Paths and Data Settings
    data_file = config['Paths']['production_data_file']
    binning_rules_file = config['Paths']['binning_rules_file']

    # Load optimized parameters
    best_lambda = config.getfloat('Optimized_Parameters', 'best_lambda')
    best_w_freq = config.getfloat('Optimized_Parameters', 'best_w_freq')
    best_w_val = config.getfloat('Optimized_Parameters', 'best_w_val')

    print("--- Running Recommender with Optimized Parameters from config.ini ---")
    print(f"  - Data file: {data_file}")
    print(f"  - Using lambda: {best_lambda:.4f}, w_freq: {best_w_freq:.2f}, w_val: {best_w_val:.2f}")

    # 2. Load production data and binning rules
    df_raw = load_data(data_file)
    try:
        with open(binning_rules_file, 'rb') as f:
            binning_artifacts = pickle.load(f)
        print(f"Loaded binning rules from '{binning_rules_file}'.")
    except FileNotFoundError:
        print(f"ERROR: Binning rules file '{binning_rules_file}' not found.")
        print("Please run 'python3 optimizer.py' first to generate the rules.")
        return

    # 3. Transform production data using pre-fitted rules
    df = transform_data(df_raw, binning_artifacts)
    if df is None:
        return

    # 4. Calculate final edge weights using optimized params
    df = df.copy()
    lessee_total_counts = df.groupby('承租人')['承租人'].transform('size')
    pair_counts = df.groupby(['承租人', '出租人'])['承租人'].transform('size')
    df['freq_prop'] = pair_counts / lessee_total_counts
    lessee_total_value = df.groupby('承租人')['财产价值（万元）'].transform('sum')
    pair_values = df.groupby(['承租人', '出租人'])['财产价值（万元）'].transform('sum')
    df['val_prop'] = (pair_values / lessee_total_value).fillna(0)
    t_max = df['披露日期'].max()
    time_decay = np.exp(-best_lambda * (t_max - df['披露日期']).dt.days)
    df['composite_score'] = (df['freq_prop'] * best_w_freq + df['val_prop'] * best_w_val) * time_decay
    lessee_lessor_weights = df.groupby(['承租人', '出租人'])['composite_score'].sum().reset_index(name='weight')

    # 5. Build graph with final weights
    print("\nBuilding final graph on full dataset...")
    G = build_heterogeneous_graph(
        df,
        lessee_lessor_weights=lessee_lessor_weights,
        province_binner=binning_artifacts['province_binner'],
        province_pivot=binning_artifacts['province_pivot']
    )

    # 6. Run a sample recommendation
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
