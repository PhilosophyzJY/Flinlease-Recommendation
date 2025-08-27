import configparser
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from itertools import product
from utils import load_and_engineer_features, build_heterogeneous_graph, get_recommendations

def calculate_lessee_lessor_weights(df, w_freq, w_val, lambda_decay):
    """Calculates the composite Lessee -> Lessor edge weights."""
    # Freq proportion
    lessee_total_counts = df.groupby('承租人')['承租人'].transform('size')
    pair_counts = df.groupby(['承租人', '出租人'])['承租人'].transform('size')
    freq_prop = pair_counts / lessee_total_counts

    # Value proportion
    lessee_total_value = df.groupby('承租人')['财产价值（万元）'].transform('sum')
    pair_values = df.groupby(['承租人', '出租人'])['财产价值（万元）'].transform('sum')
    val_prop = pair_values / lessee_total_value

    # Time decay
    t_max = df['披露日期'].max()
    time_decay = np.exp(-lambda_decay * (t_max - df['披露日期']).dt.days)

    # Combine and aggregate
    df['composite_score'] = (freq_prop * w_freq + val_prop * w_val) * time_decay
    lessee_lessor_weights = df.groupby(['承租人', '出租人'])['composite_score'].sum().reset_index(name='weight')

    return lessee_lessor_weights

def run_optimizer():
    """
    Runs the full hyperparameter tuning pipeline.
    """
    # 1. Read Config
    config = configparser.ConfigParser()
    config.read('config.ini')

    data_file = config['Paths']['data_file']
    plot_file = config['Paths']['plot_file']

    lambda_start, lambda_end, lambda_steps = map(float, config['Tuning']['lambda_space'].split(','))
    weight_start, weight_end, weight_steps = map(float, config['Tuning']['weight_space'].split(','))

    # 2. Prepare Data
    print("--- Preparing Data for Optimization ---")
    df = load_and_engineer_features(data_file)
    train_df = df[df['披露日期'].dt.month.isin([5, 6])]
    test_df = df[df['披露日期'].dt.month == 7]

    train_lessees = set(train_df['承租人'].unique())
    test_lessees = set(test_df['承租人'].unique())
    target_lessees = list(train_lessees.intersection(test_lessees))
    print(f"Found {len(target_lessees)} target lessees for validation.")

    # Create a lookup dictionary from test data for quick validation
    test_lookup = test_df.groupby('承租人')['出租人'].apply(set).to_dict()

    # 3. Grid Search
    print("\n--- Starting Hyperparameter Grid Search ---")
    results = []
    best_params = {'hit_rate': -1}

    lambda_range = np.linspace(lambda_start, lambda_end, int(lambda_steps))
    weight_range = np.linspace(weight_start, weight_end, int(weight_steps))

    for l_decay in lambda_range:
        for w_freq in weight_range:
            w_val = 1.0 - w_freq

            print(f"\nTesting params: lambda={l_decay:.4f}, w_freq={w_freq:.2f}, w_val={w_val:.2f}")

            # a. Calculate edge weights with current params
            lessee_lessor_weights = calculate_lessee_lessor_weights(train_df, w_freq, w_val, l_decay)

            # b. Build graphs for train and val sets
            G_train = build_heterogeneous_graph(train_df, lessee_lessor_weights=lessee_lessor_weights)

            # c. Get recommendations for each target lessee and validate
            hits = 0
            lessee_attributes = train_df.drop_duplicates(subset=['承租人']).set_index('承租人')

            for lessee in target_lessees:
                # Get query attributes from the training set
                try:
                    lessee_data = lessee_attributes.loc[lessee]
                except KeyError:
                    continue # Should not happen, but as a safeguard

                query = {
                    "province": lessee_data['省份'],
                    "industry": lessee_data['申万行业一级'],
                    "value_bin": lessee_data['价值分箱'],
                    "term_bin": lessee_data['期限分箱'],
                }

                # Get Top 5 recommendations based on the training graph
                recommendations = get_recommendations(G_train, query, top_n=5)
                recommended_lessors = {rec[0] for rec in recommendations}

                # d. Validate against the test set
                actual_lessors_in_july = test_lookup.get(lessee, set())

                # Check for a "hit"
                if not actual_lessors_in_july.isdisjoint(recommended_lessors):
                    hits += 1

            hit_rate = hits / len(target_lessees) if target_lessees else 0
            print(f"  => Hit Rate: {hit_rate:.4f}")
            results.append({'lambda': l_decay, 'w_freq': w_freq, 'hit_rate': hit_rate})

            if hit_rate > best_params['hit_rate']:
                best_params['hit_rate'] = hit_rate
                best_params['lambda'] = l_decay
                best_params['w_freq'] = w_freq
                best_params['w_val'] = 1.0 - w_freq

    print("\n--- Grid Search Complete ---")
    print(f"Best Hit Rate: {best_params['hit_rate']:.4f}")
    print(f"Best Parameters: lambda={best_params['lambda']:.4f}, w_freq={best_params['w_freq']:.2f}, w_val={best_params['w_val']:.2f}")

    # 4. Plot Results
    print(f"\n--- Generating Plot: {plot_file} ---")
    df_results = pd.DataFrame(results)
    fig, ax = plt.subplots(figsize=(10, 6))
    for w_freq_val in df_results['w_freq'].unique():
        subset = df_results[df_results['w_freq'] == w_freq_val]
        ax.plot(subset['lambda'], subset['hit_rate'], marker='o', linestyle='-', label=f'w_freq={w_freq_val:.1f}')
    ax.set_xlabel("Lambda (Time Decay Rate)")
    ax.set_ylabel("Hit Rate (Recommendation Stability)")
    ax.set_title("Hyperparameter Tuning Results")
    ax.legend()
    ax.grid(True)
    plt.savefig(plot_file)
    print("Plot saved.")

    # 5. Update Config File
    print("\n--- Updating Config File with Best Parameters ---")
    config['Optimized_Parameters']['best_lambda'] = str(best_params['lambda'])
    config['Optimized_Parameters']['best_w_freq'] = str(best_params['w_freq'])
    config['Optimized_Parameters']['best_w_val'] = str(best_params['w_val'])
    with open('config.ini', 'w') as configfile:
        config.write(configfile)
    print("Config file updated.")

if __name__ == '__main__':
    run_optimizer()
