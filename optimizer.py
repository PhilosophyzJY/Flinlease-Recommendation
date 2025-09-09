import configparser
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import plotly.graph_objects as go
from itertools import combinations
from utils import load_data, fit_binners, transform_data, calculate_advanced_scores, build_heterogeneous_graph, get_recommendations
from bi_report_generator import generate_bi_report
from sklearn.preprocessing import normalize

def generate_sankey_diagrams(df, output_html_path, top_n=10):
    """Generates a Sankey diagram HTML report showing top lessor preferences."""
    print(f"\n--- Generating Sankey Diagram Report for Top {top_n} Lessors ---")
    top_lessors = df.groupby('出租人')['财产价值（万元）'].sum().nlargest(top_n).index
    df_top = df[df['出租人'].isin(top_lessors)]
    figs = {}
    preference_dims = {
        '地域偏好 (Province)': '省份', '行业偏好 (Industry)': '申万行业一级',
        '财产价值偏好 (Value Bin)': '价值分箱', '期限偏好 (Term Bin)': '期限分箱'
    }
    for title, dim_col in preference_dims.items():
        sankey_data = df_top.groupby(['出租人', dim_col])['财产价值（万元）'].sum().reset_index()
        sankey_data.columns = ['source', 'target', 'value']
        all_nodes = pd.concat([sankey_data['source'], sankey_data['target']]).unique()
        node_map = {name: i for i, name in enumerate(all_nodes)}
        links = {'source': sankey_data['source'].map(node_map), 'target': sankey_data['target'].map(node_map), 'value': sankey_data['value']}
        fig = go.Figure(data=[go.Sankey(node=dict(pad=15, thickness=20, line=dict(color="black", width=0.5), label=all_nodes), link=links)])
        fig.update_layout(title_text=f"Top {top_n} 出租人 - {title}", font_size=10)
        figs[title] = fig
    with open(output_html_path, 'w', encoding='utf-8') as f:
        f.write("<h1>出租人偏好桑基图分析报告 (Lessor Preference Sankey Diagram Report)</h1>")
        for title, fig in figs.items():
            f.write(f"<h2>{title}</h2>")
            f.write(fig.to_html(full_html=False, include_plotlyjs='cdn'))
            f.write("<hr>")
    print(f"Sankey diagram report saved to '{output_html_path}'")

def generate_binning_report(artifacts, province_clusters, report_file):
    """Generates a human-readable report of the binning results."""
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("--- Binning and Clustering Analysis Report ---\n\n")
        f.write("--- 财产价值分箱 (Value Bins) ---\nCalculated using quantile-based binning.\n")
        bounds = artifacts['value_bins']
        for i in range(len(bounds) - 1):
            f.write(f"  价值分箱 {i}: {bounds[i]:.2f} - {bounds[i+1]:.2f}\n")
        f.write("\n--- 租赁期限分箱 (Term Bins) ---\nCalculated using K-Means clustering.\n")
        term_labels = artifacts['term_binner_labels']
        for i, label in enumerate(term_labels.values()):
             f.write(f"  {label} (Cluster {i})\n")
        f.write("\n--- 省份分组 (Province Clusters) ---\nCalculated using K-Means clustering on industry profiles.\n")
        for i, provinces in province_clusters.items():
            f.write(f"  省份分组 {i}: {', '.join(provinces)}\n")
    print(f"Binning report saved to '{report_file}'.")

def get_weight_combinations(step=0.1):
    """Generates combinations of 4 weights that sum to 1.0."""
    weights = []
    step = round(step, 4)
    for i in np.arange(0, 1.0 + step, step):
        for j in np.arange(0, 1.0 - i + step, step):
            for k in np.arange(0, 1.0 - i - j + step, step):
                l = 1.0 - i - j - k
                if l >= -1e-9:
                    l_r, i_r, j_r, k_r = round(l, 4), round(i, 4), round(j, 4), round(k, 4)
                    if np.isclose(i_r + j_r + k_r + l_r, 1.0):
                        weights.append({'province': i_r, 'industry': j_r, 'value': k_r, 'term': l_r})
    return weights

def evaluate_params(weights, l_decay, train_df, binning_artifacts, target_lessees, test_lookup, lessee_attributes):
    """Helper function to calculate hit rate for a given set of parameters."""
    try:
        train_df_scored = calculate_advanced_scores(train_df.copy(), l_decay, weights)
        G_train = build_heterogeneous_graph(
            train_df_scored,
            province_binner=binning_artifacts['province_binner'],
            province_pivot=binning_artifacts['province_pivot']
        )
        hits = 0
        if not target_lessees: return 0.0
        for lessee in target_lessees:
            lessee_data = lessee_attributes.loc[lessee]
            query = {"province": lessee_data['省份'], "industry": lessee_data['申万行业一级'], "value_bin": lessee_data['价值分箱'], "term_bin": lessee_data['期限分箱']}
            recommendations = get_recommendations(G_train, query, weights, top_n=5)
            recommended_lessors = {rec[0] for rec in recommendations}
            actual_lessors = test_lookup.get(lessee, set())
            if not actual_lessors.isdisjoint(recommended_lessors):
                hits += 1
        return hits / len(target_lessees)
    except Exception:
        return 0.0

def local_search(candidate, train_df, binning_artifacts, target_lessees, test_lookup, lessee_attributes):
    """Performs a fine-grained local search around a promising candidate."""
    initial_hit_rate = candidate['hit_rate']
    print(f"\n--- Running Local Search for Candidate: Hit Rate={initial_hit_rate:.4f}, Lambda={candidate['lambda']:.4f}, Weights={ {k: f'{v:.2f}' for k, v in candidate.items() if k in ['province','industry','value','term']} }")

    best_local_result = candidate.copy()
    initial_weights_dict = {k: candidate[k] for k in ['province', 'industry', 'value', 'term']}
    l_decay = candidate['lambda']

    weight_keys = ['province', 'industry', 'value', 'term']
    weights_to_test = set()
    increment = 0.01

    for i_idx, j_idx in combinations(range(len(weight_keys)), 2):
        w_i_key, w_j_key = weight_keys[i_idx], weight_keys[j_idx]

        max_delta_i = initial_weights_dict[w_i_key] * 0.20
        max_delta_j = initial_weights_dict[w_j_key] * 0.20

        for delta in np.arange(increment, max_delta_j + increment, increment):
            if initial_weights_dict[w_j_key] - delta >= -1e-9:
                new_weights = initial_weights_dict.copy()
                new_weights[w_i_key] += delta
                new_weights[w_j_key] -= delta
                weights_to_test.add(tuple(round(w, 4) for w in new_weights.values()))

        for delta in np.arange(increment, max_delta_i + increment, increment):
            if initial_weights_dict[w_i_key] - delta >= -1e-9:
                new_weights = initial_weights_dict.copy()
                new_weights[w_j_key] += delta
                new_weights[w_i_key] -= delta
                weights_to_test.add(tuple(round(w, 4) for w in new_weights.values()))

    print(f"  - Generated {len(weights_to_test)} new unique weight combinations for local search.")

    for weights_tuple in weights_to_test:
        weights = dict(zip(weight_keys, weights_tuple))
        hit_rate = evaluate_params(weights, l_decay, train_df, binning_artifacts, target_lessees, test_lookup, lessee_attributes)
        if hit_rate > best_local_result['hit_rate']:
            print(f"  - Found new local best: Hit Rate={hit_rate:.4f} (from {best_local_result['hit_rate']:.4f}), Weights={weights}")
            best_local_result = {'lambda': l_decay, 'hit_rate': hit_rate, **weights}

    if best_local_result['hit_rate'] > initial_hit_rate:
        print(f"  - Local search improved hit rate from {initial_hit_rate:.4f} to {best_local_result['hit_rate']:.4f}")
    else:
        print("  - Local search did not find a better combination.")

    return best_local_result

def run_optimizer():
    config = configparser.ConfigParser()
    config.read('config.ini')

    # Load settings
    learning_data_file, plot_file, binning_rules_file, report_file, sankey_report_file, bi_report_file = (
        config['Paths']['learning_data_file'], config['Paths']['plot_file'], config['Paths']['binning_rules_file'],
        config['Paths']['report_file'], config['Paths']['sankey_report_file'], config['Paths']['bi_report_file']
    )
    n_term_bins, n_value_bins, n_province_clusters = (
        config.getint('Data_Settings', 'n_term_bins'), config.getint('Data_Settings', 'n_value_bins'),
        config.getint('Data_Settings', 'n_province_clusters')
    )
    train_periods = [pd.to_datetime(p.strip()).to_period('M') for p in config['Optimization_Settings']['train_year_months'].split(',')]
    test_periods = [pd.to_datetime(p.strip()).to_period('M') for p in config['Optimization_Settings']['test_year_months'].split(',')]
    lambda_start, lambda_end, lambda_steps = map(float, config['Optimization_Settings']['lambda_space'].split(','))
    weight_step = config.getfloat('Optimization_Settings', 'attribute_weight_step')

    # 1. Load and prepare data
    print("--- Preparing Data for Optimization ---")
    df = load_data(learning_data_file)
    df['year_month'] = df['transaction_date'].dt.to_period('M')
    train_df_raw = df[df['year_month'].isin(train_periods)]
    test_df_raw = df[df['year_month'].isin(test_periods)]

    print("  - Fitting binning rules on training data...")
    binning_artifacts = fit_binners(train_df_raw, n_value_bins, n_term_bins, n_province_clusters)
    with open(binning_rules_file, 'wb') as f: pickle.dump(binning_artifacts, f)
    print("  - Binning rules fitted and saved.")

    province_pivot = binning_artifacts['province_pivot']
    province_binner = binning_artifacts['province_binner']
    province_profiles = normalize(province_pivot, norm='l1', axis=1)
    province_labels = province_binner.predict(province_profiles)
    province_clusters = {lbl: [] for lbl in np.unique(province_labels)}
    for i, prov in enumerate(province_pivot.index):
        province_clusters[province_labels[i]].append(prov)
    generate_binning_report(binning_artifacts, province_clusters, report_file)

    print("  - Applying pre-fitted binning rules...")
    train_df = transform_data(train_df_raw, binning_artifacts)
    test_df = transform_data(test_df_raw, binning_artifacts)

    print("\n--- Generating Comprehensive BI Report ---")
    generate_bi_report(train_df, bi_report_file)
    generate_sankey_diagrams(train_df, sankey_report_file)

    target_lessees = list(set(train_df['承租人'].unique()).intersection(set(test_df['承租人'].unique())))
    test_lookup = test_df.groupby('承租人')['出租人'].apply(set).to_dict()
    lessee_attributes = train_df.drop_duplicates(subset=['承租人']).set_index('承租人')
    print(f"Found {len(target_lessees)} target lessees for validation.")

    # STAGE 1: Grid Search
    print("\n--- STAGE 1: Starting Coarse Grid Search ---")
    all_results = []
    lambda_range = np.linspace(lambda_start, lambda_end, int(lambda_steps))
    weight_combinations = get_weight_combinations(weight_step)
    print(f"Testing {len(lambda_range)} lambda values and {len(weight_combinations)} weight combinations...")

    for l_decay in lambda_range:
        for weights in weight_combinations:
            hit_rate = evaluate_params(weights, l_decay, train_df, binning_artifacts, target_lessees, test_lookup, lessee_attributes)
            print(f"Tested params: lambda={l_decay:.4f}, w={weights} => Hit Rate: {hit_rate:.4f}")
            all_results.append({'lambda': l_decay, 'hit_rate': hit_rate, **weights})

    print("\n--- Grid Search (Stage 1) Complete ---")
    if not all_results:
        print("No results were generated. Exiting.")
        return

    df_results = pd.DataFrame(all_results)
    df_results.sort_values('hit_rate', ascending=False, inplace=True)
    stage1_best_params = df_results.iloc[0].to_dict()

    print(f"\n--- Stage 1 Analysis ---")
    print(f"Best Hit Rate from Stage 1: {stage1_best_params['hit_rate']:.4f}")

    # STAGE 2: Local Search
    top_n_candidates = 50
    num_candidates = min(top_n_candidates, len(df_results))
    top_candidates = df_results.head(num_candidates).to_dict('records')

    print(f"\n--- STAGE 2: Starting Fine-grained Local Search for Top {num_candidates} Candidates ---")
    stage2_results = []
    for i, candidate in enumerate(top_candidates):
        print(f"--- Processing candidate {i+1}/{num_candidates} ---")
        local_best = local_search(candidate, train_df, binning_artifacts, target_lessees, test_lookup, lessee_attributes)
        stage2_results.append(local_best)

    if stage2_results:
        final_best_params = max(stage2_results, key=lambda x: x['hit_rate'])
    else:
        final_best_params = stage1_best_params

    final_best_weights = {k: v for k, v in final_best_params.items() if k in ['province', 'industry', 'value', 'term']}

    # 3. Plot Results
    print(f"\n--- Generating Plot: {plot_file} ---")
    if not df_results.empty:
        grouped_by_weights = df_results.loc[df_results.groupby(['province', 'industry', 'value', 'term'])['hit_rate'].idxmax()]
        top_5_weight_combos = grouped_by_weights.nlargest(5, 'hit_rate')
        fig, ax = plt.subplots(figsize=(12, 8))
        for _, combo in top_5_weight_combos.iterrows():
            weights = {'province': combo['province'], 'industry': combo['industry'], 'value': combo['value'], 'term': combo['term']}
            subset = df_results[(np.isclose(df_results['province'], weights['province'])) & (np.isclose(df_results['industry'], weights['industry'])) & (np.isclose(df_results['value'], weights['value'])) & (np.isclose(df_results['term'], weights['term']))]
            label = (f"P={weights['province']:.2f}, I={weights['industry']:.2f}, V={weights['value']:.2f}, T={weights['term']:.2f} (Max Hit: {combo['hit_rate']:.2f})")
            ax.plot(subset['lambda'], subset['hit_rate'], marker='o', linestyle='-', label=label)
        ax.set_xlabel("Lambda (Time Decay Rate)"), ax.set_ylabel("Hit Rate"), ax.set_title("Hit Rate vs. Lambda for Top 5 Attribute Weight Combinations (from Stage 1)")
        ax.legend(title="Weights (P,I,V,T) & Max Hit Rate", bbox_to_anchor=(1.05, 1), loc='upper left'), ax.grid(True)
        plt.tight_layout(), plt.savefig(plot_file)
        print("Plot saved.")
    else:
        print("No results to plot.")

    # 4. Write learned rules to config
    print("\n--- Writing Learned Rules to Config File ---")
    if not config.has_section('Learned_Rules_Summary'): config.add_section('Learned_Rules_Summary')
    config.set('Learned_Rules_Summary', 'value_bin_boundaries', ', '.join([f'{b:.2f}' for b in binning_artifacts['value_bins']]))
    term_centers = binning_artifacts['term_binner_model'].cluster_centers_.flatten()
    term_centers.sort()
    config.set('Learned_Rules_Summary', 'term_bin_centers_years', ', '.join([f'{c:.2f}' for c in term_centers]))
    import json
    sorted_clusters = {str(k): sorted(v) for k, v in province_clusters.items()}
    config.set('Learned_Rules_Summary', 'province_cluster_map_json', json.dumps(sorted_clusters, ensure_ascii=False))
    with open('config.ini', 'w') as configfile: config.write(configfile)
    print("Config file updated with learned rules summary.")

    print("\n--- ACTION REQUIRED ---")
    print("Optimization complete. The optimizer has suggested the following parameters:")
    print(f"  - Best Hit Rate: {final_best_params['hit_rate']:.4f}")
    print(f"  - Suggested Lambda: {final_best_params.get('lambda', 'N/A'):.4f}")
    print(f"  - Suggested Weights: {final_best_weights}")
    print("\nPlease review the plot and manually update the [Optimized_Parameters] section in 'config.ini'.")

if __name__ == '__main__':
    run_optimizer()
