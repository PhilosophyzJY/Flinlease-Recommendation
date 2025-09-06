import configparser
import pandas as pd
import numpy as np
import pickle
from utils import load_data, transform_data, calculate_advanced_scores, build_heterogeneous_graph, get_recommendations

class Recommender:
    """
    A class to encapsulate the recommendation model, allowing for a one-time setup
    and subsequent calls to a recommendation method.
    """
    def __init__(self):
        print("--- Initializing Recommender ---")
        # 1. Read Config
        config = configparser.ConfigParser()
        config.read('config.ini')

        # Paths and Data Settings
        data_file = config['Paths']['production_data_file']
        binning_rules_file = config['Paths']['binning_rules_file']

        # Load optimized parameters
        self.best_lambda = config.getfloat('Optimized_Parameters', 'best_lambda')
        self.attribute_weights = {
            'province': config.getfloat('Optimized_Parameters', 'best_w_province'),
            'industry': config.getfloat('Optimized_Parameters', 'best_w_industry'),
            'value': config.getfloat('Optimized_Parameters', 'best_w_value'),
            'term': config.getfloat('Optimized_Parameters', 'best_w_term'),
        }

        print(f"  - Data file: {data_file}")
        print(f"  - Using lambda: {self.best_lambda:.4f}")
        print(f"  - Using attribute weights: {self.attribute_weights}")

        # 2. Load production data and binning rules
        df_raw = load_data(data_file)
        try:
            with open(binning_rules_file, 'rb') as f:
                self.binning_artifacts = pickle.load(f)
            print(f"Loaded binning rules from '{binning_rules_file}'.")
        except FileNotFoundError:
            raise RuntimeError(f"ERROR: Binning rules file '{binning_rules_file}' not found. Please run 'python optimizer.py' first.")

        # 3. Transform production data using pre-fitted rules
        self.df = transform_data(df_raw, self.binning_artifacts)
        if self.df is None:
            raise RuntimeError("Failed to transform data.")

        # 4. Calculate final scores using optimized params
        print("Calculating advanced scores for the graph...")
        df_scored = calculate_advanced_scores(self.df, self.best_lambda, self.attribute_weights)

        # 5. Build and store the graph
        print("\nBuilding final graph on full dataset...")
        self.graph = build_heterogeneous_graph(
            df_scored,
            province_binner=self.binning_artifacts['province_binner'],
            province_pivot=self.binning_artifacts['province_pivot']
        )
        print("--- Recommender Initialized and Ready ---")

    def get_recommendation_for_query(self, query):
        """
        Gets recommendations for a given query dictionary.
        The query must contain 'province', 'industry', 'value', and 'term'.
        """
        # The query contains raw values. We need to map them to the binned categories.
        # Map value to value_bin
        value_bins = self.binning_artifacts['value_bins']
        value_bin_labels = [f'价值{i}' for i in range(len(value_bins)-1)]
        query_value_bin = pd.cut([query['value']], bins=value_bins, labels=value_bin_labels, include_lowest=True)[0]

        # Map term to term_bin
        term_model = self.binning_artifacts['term_binner_model']
        term_label_map = self.binning_artifacts['term_binner_labels']
        raw_term_bin = term_model.predict(np.array([[query['term']]]))[0]
        query_term_bin = term_label_map[raw_term_bin]

        # Construct the final query for PageRank
        pagerank_query = {
            "province": query['province'],
            "industry": query['industry'],
            "value_bin": query_value_bin,
            "term_bin": query_term_bin,
        }

        print(f"\n--- Running Recommendation for: {pagerank_query} ---")
        recommendations = get_recommendations(self.graph, pagerank_query, self.attribute_weights)
        return recommendations

    def get_all_industries(self):
        """Returns a list of all unique industries in the dataset."""
        return self.df['申万行业一级'].unique().tolist()

    def get_all_provinces(self):
        """Returns a list of all unique provinces in the dataset."""
        return self.df['省份'].unique().tolist()

    def get_term_bin_labels(self):
        """Returns the labels for the term bins."""
        # The labels are already sorted by the cluster's mean term value in utils.fit_binners
        return list(self.binning_artifacts['term_binner_labels'].values())

if __name__ == '__main__':
    # This block is for standalone testing of the Recommender class.
    try:
        recommender = Recommender()

        # Example query
        sample_query = {
            'province': '山东',
            'industry': '农林牧渔',
            'value': 5000,  # Raw value in万元
            'term': 5       # Raw value in years
        }

        recommendations = recommender.get_recommendation_for_query(sample_query)

        print("\n--- Top 10 Recommended Lessors ---")
        for i, (lessor, score) in enumerate(recommendations, 1):
            print(f"{i}. {lessor} (Score: {score})")

        print("\nAvailable Industries:", recommender.get_all_industries())
        print("\nAvailable Term Bins:", recommender.get_term_bin_labels())

    except RuntimeError as e:
        print(e)
