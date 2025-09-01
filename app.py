from flask import Flask, request, jsonify, render_template
from recommender import Recommender
import numpy as np

# --- 1. Create Flask App ---
app = Flask(__name__, template_folder='.')

# --- 2. Initialize Recommender (Load Model) ---
# This is done once when the server starts.
try:
    recommender_model = Recommender()
except Exception as e:
    print(f"FATAL: Could not initialize the recommender model. Error: {e}")
    recommender_model = None

# --- 3. Define Routes ---

@app.route('/')
def home():
    """Serves the main HTML page."""
    return render_template('index.html')

@app.route('/api/form-options', methods=['GET'])
def get_form_options():
    """Provides data to populate the frontend dropdowns."""
    if not recommender_model:
        return jsonify({"error": "Recommender model not loaded"}), 500

    try:
        options = {
            "provinces": recommender_model.get_all_provinces(),
            "industries": recommender_model.get_all_industries(),
            "term_bins": recommender_model.get_term_bin_labels()
        }
        return jsonify(options)
    except Exception as e:
        return jsonify({"error": f"Failed to get form options: {str(e)}"}), 500


@app.route('/api/recommend', methods=['POST'])
def recommend_api():
    """Handles recommendation requests from the frontend."""
    if not recommender_model:
        return jsonify({"error": "Recommender model not loaded"}), 500

    try:
        data = request.get_json()

        # Basic validation
        required_fields = ['province', 'industry', 'value', 'term_bin']
        if not all(field in data for field in required_fields):
            return jsonify({"error": "Missing required fields"}), 400

        # The user provides the term_bin label directly, but the recommender needs a representative raw value.
        # We'll find the index of the label and use a pre-defined mapping or average.
        # For simplicity, we'll map the index to a reasonable term year.
        term_bin_labels = recommender_model.get_term_bin_labels()
        try:
            term_bin_index = term_bin_labels.index(data['term_bin'])
            # Create a representative term value. This is a simplification.
            # A more robust solution might use the cluster centers.
            # E.g. [2.8, 5.5, 9.6, 16.7, 25.0] -> index 1 is 5.5
            term_centers = [float(c) for c in recommender_model.binning_artifacts['term_binner_model'].cluster_centers_.flatten()]
            term_centers.sort()
            representative_term = term_centers[term_bin_index]
        except ValueError:
            return jsonify({"error": f"Invalid term_bin label: {data['term_bin']}"}), 400

        query = {
            "province": data['province'],
            "industry": data['industry'],
            "value": float(data['value']),
            "term": representative_term
        }

        recommendations = recommender_model.get_recommendation_for_query(query)
        return jsonify(recommendations)

    except Exception as e:
        # Log the full error for debugging
        print(f"ERROR in /api/recommend: {e}")
        return jsonify({"error": "An internal error occurred."}), 500

# --- 4. Run the App ---
if __name__ == '__main__':
    # Setting debug=False is important for production.
    # The reloader in debug mode can cause issues with loading large models.
    # We set host='0.0.0.0' to make it accessible from outside the container.
    app.run(host='0.0.0.0', port=5001, debug=False)
