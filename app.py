import pickle
from flask import Flask, jsonify, render_template, request
from graph_builder import create_province_subgraph, create_industry_subgraph
from recommender import run_pagerank_on_subgraph

# --- Configuration ---
DATA_PKL = 'data.pkl'
GRAPH_PKL = 'graph.gpickle'

# Initialize Flask app
app = Flask(__name__)

# --- Global Variables for Data and Graph ---
DATA = None
G_GLOBAL = None

def load_model_artifacts():
    """Loads the pre-computed data and graph from disk."""
    global DATA, G_GLOBAL
    print("Loading pre-computed model artifacts...")
    try:
        with open(DATA_PKL, 'rb') as f:
            DATA = pickle.load(f)
        print(f"Loaded data from '{DATA_PKL}'.")

        with open(GRAPH_PKL, 'rb') as f:
            G_GLOBAL = pickle.load(f)
        print(f"Loaded graph from '{GRAPH_PKL}'.")

    except FileNotFoundError:
        print("\n---")
        print("ERROR: Model artifact files not found.")
        print(f"Please run 'python3 train.py' first to generate '{DATA_PKL}' and '{GRAPH_PKL}'.")
        print("---\n")
        # Exit gracefully if files are not found.
        exit()


@app.route('/')
def index():
    """Renders the main HTML page."""
    return render_template('index.html')

@app.route('/options')
def options():
    """Provides options for the dropdown menus."""
    if DATA is None:
        return jsonify({"error": "Data not loaded"}), 500

    provinces = sorted(DATA['承租人所属省份'].unique().tolist())
    industries = sorted(DATA['申万行业一级'].unique().tolist())

    return jsonify({
        "provinces": provinces,
        "industries": industries
    })

@app.route('/recommend')
def recommend():
    """Runs the 4-sub-model recommendation and returns results."""
    province = request.args.get('province')
    industry = request.args.get('industry')

    if not province or not industry:
        return jsonify({"error": "Missing 'province' or 'industry' parameter"}), 400

    if G_GLOBAL is None or DATA is None:
        return jsonify({"error": "Graph or data not loaded"}), 500

    # Create Subgraphs
    G_province = create_province_subgraph(G_GLOBAL, DATA, province)
    G_industry = create_industry_subgraph(G_GLOBAL, DATA, industry)

    # Run the 4 models
    rec_prov_count = run_pagerank_on_subgraph(G_province, 'count', top_n=10)
    rec_prov_value = run_pagerank_on_subgraph(G_province, 'total_value', top_n=10)
    rec_ind_count = run_pagerank_on_subgraph(G_industry, 'count', top_n=10)
    rec_ind_value = run_pagerank_on_subgraph(G_industry, 'total_value', top_n=10)

    # Structure results
    results = {
        "province_by_count": {
            "reason": f"因在【{province}】内合作频率高而推荐",
            "recommendations": rec_prov_count
        },
        "province_by_value": {
            "reason": f"因在【{province}】内合作总金额高而推荐",
            "recommendations": rec_prov_value
        },
        "industry_by_count": {
            "reason": f"因在【{industry}】行业内合作频率高而推荐",
            "recommendations": rec_ind_count
        },
        "industry_by_value": {
            "reason": f"因在【{industry}】行业内合作总金额高而推荐",
            "recommendations": rec_ind_value
        }
    }

    return jsonify(results)


if __name__ == '__main__':
    load_model_artifacts()
    app.run(debug=True, port=5001)
