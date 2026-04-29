from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd

app = Flask(__name__)

# --- 1. Load Models and Database Safely ---
try:
    cof_model = joblib.load('random_forest_model.pkl')
    ocp_model = joblib.load('ocp_model.pkl')
    wear_db = joblib.load('wear_database.pkl')
    print("✅ All models and databases loaded successfully.")
except Exception as e:
    print(f"⚠️ Warning: Could not load one or more models. Error: {e}")
    cof_model, ocp_model, wear_db = None, None, {}

# ADDED: Load the newly generated accuracy metrics
try:
    metrics_db = joblib.load('model_metrics.pkl')
    print("✅ Model metrics loaded successfully.")
except Exception as e:
    print(f"⚠️ Warning: Could not load model_metrics.pkl. Error: {e}")
    metrics_db = {'cof_accuracy': 'N/A', 'ocp_accuracy': 'N/A'}

ALLOYS = ['Pure Mg', 'Mg-Bi', 'Mg-Sr', 'Mg-Zn']

# --- 2. Expert Insight Logic (Updated to match Manuscript Tables) ---
def generate_expert_insight(cof, ocp, wear_depth):
    comments = []
    
    # --- Friction Logic ---
    if cof < 0.20: 
        comments.append("🟢 **Low Friction (Excellent):** Indicates a stable, lubricious tribo-film (likely fluid film or boundary lubrication). This regime is safe for articulation.")
    elif cof > 0.40: 
        comments.append("🔴 **High Friction (Risk of Wear):** Indicates a breakdown of lubrication and the onset of abrasive wear; there is a high risk of debris generation and subsequent osteolysis.")
    else: 
        comments.append("🟡 **Moderate Friction (Caution):** Represents a transitional regime. While acceptable for static spacers, this level poses risks for moving joints.")

    # --- Corrosion Logic ---
    if ocp < -1.4: 
        comments.append("🔴 **Highly Active (Rapid Degradation):** The material is actively dissolving. This creates a risk of hydrogen gas pocket formation and premature mechanical failure.")
    elif ocp > -1.25: 
        comments.append("🟢 **More Noble (Stable):** Indicates the presence of a robust, protective oxide or hydroxide layer. The implant is considered 'passive' and chemically stable.")
    else: 
        comments.append("🟡 **Transition Zone (Caution):** Represents a transitional state where the protective film may be in the process of forming or is partially damaged.")

    # --- Wear Logic ---
    if wear_depth == 'N/A' or wear_depth is None:
        comments.append("⚪ **Wear:** Data not available in database.")
    else:
        try:
            wd = float(wear_depth)
            if wd > 20.0:
                comments.append(f"🔴 **Significant Material Loss (Critical Failure):** Issues a 'Red Alert.' Wear of {wd:.2f} µm compromises the structural integrity of components like screws or plates.")
            elif wd < 10.0:
                comments.append(f"🟢 **High Wear Resistance (Optimal):** Confirms the material's mechanical suitability ({wd:.2f} µm). The surface hardness is sufficient to resist abrasion during articulation.")
            else:
                comments.append(f"🟡 **Moderate Material Loss:** Wear of {wd:.2f} µm. Monitor for transition to critical structural failure.")
        except ValueError:
            comments.append("⚠️ **Wear:** Invalid data format.")

    return comments

# --- 3. Routes ---
@app.route('/')
def index():
    return render_template('index.html', alloy_types=ALLOYS)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        
        alloy = data.get('alloyType') 
        time = data.get('timestamp')

        if not alloy or time is None:
            return jsonify({'error': 'Missing alloyType or timestamp'}), 400

        input_df = pd.DataFrame({'Timestamp': [float(time)], 'Alloy_Type': [alloy]})

        cof_pred = float(cof_model.predict(input_df)[0]) if cof_model else 0.0
        ocp_pred = float(ocp_model.predict(input_df)[0]) if ocp_model else 0.0

        alloy_wear_data = wear_db.get(alloy, {'max_depth_um': 'N/A'})
        wear_depth = alloy_wear_data.get('max_depth_um', 'N/A')

        insights = generate_expert_insight(cof_pred, ocp_pred, wear_depth)

        # ADDED: Send the loaded metrics dictionary to the frontend JSON response
        response = {
            'predicted_cof': round(cof_pred, 3),
            'predicted_ocp': round(ocp_pred, 3),
            'metrics': metrics_db, 
            'wear_metrics': {
                'max_depth_um': wear_depth
            },
            'comments': insights
        }

        return jsonify(response)

    except Exception as e:
        print(f"Server Error during prediction: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)