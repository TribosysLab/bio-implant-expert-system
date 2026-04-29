import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
import os

# --- ADDED IMPORTS FOR ACCURACY METRICS ---
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# --- 1. Preprocessing for Friction (COF) ---
def preprocess_cof_data(filepath="Friction_File.csv"):
    try:
        # Read the CSV but skip the first row (the alloy names) to get actual data headers
        df = pd.read_csv(filepath, header=1)
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return pd.DataFrame()
        
    # UPDATED: Removed "Al-" prefix
    alloys = ['Pure Mg', 'Mg-Bi', 'Mg-Sr', 'Mg-Zn']
    processed_data = []
    
    for i, alloy in enumerate(alloys):
        # Columns are paired: [Time, COF, Time, COF...]
        time_col = df.columns[i*2]
        cof_col = df.columns[i*2 + 1]
        
        temp_df = df[[time_col, cof_col]].copy()
        temp_df.columns = ['Timestamp', 'COF']
        temp_df['Alloy_Type'] = alloy
        
        # Ensure data is numeric and drop any trailing blank rows
        temp_df['Timestamp'] = pd.to_numeric(temp_df['Timestamp'], errors='coerce')
        temp_df['COF'] = pd.to_numeric(temp_df['COF'], errors='coerce')
        temp_df = temp_df.dropna()
        
        processed_data.append(temp_df)
        
    return pd.concat(processed_data, ignore_index=True)

# --- 2. Preprocessing for Corrosion (OCP) ---
def preprocess_ocp_data(filepath="OCP.csv"):
    try:
        df = pd.read_csv(filepath, header=1)
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return pd.DataFrame()
        
    # UPDATED: Removed "Al-" prefix
    alloys = ['Pure Mg', 'Mg-Bi', 'Mg-Sr', 'Mg-Zn']
    processed_data = []
    
    for i, alloy in enumerate(alloys):
        time_col = df.columns[i*2]
        ocp_col = df.columns[i*2 + 1]
        
        temp_df = df[[time_col, ocp_col]].copy()
        temp_df.columns = ['Timestamp', 'OCP']
        temp_df['Alloy_Type'] = alloy
        
        # Ensure data is numeric and drop trailing blank rows
        temp_df['Timestamp'] = pd.to_numeric(temp_df['Timestamp'], errors='coerce')
        temp_df['OCP'] = pd.to_numeric(temp_df['OCP'], errors='coerce')
        temp_df = temp_df.dropna()
        
        processed_data.append(temp_df)
        
    return pd.concat(processed_data, ignore_index=True)

# --- 3a. Extract Sheets from Excel to Simple CSVs ---
def extract_excel_to_csv(excel_filepath="Wear_profile.xlsx"):
    print(f"\n--- Extracting sheets from {excel_filepath} ---")
    
    # Map the exact sheet names in the Excel file to simple, clean CSV filenames
    # Assuming the Excel sheets remain named with 'AlMg', updating the CSV outputs
    sheet_to_csv_map = {
        'Wear-profile-PureMg': 'wear_PureMg.csv',
        'Wear-profile-AlMgBi': 'wear_MgBi.csv',
        'Wear-profile-AlMgSr': 'wear_MgSr.csv',
        'Wear-profile-AlMgZn': 'wear_MgZn.csv'
    }
    
    if not os.path.exists(excel_filepath):
        print(f"🚨 Error: {excel_filepath} not found in the current directory.")
        return False

    try:
        # Load the Excel file
        xl = pd.ExcelFile(excel_filepath)
        
        for sheet_name, csv_name in sheet_to_csv_map.items():
            if sheet_name in xl.sheet_names:
                # Read the sheet and save it as a clean CSV
                df = xl.parse(sheet_name)
                df.to_csv(csv_name, index=False)
                print(f"✅ Extracted sheet '{sheet_name}' to '{csv_name}'")
            else:
                print(f"⚠️ Warning: Sheet '{sheet_name}' not found in {excel_filepath}.")
        return True
                
    except Exception as e:
        print(f"🚨 Fatal Error: Could not parse {excel_filepath}. Error: {e}")
        return False

# --- 3b. Preprocessing for Wear Database ---
def generate_wear_database():
    print("\n--- Generating Wear Database ---")
    wear_db = {}
    
    # UPDATED: Mapped the new exact alloy keys to the newly created simple CSVs
    file_map = {
        'Pure Mg': 'wear_PureMg.csv',
        'Mg-Bi': 'wear_MgBi.csv',
        'Mg-Sr': 'wear_MgSr.csv',
        'Mg-Zn': 'wear_MgZn.csv'
    }
    
    for alloy, filename in file_map.items():
        if os.path.exists(filename):
            try:
                # Read the clean CSV file
                df = pd.read_csv(filename)
                
                # Calculate max depth using the second column (index 1)
                max_depth = df.iloc[:, 1].max() - df.iloc[:, 1].min()
                
                wear_db[alloy] = {'max_depth_um': float(round(max_depth, 2))}
                print(f"✅ Success: Processed {alloy} from '{filename}'")
                
            except Exception as e:
                print(f"❌ Error processing file '{filename}': {e}")
                wear_db[alloy] = {'max_depth_um': 'N/A'}
        else:
            print(f"⚠️ Warning: File not found: '{filename}'")
            wear_db[alloy] = {'max_depth_um': 'N/A'}
            
    joblib.dump(wear_db, 'wear_database.pkl')
    print("Wear database generation complete!")

# --- Main Execution ---
if __name__ == "__main__":
    print("--- Building Multi-Parameter Expert System Backend ---")
    
    # --- ADDED: Initialize a dictionary to store our accuracies ---
    model_metrics = {}
    
    # Train COF (Friction) using Random Forest Regressor
    print("\n--- Training Friction (COF) Model ---")
    processed_cof = preprocess_cof_data("Friction_File.csv")
    
    if not processed_cof.empty:
        X_cof = processed_cof[['Timestamp', 'Alloy_Type']]
        y_cof = processed_cof['COF']
        
        # --- ADDED: Split the data 80% for training, 20% for testing ---
        X_train_cof, X_test_cof, y_train_cof, y_test_cof = train_test_split(X_cof, y_cof, test_size=0.2, random_state=42)
        
        preprocessor_cof = ColumnTransformer([
            ('num', StandardScaler(), ['Timestamp']),
            ('cat', OneHotEncoder(handle_unknown='ignore'), ['Alloy_Type'])
        ])
        
        cof_model = Pipeline([
            ('prep', preprocessor_cof), 
            ('rf', RandomForestRegressor(n_estimators=100, random_state=42))
        ])
        
        # --- UPDATED: Fit only on training data ---
        cof_model.fit(X_train_cof, y_train_cof)
        joblib.dump(cof_model, 'random_forest_model.pkl')
        print("Friction model saved successfully (random_forest_model.pkl).")
        
        # --- ADDED: Calculate test accuracy and save to dictionary ---
        cof_r2 = r2_score(y_test_cof, cof_model.predict(X_test_cof))
        model_metrics['cof_accuracy'] = round(cof_r2 * 100, 2)
        print(f"📊 Friction Model Test Accuracy (R2): {model_metrics['cof_accuracy']}%")
    else:
        print("Error: No COF data processed. Check Friction_File.csv format.")

    # Train OCP (Corrosion) using Random Forest Regressor
    print("\n--- Training Corrosion (OCP) Model ---")
    processed_ocp = preprocess_ocp_data("OCP.csv")
    
    if not processed_ocp.empty:
        X_ocp = processed_ocp[['Timestamp', 'Alloy_Type']]
        y_ocp = processed_ocp['OCP']
        
        # --- ADDED: Split the data 80% for training, 20% for testing ---
        X_train_ocp, X_test_ocp, y_train_ocp, y_test_ocp = train_test_split(X_ocp, y_ocp, test_size=0.2, random_state=42)
        
        preprocessor_ocp = ColumnTransformer([
            ('num', StandardScaler(), ['Timestamp']),
            ('cat', OneHotEncoder(handle_unknown='ignore'), ['Alloy_Type'])
        ])
        
        ocp_model = Pipeline([
            ('prep', preprocessor_ocp), 
            ('rf', RandomForestRegressor(n_estimators=100, random_state=42))
        ])
        
        # --- UPDATED: Fit only on training data ---
        ocp_model.fit(X_train_ocp, y_train_ocp)
        joblib.dump(ocp_model, 'ocp_model.pkl')
        print("OCP model saved successfully (ocp_model.pkl).")
        
        # --- ADDED: Calculate test accuracy and save to dictionary ---
        ocp_r2 = r2_score(y_test_ocp, ocp_model.predict(X_test_ocp))
        model_metrics['ocp_accuracy'] = round(ocp_r2 * 100, 2)
        print(f"📊 Corrosion Model Test Accuracy (R2): {model_metrics['ocp_accuracy']}%")
    else:
        print("Error: No OCP data processed. Check OCP.csv format.")
        
    # --- ADDED: Save the metrics dictionary for the Flask app to read ---
    if model_metrics:
        joblib.dump(model_metrics, 'model_metrics.pkl')
        print("Model metrics saved successfully (model_metrics.pkl).")
        
    # Extract sheets from Excel and Generate Wear DB
    excel_status = extract_excel_to_csv("Wear_profile.xlsx")
    
    if excel_status:
        generate_wear_database()
    else:
        print("\n⚠️ Skipping Wear Database generation because the Excel file wasn't found or couldn't be read.")
        
    print("\n✅ Backend processing complete. You can now start Flask (python app.py)!")