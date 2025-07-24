import os
import logging
import joblib
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify, send_file, Response
from datetime import datetime
import tensorflow as tf
from rdkit import Chem
from rdkit.Chem import Draw, Descriptors, AllChem, Lipinski
from rdkit.Chem import rdmolfiles
from Bio.PDB import PDBIO
from PeptideBuilder import Geometry
from Bio.PDB import PDBParser
import PeptideBuilder
from protlearn.features import aac  # Use protlearn directly
from flask import Flask, render_template, request
from feature_extractor import extract_features, compute_pcp, compute_pharmacophore
import csv
from predictor import predict_acp  # Reuse centralized prediction logic
import io
from sklearn.preprocessing import StandardScaler
from keras.models import load_model
from flask import render_template  
from feature_extractor import extract_features, aac_new, dpc_new,compute_pcp, compute_pharmacophore  # Reuse centralized prediction logic
from feature_extractor import extract_features
from protlearn.features import aac, cksaap




# Configure TensorFlow to suppress warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Initialize Flask app
app = Flask(__name__)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set up static directory for images
IMAGE_DIR = "static/images"
os.makedirs(IMAGE_DIR, exist_ok=True)

# Set up directory for storing feature CSVs
FEATURE_DIR = os.path.join("static", "features")
os.makedirs(FEATURE_DIR, exist_ok=True)

# Load dataset
def load_dataset():
    try:
        data = pd.read_csv("pepacp(2348).csv", encoding="utf-8", low_memory=False)
        rename_mapping = {
            "Molecular_Weight": "Molecular Weight",
            "Activity against": "Activity",
            "PMID": "Reference"
        }
        data.rename(columns=rename_mapping, inplace=True)

        required_columns = {"Peptide ID", "Peptide Name", "Sequence", "Molecular Weight", "Charge", "Reference"}
        for col in required_columns - set(data.columns):
            data[col] = "N/A"

        data.fillna("N/A", inplace=True)
        data["Peptide ID"] = data["Peptide ID"].astype(str)
        logger.info("✅ Dataset loaded successfully.")
        return data
    except Exception as e:
        logger.error(f"❌ Error loading dataset: {e}")
        return pd.DataFrame(columns=list(required_columns))
    
def load_model_and_scaler():
    """Load the pre-trained model and scaler."""
    try:
        model = tf.keras.models.load_model("acp_model11.keras", compile=True)
        scaler = joblib.load("scaler11.pkl") if os.path.exists("scaler11.pkl") else None
        if scaler and not hasattr(scaler, "mean_"):
            logger.error("❌ Scaler not fitted.")
            scaler = None
        logger.info(f"✅ Model loaded: {model}")
        logger.info(f"✅ Scaler loaded: {scaler}")
        return model, scaler
    except Exception as e:
        logger.error(f"❌ Model/Scaler loading error: {e}")
        return None, None
    
model, scaler = load_model_and_scaler()

def predict_acp_from_sequence(sequence, model, scaler):
    """Run feature extraction and prediction."""
    aac = aac_new(sequence)
    dpc = dpc_new(sequence)
    pcp = compute_pcp(sequence)
    ph4 = compute_pharmacophore(sequence)

    features = np.concatenate([aac, dpc, pcp, ph4])
    features = features.reshape(1, -1)
    features_scaled = scaler.transform(features)
    features_scaled = features_scaled.reshape((1, 456, 1))

    prediction = model.predict(features_scaled)
    pred_label = "ACP" if prediction[0][0] > 0.5 else "Non-ACP"
    return pred_label

@app.route("/")
def ind():
    return render_template("ind.html")

@app.route("/index")
def home():
    return render_template("index.html")


data = load_dataset()
 # Adjust path as needed


@app.route("/search")
def search():
    query = request.args.get("query", "").strip().lower()
    if not query:
        return render_template("downloads.html", peptides=[], message="Enter a valid search query.")

    filtered_data = data[
        data["Peptide Name"].str.lower().str.contains(query, na=False) | 
        data["Peptide"].str.lower().str.contains(query, na=False)
    ]
    return render_template("downloads.html", peptides=filtered_data.to_dict(orient="records"))

@app.route("/peptide/<pepid>")
def peptide_detail(pepid):
    peptide = data[data["Peptide ID"] == pepid].to_dict(orient="records")
    return render_template("peptide_detail.html", peptide=peptide[0]) if peptide else jsonify({"error": "Peptide not found"}), 404


@app.route("/", methods=["GET", "POST"])

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        sequence = data.get('sequence', '').strip()
        
        if not sequence:
            return jsonify({"error": "Sequence is empty"}), 400
        
        prediction = predict_acp_from_sequence(sequence, model, scaler)
        return jsonify({"prediction": prediction})
    
    except Exception as e:
        logger.error(f"Error in prediction: {e}")
        return jsonify({"error": "Internal server error"}), 500



@app.route("/prediction")
def prediction():
    return render_template("prediction.html")

# ----------------- Feature Download -----------------

@app.route('/download_features')
def download_features():
    try:
        feature_file = os.path.join("static", "features", "sequence_features.csv")
        return send_file(feature_file, as_attachment=True,
                         download_name="sequence_features.csv",
                         mimetype='text/csv')
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ----------------- Feature Page (manual POST) -----------------
PCP_GROUPS = {
    "Positively_charge": set("KR"),  # Positively charged
    "Negatively_charge": set("DE"),  # Negatively charged
    "Neutral_charge": set("NQ"),  # Neutral charged
    "Polar": set("STNQ"),  # Polar
    "Non-polar": set("GAVLIPFMWC"),  # Non-polar
    "Aliphatic": set("GAVLMI"),  # Aliphatic
    "Cyclic_(Proline)": set("P"),  # Cyclic (Proline)
    "Aromatic": set("FWYH"),  # Aromatic
    "Acidic": set("DE"),  # Acidic
    "Basic": set("KRH"),  # Basic
    "Neutral_pH": set("NQ"),  # Neutral at physiological pH
    "Hydrophobic": set("AFILMPVW"),
    "Hydrophilic": set("RNDQEGHKPSTY"),
    "Tiny": set("AGDSC"),
    "Hydroxylic": set("STY"),
    "Sulfur_Containing": set("CM"),
    "Helix_Forming": set("AEHLKMQR"),
    "Strand_Forming": set("VIYCWTF"),
    "Coil_Forming": set("PGND"),
    "Buried_Residues": set("AILFWV"),  # Buried residues
    "PCP_SA_EX": set("RNDQEKPSTY"),  # Exposed residues
    "PCP_SA_IN": set("CMG"),  # Intermediate accessibility
    "PCP_TN": set("GASD"),  # Tiny residues
    "PCP_SM": set("VCNTP"),  # Small residues
    "PCP_LR": set("KMHFRYW"),  # Large residues
    "PCP_Z1": set("ADEFIKLMNQRSVWY"),  # Advanced PCP Z1
    "PCP_Z2": set("CDFGHLNPQSTVWY"),  # Advanced PCP Z2
    "PCP_Z3": set("ADEGIKLMNPQRSVWY"),  # Advanced PCP Z3
    "PCP_Z4": set("CDEFHIKLMNPQRSTVWY"),  # Advanced PCP Z4
    "PCP_Z5": set("ACDEFGHIKLMNPQRSTVWY"),  # Advanced PCP Z5
}

@app.route('/features', methods=['POST'])
def features():
    try:
        print("==> Step 1: Getting sequence")
        sequence = request.form.get('sequence', '').strip().upper()
        if not sequence:
            raise ValueError("No sequence provided.")
        print(f"==> Sequence: {sequence}")

        print("==> Step 2: Extracting AAC")
        aac_features = list(aac_new(sequence))
        print("AAC done")

        print("==> Step 3: Extracting DPC")
        dpc_features = list(dpc_new(sequence))
        print("DPC done")

        print("==> Step 4: Extracting PCP")
        pcp_features = list(compute_pcp(sequence))
        print("PCP done")

        print("==> Step 5: Extracting Pharma")
        pharma_features = list(compute_pharmacophore(sequence))
        print("Pharma done")

        aac_labels = [f"AAC_{aa}" for aa in 'ACDEFGHIKLMNPQRSTVWY']
        dpc_labels = [f"DPC_{a}{b}" for a in 'ACDEFGHIKLMNPQRSTVWY' for b in 'ACDEFGHIKLMNPQRSTVWY']
        pcp_labels = list(PCP_GROUPS.keys())
        pharma_labels = [f"Pharma_{i+1}" for i in range(len(pharma_features))]

        aac_data = dict(zip(aac_labels, aac_features))
        dpc_data = dict(zip(dpc_labels, dpc_features))
        pcp_data = dict(zip(pcp_labels, pcp_features))
        pharma_data = dict(zip(pharma_labels, pharma_features))

        all_features = {**aac_data, **dpc_data, **pcp_data, **pharma_data}
        non_zero_labels = [k for k, v in all_features.items() if v != 0]
        non_zero_values = [v for k, v in all_features.items() if v != 0]

        print("==> Rendering features.html")

        return render_template(
            "features.html",
            sequence=sequence,
            aac=aac_features,
            dpc=dpc_features,
            pcp=pcp_features,
            pharmacophore=pharma_features,
            feature_presence={
                'aac': any(aac_features),
                'dpc': any(dpc_features),
                'pcp': any(pcp_features),
                'pharmacophore': any(pharma_features)
            },
            non_zero_labels=non_zero_labels,
            non_zero_values=non_zero_values
        )

    except Exception as e:
        print("❌ Caught error in features route:", e)
        logger.error(f"❌ Features page error: {e}")
        return render_template("prediction.html", error="Error displaying features.")


UPLOAD_FOLDER = os.path.join("static")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Compute molecular properties
def get_properties(mol):
    return {
        "Molecular Weight": round(Descriptors.MolWt(mol), 3),
        "LogP (Hydrophobicity)": round(Descriptors.MolLogP(mol), 5),
        "H-Bond Donors": Lipinski.NumHDonors(mol),
        "H-Bond Acceptors": Lipinski.NumHAcceptors(mol),
        "TPSA (Polar Surface Area)": round(Descriptors.TPSA(mol), 3),
        "Rotatable Bonds": Lipinski.NumRotatableBonds(mol)
    }

# Convert peptide sequence to 3D molecule using PeptideBuilder and RDKit
def peptide_to_mol(sequence):
    try:
        sequence = sequence.strip().upper().replace(" ", "").replace("\n", "")
        valid_aa = "ACDEFGHIKLMNPQRSTVWY"
        if not all(residue in valid_aa for residue in sequence):
            raise ValueError("Sequence contains invalid amino acids.")
        if len(sequence) > 50:
            raise ValueError("Sequence too long (limit: 50 amino acids).")

        # Build 3D structure
        geo = Geometry.geometry(sequence[0])
        structure = PeptideBuilder.initialize_res(geo)
        for aa in sequence[1:]:
            geo = Geometry.geometry(aa)
            PeptideBuilder.add_residue(structure, geo)

        # Save PDB file
        pdb_path = os.path.join(UPLOAD_FOLDER, "peptide.pdb")
        io = PDBIO()
        io.set_structure(structure)
        io.save(pdb_path)

        # Convert to RDKit mol
        rdkit_mol = Chem.MolFromPDBFile(pdb_path, removeHs=False)
        if rdkit_mol is None:
            raise ValueError("Failed to parse peptide into RDKit molecule.")

        Chem.SanitizeMol(rdkit_mol)
        AllChem.EmbedMolecule(rdkit_mol)
        AllChem.UFFOptimizeMolecule(rdkit_mol)

        # Save MOL file for 3Dmol.js
        mol_path = os.path.join(UPLOAD_FOLDER, "peptide_3d.mol")
        Chem.MolToMolFile(rdkit_mol, mol_path)

        return rdkit_mol

    except Exception as e:
        return str(e)


@app.route("/structure_prediction")
def structure_page():
    return render_template("structure_prediction.html")


@app.route("/predict_structure", methods=["POST"])
def predict_structure():
    sequence = request.form["sequence"]
    vis_type = request.form["vis_type"]

    mol = peptide_to_mol(sequence)
    if isinstance(mol, str):  # Error
        return render_template("structure_prediction.html", error=mol)

    properties = get_properties(mol)
    mol_block = Chem.MolToMolBlock(mol)

    return render_template("structure_prediction.html",
                           vis_type=vis_type,
                           properties=properties,
                           mol_block=mol_block,
                           sequence=sequence)


@app.route('/download_molblock', methods=['POST'])
def download_molblock():
    mol_block = request.form['mol_block']
    return Response(
        mol_block,
        mimetype='chemical/x-mdl-molfile',
        headers={"Content-Disposition": "attachment;filename=peptide_structure.mol"}
    )

    
if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)
