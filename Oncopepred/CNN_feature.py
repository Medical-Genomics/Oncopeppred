
import numpy as np
import string
from protlearn.features import aac, cksaap
from rdkit import Chem
from rdkit.Chem import Descriptors

# Define amino acid groups based on their physicochemical properties
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

def compute_pcp(sequence):
    pcp_features = np.zeros(20)
    seq_length = len(sequence)
    
    if seq_length == 0:
        return pcp_features

    pcp_groups_list = list(PCP_GROUPS.items())[:20]

    for idx, (pcp_name, residues) in enumerate(pcp_groups_list):
        count = sum(1 for aa in sequence if aa in residues)
        pcp_features[idx] = count / seq_length

    return pcp_features


from collections import Counter

def aac_new(sequence):
    amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
    sequence = sequence.upper()
    aa_counts = Counter(sequence)  # Get counts of each amino acid
    aac_values = [aa_counts.get(aa, 0) / len(sequence) for aa in amino_acids]
    return np.array(aac_values)


# Updated function for dipeptide composition (DPC)
def dpc_new(sequence, k=1):
    dipeptides = [''.join(pair) for pair in zip(string.ascii_uppercase, string.ascii_uppercase)]
    
    if len(sequence) < 2:
        return np.zeros(400)  # Return zero array for sequences that are too short

    counts = np.zeros(400)

    for i in range(len(sequence) - k):
        dipeptide = sequence[i:i + 2]  # Get pair of consecutive residues
        idx = dipeptides.index(dipeptide)
        counts[idx] += 1

    length = len(sequence) - k
    return counts / length if length > 0 else counts


# Function to compute pharmacophore features using RDKit
def compute_pharmacophore(sequence):
    """Compute 16 pharmacophore features for a given sequence."""
    # We assume that RDKit can't handle peptide sequences directly, so let's simplify.
    if not sequence.isalpha() or len(sequence) == 0:
        return np.zeros(16)  # Return zero vector for invalid or empty sequence

    features = np.zeros(16)  # Placeholder for pharmacophore features
    # We can replace this part with actual feature extraction logic for peptides (if any)

    return features


amino_acid_to_int = {'A': 0, 'C': 1, 'D': 2, 'E': 3, 'F': 4, 'G': 5, 'H': 6, 'I': 7, 'K': 8,
                         'L': 9, 'M': 10, 'N': 11, 'P': 12, 'Q': 13, 'R': 14, 'S': 15, 'T': 16, 'V': 17,
                         'W': 18, 'Y': 19}

def preprocess_sequence(seq, max_length=456):
    amino_acid_to_int = {aa: idx for idx, aa in enumerate('ACDEFGHIKLMNPQRSTVWY')}
    seq = seq.upper()
    seq_int = [amino_acid_to_int.get(aa, 0) for aa in seq]  # Default to 0 for invalid amino acids

    seq_int = seq_int[:max_length]  # Truncate if longer than max_length
    seq_int = seq_int + [0] * (max_length - len(seq_int))  # Pad with zeros if shorter

    return np.array(seq_int).reshape((1, max_length, 1))


# Main feature extraction function
def extract_features(sequence):
    try:
        # Extract physicochemical and other features
        aac_values = np.array(aac_new(sequence)).flatten()                  # (20,)
        dpc_values = np.array(dpc_new(sequence)).flatten()                  # (400,)
        pcp_values = np.array(compute_pcp(sequence)).flatten()             # (20,)
        pharmacophore_values = np.array(compute_pharmacophore(sequence)).flatten()  # (16,)

        # Preprocess sequence (convert to padded integer array)
        sequence_padded = preprocess_sequence(sequence)

        # Combine all extracted features into one feature vector
        feature_vector = np.concatenate((
            aac_values,
            dpc_values,
            pcp_values,
            pharmacophore_values,
            sequence_padded.flatten()  # Flatten the padded sequence
        ))

        # Ensure the vector has a consistent size
        if feature_vector.shape[0] < 456:
            feature_vector = np.pad(feature_vector, (0, 456 - feature_vector.shape[0]))

        return feature_vector

    except Exception as e:
        print(f"❌ Feature extraction failed: {str(e)}")
        return np.zeros(456)