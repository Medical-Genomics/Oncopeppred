import numpy as np
from collections import Counter

AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"

PCP_GROUPS = {
    "Positively_charge": set("KR"),
    "Negatively_charge": set("DE"),
    "Neutral_charge": set("NQ"),
    "Polar": set("STNQ"),
    "Non_polar": set("GAVLIPFMWC"),
    "Aliphatic": set("GAVLMI"),
    "Cyclic": set("P"),
    "Aromatic": set("FWYH"),
    "Acidic": set("DE"),
    "Basic": set("KRH"),
    "Hydrophobic": set("AFILMPVW"),
    "Hydrophilic": set("RNDQEGHKPSTY"),
    "Tiny": set("AGDSC"),
    "Hydroxylic": set("STY"),
    "Sulfur": set("CM"),
    "Helix": set("AEHLKMQR"),
    "Strand": set("VIYCWTF"),
    "Coil": set("PGND"),
    "Buried": set("AILFWV"),
    "Exposed": set("RNDQEKPSTY")
}

def aac_new(sequence):
    sequence = sequence.upper()
    length = len(sequence)

    if length == 0:
        return np.zeros(20)

    counts = Counter(sequence)

    return np.array(
        [counts.get(aa, 0) / length for aa in AMINO_ACIDS],
        dtype=np.float32
    )

DIP_INDEX = {
    aa1 + aa2: idx
    for idx, (aa1, aa2)
    in enumerate([(a, b) for a in AMINO_ACIDS for b in AMINO_ACIDS])
}

def dpc_new(sequence):
    sequence = sequence.upper()

    dpc = np.zeros(400, dtype=np.float32)

    if len(sequence) < 2:
        return dpc

    total = 0

    for i in range(len(sequence) - 1):
        dipep = sequence[i:i+2]

        if dipep in DIP_INDEX:
            dpc[DIP_INDEX[dipep]] += 1
            total += 1

    if total > 0:
        dpc /= total

    return dpc

def compute_pcp(sequence):
    sequence = sequence.upper()

    if len(sequence) == 0:
        return np.zeros(20)

    length = len(sequence)

    values = []

    for residues in list(PCP_GROUPS.values())[:20]:
        count = sum(aa in residues for aa in sequence)
        values.append(count / length)

    return np.array(values, dtype=np.float32)

def compute_pharmacophore(sequence):
    sequence = sequence.upper()

    length = len(sequence)

    if length == 0:
        return np.zeros(16)

    hydrophobic = set("AFILMPVW")
    aromatic = set("FWYH")
    positive = set("KRH")
    negative = set("DE")
    polar = set("STNQ")
    sulfur = set("CM")

    features = [
        sum(aa in hydrophobic for aa in sequence) / length,
        sum(aa in aromatic for aa in sequence) / length,
        sum(aa in positive for aa in sequence) / length,
        sum(aa in negative for aa in sequence) / length,
        sum(aa in polar for aa in sequence) / length,
        sum(aa in sulfur for aa in sequence) / length,
        length,
        sequence.count("G") / length,
        sequence.count("P") / length,
        sequence.count("W") / length,
        sequence.count("Y") / length,
        sequence.count("F") / length,
        sequence.count("K") / length,
        sequence.count("R") / length,
        sequence.count("D") / length,
        sequence.count("E") / length
    ]

    return np.array(features, dtype=np.float32)

def extract_features(sequence):
    aac = aac_new(sequence)
    dpc = dpc_new(sequence)
    pcp = compute_pcp(sequence)
    pharma = compute_pharmacophore(sequence)

    features = np.concatenate(
        [aac, dpc, pcp, pharma]
    )

    return features.astype(np.float32)
