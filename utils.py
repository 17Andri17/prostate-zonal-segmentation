import os
import re

def extract_patient_ids_from_folders(root):
    """
    Reads folder names like 001, 002, ... and returns a set of integers.
    """
    ids = set()
    for name in os.listdir(root):
        if name.isdigit() and len(name) == 3:
            ids.add(int(name))
    return ids


def extract_patient_ids_from_files(folder, pattern):
    """
    Reads filenames and extracts 4-digit patient IDs using a regex pattern.
    Example pattern: r"Seg-(\d{4})\.nrrd"
    """
    ids = set()
    for fname in os.listdir(folder):
        match = re.match(pattern, fname)
        if match:
            ids.add(int(match.group(1)))
    return ids


def find_common_patients(patients_root, prostatex_root):
    singles_root = f"{prostatex_root}/Singles"
    dup_r1_root = f"{prostatex_root}/Duplicates/R1"

    # --- Extract IDs ---
    folder_ids = extract_patient_ids_from_folders(patients_root)  # 3-digit → int
    singles_ids = extract_patient_ids_from_files(singles_root, r"Seg-(\d{4})\.nrrd")
    dup_r1_ids = extract_patient_ids_from_files(dup_r1_root, r"Seg-(\d{4})_R1\.nrrd")

    # Combine segmentation IDs
    seg_ids = singles_ids | dup_r1_ids

    # Convert folder IDs (3-digit) → 4-digit padded
    seg_ids_3 = {int(f"{i:03d}") for i in seg_ids}

    # --- Find common IDs ---
    common = seg_ids_3 & folder_ids

    # --- Write results ---
    with open("common_ids.txt", "w") as f:
        for pid in sorted(common):
            f.write(f"{pid:03d}\n")

    print("Common patient IDs written to common_ids.txt")