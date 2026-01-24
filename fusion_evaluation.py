import random

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.spatial.distance import cdist
from scipy.stats import ttest_rel, wilcoxon
from torch.utils.data import DataLoader

from Utils.AnnotationFusion import MultiAnnotatorFusion, STAPLEFusionProvider
from Utils.DataUtils import MultiAnnotatorProstateDataset
from Utils.utils import find_common_patients_3, find_common_patients

random.seed(42)


# ---------------------------------------------------------
# Utility: metrics, statistical tests
# ---------------------------------------------------------
def hausdorff95(pred, gt):
    """
    pred, gt: [H, W] integer masks for a single class
    Computes the 95th percentile Hausdorff distance.
    """
    pred_pts = np.argwhere(pred > 0)
    gt_pts = np.argwhere(gt > 0)

    if len(pred_pts) == 0 or len(gt_pts) == 0:
        return np.nan

    D = cdist(pred_pts, gt_pts)

    hd95 = max(np.percentile(D.min(axis=1), 95), np.percentile(D.min(axis=0), 95))
    return float(hd95)


def expected_calibration_error(probabilities, labels, num_bins=15):
    """
    probabilities: [C, H, W] softmax output
    labels: [H, W] ground truth class indices
    """
    probs = probabilities.permute(1, 2, 0).reshape(-1, probabilities.shape[0])
    labels = labels.reshape(-1)

    confidences, predictions = probs.max(dim=1)
    accuracies = (predictions == labels).float()

    ece = 0.0
    bin_boundaries = torch.linspace(0, 1, num_bins + 1)

    for i in range(num_bins):
        mask = (confidences >= bin_boundaries[i]) & (confidences < bin_boundaries[i + 1])
        if mask.sum() > 0:
            bin_acc = accuracies[mask].mean()
            bin_conf = confidences[mask].mean()
            ece += (mask.float().mean() * torch.abs(bin_acc - bin_conf))

    return float(ece)


def statistical_tests(fusion_scores, staple_scores):
    """
    fusion_scores, staple_scores: lists of per-patient Dice values
    """
    fusion_arr = np.array(fusion_scores)
    staple_arr = np.array(staple_scores)

    t_stat, t_p = ttest_rel(fusion_arr, staple_arr, nan_policy='omit')
    w_stat, w_p = wilcoxon(fusion_arr, staple_arr)

    return {"t_test_p": float(t_p), "wilcoxon_p": float(w_p)}


def dice_score(pred, gt, num_classes=3):
    """
    pred, gt: [H, W] integer class maps
    returns dict: {class_id: dice}
    """
    scores = {}
    for c in range(num_classes):
        pred_c = (pred == c).float()
        gt_c = (gt == c).float()

        intersection = (pred_c * gt_c).sum()
        union = pred_c.sum() + gt_c.sum()

        dice = (2 * intersection) / (union + 1e-8)
        scores[c] = dice.item()

    return scores


# ---------------------------------------------------------
# Visualization
# ---------------------------------------------------------
def visualize(image, gt, fusion_pred, staple_pred, pid, num_annotators):
    plt.figure(figsize=(16, 4))

    plt.subplot(1, 4, 1)
    plt.imshow(image[0], cmap="gray")
    plt.title(f"Image {pid}")
    plt.axis("off")

    plt.subplot(1, 4, 2)
    plt.imshow(gt, cmap="viridis")
    plt.title("Ground Truth")
    plt.axis("off")

    plt.subplot(1, 4, 3)
    plt.imshow(fusion_pred, cmap="viridis")
    plt.title(f"Fusion: {num_annotators} annotators")
    plt.axis("off")

    plt.subplot(1, 4, 4)
    plt.imshow(staple_pred, cmap="viridis")
    plt.title(f"STAPLE: {num_annotators} annotators")
    plt.axis("off")

    plt.show()


# ---------------------------------------------------------
# Evaluation Loop
# ---------------------------------------------------------
def evaluate_fusion(data_root, labels_root, prostatex_root, patient_ids, batch_size=1, visualize_results=False,
                    device="cuda" if torch.cuda.is_available() else "cpu", num_annotators=2):
    assert num_annotators == 2 or num_annotators == 3
    # Load dataset
    dataset = MultiAnnotatorProstateDataset(data_root=data_root, labels_root=labels_root, prostatex_root=prostatex_root,
                                            patient_ids=patient_ids, modalities=["t2w"], target_size=(256, 256),
                                            normalize=True, num_annotators=num_annotators,
                                            include_R2=num_annotators == 3)

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    fusion = MultiAnnotatorFusion()
    staple = STAPLEFusionProvider()

    fusion_dice_list = []
    staple_dice_list = []
    fusion_dice_list_pz = []
    staple_dice_list_pz = []
    fusion_dice_list_tz = []
    staple_dice_list_tz = []
    fusion_hd_list_pz = []
    staple_hd_list_pz = []
    fusion_hd_list_tz = []
    staple_hd_list_tz = []
    fusion_ece_list = []
    staple_ece_list = []

    for batch in loader:
        image = batch["image"].to(device)
        labels = batch["labels"].to(device)  # [B, N, C, H, W]
        base_label = batch["base_label"].to(device)  # [B, C, H, W]
        pid = batch["patient_id"][0]

        # Ground truth from dataset
        gt = base_label[0].argmax(dim=0)  # [H, W]

        # Annotator labels
        annot = labels[0]  # [N, C, H, W]
        annot_idx = annot.argmax(dim=1)  # [N, H, W]

        # Fusion method
        fused_probs = fusion.probabilistic_fusion(annot_idx)
        fusion_pred = fusion.get_hard_labels(fused_probs)

        # STAPLE
        staple_probs = staple.probabilistic_fusion(annot_idx)
        staple_pred = staple.get_hard_labels(staple_probs)

        # Dice
        fusion_dice = dice_score(fusion_pred, gt)
        staple_dice = dice_score(staple_pred, gt)

        fusion_dice_list.append(fusion_dice)
        staple_dice_list.append(staple_dice)

        fusion_dice_list_pz.append(fusion_dice[1])  # PZ
        staple_dice_list_pz.append(staple_dice[1])  # PZ

        fusion_dice_list_tz.append(fusion_dice[2])  # TZ
        staple_dice_list_tz.append(staple_dice[2])  # TZ

        # Hausdorff95
        # (PZ class)
        fusion_hd_list_pz.append(
            hausdorff95((fusion_pred.cpu().numpy() == 1).astype(np.uint8), (gt.cpu().numpy() == 1).astype(np.uint8)))
        staple_hd_list_pz.append(
            hausdorff95((staple_pred.cpu().numpy() == 1).astype(np.uint8), (gt.cpu().numpy() == 1).astype(np.uint8)))
        # (TZ class)
        fusion_hd_list_tz.append(
            hausdorff95((fusion_pred.cpu().numpy() == 2).astype(np.uint8), (gt.cpu().numpy() == 2).astype(np.uint8)))
        staple_hd_list_tz.append(
            hausdorff95((staple_pred.cpu().numpy() == 2).astype(np.uint8), (gt.cpu().numpy() == 2).astype(np.uint8)))

        # Calibration Error (ECE)
        fusion_ece_list.append(expected_calibration_error(fused_probs.cpu(), gt.cpu()))
        staple_ece_list.append(expected_calibration_error(staple_probs.cpu(), gt.cpu()))

        # Visualization
        if visualize_results:
            print(f"\nPatient {pid}")
            print("Fusion Dice:", fusion_dice)
            print("STAPLE Dice:", staple_dice)

            visualize(image[0].cpu().numpy(), gt.cpu().numpy(), fusion_pred.cpu().numpy(), staple_pred.cpu().numpy(),
                      pid, num_annotators)

    print("\n==================== SUMMARY: Dice ====================")

    def mean_class_dice(results, class_id):
        return sum(r[class_id] for r in results) / len(results)

    # Per‑class Dice
    for c in range(3):
        fusion_mean = mean_class_dice(fusion_dice_list, c)
        staple_mean = mean_class_dice(staple_dice_list, c)

        print(f"\nClass {c} Dice:")
        print(f"  Fusion : {fusion_mean:.4f}")
        print(f"  STAPLE : {staple_mean:.4f}")

    # Overall mean Dice across all classes
    fusion_overall = sum(sum(r.values()) for r in fusion_dice_list) / (3 * len(fusion_dice_list))
    staple_overall = sum(sum(r.values()) for r in staple_dice_list) / (3 * len(staple_dice_list))

    print("\nMean Dice (all classes):")
    print(f"  Fusion : {fusion_overall:.4f}")
    print(f"  STAPLE : {staple_overall:.4f}")

    print("\n==================== SUMMARY: Hausdorff95 ====================")

    def mean(values):
        return sum(values) / len(values) if len(values) > 0 else float("nan")

    # --- Hausdorff95 for PZ ---
    print("\nHausdorff95 (PZ):")
    print(f"  Fusion : {np.nanmean(np.array(fusion_hd_list_pz)):.4f}")
    print(f"  STAPLE : {np.nanmean(np.array(staple_hd_list_pz)):.4f}")

    # --- Hausdorff95 for TZ ---
    print("\nHausdorff95 (TZ):")
    print(f"  Fusion : {np.nanmean(np.array(fusion_hd_list_tz)):.4f}")
    print(f"  STAPLE : {np.nanmean(np.array(staple_hd_list_tz)):.4f}")

    print("\n==================== SUMMARY: Calibration Error (ECE) ====================")

    print(f"\nECE:")
    print(f"  Fusion : {mean(fusion_ece_list):.4f}")
    print(f"  STAPLE : {mean(staple_ece_list):.4f}")

    print("\n=== Statistical Significance Tests ===")

    dice_stats_pz = statistical_tests(fusion_dice_list_pz, staple_dice_list_pz)
    dice_stats_tz = statistical_tests(fusion_dice_list_tz, staple_dice_list_tz)
    hd_stats_pz = statistical_tests(fusion_hd_list_pz, staple_hd_list_pz)
    hd_stats_tz = statistical_tests(fusion_hd_list_tz, staple_hd_list_tz)
    ece_stats = statistical_tests(fusion_ece_list, staple_ece_list)

    print("Dice PZ p-values:", dice_stats_pz)
    print("Dice TZ p-values:", dice_stats_tz)
    print("Hausdorff PZ p-values:", hd_stats_pz)
    print("Hausdorff TZ p-values:", hd_stats_tz)
    print("ECE p-values:", ece_stats)


if __name__ == "__main__":
    DATA_PATH = r"AI4AR_cont/Data"
    LABELS_PATH = r"AI4AR_cont/Anatomical_Labels"
    PROSTATEX_PATH = r"ProstateZones"

    # 3 annotators comparison
    print("================== 3 annotators comparison ==================")
    find_common_patients_3(DATA_PATH, PROSTATEX_PATH)
    with open("common_ids_3.txt", "r") as f:
        patient_ids = [line.strip() for line in f if line.strip()]

    evaluate_fusion(data_root=DATA_PATH, labels_root=LABELS_PATH, prostatex_root=PROSTATEX_PATH,
                    patient_ids=patient_ids,  # random.sample(patient_ids, 10), for visualize_results = True
                    visualize_results=True, num_annotators=3)

    # 2 annotators comparison
    print("\n\n================== 2 annotators comparison ==================")
    find_common_patients(DATA_PATH, PROSTATEX_PATH)
    with open("common_ids.txt", "r") as f:
        patient_ids = [line.strip() for line in f if line.strip()]

    evaluate_fusion(data_root=DATA_PATH, labels_root=LABELS_PATH, prostatex_root=PROSTATEX_PATH,
                    patient_ids=patient_ids,  # random.sample(patient_ids, 10), for visualize_results = True
                    visualize_results=True, num_annotators=2)
