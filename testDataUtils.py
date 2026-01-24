import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from Utils.DataUtils import MultiAnnotatorProstateDataset


def visualize_sample(sample):
    """
    Visualize:
    - Input image
    - Annotator 1 mask (3 classes)
    - Annotator 2 mask (3 classes)
    - Difference map between annotators
    """

    image = sample["image"].numpy()  # [C, H, W]
    labels = sample["labels"].numpy()  # [num_annotators, 3, H, W]
    pid = sample["patient_id"]

    ann1 = labels[0]  # [3, H, W]
    ann2 = labels[1]  # [3, H, W]

    # Convert multi‑channel masks to single‑label maps for visualization
    ann1_map = ann1.argmax(axis=0)
    ann2_map = ann2.argmax(axis=0)

    diff_map = (ann1_map != ann2_map).astype(float)

    fig, axs = plt.subplots(1, 5, figsize=(22, 6))

    axs[0].imshow(image[0], cmap="gray")
    axs[0].set_title(f"Image (PID {pid})")
    axs[0].axis("off")

    axs[1].imshow(ann1_map, cmap="viridis")
    axs[1].set_title("Annotator 1 Mask")
    axs[1].axis("off")

    axs[2].imshow(ann2_map, cmap="viridis")
    axs[2].set_title("Annotator 2 Mask")
    axs[2].axis("off")

    axs[3].imshow(diff_map, cmap="hot")
    axs[3].set_title("Difference Map")
    axs[3].axis("off")

    # Overlay annotator 1 mask on image
    axs[4].imshow(image[0], cmap="gray")
    axs[4].imshow(ann1_map, cmap="jet", alpha=0.4)
    axs[4].set_title("Overlay (Annotator 1)")
    axs[4].axis("off")

    plt.tight_layout()
    plt.show()


def test_dataset():
    with open("common_ids.txt", "r") as f:
        patient_ids = [line.strip() for line in f if line.strip()]

    dataset = MultiAnnotatorProstateDataset(
        data_root=DATA_PATH,
        labels_root=LABELS_PATH,
        prostatex_root=PROSTATEX_PATH,
        patient_ids=patient_ids,
        modalities=["t2w"],
        target_size=(256, 256),
        normalize=True,
        num_annotators=2
    )

    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    i = 0
    for batch in loader:
        sample = {
            "image": batch["image"][0],
            "labels": batch["labels"][0],
            "patient_id": batch["patient_id"][0]
        }

        print(f"Patient {sample['patient_id']}")
        print("Image shape:", sample["image"].shape)
        print("Labels shape:", sample["labels"].shape)  # [2, 3, H, W]

        visualize_sample(sample)
        i += 1
        if i == 2:
            break


if __name__ == "__main__":
    DATA_PATH = r"AI4AR_cont/Data"
    LABELS_PATH = r"AI4AR_cont/Anatomical_Labels"
    PROSTATEX_PATH = r"ProstateZones"

    test_dataset()
