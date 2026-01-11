import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from DataUtils import MultiAnnotatorProstateDataset


class MultiAnnotatorFusion:
    """
    Implements our multi-annotator fusion approach.
    Handles pixel-level ambiguity, annotator reliability, and probabilistic fusion.
    """

    def __init__(self, num_classes: int = 7, epsilon: float = 1e-8):
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.annotator_reliabilities = None

    def compute_pixel_ambiguity(self, annotations: torch.Tensor) -> torch.Tensor:
        """
        Compute pixel-level ambiguity weights.

        Args:
            annotations: Tensor of shape [N, H, W] where N is number of annotators
                        Each pixel contains class index (0-6)

        Returns:
            ambiguity_weights: Tensor of shape [H, W] with values in [0, 1]
        """
        N, H, W = annotations.shape

        # Compute class distribution for each pixel
        pixel_ambiguity = torch.zeros(H, W, device=annotations.device)

        for i in range(H):
            for j in range(W):
                pixel_votes = annotations[:, i, j]
                # Count votes for each class
                class_counts = torch.zeros(self.num_classes, device=annotations.device)
                for c in range(self.num_classes):
                    class_counts[c] = (pixel_votes == c).sum().float()

                # Probability distribution
                p = class_counts / N
                # Ambiguity weight: 1 - max probability
                pixel_ambiguity[i, j] = 1 - p.max()

        return pixel_ambiguity

    def compute_annotator_reliability(self, annotations: torch.Tensor) -> torch.Tensor:
        """
        Compute weighted annotatbbor reliability scores.

        Args:
            annotations: Tensor of shape [N, H, W] where N is number of annotators

        Returns:
            reliability_scores: Tensor of shape [N] with reliability scores
        """
        N, H, W = annotations.shape

        # Compute pixel ambiguity weights
        ambiguity_weights = self.compute_pixel_ambiguity(annotations)

        # Compute reliability for each annotator
        reliability_scores = torch.zeros(N, device=annotations.device)

        for i in range(N):
            total_weighted_agreement = 0
            total_agreement_pairs = 0

            for j in range(N):
                if i != j:
                    # Compute agreement between annotator i and j
                    agreement_mask = (annotations[i] == annotations[j]).float()

                    # Weighted agreement
                    weighted_agreement = (ambiguity_weights * agreement_mask).sum()
                    total_weight = ambiguity_weights.sum()

                    if total_weight > self.epsilon:
                        total_weighted_agreement += weighted_agreement / total_weight
                        total_agreement_pairs += 1

            if total_agreement_pairs > 0:
                reliability_scores[i] = total_weighted_agreement / total_agreement_pairs

        self.annotator_reliabilities = reliability_scores
        return reliability_scores

    def probabilistic_fusion(self, annotations: torch.Tensor,
                             reliabilities: torch.Tensor = None) -> torch.Tensor:
        """
        Fuse multiple annotations into probabilistic label maps.

        Args:
            annotations: Tensor of shape [N, H, W] with class indices
            reliabilities: Optional reliability scores (if None, compute them)

        Returns:
            fused_probabilities: Tensor of shape [num_classes, H, W] with probabilities
        """
        N, H, W = annotations.shape

        if reliabilities is None:
            reliabilities = self.compute_annotator_reliability(annotations)

        # Initialize probability map
        fused_probabilities = torch.zeros(self.num_classes, H, W,
                                          device=annotations.device)

        # Normalize reliabilities
        if reliabilities.sum() > self.epsilon:
            normalized_reliabilities = reliabilities / reliabilities.sum()
        else:
            normalized_reliabilities = torch.ones_like(reliabilities) / N

        # Aggregate votes weighted by reliability
        for i in range(N):
            annotator_mask = annotations[i]
            reliability_weight = normalized_reliabilities[i]

            for c in range(self.num_classes):
                class_mask = (annotator_mask == c).float()
                fused_probabilities[c] += reliability_weight * class_mask

        return fused_probabilities

    def get_hard_labels(self, probabilities: torch.Tensor) -> torch.Tensor:
        """Convert probabilities to hard labels (argmax)"""
        return torch.argmax(probabilities, dim=0)


class MultiAnnotatorUNetDataset(Dataset):
    """
    Dataset wrapper that applies multi-annotator fusion for training.
    """

    def __init__(self, base_dataset: MultiAnnotatorProstateDataset,
                 fusion_method: MultiAnnotatorFusion = None,
                 use_probabilistic_labels: bool = True):
        self.base_dataset = base_dataset
        self.fusion_method = fusion_method or MultiAnnotatorFusion()
        self.use_probabilistic_labels = use_probabilistic_labels

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        # Get multi-annotator data
        sample = self.base_dataset[idx]

        # Convert one-hot labels to class indices for fusion
        # Shape: [num_annotators, num_classes, H, W] -> [num_annotators, H, W]
        annotations = sample['labels']  # [N, C, H, W]
        N, C, H, W = annotations.shape

        # Convert to class indices
        annotations_idx = torch.argmax(annotations, dim=1)  # [N, H, W]

        # Apply fusion
        fused_probabilities = self.fusion_method.probabilistic_fusion(annotations_idx)

        if self.use_probabilistic_labels:
            # Use soft probabilistic labels
            labels = fused_probabilities
        else:
            # Use hard labels
            labels = self.fusion_method.get_hard_labels(fused_probabilities)
            # Convert back to one-hot
            labels = F.one_hot(labels.long(), num_classes=C).permute(2, 0, 1).float()

        return {
            'image': sample['image'],
            'labels': labels,
            'patient_id': sample['patient_id'],
            'original_labels': sample['labels'],  # Keep original for analysis
            'reliabilities': self.fusion_method.annotator_reliabilities,
            'label_names': sample['label_names']
        }
