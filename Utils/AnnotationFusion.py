import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from Utils.DataUtils import MultiAnnotatorProstateDataset


class MultiAnnotatorFusion:
    """
    Implements our multi-annotator fusion approach.
    Handles pixel-level ambiguity, annotator reliability, and probabilistic fusion.
    Matches the mathematical formulation exactly.
    """

    def __init__(self, num_classes: int = 3, epsilon: float = 1e-8):
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.annotator_reliabilities = None

    def compute_pixel_ambiguity(self, annotations: torch.Tensor) -> torch.Tensor:
        """
        Compute pixel-level ambiguity weights.
        
        Equation: w(x) = 1 - max_c p_c(x)
        where p_c(x) = (# annotators labeling x as class c) / N
        
        Args:
            annotations: Tensor of shape [N, H, W]
            
        Returns:
            ambiguity_weights: Tensor of shape [H, W] with values in [0, 1]
        """
        N, H, W = annotations.shape
        device = annotations.device

        one_hot = F.one_hot(annotations.long(), num_classes=self.num_classes).float()  # [N, H, W, C]
        class_counts = one_hot.sum(dim=0)  # [H, W, C]

        # Probability distribution p_c(x)
        p = class_counts / max(N, 1)  # [H, W, C]

        # Ambiguity weight: 1 - max_c p_c(x)
        max_probs, _ = p.max(dim=-1)  # [H, W]
        pixel_ambiguity = 1 - max_probs  # [H, W]

        return pixel_ambiguity

    def compute_annotator_reliability(self, annotations: torch.Tensor) -> torch.Tensor:
        """
        Compute weighted annotator reliability scores.
        
        Equation: R_i = (1/(N-1)) * Σ_{j≠i} [Σ_x w(x) * 1[A_i(x) = A_j(x)] / Σ_x w(x)]
        
        Args:
            annotations: Tensor of shape [N, H, W]
            
        Returns:
            reliability_scores: Tensor of shape [N] with reliability scores
        """
        N, H, W = annotations.shape
        device = annotations.device

        # Compute pixel ambiguity weights
        ambiguity_weights = self.compute_pixel_ambiguity(annotations)  # [H, W]

        # Total weight Σ_x w(x)
        total_weight = ambiguity_weights.sum()

        if N <= 1:
            # If only 1 annotator, reliability is 1.0
            reliability_scores = torch.ones(N, device=device)
            self.annotator_reliabilities = reliability_scores
            return reliability_scores

        # Pre-calculate total weight for normalization
        if total_weight > self.epsilon:
            inv_total_weight = 1.0 / total_weight
        else:
            # If total weight is too small, use uniform reliability
            reliability_scores = torch.ones(N, device=device) / N
            self.annotator_reliabilities = reliability_scores
            return reliability_scores

        # Initialize reliability scores
        reliability_scores = torch.zeros(N, device=device)

        # Compute agreement for each pair of annotators
        # Use broadcasting for efficiency
        annotations_i = annotations.unsqueeze(1)  # [N, 1, H, W]
        annotations_j = annotations.unsqueeze(0)  # [1, N, H, W]

        # Agreement matrix: 1[A_i(x) = A_j(x)] for all i,j pairs
        agreement = (annotations_i == annotations_j).float()  # [N, N, H, W]

        # Weighted agreement: w(x) * 1[A_i(x) = A_j(x)]
        # Expand ambiguity_weights to match shape: [1, 1, H, W]
        w_expanded = ambiguity_weights.unsqueeze(0).unsqueeze(0)
        weighted_agreement = agreement * w_expanded  # [N, N, H, W]

        # Sum over spatial dimensions: Σ_x w(x) * 1[A_i(x) = A_j(x)]
        weighted_agreement_sum = weighted_agreement.sum(dim=(2, 3))  # [N, N]

        # Normalize by total weight
        normalized_agreement = weighted_agreement_sum * inv_total_weight  # [N, N]

        # Exclude self-comparisons (i ≠ j)
        # Create mask with False on diagonal
        mask = torch.eye(N, device=device, dtype=torch.bool)

        # Set diagonal to 0 (self-agreement doesn't count)
        normalized_agreement_masked = normalized_agreement.masked_fill(mask, 0)

        # Sum over j ≠ i and divide by (N-1)
        reliability_scores = normalized_agreement_masked.sum(dim=1) / (N - 1)

        self.annotator_reliabilities = reliability_scores
        return reliability_scores

    def probabilistic_fusion(self, annotations: torch.Tensor,
                             reliabilities: torch.Tensor = None) -> torch.Tensor:
        """
        Fuse multiple annotations into probabilistic label maps.
        
        Equation: P(c|x) = Σ_i R_i * 1[A_i(x) = c] / Σ_i R_i
        
        Args:
            annotations: Tensor of shape [N, H, W]
            reliabilities: Optional reliability scores
            
        Returns:
            fused_probabilities: Tensor of shape [num_classes, H, W]
        """
        N, H, W = annotations.shape
        device = annotations.device

        if reliabilities is None:
            reliabilities = self.compute_annotator_reliability(annotations)

        # Normalize reliabilities for weighted sum
        if reliabilities.sum() > self.epsilon:
            normalized_reliabilities = reliabilities / reliabilities.sum()
        else:
            normalized_reliabilities = torch.ones_like(reliabilities) / N

        # Convert to one-hot encoding: [N, C, H, W]
        one_hot = F.one_hot(annotations.long(), num_classes=self.num_classes)
        one_hot = one_hot.permute(0, 3, 1, 2).float()

        # Weight each annotator's one-hot encoding by their reliability
        # Expand weights to shape [N, 1, 1, 1] for broadcasting
        weights = normalized_reliabilities.view(N, 1, 1, 1)

        # Weighted sum: Σ_i R_i * 1[A_i(x) = c]
        weighted_one_hot = one_hot * weights

        # Sum across annotators to get P(c|x)
        fused_probabilities = weighted_one_hot.sum(dim=0)  # [C, H, W]

        return fused_probabilities

    def get_hard_labels(self, probabilities: torch.Tensor) -> torch.Tensor:
        """Convert probabilities to hard labels (argmax)"""
        return torch.argmax(probabilities, dim=0)


class STAPLEFusionProvider:
    """
    Alternative fusion provider using the STAPLE algorithm.
    Matches the interface of MultiAnnotatorFusion.
    """

    def __init__(self, num_classes: int = 3, max_iter: int = 20, tol: float = 1e-4):
        self.num_classes = num_classes
        self.max_iter = max_iter
        self.tol = tol
        self.sensitivity = None  # p
        self.specificity = None  # q

    def probabilistic_fusion(self, annotations: torch.Tensor) -> torch.Tensor:
        # annotations: [N, H, W]
        N, H, W = annotations.shape
        C = self.num_classes
        device = annotations.device

        # We cast them to int() to prevent the TypeError
        shape_tuple = (int(C), int(H), int(W))

        # Convert to One-Hot: [N, C, H, W]
        D = F.one_hot(annotations.long(), num_classes=C).permute(0, 3, 1, 2).float()

        # Initial estimate: Majority Vote
        W_map = D.mean(dim=0)

        for i in range(self.max_iter):
            old_W = W_map.clone()

            # Using W_map instead of W to avoid confusion with W variable
            sum_W = W_map.sum(dim=(1, 2)) + 1e-8
            sum_W_inv = (1 - W_map).sum(dim=(1, 2)) + 1e-8

            self.sensitivity = (D * W_map).sum(dim=(2, 3)) / sum_W
            self.specificity = ((1 - D) * (1 - W_map)).sum(dim=(2, 3)) / sum_W_inv

            p = torch.clamp(self.sensitivity, 0.01, 0.99).view(N, C, 1, 1)
            q = torch.clamp(self.specificity, 0.01, 0.99).view(N, C, 1, 1)

            log_W = torch.full(shape_tuple, torch.log(torch.tensor(1.0 / C)), device=device)

            term = (D * torch.log(p)) + ((1 - D) * torch.log(1 - q))
            log_W += term.sum(dim=0)

            W_map = F.softmax(log_W, dim=0)

            if torch.abs(W_map - old_W).mean() < self.tol:
                break

        return W_map

    def get_hard_labels(self, probabilities: torch.Tensor) -> torch.Tensor:
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

        reliabilities = getattr(self.fusion_method, 'annotator_reliabilities', torch.tensor(0))
        sensitivity = getattr(self.fusion_method, 'sensitivity', torch.tensor(0))
        specificity = getattr(self.fusion_method, 'specificity', torch.tensor(0))

        return {
            'image': sample['image'],
            'labels': labels,
            'patient_id': sample['patient_id'],
            'original_labels': sample['labels'],  # Keep original for analysis
            'reliabilities': reliabilities,
            'sensitivity': sensitivity,
            'specificity': specificity,
            'label_names': sample['label_names']
        }
