import copy
import logging
import os
import time
from collections import defaultdict
from typing import List, Tuple, Dict

import numpy as np
import torch
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap, BoundaryNorm
from sklearn.model_selection import train_test_split
from torch import GradScaler, autocast, nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from AnnotationFusion import MultiAnnotatorFusion, MultiAnnotatorUNetDataset
from DataUtils import MultiAnnotatorProstateDataset
from UNet import UNet
from Loss import CombinedLoss


log_filename = f"training_{time.strftime('%Y%m%d_%H%M%S')}.log"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler()            
    ]
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logger.info(f"Logging initialized. Saving to {log_filename}")

class ProstateSegmentationTrainer:
    """Complete training and evaluation pipeline"""

    def __init__(self,
                 model: nn.Module,
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 test_loader: DataLoader,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
                 learning_rate: float = 1e-4,
                 weight_decay: float = 1e-5):

        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = device

        # Loss function - use combined loss
        self.criterion = CombinedLoss(dice_weight=0.7, ce_weight=0.3).to(device)

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )

        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5
        )

        # Gradient scaler for mixed precision
        self.scaler = GradScaler() if device == 'cuda' else None

        # Metrics tracking
        self.train_losses = []
        self.val_losses = []
        self.train_dice_scores = []
        self.val_dice_scores = []
        self.learning_rates = []

        # Best model tracking
        self.best_model_state = None
        self.best_val_loss = float('inf')
        self.best_val_dice = 0.0

        logger.info(f"Initialized trainer on device: {device}")
        logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    def train_epoch(self) -> Tuple[float, float]:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        total_dice = 0
        num_batches = 0

        for batch_idx, batch in enumerate(self.train_loader):
            # Move data to device
            images = batch['image'].to(self.device)
            labels = batch['labels'].to(self.device)

            # Forward pass with mixed precision
            self.optimizer.zero_grad()

            if self.scaler:
                with autocast(device_type=self.device):
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)

                # Backward pass with gradient scaling
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

            # Compute dice score
            with torch.no_grad():
                probs = F.softmax(outputs, dim=1)
                dice_score = self.compute_dice_score(probs, labels)

            total_loss += loss.item()
            total_dice += dice_score
            num_batches += 1

            # Log progress
            if True:  #(batch_idx + 1) % 10 == 0:
                logger.info(f"  Batch {batch_idx + 1}/{len(self.train_loader)}: "
                            f"Loss={loss.item():.4f}, Dice={dice_score:.4f}")

        avg_loss = total_loss / num_batches
        avg_dice = total_dice / num_batches

        return avg_loss, avg_dice

    def validate(self) -> Tuple[float, float, Dict]:
        """Validate model performance"""
        self.model.eval()
        total_loss = 0
        total_dice = 0
        num_batches = 0

        # Store per-class metrics
        class_dice_scores = defaultdict(list)

        with torch.no_grad():
            for batch in self.val_loader:
                images = batch['image'].to(self.device)
                labels = batch['labels'].to(self.device)

                # Forward pass
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

                # Compute probabilities
                probs = F.softmax(outputs, dim=1)

                # Compute overall dice
                dice_score = self.compute_dice_score(probs, labels)

                # Compute per-class dice
                batch_class_dice = self.compute_per_class_dice(probs, labels)
                for class_idx, dice in enumerate(batch_class_dice):
                    class_dice_scores[class_idx].append(dice)

                total_loss += loss.item()
                total_dice += dice_score
                num_batches += 1

        avg_loss = total_loss / num_batches
        avg_dice = total_dice / num_batches

        # Compute average per-class dice
        avg_class_dice = {}
        for class_idx, scores in class_dice_scores.items():
            avg_class_dice[class_idx] = np.mean(scores)

        return avg_loss, avg_dice, avg_class_dice

    def compute_dice_score(self, pred_probs: torch.Tensor,
                           target: torch.Tensor,
                           smooth: float = 1e-6) -> float:
        """Compute Dice similarity coefficient"""
        # Convert to hard predictions
        pred_classes = torch.argmax(pred_probs, dim=1)

        # Convert target to class indices if one-hot
        if target.dim() == 4:
            target_classes = torch.argmax(target, dim=1)
        else:
            target_classes = target

        # Compute dice per class and average
        num_classes = pred_probs.shape[1]
        dice_scores = []

        for class_idx in range(num_classes):
            pred_mask = (pred_classes == class_idx).float()
            target_mask = (target_classes == class_idx).float()

            intersection = (pred_mask * target_mask).sum()
            union = pred_mask.sum() + target_mask.sum()

            if union > 0:
                dice = (2. * intersection + smooth) / (union + smooth)
                dice_scores.append(dice.item())

        return np.mean(dice_scores) if dice_scores else 0.0

    def compute_per_class_dice(self, pred_probs: torch.Tensor,
                               target: torch.Tensor,
                               smooth: float = 1e-6) -> List[float]:
        """Compute Dice score for each class"""
        pred_classes = torch.argmax(pred_probs, dim=1)

        if target.dim() == 4:
            target_classes = torch.argmax(target, dim=1)
        else:
            target_classes = target

        num_classes = pred_probs.shape[1]
        dice_scores = []

        for class_idx in range(num_classes):
            pred_mask = (pred_classes == class_idx).float()
            target_mask = (target_classes == class_idx).float()

            intersection = (pred_mask * target_mask).sum()
            union = pred_mask.sum() + target_mask.sum()

            if union > 0:
                dice = (2. * intersection + smooth) / (union + smooth)
                dice_scores.append(dice.item())
            else:
                dice_scores.append(0.0)

        return dice_scores

    def train(self, num_epochs: int = 50, patience: int = 10):
        """Complete training loop with early stopping"""
        logger.info(f"Starting training for {num_epochs} epochs")

        early_stop_counter = 0
        current_lr = self.optimizer.param_groups[0]['lr']
        self.learning_rates.append(current_lr)

        for epoch in range(num_epochs):
            start_time = time.time()

            # Train
            train_loss, train_dice = self.train_epoch()
            self.train_losses.append(train_loss)
            self.train_dice_scores.append(train_dice)

            # Validate
            val_loss, val_dice, class_dice = self.validate()
            self.val_losses.append(val_loss)
            self.val_dice_scores.append(val_dice)

            # Update learning rate
            self.scheduler.step(val_loss)

            # Check if learning rate changed
            new_lr = self.optimizer.param_groups[0]['lr']
            self.learning_rates.append(new_lr)
            if new_lr != current_lr:
                logger.info(f"  Learning rate changed from {current_lr:.2e} to {new_lr:.2e}")
                current_lr = new_lr

            # Save best model
            if val_dice > self.best_val_dice:
                self.best_val_dice = val_dice
                self.best_val_loss = val_loss
                self.best_model_state = copy.deepcopy(self.model.state_dict())
                early_stop_counter = 0

                # Save model checkpoint
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_loss': val_loss,
                    'val_dice': val_dice,
                    'train_loss': train_loss,
                    'train_dice': train_dice,
                }, 'best_model.pth')
                logger.info(f"  Saved best model with Dice: {val_dice:.4f}")
            else:
                early_stop_counter += 1

            # Log epoch results
            epoch_time = time.time() - start_time
            logger.info(f"\nEpoch {epoch + 1}/{num_epochs} ({epoch_time:.1f}s):")
            logger.info(f"  Train - Loss: {train_loss:.4f}, Dice: {train_dice:.4f}")
            logger.info(f"  Val   - Loss: {val_loss:.4f}, Dice: {val_dice:.4f}")

            # Log per-class dice
            logger.info("  Per-class Dice:")
            for class_idx, dice in class_dice.items():
                logger.info(f"    Class {class_idx}: {dice:.4f}")

            # Early stopping
            if early_stop_counter >= patience:
                logger.info(f"\nEarly stopping triggered after {epoch + 1} epochs")
                break

        # Load best model
        if self.best_model_state:
            self.model.load_state_dict(self.best_model_state)

        logger.info(f"\nTraining completed. Best validation Dice: {self.best_val_dice:.4f}")

    def load_pretrained_model(self):
        self.model.load_state_dict(torch.load('best_model.pth', weights_only=True))

    def evaluate(self, loader: DataLoader = None) -> Dict:
        """Evaluate model on test set"""
        if loader is None:
            loader = self.test_loader

        self.model.eval()
        results = {
            'overall_dice': [],
            'per_class_dice': defaultdict(list),
            'predictions': [],
            'ground_truths': [],
            'patient_ids': []
        }

        with torch.no_grad():
            for batch in loader:
                images = batch['image'].to(self.device)
                labels = batch['labels'].to(self.device)

                # Forward pass
                outputs = self.model(images)
                probs = F.softmax(outputs, dim=1)

                # Compute dice
                dice_score = self.compute_dice_score(probs, labels)
                results['overall_dice'].append(dice_score)

                # Compute per-class dice
                batch_class_dice = self.compute_per_class_dice(probs, labels)
                for class_idx, dice in enumerate(batch_class_dice):
                    results['per_class_dice'][class_idx].append(dice)

                # Store predictions and ground truth for visualization
                results['predictions'].append(probs.cpu())
                results['ground_truths'].append(labels.cpu())
                results['patient_ids'].extend(batch['patient_id'])

        # Compute average metrics
        avg_results = {
            'mean_dice': np.mean(results['overall_dice']),
            'std_dice': np.std(results['overall_dice']),
            'per_class_mean_dice': {},
            'per_class_std_dice': {}
        }

        for class_idx, scores in results['per_class_dice'].items():
            avg_results['per_class_mean_dice'][class_idx] = np.mean(scores)
            avg_results['per_class_std_dice'][class_idx] = np.std(scores)

        return avg_results, results

    def plot_training_history(self):
        """Plot training and validation metrics"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # Plot losses
        axes[0, 0].plot(self.train_losses, label='Train Loss', linewidth=2)
        axes[0, 0].plot(self.val_losses, label='Val Loss', linewidth=2)
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Training and Validation Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # Plot dice scores
        axes[0, 1].plot(self.train_dice_scores, label='Train Dice', linewidth=2)
        axes[0, 1].plot(self.val_dice_scores, label='Val Dice', linewidth=2)
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Dice Score')
        axes[0, 1].set_title('Training and Validation Dice Score')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # Plot learning rate
        axes[1, 0].plot(self.learning_rates, marker='o', linewidth=2)
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Learning Rate')
        axes[1, 0].set_title('Learning Rate Schedule')
        axes[1, 0].set_yscale('log')
        axes[1, 0].grid(True, alpha=0.3)

        # Plot train vs val dice comparison
        axes[1, 1].scatter(self.train_dice_scores, self.val_dice_scores, alpha=0.6)
        axes[1, 1].plot([0, 1], [0, 1], 'r--', alpha=0.5)
        axes[1, 1].set_xlabel('Train Dice')
        axes[1, 1].set_ylabel('Val Dice')
        axes[1, 1].set_title('Train vs Validation Dice Correlation')
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f'training_history_{time.strftime('%d-%m_%H:%M')}.png', dpi=150, bbox_inches='tight')
        plt.show()

    def visualize_predictions(self, results: Dict, num_samples: int = 3):
        """Visualize model predictions"""
        label_names = ['NO-PG', 'AFS', 'CZ', 'PZ', 'SV_L', 'SV_R', 'TZ']  #['NO-PG', 'AFS', 'CZ', 'PG', 'PZ', 'SV_L', 'SV_R', 'TZ']
        colors = ['black','red', 'cyan', 'green', 'yellow', 'purple', 'orange'] #'blue'
        cmap = ListedColormap(colors)
        norm = BoundaryNorm(range(len(colors) + 1), cmap.N)

        for sample_idx in range(min(num_samples, len(results['predictions']))):
            # Get sample
            pred_probs = results['predictions'][sample_idx][0]  # First in batch
            gt = results['ground_truths'][sample_idx][0]
            patient_id = results['patient_ids'][sample_idx]

            # Convert to class indices
            pred_classes = torch.argmax(pred_probs, dim=0)

            # Convert ground truth (handle both one-hot and class indices)
            if gt.dim() == 3:  # One-hot
                gt_classes = torch.argmax(gt, dim=0)
            else:  # Class indices
                gt_classes = gt

            # Create visualization
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            fig.suptitle(f'Patient {patient_id} - Model Predictions', fontsize=16, fontweight='bold')

            # Plot ground truth
            im0 = axes[0].imshow(gt_classes, cmap=cmap, norm=norm)
            axes[0].set_title('Ground Truth')
            axes[0].axis('off')

            # Predictions
            im1 = axes[1].imshow(pred_classes, cmap=cmap, norm=norm)
            axes[1].set_title('Model Prediction')
            axes[1].axis('off')

            # Overlay
            axes[2].imshow(gt_classes, cmap='gray', alpha=0.5)
            axes[2].imshow(pred_classes, cmap=cmap, norm=norm, alpha=0.5)
            axes[2].set_title('Overlay (GT in gray, Pred in color)')
            axes[2].axis('off')

            # Add colorbar
            plt.colorbar(im1, ax=axes[0:2], orientation='horizontal',
                         fraction=0.05, pad=0.04, ticks=range(7))

            # Add legend
            legend_patches = [
                mpatches.Patch(color=colors[i], label=label_names[i])
                for i in range(len(label_names))
            ]

            axes[2].legend(
                handles=legend_patches,
                loc='center left',
                bbox_to_anchor=(1.05, 0.5),
                fontsize=9
            )

            #plt.tight_layout()
            plt.savefig(f'prediction_sample_{sample_idx}_{time.strftime('%d-%m_%H:%M')}.png', dpi=150, bbox_inches='tight')
            plt.show()


def run_training(
    batch_size=16, 
    num_epochs=50, 
    load_pretrained=False, 
    data_subset=False, 
    num_workers=16,
    data_root="/root/data/AI4AR_cont/Data",
    labels_root="/root/data/AI4AR_cont/Anatomical_Labels"
):

    if not os.path.exists(data_root) or not os.path.exists(labels_root):
        print(f"Error: Paths not found!\nData: {data_root}\nLabels: {labels_root}")
        return

    # Get patient IDs
    patient_ids = [d for d in os.listdir(data_root)
                   if os.path.isdir(os.path.join(data_root, d))]

    # Sort numerically
    patient_ids = sorted(patient_ids, key=lambda x: int(x))
    logger.info(f"Found {len(patient_ids)} patients")

    if data_subset:
        patient_ids, _ = train_test_split(patient_ids, test_size=0.75, random_state=42)
        print(f"--- DEBUG MODE: Using subset of {len(patient_ids)} patients ---")

    # Split data
    train_ids, test_ids = train_test_split(patient_ids, test_size=0.2, random_state=42)
    train_ids, val_ids = train_test_split(train_ids, test_size=0.2, random_state=42)

    logger.info(f"Train: {len(train_ids)} patients")
    logger.info(f"Val: {len(val_ids)} patients")
    logger.info(f"Test: {len(test_ids)} patients")

    # Create multi-annotator datasets
    train_dataset = MultiAnnotatorProstateDataset(
        data_root=data_root,
        labels_root=labels_root,
        patient_ids=train_ids,
        modalities=['t2w'], # 'cor', 'sag',
        target_size=(256, 256),
        normalize=True,
        num_annotators=3,
        annotator_variance=0.1
    )

    val_dataset = MultiAnnotatorProstateDataset(
        data_root=data_root,
        labels_root=labels_root,
        patient_ids=val_ids,
        modalities=['t2w'], # 'cor', 'sag',
        target_size=(256, 256),
        normalize=True,
        num_annotators=3,
        annotator_variance=0.1
    )

    test_dataset = MultiAnnotatorProstateDataset(
        data_root=data_root,
        labels_root=labels_root,
        patient_ids=test_ids,
        modalities=['t2w'], # 'cor', 'sag',
        target_size=(256, 256),
        normalize=True,
        num_annotators=3,
        annotator_variance=0.1
    )

    # Apply multi-annotator fusion
    fusion_method = MultiAnnotatorFusion(num_classes=7)

    train_fused_dataset = MultiAnnotatorUNetDataset(
        base_dataset=train_dataset,
        fusion_method=fusion_method,
        use_probabilistic_labels=True
    )

    val_fused_dataset = MultiAnnotatorUNetDataset(
        base_dataset=val_dataset,
        fusion_method=fusion_method,
        use_probabilistic_labels=False  # Use hard labels for validation
    )

    test_fused_dataset = MultiAnnotatorUNetDataset(
        base_dataset=test_dataset,
        fusion_method=fusion_method,
        use_probabilistic_labels=False  # Use hard labels for testing
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_fused_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,  # Set to 0 if you have issues with multiprocessing
        pin_memory=True if torch.cuda.is_available() else False
    )

    val_loader = DataLoader(
        val_fused_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )

    test_loader = DataLoader(
        test_fused_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )

    # Test data loading
    logger.info("\nTesting data loading...")
    test_batch = next(iter(train_loader))
    logger.info(f"Batch image shape: {test_batch['image'].shape}")
    logger.info(f"Batch labels shape: {test_batch['labels'].shape}")

    # Create model
    model = UNet(n_channels=1, n_classes=7, bilinear=True)

    # Create trainer
    trainer = ProstateSegmentationTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        learning_rate=1e-4,
        weight_decay=1e-5
    )

    if not load_pretrained:
        # Train model
        trainer.train(num_epochs=num_epochs, patience=10)  # Reduced epochs for testing
        # Plot training history
        trainer.plot_training_history()
    else:
        trainer.load_pretrained_model()


    # Evaluate on test set
    logger.info("\n" + "=" * 60)
    logger.info("EVALUATION ON TEST SET")
    logger.info("=" * 60)

    avg_results, detailed_results = trainer.evaluate()

    logger.info(f"\nOverall Dice Score: {avg_results['mean_dice']:.4f} ± {avg_results['std_dice']:.4f}")
    logger.info("\nPer-class Dice Scores:")

    label_names = ['NO', 'AFS', 'CZ', 'PG', 'PZ', 'SV_L', 'SV_R', 'TZ']
    for class_idx in range(7):
        mean_dice = avg_results['per_class_mean_dice'][class_idx]
        std_dice = avg_results['per_class_std_dice'][class_idx]
        logger.info(f"  {label_names[class_idx]:6s}: {mean_dice:.4f} ± {std_dice:.4f}")

    # Visualize predictions
    logger.info("\nGenerating prediction visualizations...")
    trainer.visualize_predictions(detailed_results, num_samples=min(3, len(test_ids)))

    # Save final model
    torch.save({
        'model_state_dict': trainer.model.state_dict(),
        'model_config': {
            'n_channels': 1,
            'n_classes': 7,
            'bilinear': True
        },
        'metrics': avg_results
    }, 'final_model.pth')

    logger.info("\nTraining completed successfully!")
    logger.info(f"Best validation Dice: {trainer.best_val_dice:.4f}")
    logger.info(f"Test Dice: {avg_results['mean_dice']:.4f}")

if __name__ == '__main__':
    #TO RUN ON SMALL SUBSET OF DATA, SET DATA_SUBSET=True
    DATA_PATH = r"/root/data/AI4AR_cont/Data"
    LABELS_PATH = r"/root/data/AI4AR_cont/Anatomical_Labels"

    run_training(
        batch_size=64, 
        num_epochs=20, 
        data_subset=False, 
        num_workers=16,
        data_root=DATA_PATH,
        labels_root=LABELS_PATH
    )