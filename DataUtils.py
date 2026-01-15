import os
from typing import List, Tuple, Dict

import SimpleITK as sitk
import numpy as np
import torch
from torch.utils.data import Dataset


class MultiAnnotatorProstateDataset(Dataset):
    """
    Dataset for loading prostate MRI data with simulated multi-annotator support.
    In real scenario, this would load annotations from different doctors.
    """

    def __init__(self,
                 data_root: str,
                 labels_root: str,
                 prostatex_root: str,
                 patient_ids: List[str],
                 modalities: List[str] = ['t2w'],  # ['cor', 'sag', 't2w'],
                 transform=None,
                 target_size: Tuple[int, int] = (256, 256),
                 normalize: bool = True,
                 num_annotators: int = 2,
                 annotator_variance: float = 0.1):
        """
        Args:
            data_root: Root directory containing patient data folders
            labels_root: Root directory containing anatomical labels
            patient_ids: List of patient IDs to include
            modalities: List of modalities to load
            transform: Optional transform to apply
            target_size: Target size for resizing (height, width)
            normalize: Whether to normalize images to [0, 1]
            num_annotators: Number of simulated annotators
            annotator_variance: Amount of noise to add to simulate different annotators
        """
        # self.region_map = {
        #     "afs": [4],
        #     "cz": [2],
        #     "pg": [0],
        #     "pz": [1],
        #     "tz": [3,5],
        # }
        # self.final_zones = {
        #     "PZ": ["pz"],
        #     "TZ": ["tz", "afs", "cz"]
        # }
        self.data_root = data_root
        self.labels_root = labels_root
        self.prostatex_root = prostatex_root
        self.patient_ids = patient_ids
        self.modalities = modalities
        self.transform = transform
        self.target_size = target_size
        self.normalize = normalize
        self.num_annotators = num_annotators
        self.annotator_variance = annotator_variance

        # Store file paths
        self.samples = []

        for pid in patient_ids:
            # Construct file paths
            patient_path = os.path.join(data_root, pid)
            label_path = os.path.join(labels_root, pid)

            if not os.path.exists(patient_path) or not os.path.exists(label_path):
                print(f"Missing data for patient {pid}")
                continue

            # Get all available modalities
            available_files = os.listdir(patient_path)

            # Check if required modalities exist
            modality_files = {}
            for modality in modalities:
                # Try with and without leading zeros
                pattern_variants = [f"{pid.lstrip('0')}_{modality}.mha", f"{int(pid)}_{modality}.mha",
                    f"{pid}_{modality}.mha"]

                for pattern in pattern_variants:
                    matching_files = [f for f in available_files if f == pattern]
                    if matching_files:
                        modality_files[modality] = os.path.join(patient_path, matching_files[0])
                        break

                if modality not in modality_files:
                    print(f"Missing {modality} for patient {pid}")

            if len(modality_files) == len(modalities):
                # Get anatomical labels
                label_files = {}
                available_labels = os.listdir(label_path)
                anatomical_regions = ['afs', 'cz', 'pg', 'pz', 'tz']

                for region in anatomical_regions:
                    # Try with and without leading zeros
                    pattern_variants = [f"{pid.lstrip('0')}_{region}_t2w.nii.gz", f"{int(pid)}_{region}_t2w.nii.gz",
                        f"{pid}_{region}_t2w.nii.gz"]

                    for pattern in pattern_variants:
                        matching_files = [f for f in available_labels if pattern in f]
                        if matching_files:
                            label_files[region] = os.path.join(label_path, matching_files[0])
                            break

                self.samples.append({'patient_id': pid, 'modality_files': modality_files, 'label_files': label_files})

        print(
            f"Loaded {len(self.samples)} patients with {len(modalities)} modalities each")  # print(f"Simulating {num_annotators} annotators with variance {annotator_variance}")

    def __len__(self):
        return len(self.samples)

    def load_image(self, filepath: str) -> np.ndarray:
        """Load medical image using SimpleITK"""
        try:
            if filepath.endswith('.mha') or filepath.endswith('.nii.gz'):
                image = sitk.ReadImage(filepath)
                array = sitk.GetArrayFromImage(image)

                # Handle 3D volumes - take middle slice
                if len(array.shape) == 3:
                    # For MRI, typically shape is (slices, height, width)
                    middle_slice = array.shape[0] // 2
                    array = array[middle_slice]

                # Resize if needed
                if self.target_size and array.shape != self.target_size:
                    from scipy.ndimage import zoom
                    zoom_factors = (self.target_size[0] / array.shape[0], self.target_size[1] / array.shape[1])
                    array = zoom(array, zoom_factors, order=1)  # Linear interpolation

                # Normalize to [0, 1]
                if self.normalize:
                    array_min = array.min()
                    array_max = array.max()
                    if array_max > array_min:
                        array = (array - array_min) / (array_max - array_min)
                    else:
                        array = np.zeros_like(array)

                return array
            else:
                raise ValueError(f"Unsupported file format: {filepath}")
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
            return np.zeros(self.target_size)

    def load_labels_nii(self, label_files: Dict) -> Dict[str, np.ndarray]:
        """Load all anatomical labels"""
        labels = {}
        for region, filepath in label_files.items():
            try:
                label = sitk.ReadImage(filepath)
                array = sitk.GetArrayFromImage(label)

                # Handle 3D volumes - take middle slice
                if len(array.shape) == 3:
                    middle_slice = array.shape[0] // 2
                    array = array[middle_slice]

                # Binarize (some labels might have different values)
                array = (array > 0).astype(np.float32)

                # Resize if needed
                if self.target_size and array.shape != self.target_size:
                    from scipy.ndimage import zoom
                    zoom_factors = (self.target_size[0] / array.shape[0], self.target_size[1] / array.shape[1])
                    array = zoom(array, zoom_factors, order=0)  # Nearest neighbor for masks

                labels[region] = array
            except Exception as e:
                print(f"Error loading label {region}: {e}")
                labels[region] = np.zeros(self.target_size)

        return labels

    def load_labels_nrrd(self, filepath) -> Dict[str, np.ndarray]:
        """
        Load anatomical labels from a multi-label .nrrd file.
        Extracts separate 2D masks for each region defined in self.region_map.
        """
        labels = {}

        try:
            seg_img = sitk.ReadImage(filepath)
            seg_np = sitk.GetArrayFromImage(seg_img)  # shape: [z, y, x]

            # Middle slice
            if seg_np.ndim != 3:
                raise ValueError("Expected a 3D segmentation volume.")
            z_mid = seg_np.shape[0] // 2
            slice_2d = seg_np[z_mid]
            labels = np.unique(seg_np)
            masks = {}
            for label in labels:
                # if label == 0:
                #     continue  # skip background
                # Resize if needed
                mask = (slice_2d == label).astype(np.float32)
                if self.target_size and mask.shape != self.target_size:
                    from scipy.ndimage import zoom
                    zoom_factors = (self.target_size[0] / mask.shape[0], self.target_size[1] / mask.shape[1])
                masks[label] = zoom(mask, zoom_factors, order=0)
            return masks

            # # Extract each region mask  # for region_name, region_labels in self.region_map.items():  #     combined_mask = np.zeros(self.target_size, dtype=np.float32)  #     for region_label in region_labels:  #         mask = (slice_2d == region_label).astype(np.float32)  #  #         # Resize if needed  #         if self.target_size and mask.shape != self.target_size:  #             from scipy.ndimage import zoom  #             zoom_factors = (  #                 self.target_size[0] / mask.shape[0],  #                 self.target_size[1] / mask.shape[1]  #             )  #             mask = zoom(mask, zoom_factors, order=0)  #             combined_mask+=mask  #  #     labels[region_name] = combined_mask

        except Exception as e:
            print(f"Error loading .nrrd segmentation: {e}")
            # Return empty masks for each region
            for label in labels:
                labels[label] = np.zeros(self.target_size, dtype=np.float32)

        return labels

    def add_annotator_noise(self, label_tensor: torch.Tensor, annotator_id: int) -> torch.Tensor:
        """
        Add simulated noise to annotations to represent different annotators.
        In real scenario, this would be actual annotations from different doctors.
        """
        if annotator_id == 0:
            # Annotator 0: Original (most reliable)
            return label_tensor.clone()
        else:
            # Add increasing noise for other annotators
            noisy_labels = label_tensor.clone()
            noise_level = self.annotator_variance * annotator_id

            # Randomly flip some pixels
            flip_mask = torch.rand_like(noisy_labels) < noise_level
            # For simplicity, we just zero out some pixels
            # In real scenario, you might have more complex noise patterns
            noisy_labels[flip_mask] = 0

            return noisy_labels

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # Load all modalities
        modality_images = []
        for modality in self.modalities:
            if modality in sample['modality_files']:
                img = self.load_image(sample['modality_files'][modality])
                modality_images.append(img)
            else:
                # Fill with zeros if modality missing
                modality_images.append(np.zeros(self.target_size))

        # Stack modalities along channel dimension
        image = np.stack(modality_images, axis=0).astype(np.float32)
        # Apply transformations
        if self.transform:
            image = self.transform(image)
        # Convert to PyTorch tensors
        image_tensor = torch.from_numpy(image)
        # Load labels
        labels_dict = self.load_labels_nii(sample['label_files'])
        base_labels_tensor = self.transform_labels_nii(labels_dict)

        pid4 = f"{int(sample['patient_id']):04d}"
        singles_path = f"ProstateZones/Singles/Seg-{pid4}.nrrd"
        dup_r1_path = f"ProstateZones/Duplicates/R1/Seg-{pid4}_R1.nrrd"
        # dup_r2_path = f"ProstateZones/Duplicates/R2/Seg-{pid4}_R2.nrrd"

        if os.path.exists(singles_path):
            labels_dictx = self.load_labels_nrrd(singles_path)
        elif os.path.exists(dup_r1_path):
            labels_dictx = self.load_labels_nrrd(dup_r1_path)
        else:
            raise Exception(f"Prostatex segmentation for patient {pid4} not found")

        # labels_dictx = self.load_labels_nrrd(dup_r1_path)
        prostatex_tensor = self.transform_labels_nrrd(labels_dictx)
        # labels_dictx_2 = self.load_labels_nrrd(dup_r2_path)
        # prostatex_tensor_2 = self.transform_labels_nrrd(labels_dictx_2)
        #
        # # Stack annotator labels
        all_labels = torch.stack([base_labels_tensor, prostatex_tensor], dim=0)  # [num_annotators, num_classes, H, W]

        return {'image': image_tensor, 'labels': all_labels,  # Multiple annotator labels
            'base_label': base_labels_tensor,  # Clean/original label
            'patient_id': sample['patient_id'], 'label_names': ['NO-PG', 'PZ', 'TZ'],
            'num_annotators': self.num_annotators}

    def transform_labels_nrrd(self, labels_dict):
        # Background: pg == 0
        background = labels_dict.get(0, np.zeros(self.target_size)).astype(np.float32)

        # PZ class (single source)
        pz = labels_dict.get(1, np.zeros(self.target_size)).astype(np.float32)

        # TZ class = union of tz + afs + cz
        tz = (labels_dict.get(3, np.zeros(self.target_size)) + labels_dict.get(4, np.zeros(
            self.target_size)) + labels_dict.get(2, np.zeros(self.target_size)) + labels_dict.get(5, np.zeros(
            self.target_size)))
        tz = (tz > 0).astype(np.float32)  # ensure binary mask

        # Stack into 3‑class output
        base_labels = np.stack([background,  # class 0
            pz,  # class 1
            tz  # class 2
        ], axis=0).astype(np.float32)

        # Optional transforms
        if self.transform:
            base_labels = self.transform(base_labels)

        return torch.from_numpy(base_labels)

    def transform_labels_nii(self, labels_dict):
        # Background: pg == 0
        background = (labels_dict.get("pg", np.zeros(self.target_size)) == 0).astype(np.float32)

        # PZ class (single source)
        pz = labels_dict.get("pz", np.zeros(self.target_size)).astype(np.float32)

        # TZ class = union of tz + afs + cz
        tz = (labels_dict.get("tz", np.zeros(self.target_size)) + labels_dict.get("afs", np.zeros(
            self.target_size)) + labels_dict.get("cz", np.zeros(self.target_size)))
        tz = (tz > 0).astype(np.float32)  # ensure binary mask

        # Stack into 3‑class output
        base_labels = np.stack([background,  # class 0
            pz,  # class 1
            tz  # class 2
        ], axis=0).astype(np.float32)

        # Optional transforms
        if self.transform:
            base_labels = self.transform(base_labels)

        return torch.from_numpy(base_labels)
