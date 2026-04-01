import logging
import math
import random
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from skimage import color
from skimage import io
from skimage.transform import rotate, resize
from torch.utils.data import Dataset
from torch.utils.data.dataloader import default_collate
from torchvision import transforms

from ace_network_dinov2 import Regressor

_logger = logging.getLogger(__name__)


class CamLocDatasetDINOv2(Dataset):
    """
    Camera localization dataset adapted for DINOv2.

    Key differences from original:
    - Uses RGB images (3 channels) instead of grayscale
    - Ensures image dimensions are multiples of 14 (DINOv2 patch size)
    - Adjusts camera intrinsics accordingly
    """

    def __init__(self,
                 root_dir,
                 mode=0,
                 sparse=False,
                 augment=False,
                 aug_rotation=15,
                 aug_scale_min=2 / 3,
                 aug_scale_max=3 / 2,
                 aug_black_white=0.1,
                 aug_color=0.3,
                 image_height=518,  # Changed default to 518 (37*14) for DINOv2
                 image_width=None,  # If set, all images resized to (image_height, image_width) for batching
                 use_half=True,
                 num_clusters=None,
                 cluster_idx=None,
                 ):
        """Constructor.

        Parameters:
            root_dir: Folder of the data (training or test).
            mode:
                0 = RGB only, load no initialization targets. Default for the ACE paper.
                1 = RGB + ground truth scene coordinates
                2 = RGB-D, load camera coordinates
            sparse: for mode = 1, load sparse initialization targets
            augment: Use random data augmentation
            aug_rotation: Max 2D image rotation angle (degrees)
            aug_scale_min: Lower limit of image scale factor
            aug_scale_max: Upper limit of image scale factor
            aug_black_white: Max relative scale factor for brightness/contrast
            aug_color: Max relative scale factor for saturation/hue
            image_height: RGB images rescaled to this height (must be multiple of 14)
            use_half: Enable half-precision floats
            num_clusters: Split frames into clusters for ensemble training
            cluster_idx: Which cluster to use for training
        """

        self.use_half = use_half
        self.init = (mode == 1)
        self.sparse = sparse
        self.eye = (mode == 2)

        # Ensure image_height is multiple of 14
        self.patch_size = 14
        self.image_height = self._round_to_patch_size(image_height)
        if self.image_height != image_height:
            _logger.warning(f"Image height adjusted from {image_height} to {self.image_height} "
                          f"(must be multiple of {self.patch_size})")
        self.image_width = self._round_to_patch_size(image_width) if image_width is not None else None

        self.augment = augment
        self.aug_rotation = aug_rotation
        self.aug_scale_min = aug_scale_min
        self.aug_scale_max = aug_scale_max
        self.aug_black_white = aug_black_white
        self.aug_color = aug_color

        self.num_clusters = num_clusters
        self.cluster_idx = cluster_idx

        if self.num_clusters is not None:
            if self.num_clusters < 1:
                raise ValueError("num_clusters must be at least 1")
            if self.cluster_idx is None:
                raise ValueError("cluster_idx needs to be specified when num_clusters is set")
            if self.cluster_idx < 0 or self.cluster_idx >= self.num_clusters:
                raise ValueError(f"cluster_idx needs to be between 0 and {self.num_clusters - 1}")

        if self.eye and self.augment and (self.aug_rotation > 0 or self.aug_scale_min != 1 or self.aug_scale_max != 1):
            _logger.warning("WARNING: Check your augmentation settings. Camera coordinates will not be augmented.")

        # Setup data paths
        root_dir = Path(root_dir)
        rgb_dir = root_dir / 'rgb'
        pose_dir = root_dir / 'poses'
        calibration_dir = root_dir / 'calibration'

        if self.eye:
            coord_dir = root_dir / 'eye'
        elif self.sparse:
            coord_dir = root_dir / 'init'
        else:
            coord_dir = root_dir / 'depth'

        # Find all files
        self.rgb_files = sorted(rgb_dir.iterdir())
        self.pose_files = sorted(pose_dir.iterdir())
        self.calibration_files = sorted(calibration_dir.iterdir())

        if self.init or self.eye:
            self.coord_files = sorted(coord_dir.iterdir())
        else:
            self.coord_files = None

        # Validation
        if len(self.rgb_files) != len(self.pose_files):
            raise RuntimeError('RGB file count does not match pose file count!')
        if len(self.rgb_files) != len(self.calibration_files):
            raise RuntimeError('RGB file count does not match calibration file count!')
        if self.coord_files and len(self.rgb_files) != len(self.coord_files):
            raise RuntimeError('RGB file count does not match coordinate file count!')

        # Create prediction grid if needed
        if self.init and not self.sparse:
            self.prediction_grid = self._create_prediction_grid()
        else:
            self.prediction_grid = None

        # Image transformations for DINOv2 (RGB, ImageNet normalization)
        if self.augment:
            self.image_transform = transforms.Compose([
                transforms.ColorJitter(
                    brightness=self.aug_black_white,
                    contrast=self.aug_black_white,
                    saturation=self.aug_color,
                    hue=self.aug_color * 0.5
                ),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            self.image_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

        # Valid file indices
        self.valid_file_indices = np.arange(len(self.rgb_files))

        # Clustering if enabled
        if self.num_clusters is not None:
            _logger.info(f"Clustering the {len(self.rgb_files)} into {num_clusters} clusters.")
            _, _, cluster_labels = self._cluster(num_clusters)
            self.valid_file_indices = np.flatnonzero(cluster_labels == cluster_idx)
            _logger.info(f"After clustering, chosen cluster: {cluster_idx}, Using {len(self.valid_file_indices)} images.")

        # Calculate mean camera center
        self.mean_cam_center = self._compute_mean_camera_center()

    def _round_to_patch_size(self, size):
        """Round size to nearest multiple of patch_size."""
        return int(round(size / self.patch_size) * self.patch_size)

    @staticmethod
    def _create_prediction_grid():
        """Create grid of 2D pixel positions for generating scene coordinates from depth."""
        prediction_grid = np.zeros((2,
                                    math.ceil(5000 / Regressor.OUTPUT_SUBSAMPLE),
                                    math.ceil(5000 / Regressor.OUTPUT_SUBSAMPLE)))

        for x in range(0, prediction_grid.shape[2]):
            for y in range(0, prediction_grid.shape[1]):
                prediction_grid[0, y, x] = x * Regressor.OUTPUT_SUBSAMPLE
                prediction_grid[1, y, x] = y * Regressor.OUTPUT_SUBSAMPLE

        return prediction_grid

    @staticmethod
    def _resize_image(image, image_height):
        """Resize image to target height, maintaining aspect ratio."""
        image = TF.to_pil_image(image)
        image = TF.resize(image, image_height)
        return image

    @staticmethod
    def _rotate_image(image, angle, order, mode='constant'):
        """Rotate image by given angle."""
        image = image.permute(1, 2, 0).numpy()
        image = rotate(image, angle, order=order, mode=mode)
        image = torch.from_numpy(image).permute(2, 0, 1).float()
        return image

    def _load_image(self, idx):
        """Load RGB image."""
        image = io.imread(self.rgb_files[idx])
        if len(image.shape) < 3:
            image = color.gray2rgb(image)
        return image

    def _load_pose(self, idx):
        """Load camera pose as 4x4 matrix."""
        pose = np.loadtxt(self.pose_files[idx])
        pose = torch.from_numpy(pose).float()
        return pose

    def _compute_mean_camera_center(self):
        """Compute mean camera center across all valid frames."""
        mean_cam_center = torch.zeros((3,))
        for idx in self.valid_file_indices:
            pose = self._load_pose(idx)
            mean_cam_center += pose[0:3, 3]
        mean_cam_center /= len(self)
        return mean_cam_center

    def _cluster(self, num_clusters):
        """Cluster dataset using hierarchical kMeans."""
        num_images = len(self.pose_files)
        _logger.info(f'Clustering {num_images} frames into {num_clusters} clusters.')

        cam_centers = np.zeros((num_images, 3), dtype=np.float32)
        for i in range(num_images):
            pose = self._load_pose(i)
            cam_centers[i] = pose[:3, 3]

        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.1)
        flags = cv2.KMEANS_PP_CENTERS
        label_counter = 0
        clusters = [(cam_centers, label_counter, np.zeros(3))]
        labels = np.zeros(num_images)

        while len(clusters) < num_clusters:
            cur_cluster = clusters.pop(0)
            label_counter += 1
            cur_error, cur_labels, cur_centroids = cv2.kmeans(cur_cluster[0], 2, None, criteria, 10, flags)

            cur_mask = (cur_labels == 0)[:, 0]
            cur_cam_centers0 = cur_cluster[0][cur_mask, :]
            clusters.append((cur_cam_centers0, cur_cluster[1], cur_centroids[0]))

            cur_mask = (cur_labels == 1)[:, 0]
            cur_cam_centers1 = cur_cluster[0][cur_mask, :]
            clusters.append((cur_cam_centers1, label_counter, cur_centroids[1]))

            cluster_labels = labels[labels == cur_cluster[1]]
            cluster_labels[cur_mask] = label_counter
            labels[labels == cur_cluster[1]] = cluster_labels
            clusters = sorted(clusters, key=lambda cluster: cluster[0].shape[0], reverse=True)

        remapped_labels = np.zeros(num_images)
        remapped_clusters = []
        for cluster_idx_new, cluster in enumerate(clusters):
            cluster_idx_old = cluster[1]
            remapped_labels[labels == cluster_idx_old] = cluster_idx_new
            remapped_clusters.append((cluster[0], cluster_idx_new, cluster[2]))

        labels = remapped_labels
        clusters = remapped_clusters
        cluster_centers = np.zeros((num_clusters, 3))
        cluster_sizes = np.zeros((num_clusters, 1))

        for cluster in clusters:
            cam_num = cluster[0].shape[0]
            cam_data = np.zeros((cam_num, 3))
            cam_count = 0
            for i, cam_center in enumerate(cam_centers):
                if labels[i] == cluster[1]:
                    cam_data[cam_count] = cam_center
                    cam_count += 1
            cluster_centers[cluster[1]] = cam_data.mean(0)
            cam_dists = np.broadcast_to(cluster_centers[cluster[1]][np.newaxis, :], (cam_num, 3))
            cam_dists = cam_data - cam_dists
            cam_dists = np.linalg.norm(cam_dists, axis=1)
            cam_dists = cam_dists ** 2
            cluster_sizes[cluster[1]] = cam_dists.mean()
            _logger.info(f"Cluster {cluster[1]}: images={cluster[0].shape[0]}, dist={cluster_sizes[cluster[1]]}")

        _logger.info('Clustering done.')
        return cluster_centers, cluster_sizes, labels

    def _get_single_item(self, idx, image_height):
        """Load and preprocess a single data item."""
        # Apply index indirection
        idx = self.valid_file_indices[idx]

        # Load RGB image
        image = self._load_image(idx)

        # Load camera intrinsics
        k = np.loadtxt(self.calibration_files[idx])
        if k.size == 1:
            focal_length = float(k)
            centre_point = None
        elif k.shape == (3, 3):
            k = k.tolist()
            focal_length = [k[0][0], k[1][1]]
            centre_point = [k[0][2], k[1][2]]
        else:
            raise Exception("Calibration file must contain either a 3x3 camera intrinsics matrix or a single float")

        # Ensure image_height is multiple of 14
        image_height = self._round_to_patch_size(image_height)

        # Scale image and adjust focal length
        f_scale_factor = image_height / image.shape[0]
        if centre_point:
            centre_point = [c * f_scale_factor for c in centre_point]
            focal_length = [f * f_scale_factor for f in focal_length]
        else:
            focal_length *= f_scale_factor

        # Resize image
        image = self._resize_image(image, image_height)

        # Ensure width is also multiple of 14 (or use fixed image_width for batching)
        current_width = image.size[0]
        if self.image_width is not None:
            target_width = self.image_width
        else:
            target_width = self._round_to_patch_size(current_width)
        if target_width != current_width:
            image = TF.resize(image, (image_height, target_width))
            # Adjust focal length and center point for width change
            w_scale = target_width / current_width
            if centre_point:
                centre_point[0] *= w_scale
                if isinstance(focal_length, list):
                    focal_length[0] *= w_scale
            else:
                focal_length *= w_scale

        # Create mask
        image_mask = torch.ones((1, image.size[1], image.size[0]))

        # Apply transforms (RGB for DINOv2)
        image = self.image_transform(image)

        # Load pose
        pose = self._load_pose(idx)

        # Load ground truth scene coordinates if needed
        if self.init:
            if self.sparse:
                try:
                    coords = torch.load(self.coord_files[idx], weights_only=True)
                except TypeError:
                    coords = torch.load(self.coord_files[idx])
            else:
                depth = io.imread(self.coord_files[idx])
                depth = depth.astype(np.float64)
                depth /= 1000
        elif self.eye:
            try:
                coords = torch.load(self.coord_files[idx], weights_only=True)
            except TypeError:
                coords = torch.load(self.coord_files[idx])
        else:
            coords = 0

        # Apply data augmentation
        if self.augment:
            angle = random.uniform(-self.aug_rotation, self.aug_rotation)
            image = self._rotate_image(image, angle, 1, 'reflect')
            image_mask = self._rotate_image(image_mask, angle, order=1, mode='constant')

            if self.init:
                if self.sparse:
                    coords_w = math.ceil(image.size(2) / Regressor.OUTPUT_SUBSAMPLE)
                    coords_h = math.ceil(image.size(1) / Regressor.OUTPUT_SUBSAMPLE)
                    coords = F.interpolate(coords.unsqueeze(0), size=(coords_h, coords_w))[0]
                    coords = self._rotate_image(coords, angle, 0)
                else:
                    depth = resize(depth, image.shape[1:], order=0)
                    depth = rotate(depth, angle, order=0, mode='constant')

            # Rotate pose
            angle_rad = angle * math.pi / 180.
            pose_rot = torch.eye(4)
            pose_rot[0, 0] = math.cos(angle_rad)
            pose_rot[0, 1] = -math.sin(angle_rad)
            pose_rot[1, 0] = math.sin(angle_rad)
            pose_rot[1, 1] = math.cos(angle_rad)
            pose = torch.matmul(pose, pose_rot)

        # Generate initialization targets from depth if needed
        if self.init and not self.sparse:
            offsetX = int(Regressor.OUTPUT_SUBSAMPLE / 2)
            offsetY = int(Regressor.OUTPUT_SUBSAMPLE / 2)
            coords = torch.zeros((3,
                                math.ceil(image.shape[1] / Regressor.OUTPUT_SUBSAMPLE),
                                math.ceil(image.shape[2] / Regressor.OUTPUT_SUBSAMPLE)))
            depth = depth[offsetY::Regressor.OUTPUT_SUBSAMPLE, offsetX::Regressor.OUTPUT_SUBSAMPLE]
            xy = self.prediction_grid[:, :depth.shape[0], :depth.shape[1]].copy()
            xy[0] += offsetX
            xy[1] += offsetY
            xy[0] -= image.shape[2] / 2
            xy[1] -= image.shape[1] / 2
            xy /= focal_length
            xy[0] *= depth
            xy[1] *= depth
            eye = np.ndarray((4, depth.shape[0], depth.shape[1]))
            eye[0:2] = xy
            eye[2] = depth
            eye[3] = 1
            sc = np.matmul(pose.numpy(), eye.reshape(4, -1))
            sc = sc.reshape(4, depth.shape[0], depth.shape[1])
            sc[:, depth == 0] = 0
            sc[:, depth > 1000] = 0
            sc = torch.from_numpy(sc[0:3])
            coords[:, :sc.shape[1], :sc.shape[2]] = sc

        # Convert to half precision if needed
        if self.use_half and torch.cuda.is_available():
            image = image.half()

        # Binarize mask
        image_mask = image_mask > 0

        # Invert pose
        pose_inv = pose.inverse()

        # Create intrinsics matrix
        intrinsics = torch.eye(3)
        if centre_point:
            intrinsics[0, 2] = centre_point[0]
            intrinsics[1, 2] = centre_point[1]
            intrinsics[0, 0] = focal_length[0]
            intrinsics[1, 1] = focal_length[1]
        else:
            intrinsics[0, 2] = image.shape[2] / 2
            intrinsics[1, 2] = image.shape[1] / 2
            intrinsics[0, 0] = focal_length
            intrinsics[1, 1] = focal_length

        intrinsics_inv = intrinsics.inverse()

        # Match dataset_origin: return filename as 8th element for test script compatibility.
        return image, image_mask, pose, pose_inv, intrinsics, intrinsics_inv, coords, str(self.rgb_files[idx])

    def __len__(self):
        return len(self.valid_file_indices)

    def __getitem__(self, idx):
        """Support single index (int) or batch of indices (list/tuple) for DataLoader with BatchSampler."""
        if isinstance(idx, (list, tuple)):
            items = [self._get_single_item(int(i), self.image_height) for i in idx]
            coords_list = [x[6] for x in items]
            coords_out = torch.stack(coords_list) if isinstance(coords_list[0], torch.Tensor) else coords_list[0]
            return (
                torch.stack([x[0] for x in items]),
                torch.stack([x[1] for x in items]),
                torch.stack([x[2] for x in items]),
                torch.stack([x[3] for x in items]),
                torch.stack([x[4] for x in items]),
                torch.stack([x[5] for x in items]),
                coords_out,
                [x[7] for x in items],
            )
        return self._get_single_item(idx, self.image_height)
