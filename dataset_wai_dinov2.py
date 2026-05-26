import logging
import math
import random
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision.transforms.functional as TF
from skimage import color
from skimage.transform import rotate
from torch.utils.data import Dataset
from torchvision import transforms

_logger = logging.getLogger(__name__)


class CamLocDatasetWAIDINOv2(Dataset):
    """
    WAI-backed camera localization dataset for DINOv2 training.

    Contract intentionally matches CamLocDatasetDINOv2(mode=0):
    returns (image, image_mask, pose, pose_inv, intrinsics, intrinsics_inv, coords, rgb_path_str).
    """

    def __init__(
        self,
        root_dir,
        mode=0,
        sparse=False,
        augment=False,
        aug_rotation=15,
        aug_scale_min=2 / 3,
        aug_scale_max=3 / 2,
        aug_black_white=0.1,
        aug_color=0.3,
        image_height=518,
        image_width=None,
        use_half=True,
        num_clusters=None,
        cluster_idx=None,
        wai_repo_root=None,
        wai_image_modality="image",
    ):
        if mode != 0:
            raise ValueError("CamLocDatasetWAIDINOv2 currently supports mode=0 only.")
        if sparse:
            raise ValueError("CamLocDatasetWAIDINOv2 does not support sparse/init coordinates.")

        self.use_half = use_half
        self.init = False
        self.sparse = False
        self.eye = False

        self.patch_size = 14
        self.image_height = self._round_to_patch_size(image_height)
        if self.image_height != image_height:
            _logger.warning(
                "Image height adjusted from %s to %s (must be multiple of %s)",
                image_height,
                self.image_height,
                self.patch_size,
            )
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

        self.scene_root = Path(root_dir)
        self.scene_meta_path = self.scene_root / "scene_meta.json"
        if not self.scene_meta_path.exists():
            raise FileNotFoundError(f"WAI scene_meta.json not found: {self.scene_meta_path}")

        self.wai_repo_root = Path(wai_repo_root) if wai_repo_root is not None else (Path(__file__).resolve().parents[1] / "map-anything")
        self.wai_image_modality = wai_image_modality
        self._load_data_impl, self._load_frame_impl = self._import_wai_io(self.wai_repo_root)
        self.scene_meta = self._load_data_impl(str(self.scene_meta_path), "scene_meta")

        frames = self.scene_meta.get("frames", [])
        frame_names = [f.get("frame_name") for f in frames if "frame_name" in f]
        self.frame_names = sorted(frame_names)
        if len(self.frame_names) == 0:
            raise RuntimeError(f"No frames found in {self.scene_meta_path}")

        self.frame_name_to_meta = {f["frame_name"]: f for f in frames if "frame_name" in f}

        if self.augment:
            self.image_transform = transforms.Compose([
                transforms.ColorJitter(
                    brightness=self.aug_black_white,
                    contrast=self.aug_black_white,
                    saturation=self.aug_color,
                    hue=self.aug_color * 0.5,
                ),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        else:
            self.image_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

        self.valid_file_indices = np.arange(len(self.frame_names))
        if self.num_clusters is not None:
            _logger.info("Clustering the %s into %s clusters.", len(self.frame_names), num_clusters)
            _, _, cluster_labels = self._cluster(num_clusters)
            self.valid_file_indices = np.flatnonzero(cluster_labels == cluster_idx)
            _logger.info("After clustering, chosen cluster: %s, using %s images.", cluster_idx, len(self.valid_file_indices))

        self.mean_cam_center = self._compute_mean_camera_center()

    @staticmethod
    def _import_wai_io(wai_repo_root: Path):
        wai_repo_root = wai_repo_root.resolve()
        if not wai_repo_root.exists():
            raise FileNotFoundError(f"WAI repo root not found: {wai_repo_root}")
        if str(wai_repo_root) not in sys.path:
            sys.path.insert(0, str(wai_repo_root))
        from mapanything.utils.wai.core import load_data, load_frame
        return load_data, load_frame

    def _round_to_patch_size(self, size):
        return int(round(size / self.patch_size) * self.patch_size)

    def _resize_image_to_height(self, image, image_height):
        """Resize to an explicit target height while preserving aspect ratio."""
        image_height = self._round_to_patch_size(image_height)
        image = TF.to_pil_image(image)
        orig_width, orig_height = image.size
        target_width = self._round_to_patch_size(orig_width * image_height / max(1, orig_height))
        return TF.resize(image, (image_height, target_width))

    @staticmethod
    def _rotate_image(image, angle, order, mode="constant"):
        image = image.permute(1, 2, 0).numpy()
        image = rotate(image, angle, order=order, mode=mode)
        image = torch.from_numpy(image).permute(2, 0, 1).float()
        return image

    @staticmethod
    def _normalize_image_array(image):
        if torch.is_tensor(image):
            image = image.detach().cpu().numpy()
        image = np.asarray(image)
        if image.ndim == 3 and image.shape[0] in (3, 4) and image.shape[-1] not in (3, 4):
            image = np.transpose(image, (1, 2, 0))
        if image.ndim == 2:
            image = color.gray2rgb(image)
        if image.ndim != 3:
            raise RuntimeError(f"Unexpected WAI image shape: {image.shape}")
        if image.shape[2] == 1:
            image = color.gray2rgb(image[..., 0])
        if image.shape[2] > 3:
            image = image[..., :3]
        if image.dtype != np.uint8:
            image = np.asarray(image, dtype=np.float32)
            if image.max() <= 1.0:
                image = image * 255.0
            image = np.clip(image, 0.0, 255.0).astype(np.uint8)
        return image

    @staticmethod
    def _to_np_44(x):
        if torch.is_tensor(x):
            x = x.detach().cpu().numpy()
        x = np.asarray(x, dtype=np.float32)
        return x.reshape(4, 4)

    @staticmethod
    def _to_np_33(x):
        if torch.is_tensor(x):
            x = x.detach().cpu().numpy()
        x = np.asarray(x, dtype=np.float32)
        return x.reshape(3, 3)

    def _get_pose_from_meta(self, frame_name: str) -> torch.Tensor:
        frame = self.frame_name_to_meta[frame_name]
        pose_raw = frame.get("transform_matrix", frame.get("extrinsics"))
        if pose_raw is None:
            raise RuntimeError(f"Frame {frame_name} missing transform_matrix/extrinsics in scene_meta.")
        pose = torch.from_numpy(np.asarray(pose_raw, dtype=np.float32).reshape(4, 4)).float()
        return pose

    def _compute_mean_camera_center(self):
        mean_cam_center = torch.zeros((3,), dtype=torch.float32)
        for idx in self.valid_file_indices:
            frame_name = self.frame_names[int(idx)]
            pose = self._get_pose_from_meta(frame_name)
            mean_cam_center += pose[0:3, 3]
        mean_cam_center /= len(self)
        return mean_cam_center

    def _cluster(self, num_clusters):
        num_images = len(self.frame_names)
        _logger.info("Clustering %s frames into %s clusters.", num_images, num_clusters)

        cam_centers = np.zeros((num_images, 3), dtype=np.float32)
        for i, frame_name in enumerate(self.frame_names):
            pose = self._get_pose_from_meta(frame_name)
            cam_centers[i] = pose[:3, 3].numpy()

        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.1)
        flags = cv2.KMEANS_PP_CENTERS
        label_counter = 0
        clusters = [(cam_centers, label_counter, np.zeros(3))]
        labels = np.zeros(num_images)

        while len(clusters) < num_clusters:
            cur_cluster = clusters.pop(0)
            label_counter += 1
            _, cur_labels, cur_centroids = cv2.kmeans(cur_cluster[0], 2, None, criteria, 10, flags)

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
        return None, None, labels

    def _get_single_item(self, idx, image_height):
        idx = self.valid_file_indices[idx]
        frame_name = self.frame_names[int(idx)]
        view = self._load_frame_impl(
            self.scene_root,
            frame_name,
            modalities=[self.wai_image_modality],
            scene_meta=self.scene_meta,
            load_intrinsics=True,
            load_extrinsics=True,
            fmt="np",
        )

        image = self._normalize_image_array(view[self.wai_image_modality])
        intr_np = self._to_np_33(view["intrinsics"])
        pose = torch.from_numpy(self._to_np_44(view["extrinsics"])).float()  # c2w

        focal_length = [float(intr_np[0, 0]), float(intr_np[1, 1])]
        centre_point = [float(intr_np[0, 2]), float(intr_np[1, 2])]

        image_height = self._round_to_patch_size(image_height)
        f_scale_factor = image_height / image.shape[0]
        centre_point = [c * f_scale_factor for c in centre_point]
        focal_length = [f * f_scale_factor for f in focal_length]

        image = self._resize_image_to_height(image, image_height)

        current_width = image.size[0]
        target_width = self.image_width if self.image_width is not None else self._round_to_patch_size(current_width)
        if target_width != current_width:
            image = TF.resize(image, (image_height, target_width))
            w_scale = target_width / current_width
            centre_point[0] *= w_scale
            focal_length[0] *= w_scale

        image_mask = torch.ones((1, image.size[1], image.size[0]))
        image = self.image_transform(image)

        if self.augment:
            angle = random.uniform(-self.aug_rotation, self.aug_rotation)
            image = self._rotate_image(image, angle, 1, "reflect")
            image_mask = self._rotate_image(image_mask, angle, order=1, mode="constant")

            angle_rad = angle * math.pi / 180.0
            pose_rot = torch.eye(4)
            pose_rot[0, 0] = math.cos(angle_rad)
            pose_rot[0, 1] = -math.sin(angle_rad)
            pose_rot[1, 0] = math.sin(angle_rad)
            pose_rot[1, 1] = math.cos(angle_rad)
            pose = torch.matmul(pose, pose_rot)

        if self.use_half and torch.cuda.is_available():
            image = image.half()

        image_mask = image_mask > 0
        pose_inv = pose.inverse()

        intrinsics = torch.eye(3)
        intrinsics[0, 2] = centre_point[0]
        intrinsics[1, 2] = centre_point[1]
        intrinsics[0, 0] = focal_length[0]
        intrinsics[1, 1] = focal_length[1]
        intrinsics_inv = intrinsics.inverse()

        coords = 0
        rgb_path = str(view.get(f"{self.wai_image_modality}_path", frame_name))
        return image, image_mask, pose, pose_inv, intrinsics, intrinsics_inv, coords, rgb_path

    def __len__(self):
        return len(self.valid_file_indices)

    def __getitem__(self, idx):
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
