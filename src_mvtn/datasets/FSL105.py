import os
import math
import torch
import cv2
import numpy as np
from pathlib import Path
from torch.utils.data.dataset import Dataset
from imgaug import augmenters as iaa
import json

class FSL105(Dataset):
    """FSL105 Dataset class: Reads video files and extracts n_frames clips (RGB only)."""
    def __init__(self, configer, path, split="train", data_type='rgb', transforms=None, n_frames=40, optical_flow=False):
        super().__init__()

        self.dataset_path = Path(path)
        self.split = split
        self.data_type = data_type
        self.transforms = transforms
        self.n_frames = n_frames

        # If no transform is provided, define a default augmentation (for training only)
        if self.transforms is None and split == "train":
            self.transforms = iaa.Sequential([
                iaa.Multiply((0.8, 1.2)),                           # brightness variation
                iaa.LinearContrast((0.75, 1.25)),                   # contrast variation
                # iaa.Crop(percent=(0, 0.1)),                       # random cropping; commented out to avoid size mismatch
                iaa.CropToFixedSize(224, 224),                      # fixed output size
                iaa.Affine(
                    scale=(0.9, 1.1),                               # scaling
                    rotate=(-10, 10),                               # rotation ±10°
                    mode='reflect'
                ),
                iaa.MotionBlur(k=(3, 5)),                           # motion blur
                iaa.AdditiveGaussianNoise(scale=(0, 0.03 * 255)),   # noise
                iaa.Fliplr(0.5)                                     # horizontal flip
            ])

        # Load JSON split file
        split_file = self.dataset_path / "splits" / f"{split}.json"
        with open(split_file, "r") as f:
            self.data = np.array(json.load(f))

        print(f"Loaded FSL105 {split.upper()} dataset ({len(self.data)} samples)")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        entry = self.data[idx]
        video_path = entry['video_path']
        label = entry['label']

        # Read video and extract frames
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_indices = self._get_center_frames(total_frames, self.n_frames)

        clip = []
        for i in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if not ret:
                frame = clip[-1] if len(clip) > 0 else np.zeros((224, 224, 3), dtype=np.uint8)
            else:
                frame = cv2.resize(frame, (224, 224))   # <- ensure fixed shape
            clip.append(frame)

        cap.release()

        clip = np.array(clip).transpose(1, 2, 3, 0)  # HWC -> HWC×T

        # Apply augmentations; issue here: double resizing if uncommented
        # if self.transforms is not None:
        #     aug_det = self.transforms.to_deterministic()
        #     clip = np.array([
        #         cv2.resize(aug_det.augment_image(clip[..., i]), (224, 224))  # <-- resize again here
        #         for i in range(clip.shape[-1])
        #     ]).transpose(1, 2, 3, 0)

        # normalize RGB
        clip = clip.astype(np.float32) / 255.0

        # apply augmentation if any
        if self.transforms is not None:
            aug_det = self.transforms.to_deterministic()
            clip = np.array([aug_det.augment_image(clip[..., i]) for i in range(clip.shape[-1])]).transpose(1, 2, 3, 0)

        # convert to torch tensor (C × H × W × T)
        clip = torch.from_numpy(clip.reshape(clip.shape[0], clip.shape[1], -1).transpose(2, 0, 1))
        label = torch.LongTensor(np.asarray([label]))

        return clip.float(), label

    def _get_center_frames(self, total_frames, n_frames):
        """Return n_frames indices centered in the video."""
        if total_frames <= n_frames:
            # repeat last frame if video too short
            return list(range(total_frames)) + [total_frames - 1] * (n_frames - total_frames)
        center = total_frames // 2
        half = n_frames // 2
        start = max(center - half, 0)
        end = start + n_frames
        return list(range(start, end))
