"""
dataset.py — VOC and COCO Dataset Loaders
==========================================

Lightweight wrappers around torchvision datasets for YOLO training.
"""

import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
import torchvision.datasets as datasets
import xml.etree.ElementTree as ET
import random
import os
from PIL import Image

# Original YOLO VOC Classes
VOC_CLASSES = (
    "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", 
    "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", 
    "pottedplant", "sheep", "sofa", "train", "tvmonitor"
)

class VOCDataset(Dataset):
    def __init__(self, root, year="2007", image_set="train", input_size=224, transform=False, subset_size=None):
        self.root = root
        self.year = year
        self.image_set = image_set
        self.input_size = input_size
        self.transform = transform
        
        os.makedirs(root, exist_ok=True)
        # Check if dataset already exists to prevent re-extracting
        needs_download = not os.path.exists(os.path.join(root, "VOCdevkit"))
        self.dataset = datasets.VOCDetection(
            root, year=year, image_set=image_set, download=needs_download
        )
        
        if subset_size is not None:
            self.indices = list(range(min(subset_size, len(self.dataset))))
        else:
            self.indices = list(range(len(self.dataset)))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        image, target = self.dataset[self.indices[idx]]
        
        width, height = image.size
        
        # Parse XML annotations
        boxes = []
        for obj in target['annotation']['object']:
            class_name = obj['name'].lower().strip()
            if class_name not in VOC_CLASSES:
                continue
                
            class_id = VOC_CLASSES.index(class_name)
            
            bndbox = obj['bndbox']
            xmin = float(bndbox['xmin'])
            ymin = float(bndbox['ymin'])
            xmax = float(bndbox['xmax'])
            ymax = float(bndbox['ymax'])
            
            # Convert to YOLO format (normalized center_x, center_y, w, h)
            x_center = ((xmin + xmax) / 2) / width
            y_center = ((ymin + ymax) / 2) / height
            box_w = (xmax - xmin) / width
            box_h = (ymax - ymin) / height
            
            boxes.append([class_id, x_center, y_center, box_w, box_h])
            
        boxes = torch.tensor(boxes, dtype=torch.float32)

        # Augmentation
        if self.transform:
            # Random horizontal flip
            if random.random() > 0.5:
                image = TF.hflip(image)
                if boxes.shape[0] > 0:
                    boxes[:, 1] = 1.0 - boxes[:, 1] # Flip center_x
            
            # Color jitter (exposure/saturation from cfg)
            if random.random() > 0.5:
                image = TF.adjust_saturation(image, random.uniform(0.5, 1.5))
            if random.random() > 0.5:
                image = TF.adjust_brightness(image, random.uniform(0.5, 1.5))

        # Resize image
        image = TF.resize(image, (self.input_size, self.input_size))
        image = TF.to_tensor(image)
        
        return image, boxes

def collate_fn(batch):
    """
    Custom collate function because target boxes arrays have different lengths.
    """
    images = []
    targets = []
    
    for img, target in batch:
        images.append(img)
        targets.append(target)
        
    return torch.stack(images, 0), targets
