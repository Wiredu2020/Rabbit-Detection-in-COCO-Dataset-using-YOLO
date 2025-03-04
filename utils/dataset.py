import torch
from pathlib import Path
from PIL import Image
from torchvision.transforms import v2
from torchvision.tv_tensors import BoundingBoxes
from torch.utils.data import Dataset
from pycocotools.coco import COCO
import utils.config as config

class RabbitDataset(Dataset):
    def __init__(self, ann_file: Path, img_size=224, apply_transform=False):
        self.coco = COCO(ann_file)
        self.img_dir = ann_file.parent
        self.image_ids = list(self.coco.imgs.keys())
        self.img_size = img_size
        self.apply_transform = apply_transform
        
        # Define transformations
        self.to_tensor = v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True)
        ])
        self.resize = v2.Resize((img_size, img_size), interpolation=3)
        self.transform = v2.Compose([
            v2.RandomAffine(degrees=0, translate=(0.3, 0.3)),
            v2.RandomHorizontalFlip(p=0.5)
        ])
        self.normalize = v2.Normalize(mean=[0.5] * 3, std=[0.25] * 3)

        # Preload image paths and annotations
        self.image_paths = [self.img_dir / self.coco.loadImgs(img_id)[0]['file_name'] for img_id in self.image_ids]
        self.annotations = {img_id: self.coco.loadAnns(self.coco.getAnnIds(imgIds=img_id)) for img_id in self.image_ids}

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        img_path = self.image_paths[idx]
        
        # Load image
        image = Image.open(str(img_path))

        # Load annotations
        annotations = self.annotations[img_id][0]
        bbox = BoundingBoxes(annotations['bbox'], format="XYWH", canvas_size=image.size[::-1])
        
        # Apply transformations
        image, bbox = self.resize(image, bbox)
        
        # Apply further transformations if apply_transformation is True
        if self.apply_transform:
            image, bbox = self.transform(image, bbox)
            
        image = self.to_tensor(image)
        image = self.normalize(image)
        
        # Initialize target tensor
        target = torch.zeros((config.GRID_SIZE, config.GRID_SIZE, 5 * config.N_BBOX + config.N_CLASSES))
        
        # Convert bounding box to YOLO format
        bbox = bbox[0]
        x_center = (bbox[0] + bbox[2] / 2) / self.img_size
        y_center = (bbox[1] + bbox[3] / 2) / self.img_size
        width = bbox[2] / self.img_size
        height = bbox[3] / self.img_size
        
        # Determine grid cell location
        grid_x = min(int(x_center * config.GRID_SIZE), config.GRID_SIZE - 1)
        grid_y = min(int(y_center * config.GRID_SIZE), config.GRID_SIZE - 1)
        
        # Compute relative position within the grid cell
        x_cell = x_center * config.GRID_SIZE - grid_x
        y_cell = y_center * config.GRID_SIZE - grid_y
        
        # Set target tensor values
        target[grid_y, grid_x, :4] = torch.tensor([x_cell, y_cell, width, height])
        target[grid_y, grid_x, 4] = 1  # Confidence score
        target[grid_y, grid_x, -1] = 1  # Class label (rabbit)
        
        return image, target
