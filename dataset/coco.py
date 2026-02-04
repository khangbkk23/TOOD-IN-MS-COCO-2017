import torch
import cv2
import numpy as np
import random
import os
from torch.utils.data import Dataset
from pycocotools.coco import COCO

class COCODataset(Dataset):
    def __init__(self, root, ann_file, img_size=640, mosaic=True, transform=None):
        """
        Args:
            root (str): Path to image directory
            ann_file (str): Path to json annotation file
            img_size (int): Target size (e.g., 640 or 512 for 6GB VRAM)
            mosaic (bool): Enable Mosaic Augmentation (Best for SOTA training)
            transform (bool): Basic transform (Normalize, ToTensor)
        """
        self.root = root
        self.coco = COCO(ann_file)
        self.ids = list(sorted(self.coco.imgs.keys()))
        self.img_size = img_size
        self.mosaic = mosaic
        self.transform = transform
        
        # Map category IDs to continuous index (0-79)
        self.cats = {i: v['name'] for i, v in self.coco.cats.items()}
        self.cat_id_to_num = {cat_id: i for i, cat_id in enumerate(self.cats.keys())}

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        # 1. Apply Mosaic Augmentation with 50% probability if enabled
        if self.mosaic and random.random() < 0.5:
            img, boxes, labels = self.load_mosaic(index)
        else:
            # Standard load + Resize (Letterbox)
            img, boxes, labels = self.load_image_and_boxes(index)
            img, boxes = self.letterbox_resize(img, boxes, self.img_size)

        # 2. Convert to Tensor & Normalize
        # Normalize to [0, 1]
        img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = np.ascontiguousarray(img)
        
        img_tensor = torch.from_numpy(img).float() / 255.0
        
        # Create Target Dict
        target = {}
        target['boxes'] = torch.as_tensor(boxes, dtype=torch.float32)
        target['labels'] = torch.as_tensor(labels, dtype=torch.int64)
        
        # Handle empty boxes (crucial for stability)
        if target['boxes'].shape[0] == 0:
            target['boxes'] = torch.zeros((0, 4), dtype=torch.float32)
            target['labels'] = torch.zeros((0,), dtype=torch.int64)
            
        return img_tensor, target

    def load_image_and_boxes(self, index):
        """Helper to load 1 image and its annotations"""
        img_id = self.ids[index]
        file_name = self.coco.loadImgs(img_id)[0]['file_name']
        img_path = os.path.join(self.root, file_name)
        
        # Load image using OpenCV
        img = cv2.imread(img_path)
        h, w = img.shape[:2]
        
        # Load annotations
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)
        
        boxes = []
        labels = []
        for ann in anns:
            if ann.get('iscrowd', 0): continue
            x, y, wb, hb = ann['bbox']
            boxes.append([x, y, x + wb, y + hb])
            labels.append(self.cat_id_to_num[ann['category_id']])
            
        return img, np.array(boxes), np.array(labels)

    def letterbox_resize(self, img, boxes, target_size):
        """Resize image with padding to keep aspect ratio (Letterbox)"""
        h, w = img.shape[:2]
        scale = min(target_size / h, target_size / w)
        nw, nh = int(w * scale), int(h * scale)
        
        img_resized = cv2.resize(img, (nw, nh))
        
        # Create padded image (gray background)
        img_padded = np.full((target_size, target_size, 3), 114, dtype=np.uint8)
        
        # Center the image
        dx = (target_size - nw) // 2
        dy = (target_size - nh) // 2
        img_padded[dy:dy+nh, dx:dx+nw] = img_resized
        
        # Adjust boxes
        if len(boxes) > 0:
            boxes *= scale
            boxes[:, [0, 2]] += dx
            boxes[:, [1, 3]] += dy
            
        return img_padded, boxes

    def load_mosaic(self, index):
        """
        MOSAIC AUGMENTATION: Loads 4 images and stitches them into 1.
        This is the SOTA standard for object detection.
        """
        labels4, boxes4 = [], []
        s = self.img_size
        
        # 1. Center point of the mosaic
        xc, yc = [int(random.uniform(s * 0.5, s * 1.5)) for _ in range(2)]
        
        # 2. Pick 3 other random indices
        indices = [index] + [random.randint(0, len(self.ids) - 1) for _ in range(3)]
        
        # 3. Create large canvas (2x size)
        img4 = np.full((s * 2, s * 2, 3), 114, dtype=np.uint8)
        
        for i, idx in enumerate(indices):
            # Load image
            img, boxes, labels = self.load_image_and_boxes(idx)
            h, w = img.shape[:2]
            
            # Place image into the canvas
            if i == 0:  # top-left
                x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc
                x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h
            elif i == 1:  # top-right
                x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s * 2), yc
                x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
            elif i == 2:  # bottom-left
                x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(yc + h, s * 2)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(h, y2a - y1a)
            elif i == 3:  # bottom-right
                x1a, y1a, x2a, y2a = xc, yc, min(xc + w, s * 2), min(yc + h, s * 2)
                x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(h, y2a - y1a)

            img4[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]
            padw = x1a - x1b
            padh = y1a - y1b
            
            # Shift boxes
            if len(boxes) > 0:
                boxes[:, [0, 2]] += padw
                boxes[:, [1, 3]] += padh
                labels4.append(labels)
                boxes4.append(boxes)
                
        # Concat labels & boxes
        if len(labels4) > 0:
            labels4 = np.concatenate(labels4, 0)
            boxes4 = np.concatenate(boxes4, 0)
            
            # Clip boxes to fit inside the mosaic image (0 to 2*s)
            np.clip(boxes4, 0, 2 * s, out=boxes4)
        
        # 4. Resize back to target size (s)
        # We created a 2x canvas, now resize it down to input size (s)
        # This acts like a random zoom/crop
        img4_resized = cv2.resize(img4, (s, s))
        if len(boxes4) > 0:
            boxes4 *= 0.5 # Scale boxes down by 2
            
        return img4_resized, boxes4, labels4