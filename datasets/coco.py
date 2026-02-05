import torch
import cv2
import numpy as np
import random
import os
from torch.utils.data import Dataset
from pycocotools.coco import COCO

class COCODataset(Dataset):
    def __init__(self, root, ann_file, img_size=640, mosaic=True):
        self.root = root
        self.coco = COCO(ann_file)
        self.ids = sorted(list(self.coco.imgs.keys()))
        self.img_size = img_size
        self.mosaic = mosaic

        self.cats = {i: v['name'] for i, v in self.coco.cats.items()}
        self.cat_id_to_num = {cat_id: i for i, cat_id in enumerate(self.cats.keys())}

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        if self.mosaic and random.random() < 0.5:
            img, boxes, labels = self.load_mosaic(index)
        else:
            img, boxes, labels = self.load_image_and_boxes(index)
            img, boxes = self.letterbox_resize(img, boxes, self.img_size)

        # BGR->RGB, HWC->CHW, Normalize 0-1
        img = img.transpose((2, 0, 1))[::-1] 
        img = np.ascontiguousarray(img)
        img_tensor = torch.from_numpy(img).float() / 255.0
        
        target = {
            'boxes': torch.as_tensor(boxes, dtype=torch.float32),
            'labels': torch.as_tensor(labels, dtype=torch.int64)
        }
        
        # Handle empty targets
        if target['boxes'].shape[0] == 0:
            target['boxes'] = torch.zeros((0, 4), dtype=torch.float32)
            target['labels'] = torch.zeros((0,), dtype=torch.int64)
            
        return img_tensor, target

    def load_image_and_boxes(self, index):
        img_id = self.ids[index]
        file_name = self.coco.loadImgs(img_id)[0]['file_name']
        img = cv2.imread(os.path.join(self.root, file_name))
        
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)
        
        boxes, labels = [], []
        for ann in anns:
            if ann.get('iscrowd', 0): continue
            x, y, w, h = ann['bbox']
            boxes.append([x, y, x + w, y + h])
            labels.append(self.cat_id_to_num[ann['category_id']])
            
        return img, np.array(boxes), np.array(labels)

    def letterbox_resize(self, img, boxes, target_size):
        h, w = img.shape[:2]
        scale = min(target_size / h, target_size / w)
        nw, nh = int(w * scale), int(h * scale)
        
        img_resized = cv2.resize(img, (nw, nh))
        img_padded = np.full((target_size, target_size, 3), 114, dtype=np.uint8)
        
        dx, dy = (target_size - nw) // 2, (target_size - nh) // 2
        img_padded[dy:dy+nh, dx:dx+nw] = img_resized
        
        if len(boxes) > 0:
            boxes *= scale
            boxes[:, [0, 2]] += dx
            boxes[:, [1, 3]] += dy
            
        return img_padded, boxes

    def load_mosaic(self, index):
        s = self.img_size
        xc, yc = [int(random.uniform(s * 0.5, s * 1.5)) for _ in range(2)]
        indices = [index] + [random.randint(0, len(self.ids) - 1) for _ in range(3)]
        
        img4 = np.full((s * 2, s * 2, 3), 114, dtype=np.uint8)
        labels4, boxes4 = [], []
        
        for i, idx in enumerate(indices):
            img, boxes, labels = self.load_image_and_boxes(idx)
            h, w = img.shape[:2]
            
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
            padw, padh = x1a - x1b, y1a - y1b
            
            if len(boxes) > 0:
                boxes[:, [0, 2]] += padw
                boxes[:, [1, 3]] += padh
                labels4.append(labels)
                boxes4.append(boxes)
                
        if len(labels4) > 0:
            labels4 = np.concatenate(labels4, 0)
            boxes4 = np.concatenate(boxes4, 0)
            np.clip(boxes4, 0, 2 * s, out=boxes4)
        
        # Resize from 2x canvas back to target size
        img4_resized = cv2.resize(img4, (s, s))
        if len(boxes4) > 0: boxes4 *= 0.5
            
        return img4_resized, boxes4, labels4