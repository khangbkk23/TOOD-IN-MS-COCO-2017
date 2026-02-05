#Utils
import torch
# Config
from utils.config import load_config
# Data
from datasets.coco import COCODataset
from datasets.collate import collate_fn
from torch.utils.data import DataLoader
# Model
from models.backbones.resnet import ResNet
from models.necks.fpn import FPN
from models.detectors.tood import TOOD
cfg = load_config("./config/config.yaml")

train_dataset = COCODataset(
    root=f"{cfg.data.root_dir}/{cfg.data.train_imgs}",
    ann_file=f"{cfg.data.root_dir}/{cfg.data.train_ann}",
    img_size=cfg.data.img_size,
    mosaic=cfg.data.use_mosaic
)

train_loader = DataLoader(
    train_dataset, 
    batch_size=cfg.data.batch_size,
    num_workers=cfg.data.num_workers,
    collate_fn=collate_fn,
    shuffle=True
)

# Building model
model = TOOD(cfg).to(cfg.device)

images, targets = next(iter(train_loader))
images = images.to(cfg.device)

with torch.no_grad():
    cls_outs, reg_outs = model(images)

print(f"Model TOOD initialized!")
print(f"Num levels: {len(cls_outs)}")
print(f"Class output shape [P3]: {cls_outs[0].shape}")