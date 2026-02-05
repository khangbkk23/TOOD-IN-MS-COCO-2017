# Config
from utils.config import load_config
# Data
from datasets.coco import COCODataset
from datasets.collate import collate_fn
from torch.utils.data import DataLoader
# Model
from models.backbones.resnet import ResNet
from models.necks.fpn import FPN
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

backbone = ResNet(
    pretrained=cfg.model.pretrained, 
    frozen_bn=cfg.model.frozen_bn
).to(cfg.device)

neck = FPN(
    in_channels=[512, 1024, 2048], 
    out_channels=cfg.model.neck.out_channels
).to(cfg.device)

print("Model initialized successfully!")
try:
    images, targets = next(iter(train_loader))
    images = images.to(cfg.device)
    
    features = backbone(images)
    print(f"Backbone outputs: {[f.shape for f in features]}")
    fpn_feats = neck(features)
    print(f"FPN outputs: {[f.shape for f in fpn_feats]}")
    
    print("Architecture shapes are correct!")
    
except Exception as e:
    print(f"Error during dry run: {e}")
    
print(f"Running on {cfg.device} with Image Size {cfg.data.img_size}")