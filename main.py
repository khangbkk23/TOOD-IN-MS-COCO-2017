from utils.config import load_config
from datasets.coco import COCODataset
from datasets.collate import collate_fn
from torch.utils.data import DataLoader

cfg = load_config("./config/config.yaml")

# 2. Setup Dataset using Config
train_dataset = COCODataset(
    root=f"{cfg.data.root_dir}/{cfg.data.train_imgs}",
    ann_file=f"{cfg.data.root_dir}/{cfg.data.train_ann}",
    img_size=cfg.data.img_size,  # 640 loaded from yaml
    mosaic=cfg.data.use_mosaic
)

train_loader = DataLoader(
    train_dataset, 
    batch_size=cfg.data.batch_size, # 4 loaded from yaml
    num_workers=cfg.data.num_workers,
    collate_fn=collate_fn,
    shuffle=True
)

print(f"Running on {cfg.device} with Image Size {cfg.data.img_size}")