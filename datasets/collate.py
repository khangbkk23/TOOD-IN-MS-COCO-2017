import torch

def collate_fn(batch):
    # batch is a list of tuples: [(img1, target1), (img2, target2), ...]
    images, targets = tuple(zip(*batch))

    # Stack images: [3, 640, 640] -> [Batch, 3, 640, 640]
    images = torch.stack(images, dim=0)
    return images, targets