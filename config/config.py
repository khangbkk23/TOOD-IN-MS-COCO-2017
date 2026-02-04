import yaml
from argparse import Namespace

def load_config(path="./config/config.yaml"):
    with open(path, 'r') as f:
        cfg_dict = yaml.safe_load(f)
        
    def dict_to_namespace(d):
        x = Namespace()
        for k, v in d.items():
            if isinstance(v, dict):
                setattr(x, k, dict_to_namespace(v))
            else:
                setattr(x, k, v)
        return x

    return dict_to_namespace(cfg_dict)

if __name__ == "__main__":
    cfg = load_config()
    print(f"Target image size: {cfg.data.img_size}")
    print(f"Backbone: {cfg.model.backbone}")