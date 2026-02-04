import yaml

class Config(dict):
    def __init__(self, config_dict):
        super().__init__(config_dict)
        for k, v in config_dict.items():
            if isinstance(v, dict):
                self[k] = Config(v)
            else:
                self[k] = v

    def __getattr__(self, name):
        if name in self:
            return self[name]
        else:
            raise AttributeError(f"Config object has no attribute '{name}'. Available keys: {list(self.keys())}")

    def __setattr__(self, name, value):
        self[name] = value

def load_config(path):
    try:
        with open(path, 'r') as f:
            cfg_dict = yaml.safe_load(f)
            
        if cfg_dict is None:
            raise ValueError(f"File {path} is empty!")
            
        return Config(cfg_dict)
        
    except FileNotFoundError:
        raise FileNotFoundError(f"Could not find config file at: {path}")
    except yaml.YAMLError as exc:
        raise ValueError(f"Error parsing YAML file: {exc}")