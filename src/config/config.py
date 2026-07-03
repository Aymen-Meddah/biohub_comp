from pathlib import Path 
import yaml

class Config :
    def __init__(self , config_path):
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        with open(config_path, 'r') as file:
            self.data = yaml.safe_load(file)
    def __getitem__(self, key):
        return self.data[key]
    def get(self, key, default=None):
        return self.data.get(key, default)
    def __key__(self):
        return self.data.keys()
    def __values__(self):
        return self.data.values()
    def items(self):
        return self.data.items()
    def __contains__(self, key):
        return key in self.data
    def __len__(self):
        return len(self.data)
    def __repr__(self):
        return f"Config({self.config_path.name})"
    