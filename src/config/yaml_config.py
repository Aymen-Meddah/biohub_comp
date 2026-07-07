from pathlib import Path


class YamlConfig:

    def __init__(self, config_path):
        self.config_path = Path(config_path)
        if not self.config_path.exists():
            raise FileNotFoundError(
                f"Config file not found: {self.config_path}"
            )

        try:
            import yaml
        except ImportError as exc:
            raise ImportError(
                "PyYAML is required to load YAML config files. Install the "
                "project requirements or run `pip install PyYAML`."
            ) from exc

        with self.config_path.open("r") as file:
            self.data = yaml.safe_load(file) or {}

    def __getitem__(self, key):
        return self.data[key]

    def get(self, key, default=None):
        return self.data.get(key, default)

    def keys(self):
        return self.data.keys()

    def values(self):
        return self.data.values()

    def items(self):
        return self.data.items()

    def __contains__(self, key):
        return key in self.data

    def __len__(self):
        return len(self.data)

    def __repr__(self):
        return f"YamlConfig({self.config_path.name})"
