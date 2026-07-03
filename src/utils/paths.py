from pathlib import Path

class ProjectPaths:
    def __init__(self, project_root):
        self.project_root = Path(project_root)

        self.configs = self.project_root / "configs"
        self.data = self.project_root / "data"
        self.logs = self.project_root / "logs"
        self.docs = self.project_root / "docs"
        self.outputs = self.project_root / "outputs"
        self.checkpoints = self.project_root / "checkpoints"
        self.experiments = self.project_root / "experiments"
        self.notebooks = self.project_root / "notebooks"
        self.visualization = self.project_root / "visualization"

        self._create_directories()

    def _create_directories(self):
        directories = [
            self.configs,
            self.data,
            self.logs,
            self.docs,
            self.outputs,
            self.checkpoints,
            self.experiments,
            self.notebooks,
            self.visualization
        ]
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
        def experiments(self , experiment_name):

            experiment_path = self.experiments / experiment_name
            experiment_path.mkdir(parents=True, exist_ok=True)
            
            (experiment_path / "checkpoints").mkdir(parents=True, exist_ok=True)
            (experiment_path / "logs").mkdir(parents=True, exist_ok=True)
            (experiment_path / "logs").mkdir(parents=True, exist_ok=True)
            (experiment_path / "predictions").mkdir(parents=True, exist_ok=True)

            return experiment_path
        def __repr__(self):
            return f"ProjectPaths({self.project_root})"
        