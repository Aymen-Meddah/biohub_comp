from pathlib import Path
import kagglehub


def get_dataset_path() -> Path:
    path = kagglehub.competition_download(
        "biohub-cell-tracking-during-development"
    )
    return Path(path)