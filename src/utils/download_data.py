from pathlib import Path
import os

KAGGLE_DATASET_PATHS = [
    Path("/kaggle/input/biohub-cell-tracking-during-development"),
    Path("/kaggle/input/competitions/biohub-cell-tracking-during-development"),
]


def get_dataset_path() -> Path:
    for path in KAGGLE_DATASET_PATHS:
        if path.exists():
            return path

    env_path = os.environ.get("BIOHUB_DATA_ROOT")
    if env_path:
        return Path(env_path)

    try:
        import kagglehub

        return Path(
            kagglehub.competition_download(
                "biohub-cell-tracking-during-development"
            )
        )
    except ImportError as exc:
        raise FileNotFoundError(
            "BioHub dataset not found. Either:\n"
            "- Add the competition as a Kaggle Notebook Input,\n"
            "- Set the BIOHUB_DATA_ROOT environment variable,\n"
            "- Or install kagglehub to download it automatically."
        ) from exc