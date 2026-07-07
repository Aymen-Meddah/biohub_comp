from pathlib import Path


KAGGLE_COMPETITION_PATH = Path(
    "/kaggle/input/competitions/biohub-cell-tracking-during-development"
)


def get_dataset_path() -> Path:
    if KAGGLE_COMPETITION_PATH.exists():
        return KAGGLE_COMPETITION_PATH

    import os

    env_path = os.environ.get("BIOHUB_DATA_ROOT")
    if env_path:
        return Path(env_path)

    try:
        import kagglehub

        path = kagglehub.competition_download(
            "biohub-cell-tracking-during-development"
        )
        return Path(path)
    except ImportError as exc:
        raise FileNotFoundError(
            "BioHub dataset not found at the Kaggle input path. Set "
            "BIOHUB_DATA_ROOT to the directory containing train/ and test/."
        ) from exc
