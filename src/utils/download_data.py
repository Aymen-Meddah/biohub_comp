import os
from pathlib import Path

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
        env_path = Path(env_path)
        if env_path.exists():
            if (env_path / "train").exists() and (env_path / "test").exists():
                return env_path
            if env_path.name in {"train", "test"} and env_path.parent.exists():
                return env_path.parent
        return env_path

    kaggle_root = Path("/kaggle/input")
    if kaggle_root.exists():
        for child in sorted(kaggle_root.iterdir()):
            if not child.is_dir():
                continue
            if (child / "train").exists() and (child / "test").exists():
                return child
        for child in sorted(kaggle_root.iterdir()):
            if not child.is_dir():
                continue
            lower_name = child.name.lower()
            if "biohub" in lower_name and "train" not in lower_name:
                if (child / "train").exists() and (child / "test").exists():
                    return child

    is_kaggle_environment = any(
        os.environ.get(name) is not None
        for name in ("KAGGLE_KERNEL_RUN_TYPE", "KAGGLE_URL_BASE")
    ) or Path("/kaggle").exists()

    if not is_kaggle_environment:
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

    raise FileNotFoundError(
        "BioHub dataset not found in the Kaggle input directories. Set "
        "BIOHUB_DATA_ROOT to the directory containing train/ and test/."
    )