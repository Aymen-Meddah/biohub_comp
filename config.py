from pathlib import Path
import os

from src.utils.download_data import get_dataset_path


def _worker_count(default=2, maximum=4):
    cpu_count = os.cpu_count() or default
    return max(0, min(maximum, cpu_count))


class ConfigMeta(type):

    _data_root = None

    @property
    def DATA_ROOT(cls):
        if cls._data_root is None:
            cls._data_root = Path(get_dataset_path())
        return cls._data_root

    @property
    def TRAIN_DIR(cls):
        return cls.DATA_ROOT / "train"

    @property
    def TEST_DIR(cls):
        return cls.DATA_ROOT / "test"

    @property
    def SAMPLE_SUBMISSION(cls):
        return cls.DATA_ROOT / "sample_submission.csv"

    @property
    def DEVICE(cls):
        import torch

        return torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )


class Config(metaclass=ConfigMeta):

    PROJECT_ROOT = Path(__file__).resolve().parent

    OUTPUT_DIR = PROJECT_ROOT / "outputs"

    CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"

    LOG_DIR = OUTPUT_DIR / "logs"

    SUBMISSION_DIR = OUTPUT_DIR / "submission"

    CACHE_DIR = OUTPUT_DIR / "cache"

    BATCH_SIZE = 4

    VALIDATION_BATCH_SIZE = 4

    NUM_WORKERS = 0

    VALIDATION_NUM_WORKERS = 0

    PIN_MEMORY = True

    PREFETCH_FACTOR = 2

    PERSISTENT_WORKERS = True

    EPOCHS = 100

    LEARNING_RATE = 5e-4

    WEIGHT_DECAY = 1e-4

    SCHEDULER_T_MAX = EPOCHS

    IMAGE_SIZE = (32, 128, 128)

    HEATMAP_SIGMA = 1.5

    EMBEDDING_DIM = 16

    MAX_CELL_RADIUS = 20

    CONFIDENCE_THRESHOLD = 0.5

    LR_SCHEDULER_MODE = "max"

    LR_SCHEDULER_FACTOR = 0.5

    LR_SCHEDULER_PATIENCE = 5

    LR_SCHEDULER_MIN_LR = 1e-6

    MATCHING_DISTANCE = 10.0

    MAX_TRACK_MISSING = 5

    RANDOM_SEED = 42

    AMP = True

    SAVE_BEST_ONLY = True

    EARLY_STOPPING_PATIENCE = 10

    EARLY_STOPPING_MIN_DELTA = 1e-4

    CHECKPOINT_NAME = "best_model.pth"

    LAST_CHECKPOINT_NAME = "last_model.pth"

    CHECKPOINT_PATH = CHECKPOINT_DIR / CHECKPOINT_NAME

    LAST_CHECKPOINT_PATH = CHECKPOINT_DIR / LAST_CHECKPOINT_NAME

    SUBMISSION_NAME = "submission.csv"

    @classmethod
    def validate_dataset_exists(cls, require_test=False):
        missing = []
        if not cls.DATA_ROOT.exists():
            missing.append(cls.DATA_ROOT)
        if not cls.TRAIN_DIR.exists():
            missing.append(cls.TRAIN_DIR)
        if require_test and not cls.TEST_DIR.exists():
            missing.append(cls.TEST_DIR)

        if missing:
            paths = ", ".join(str(path) for path in missing)
            raise FileNotFoundError(
                "BioHub dataset is incomplete or missing. Expected Kaggle "
                f"dataset structure with train/ and test/. Missing: {paths}"
            )

        return True

    @classmethod
    def create_directories(cls):

        cls.OUTPUT_DIR.mkdir(

            parents=True,

            exist_ok=True

        )

        cls.CHECKPOINT_DIR.mkdir(

            parents=True,

            exist_ok=True

        )

        cls.LOG_DIR.mkdir(

            parents=True,

            exist_ok=True

        )

        cls.SUBMISSION_DIR.mkdir(

            parents=True,

            exist_ok=True

        )

        cls.CACHE_DIR.mkdir(

            parents=True,

            exist_ok=True

        )
