from pathlib import Path
import torch

from src.utils.download_data import get_dataset_path


class Config:

    PROJECT_ROOT = Path(__file__).resolve().parent

    DATA_ROOT = Path(get_dataset_path())

    TRAIN_DIR = DATA_ROOT / "train"

    TEST_DIR = DATA_ROOT / "test"

    SAMPLE_SUBMISSION = DATA_ROOT / "sample_submission.csv"

    OUTPUT_DIR = PROJECT_ROOT / "outputs"

    CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"

    LOG_DIR = OUTPUT_DIR / "logs"

    SUBMISSION_DIR = OUTPUT_DIR / "submission"

    CACHE_DIR = OUTPUT_DIR / "cache"

    BATCH_SIZE = 2

    VALIDATION_BATCH_SIZE = 1

    NUM_WORKERS = 4

    VALIDATION_NUM_WORKERS = 2

    PIN_MEMORY = True

    PREFETCH_FACTOR = 2

    PERSISTENT_WORKERS = True

    EPOCHS = 100

    LEARNING_RATE = 1e-4

    WEIGHT_DECAY = 1e-4

    SCHEDULER_T_MAX = EPOCHS

    IMAGE_SIZE = (64, 256, 256)

    HEATMAP_SIGMA = 2.5

    EMBEDDING_DIM = 16

    MAX_CELL_RADIUS = 20

    CONFIDENCE_THRESHOLD = 0.30

    MATCHING_DISTANCE = 30.0

    MAX_TRACK_MISSING = 5

    RANDOM_SEED = 42

    DEVICE = torch.device(

        "cuda"

        if torch.cuda.is_available()

        else "cpu"

    )

    AMP = True

    SAVE_BEST_ONLY = True

    CHECKPOINT_NAME = "best_model.pth"

    LAST_CHECKPOINT_NAME = "last_model.pth"

    CHECKPOINT_PATH = CHECKPOINT_DIR / CHECKPOINT_NAME

    LAST_CHECKPOINT_PATH = CHECKPOINT_DIR / LAST_CHECKPOINT_NAME

    SUBMISSION_NAME = "submission.csv"

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