from pathlib import Path


class Config:

    PROJECT_ROOT = Path(__file__).resolve().parent

    DATA_DIR = PROJECT_ROOT / "data"

    RAW_DATA_DIR = DATA_DIR / "raw"

    TRAIN_DIR = RAW_DATA_DIR / "train"

    VALIDATION_DIR = RAW_DATA_DIR / "validation"

    TEST_DIR = RAW_DATA_DIR / "test"

    OUTPUT_DIR = PROJECT_ROOT / "outputs"

    CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"

    LOG_DIR = OUTPUT_DIR / "logs"

    SUBMISSION_DIR = OUTPUT_DIR / "submission"

    BATCH_SIZE = 2

    VALIDATION_BATCH_SIZE = 1

    NUM_WORKERS = 4

    VALIDATION_NUM_WORKERS = 2

    PIN_MEMORY = True

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

    DEVICE = "cuda"

    RANDOM_SEED = 42

    SAVE_BEST_ONLY = True

    CHECKPOINT_NAME = "best_model.pth"

    CHECKPOINT_PATH = CHECKPOINT_DIR / CHECKPOINT_NAME

    SUBMISSION_NAME = "submission.csv"
