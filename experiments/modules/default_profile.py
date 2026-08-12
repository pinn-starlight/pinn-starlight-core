class DefaultProfile:
    def __init__(self):
        IMAGE_PATH = "data/image/"
        OUTPUT_DIR = "experiments/outputs/e1_learnable_alpha"
        HIDDEN_DIMS = [128, 128]
        KERNEL_SIZE = 31
        BATCH_SIZE = 8192
        MAX_STEPS = 3000
        PHYSICS_WEIGHT = 0.4
        MODEL_LR = 1e-3
        ICITY_LR = 1e-3
        ALPHA_MODE = "learnable"
        ALPHA_VALUE = 0.5
        ALPHA_INIT = 0.55
        ALPHA_MIN = 0.4
        ALPHA_MAX = 0.6
        ALPHA_LR = 1e-4
        RENDER_CHUNK_SIZE = 50_000
        SEED = 20260728
        LOG_INTERVAL = 50