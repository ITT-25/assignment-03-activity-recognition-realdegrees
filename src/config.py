class Config:
    window_width: int = 800
    window_height: int = 550
    COLUMN_GAP: int = 10
    IMAGE_ANIMATION_INTERVAL: float = 0.5
    UPDATE_RATE: int = 60
    STAGE_TRANSITION_DURATION: float = 1.5
    TEXT_COLOR: tuple[int, int, int] = (255, 255, 255, 255)
    PRIMARY_COLOR: tuple[int, int, int] = (31, 31, 31, 255)
    SECONDARY_COLOR: tuple[int, int, int] = (49, 49, 49, 255)
    SUCCESS_COLOR: tuple[int, int, int] = (32, 253, 58, 255)
    ERROR_COLOR: tuple[int, int, int] = (253, 65, 32, 255)
    WARNING_COLOR: tuple[int, int, int] = (253, 132, 32, 255)
    TRAINING_DATA_SUBSET_SIZE: int = 100
    LIVE_DATA_SUBSET_SIZE: int = 100
    IDLE_DATA_SUBSET_SIZE: int = UPDATE_RATE // 2
