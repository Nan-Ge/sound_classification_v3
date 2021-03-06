import numpy as np


# Dataset Related
SUPPORT_SET_LABEL = set()  # 支撑集样本类别
LABEL_DICT = {
    '2-1': np.int64(1), '2-2': np.int64(2),
    '3-1': np.int64(3), '3-2': np.int64(4), '3-3': np.int64(5),
    '4-1': np.int64(6), '4-2': np.int64(7),
    '5-1': np.int64(8), '5-2': np.int64(9), '5-3': np.int64(10), '5-4': np.int64(46),
    '6-1': np.int64(11), '6-2': np.int64(12), '6-3': np.int64(13),
    '7-1': np.int64(14), '7-2': np.int64(15), '7-3': np.int64(45),
    '8-1': np.int64(16), '8-2': np.int64(17),
    '9-1': np.int64(18), '9-2': np.int64(19),
    '10-1': np.int64(20), '10-2': np.int64(21),
    '12-1': np.int64(22), '12-2': np.int64(23),
    '14-1': np.int64(24), '14-2': np.int64(25), '14-3': np.int64(26),
    '15-1': np.int64(27), '15-2': np.int64(28),
    '16-1': np.int64(29), '16-2': np.int64(30), '16-3': np.int64(31),
    '17-1': np.int64(32),
    '18-1': np.int64(33), '18-2': np.int64(34), '18-3': np.int64(35),
    '19-1': np.int64(36), '19-2': np.int64(37), '19-3': np.int64(38), '19-4': np.int64(39),
    '20-3': np.int64(40),
    '21-2': np.int64(41), '21-3': np.int64(42),
    '22-1': np.int64(43), '22-2': np.int64(44)
}

# Network Parameters
EMBEDDING_SIZE = 256  # Feature Extractor得到的embedding的大小（一维）
LP_OUTPUT_SIZE = 128  # Label Predictor输出的大小（一维）
DC_OUTPUT_SIZE = 128  # Domain Classifier输出的大小（一维）

# Loss Related
TRIPLET_MARGIN = 1  # Triplet loss的margin
PAIR_MARGIN = 0.25  # Pair-wise loss的margin
TRIPLET_PAIR_RATIO = 1  # 权重比，Triplet_loss / Pair_wise_loss

# Offline-training Related
OFF_INITIAL_LR = 1e-3  # 初始学习率
OFF_WEIGHT_DECAY = 1e-4  # 权重衰减率
OFF_LR_ADJUST_STEP = 50  # 学习率调整步长，单位：epoch
OFF_LR_ADJUST_RATIO = 0.1  # 学习率调整比例，每OFF_LR_ADJUST_STEP个epoch，调整至原来的0.1

OFFLINE_EPOCH = 200  # 离线训练epoch数
NUM_SAMPLES_PER_CLASS = 2  # Triplet-loss BalancedBatchSampler中每类样本的取样个数，决定了triplet的batch大小
PAIR_WISE_BATCH = 40  # Pair-wise Dataset的batch大小


# Online-training Related
ON_INITIAL_LR = 1e-3  # 初始学习率
ON_WEIGHT_DECAY = 1e-4  # 权重衰减率
ON_LR_ADJUST_STEP = 50  # 学习率调整步长，单位：epoch
ON_LR_ADJUST_RATIO = 0.1  # 学习率调整比例，每ON_LR_ADJUST_STEP个epoch，调整至原来的0,1

ONLINE_EPOCH = 100  # 在线训练epoch数
FINE_TUNE_BATCH = 19 # Fine-tuning Dataset的batch大小




