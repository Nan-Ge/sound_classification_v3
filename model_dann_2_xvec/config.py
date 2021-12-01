import numpy as np

# Flow control
TRAIN_STAGE = 1
FINE_TUNE_STAGE = 0

# Dataset Related
SUPPORT_SET_LABEL = set((6, 7, 20, 21, 33, 34, 35))  # 支撑集样本类别
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
EMBEDDING_SIZE = 128  # Feature Extractor得到的embedding的大小（一维）
XVEC_VERSION = 2  # X-vector实现版本

# Loss Related
LOSS_WEIGHTS = (1.0, 1.0, 1.0)  # 三个损失(src_lp_err, tgt_lp_loss, dc_err)的权重

# Offline-training Related
OFF_INITIAL_LR = 1e-2  # 初始学习率
OFF_WEIGHT_DECAY = 1e-3  # 权重衰减率
OFF_LR_ADJUST_STEP = 100  # 学习率调整步长，单位：epoch
OFF_LR_ADJUST_RATIO = 0.1  # 学习率调整比例，每OFF_LR_ADJUST_STEP个epoch，调整至原来的0.1
OFFLINE_EPOCH = 300  # 离线训练epoch数
NUM_SAMPLES_PER_CLASS = 2  # Triplet-loss BalancedBatchSampler中每类样本的取样个数，决定了triplet的batch大小
NOISE_EPS = 0  # 离线训练噪声强度
P_DROP = 0.5  # 离线训练dropout概率

# Online-training Related
ON_INITIAL_LR = 1e-3  # 初始学习率
ON_WEIGHT_DECAY = 1e-4  # 权重衰减率
ON_LR_ADJUST_STEP = 50  # 学习率调整步长，单位：epoch
ON_LR_ADJUST_RATIO = 0.1  # 学习率调整比例，每ON_LR_ADJUST_STEP个epoch，调整至原来的0,1

ONLINE_EPOCH = 100  # 在线训练epoch数
FINE_TUNE_BATCH = 20  # Fine-tuning Dataset的batch大小




