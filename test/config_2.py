# Dataset Related
OBJ_DICT = {
    'top_razor-core-x': '2-1',
    'side_razor-core-x': '2-2',
    'razor-core-x-back': '2-3',  # 新增

    'xiaomi-lamp-top': '3-1',
    'xiaomi-lamp-body': '3-2',
    'xiaomi-lamp-bottom': '3-3',

    'dehumidifier-top': '4-1',
    'dehumidifier-body': '4-2',

    'top2_dell-inspire-case-right-side': '5-1',
    'top1_dell-inspire-case-left-side': '5-2',
    'side_dell-inspire-case-right-side': '5-3',
    'dell-inspire-case-front': '5-4',

    'top_galanz-microwave-oven-body': '6-1',
    'side_galanz-microwave-oven-body': '6-2',
    'galanz-microwave-oven-front': '6-3',

    'white-kettle-top': '7-1',
    'white-kettle-handle': '7-2',
    'white-kettle-body': '7-3',

    'top_philips-speaker-body': '8-1',
    'side_philips-speaker-body': '8-2',

    'top_yamaha-speaker': '9-1',
    'side_yamaha-speaker': '9-2',

    'top_mitsubishi-projector': '10-1',
    'front_mitsubishi-projector': '10-2',

    'hp-printer-top': '12-1',
    'hp-printer-front': '12-2',

    'electic-kettle-top': '14-1',
    'electic-kettle-handle': '14-2',
    'electic-kettle-body': '14-3',

    'top_dual-microwave-oven-side': '15-1',
    'side_dual-microwave-oven-side': '15-2',
    'dual-microwave-oven-front-front': '15-3',

    'side1_hair-dryer': '16-1',
    'side2_hair-dryer': '16-1',
    'side3_hair-dryer': '16-1',

    'weight-meter': '17-1',

    'rice-cooker-top': '18-1',
    'side1_rice-cooker-side': '18-2',
    'side2_rice-cooker-side': '18-3',

    'top_oven-body': '19-1',
    'side_oven-body': '19-2',
    'oven-front': '19-3',
    'oven-panel': '19-4',

    'tv-base': '20-3',

    'coffee-machine-top2': '21-2',
    'coffee-machine-front': '21-3',

    'imac-screen': '22-1',
    'imac-body': '22-2',
}

SUPPORT_SET_LABEL = {0}  # 支撑集样本类别
LABEL_DICT = {
    '2-1': 1, '2-2': 2,
    '3-1': 3, '3-2': 4, '3-3': 5,
    '4-1': 6, '4-2': 7,
    '5-1': 8, '5-2': 9, '5-3': 10, '5-4': 46,
    '6-1': 11, '6-2': 12, '6-3': 13,
    '7-1': 14, '7-2': 15, '7-3': 45,
    '8-1': 16, '8-2': 17,
    '9-1': 18, '9-2': 19,
    '10-1': 20, '10-2': 21,
    '12-1': 22, '12-2': 23,
    '14-1': 24, '14-2': 25, '14-3': 26,
    '15-1': 27, '15-2': 28,
    '16-1': 29, '16-2': 30, '16-3': 31,
    '17-1': 32,
    '18-1': 33, '18-2': 34, '18-3': 35,
    '19-1': 36, '19-2': 37, '19-3': 38, '19-4': 39,
    '20-3': 40,
    '21-2': 41, '21-3': 42,
    '22-1': 43, '22-2': 44
}

# Network Parameters
EMBEDDING_SIZE = 512  # Feature Extractor得到的embedding的大小（一维）
LP_OUTPUT_SIZE = 256  # Label Predictor输出的大小（一维）
DC_OUTPUT_SIZE = 256  # Domain Classifier输出的大小（一维）

# Loss Related
TRIPLET_MARGIN = 10.0  # Triplet loss的margin
PAIR_MARGIN = 0.25  # Pair-wise loss的margin
TRIPLET_PAIR_RATIO = 100  # 两种Loss的权重

# Offline-training Related
OFF_INITIAL_LR = 1e-3  # 初始学习率
OFF_WEIGHT_DECAY = 1e-4  # 权重衰减率
OFF_LR_ADJUST_STEP = 50  # 学习率调整步长，单位：epoch
OFF_LR_ADJUST_RATIO = 0.1  # 学习率调整比例，每OFF_LR_ADJUST_STEP个epoch，调整至原来的0.1

OFFLINE_EPOCH = 100  # 离线训练epoch数
NUM_SAMPLES_PER_CLASS = 5  # Triplet-loss BalancedBatchSampler中每类样本的取样个数，决定了triplet的batch大小
PAIR_WISE_BATCH = 40  # Pair-wise Dataset的batch大小


# Online-training Related
ON_INITIAL_LR = 1e-3  # 初始学习率
ON_WEIGHT_DECAY = 1e-4  # 权重衰减率
ON_LR_ADJUST_STEP = 50  # 学习率调整步长，单位：epoch
ON_LR_ADJUST_RATIO = 0.1  # 学习率调整比例，每ON_LR_ADJUST_STEP个epoch，调整至原来的0,1

ONLINE_EPOCH = 100  # 在线训练epoch数
FINE_TUNE_BATCH = 20  # Fine-tuning Dataset的batch大小
