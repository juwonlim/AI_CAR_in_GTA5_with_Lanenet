# global_config.py
from easydict import EasyDict as edict

cfg = edict()

# GPU 설정
cfg.GPU = edict()
cfg.GPU.TF_ALLOW_GROWTH = True  # GPU 메모리 자동 조절

# 모델 설정
cfg.MODEL = edict()
cfg.MODEL.FRONT_END = 'bisenetv2'  # 'vgg' 또는 'bisenetv2' 중 선택

# 데이터셋 설정
cfg.DATASET = edict()
cfg.DATASET.NUM_CLASSES = 4  # lane class + background

# 학습 관련 설정
cfg.SOLVER = edict()
cfg.SOLVER.LOSS_TYPE = 'softmax'  # 또는 'dice', 사용 시 변경 가능
cfg.SOLVER.WEIGHT_DECAY = 0.0001
cfg.SOLVER.OHEM = edict()
cfg.SOLVER.OHEM.ENABLE = False

# Moving Average
cfg.SOLVER.MOVING_AVE_DECAY = 0.9999

# Discriminative Loss (postprocess 등에서 쓰일 가능성 있음)
cfg.POSTPROCESS = edict()
cfg.POSTPROCESS.DIS_THRESH = 0.8
cfg.POSTPROCESS.CLUSTER_MIN_SAMPLES = 100

# 필요한 경우 추가 설정 가능
cfg.MODEL.EMBEDDING_FEATS_DIMS = 4  # 일반적으로 4 또는 3 사용됨


# BISENETV2 모델 세부 설정 추가
cfg.MODEL.BISENETV2 = edict()
cfg.MODEL.BISENETV2.GE_EXPAND_RATIO = 6
cfg.MODEL.BISENETV2.WITHOUT_CONTEXT = False
cfg.MODEL.BISENETV2.USE_EMBEDDING = True



CFG = cfg