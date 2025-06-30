from easydict import EasyDict as edict

cfg = edict()

# -------- SOLVER --------
cfg.SOLVER = edict()
cfg.SOLVER.LR = 0.001
cfg.SOLVER.MOVING_AVE_DECAY = 0.9999
cfg.SOLVER.WEIGHT_DECAY = 0.0005
cfg.SOLVER.LOSS_TYPE = 'dice'        # 또는 'cross_entropy'
cfg.SOLVER.LOSS_TYPE = 'softmax'
cfg.SOLVER.OHEM = False  # 또는 True, 필요에 따라 설정 #OHEM(Online Hard Example Mining)은 어려운 샘플만을 골라 학습에 집중하는 방법,일반적으로 loss.py 또는 bisenet_v2.py 등에서 loss 계산 시 사용 여부를 체크
cfg.SOLVER.MOVING_AVE_DECAY = 0.9999
cfg.SOLVER.MOMENTUM = 0.9
cfg.SOLVER.OPTIMIZER = 'sgd'

cfg.SOLVER.OHEM = edict()
cfg.SOLVER.OHEM.ENABLE = False  # 또는 True

# -------- GPU --------
cfg.GPU = edict()
cfg.GPU.TF_ALLOW_GROWTH = True

# -------- MODEL --------
cfg.MODEL = edict()
cfg.MODEL.FRONT_END = 'bisenetv2'     # 또는 'vgg', 'enet' 등 사용 모델에 따라

# -------- DATASET --------
cfg.DATASET = edict()
cfg.DATASET.NAME = 'tusimple'
cfg.DATASET.TUSIMPLE_ROOT = ''
cfg.DATASET.CULANE_ROOT = ''
cfg.DATASET.BATCH_SIZE = 1
cfg.DATASET.NUM_CLASSES = 4          # Lane + background (보통 4 또는 2)
cfg.DATASET.IMG_HEIGHT = 256
cfg.DATASET.IMG_WIDTH = 512



CFG = cfg