import os

from yacs.config import CfgNode as CN

cfg = CN()
cfg.RESUME = False
cfg.OUTPUT_DIR = ''
cfg.PRINT_FREQ = 1
cfg.DATA = CN()
cfg.DATA.PATH = './dataset/reviews_Beauty_5/min_count1/'
cfg.DATA.MODE = 'train'
cfg.DATA.SAMPLING_RATE = 1e-4
cfg.DATA.NEGATIVE_SAMPLES = 5
cfg.DATA.SHUFFLE = True

cfg.MODEL = CN()
cfg.MODEL.FUSION = 'cat'  #
cfg.MODEL.NEGATIVE_SAMPLES = 5
cfg.MODEL.EMBED_SIZE = 300
cfg.MODEL.IMG_EMBED_SIZE = 300
cfg.MODEL.NEED_TEXT = True
cfg.MODEL.NEED_IMAGE = True
cfg.MODEL.NEED_REVIEW = False
cfg.MODEL.NEED_BPR = True
cfg.MODEL.NEED_EXTEND = False
cfg.MODEL.NEED_MASK = False
cfg.MODEL.IMAGE_WEIGHT = 1e-4
cfg.MODEL.WEIGHT_FILE = ""
cfg.MODEL.SIMILARITY_FUNC = "product"
cfg.MODEL.RANK_CUTOFF = 100
cfg.MODEL.QUANTIZE = False
cfg.MODEL.PRETRAINED = ''
cfg.MODEL.NAS = CN()
cfg.MODEL.NAS.COMBINE_FUNC = ['concat']
cfg.MODEL.NAS.STEPS = 10
cfg.MODEL.NAS.INVERT_WEIGHT = False
cfg.MODEL.NAS.SELECTION = 'cutoff'
cfg.MODEL.NAS.DERIVATION = 'selected'
cfg.MODEL.NAS.TAU = [10.0, 0.001]
cfg.MODEL.NAS.LR = CN()
cfg.MODEL.NAS.LR.MAX = 1e-3
cfg.MODEL.NAS.LR.MIN = 1e-5
cfg.MODEL.NAS.GRAD_NORM = 5.0
cfg.MODEL.NAS.BUDGET = CN()
cfg.MODEL.NAS.BUDGET.MAX = 0.5
cfg.MODEL.NAS.BUDGET.MAX_W = 1.0
cfg.MODEL.NAS.BUDGET.MIN_W = 1.0
cfg.MODEL.NAS.BUDGET.MIN = 0.5
cfg.MODEL.NAS.BUDGET.ENABLED = True
cfg.MODEL.NAS.BUDGET.SCALE = 2e-7
cfg.MODEL.NAS.SELF_DISTILL = False
cfg.MODEL.NAS.DISTILL = False
cfg.TRAIN = CN()
cfg.TRAIN.BEGIN_EPOCH = 0
cfg.TRAIN.BATCH_SIZE = 100
cfg.TRAIN.END_EPOCH = 5
cfg.TRAIN.OPTIMIZER = 'sgd'
cfg.TRAIN.LR = 0.5
cfg.TRAIN.LR_MIN = 0.0005
cfg.TRAIN.WEIGHT_DECAY = 0.09
cfg.TRAIN.CLIP_NORM = 5.0
cfg.TRAIN.SCHEDULER = 'linear'
cfg.TRAIN.LR_DECAY = 0.90
cfg.TRAIN.DISTILL = CN()
cfg.TRAIN.DISTILL.ENABLED = False
cfg.TRAIN.DISTILL.PATH = ''
cfg.TRAIN.DISTILL.SCALE = 1.0
cfg.TRAIN.MASK = CN()
cfg.TRAIN.MASK.RHO = 0.001
cfg.TRAIN.MASK.RHO_IMG = 0.0001
cfg.TRAIN.MASK.PATH = ''
cfg.TRAIN.MASK.BUFFER = False
cfg.TRAIN.MASK.BINARY = False
cfg.TEST = CN()
cfg.TEST.BATCH_SIZE = 64

cfg.DEBUG = CN()
cfg.DEBUG.ENABLED = False
cfg.DEBUG.CHECK_NORM = False


def update_config(cfg, args):
    cfg.defrost()
    cfg.merge_from_file(args.cfg)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
