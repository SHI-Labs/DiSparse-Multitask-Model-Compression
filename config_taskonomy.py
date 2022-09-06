'''
Configuration File Used for Cityscapes Training & Evaluation
'''

DATA_ROOT = "/data/alexsun/taskonomy_dataset/"
CROP_H = 224
CROP_W = 224
TASKS = ["seg", "sn", "depth", "keypoint", "edge"]
TASKS_NUM_CLASS = [17, 3, 1, 1, 1]

LAMBDAS = [1, 3, 2, 7, 7]
NUM_GPUS = 1
BATCH_SIZE = 16 * NUM_GPUS
MAX_ITERS = 100000 / NUM_GPUS
DECAY_LR_FREQ = 12000 / NUM_GPUS
DECAY_LR_RATE = 0.3
    
INIT_LR = 1e-4
WEIGHT_DECAY = 5e-4
IMAGE_SHAPE = (256, 256) 

END = 75000
INT = 250
# INT = 20
PRUNE_RATE = 0.5

RETRAIN_EPOCH = 4000
RETRAIN_DECAY_LR_FREQ = 1500
RETRAIN_LR = 1e-6