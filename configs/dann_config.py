LOSS_GAMMA = 10  # from authors, not optimized
LOSS_NEED_INTERMEDIATE_LAYERS = False
UNK_VALUE = -100  # torch default
IS_UNSUPERVISED = True

GRADIENT_REVERSAL_LAYER_ALPHA = 1.0
FREZE_BACKBONE_FEATURES = True

BATCH_SIZE = 32

NUM_WORKERS = 4
N_EPOCHS = 200
STEPS_PER_EPOCH = 20
VAL_FREQ = 1
SAVE_MODEL_FREQ = 199

################### Model dependent parameters #########################

CLASSES_CNT = 31
MODEL_BACKBONE = "alexnet" # alexnet resnet50 vanilla_dann
DOMAIN_HEAD = "vanilla_dann"
BACKBONE_PRETRAINED = True
NEED_ADAPTATION_BLOCK = True # ="True" only for alexnet, ="False" for other types
BLOCKS_WITH_SMALLER_LR = 2 # ="2" only for alexnet, ="0" for other types
IMAGE_SIZE = 224
DATASET = "office-31"
SOURCE_DOMAIN = "amazon"
TARGET_DOMAIN = "webcam"

# CLASSES_CNT = 10
# MODEL_BACKBONE = "mnist_dann"
# DOMAIN_HEAD = "mnist_dann"
# BACKBONE_PRETRAINED = False
# NEED_ADAPTATION_BLOCK = False
# BLOCKS_WITH_SMALLER_LR = 0
# IMAGE_SIZE = 28
# DATASET = "mnist"
# SOURCE_DOMAIN = "mnist"
# TARGET_DOMAIN = "mnist-m"

assert (MODEL_BACKBONE == "alexnet" or \
        (MODEL_BACKBONE != "alexnet" and not NEED_ADAPTATION_BLOCK and BLOCKS_WITH_SMALLER_LR == 0)), \
       "can't use adaptation block with non-alexnet"
