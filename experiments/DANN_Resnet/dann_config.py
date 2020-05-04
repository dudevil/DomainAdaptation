LOSS_GAMMA = 10  # from authors, not optimized
LOSS_NEED_INTERMEDIATE_LAYERS = False
UNK_VALUE = -100  # torch default
IS_UNSUPERVISED = True

GRADIENT_REVERSAL_LAYER_ALPHA = 1.0
FREZE_BACKBONE_FEATURES = True

BATCH_SIZE = 64

NUM_WORKERS = 4
N_EPOCHS = 31
STEPS_PER_EPOCH = 10
VAL_FREQ = 1
SAVE_MODEL_FREQ = 20

################### Model dependent parameters #########################

CLASSES_CNT = 31
MODEL_BACKBONE = "resnet50" # alexnet resnet50 vanilla_dann
DOMAIN_HEAD = "vanilla_dann"
BACKBONE_PRETRAINED = True
IMAGE_SIZE = 224
DATASET = "office-31"
SOURCE_DOMAIN = "amazon"
TARGET_DOMAIN = "dslr"

# CLASSES_CNT = 10
# MODEL_BACKBONE = "mnist_dann"
# DOMAIN_HEAD = "mnist_dann"
# BACKBONE_PRETRAINED = False
# IMAGE_SIZE = 28
# DATASET = "mnist"
# SOURCE_DOMAIN = "mnist"
# TARGET_DOMAIN = "mnist-m"
