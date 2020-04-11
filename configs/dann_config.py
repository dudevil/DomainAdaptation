LOSS_NEED_INTERMEDIATE_LAYERS = False
CLASSES_CNT = 31
MODEL_BACKBONE = "alexnet" # alexnet resnet50 vanilla_dann
BACKBONE_PRETRAINED = True
GRADIENT_REVERSAL_LAYER_ALPHA = 1.0
IMAGE_SIDE = 256

LOSS_GAMMA = 10  # from authors, not optimized
# BATCH_SIZE = 128  # from authors # is it ever used?
UNK_VALUE = -100  # torch default
