PYTHON="/opt/conda/bin/python"

CONFIG="panoptic_deeplab_H48_os4_cityscapes"

# training
$PYTHON -m torch.distributed.launch \
                --nproc_per_node=4 \
                tools/train_net.py \
                --cfg configs/${CONFIG}.yaml

# evaluation
# $PYTHON tools/test_net_single_core.py \
#                 --cfg configs/${CONFIG}.yaml 