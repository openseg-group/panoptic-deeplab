PYTHON="/data/anaconda/envs/pytorch1.5/bin/python"
PYTHON="/data/anaconda/envs/pytorch1.5.1/bin/python"

CONFIG=$1
$PYTHON -m pip install git+https://github.com/mcordts/cityscapesScripts.git
$PYTHON -m pip install fvcore

# export mapillary_pretrain=1
export CUDA_VISIBLE_DEVICES=0,1,2,3

# training
$PYTHON -m torch.distributed.launch \
                --nproc_per_node=4 \
                tools/train_net.py \
                --cfg configs/${CONFIG}.yaml

# evaluation
# $PYTHON tools/test_net_single_core.py \
#                 --cfg configs/${CONFIG}.yaml \
