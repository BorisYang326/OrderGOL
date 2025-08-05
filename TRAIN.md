# 🚀 Training Guide

This guide provides detailed instructions for training the OrderGOL model. The training process consists of three stages:

1. **TRAIN-BASELINE**: Training baseline ordering strategies (random, raster, saliency, layer, lexical)
2. **JOINT-TRAINING**: Joint training of ordering network and generator
3. **TRAIN-OURS**: Training generator with learned neural ordering

## 💾 Hardware Requirements

- **Single Generator Training**: ~24GB GPU RAM (batch size 256)
- **Joint Training**: ~48GB GPU RAM (batch size 256)

## 🎯 Training Stages

### Stage 1: Training Generator under Baseline Ordering

```bash
python main.py --multirun \
    +experiment=crello-10k-corrv3 \
    data.elem_order=random,raster,saliency,layer,lexical \
    trainer.max_epochs=800 \
    data.batch_size=256 \
    hydra.job.chdir=true \
    trainer.learning_rate=5e-5 \
    trainer.save_model=True \
    trainer.run_mode=full \
    trainer.top_p=0.95 \
    trainer.split_ratio=1.0 \
    trainer.eval_fid_steplist='[800]' \
    trainer.wandb_mode=disabled \
    "trainer.exp_name=crello-bs\${data.batch_size}_epochs\${trainer.max_epochs}_5e-5" \
    hydra.sweep.subdir='crello_bs${data.batch_size}_epochs${trainer.max_epochs}-${data.elem_order}'
```

### Stage 2: Joint Training

Train the ordering network jointly with the generator:

```bash
python main.py --multirun \
    +experiment=crello-10k-corrv3 \
    data.elem_order=neural \
    trainer.max_epochs=800 \
    data.batch_size=256 \
    hydra.job.chdir=true \
    trainer.learning_rate=5e-5 \
    trainer.save_model=True \
    trainer.run_mode=full \
    trainer.top_p=0.95 \
    trainer.split_ratio=1.0 \
    trainer.eval_fid_steplist='[800]' \
    trainer.wandb_mode=disabled \
    "trainer.exp_name=crello-bs\${data.batch_size}_epochs\${trainer.max_epochs}_5e-5" \
    hydra.sweep.subdir='crello_bs${data.batch_size}_epochs${trainer.max_epochs}-${data.elem_order}'
```

### Stage 3: Training Generator under Neural Ordering


```bash
# Set the path to order pretraining data
ORDER_PRETRAIN_PATH="$(pwd)/data/crello/order_pretrain"

python main.py --multirun \
    +experiment=crello-10k-corrv3-optimized \
    data.elem_order=optimized \
    trainer.max_epochs=800 \
    data.batch_size=256 \
    hydra.job.chdir=true \
    trainer.learning_rate=5e-5 \
    trainer.save_model=True \
    trainer.run_mode=full \
    data.order_pretrain_flag=false \
    data.order_pretrain_path=$ORDER_PRETRAIN_PATH \
    data.order_sweep_dir=$CKPT_PATH/optimized \
    data.order_selected='[48]' \
    trainer.top_p=0.95 \
    trainer.split_ratio=1.0 \
    trainer.eval_fid_steplist='[800]' \
    trainer.wandb_mode=disabled \
    "trainer.exp_name=crello-bs\${data.batch_size}_epochs\${trainer.max_epochs}_5e-5" \
    hydra.sweep.subdir='crello_bs${data.batch_size}_epochs${trainer.max_epochs}-${data.elem_order}'
```

- `order_selected='[48]'` specifies which learned ordering to use