# 🎯 Evaluation Guide

This guide provides detailed instructions for evaluating trained OrderGOL models and generating design samples.

## 📦 Checkpoint Setup

### Step 1: Download Pretrained Checkpoints

Download the pretrained model checkpoints from our Hugging Face repository:

🤗 **Hugging Face Repository**: [https://huggingface.co/BorisYang326/OrderGOL/tree/main](https://huggingface.co/BorisYang326/OrderGOL/tree/main)

Available checkpoints:
- `random` - Random ordering baseline
<!-- - `raster/` - Raster ordering baseline   -->
<!-- - `saliency/` - Saliency-based ordering baseline -->
<!-- - `layer/` - Layer-based ordering baseline -->
- `optimized` - Our proposed neural ordering

### Step 2: Organize Checkpoints

Create the checkpoint directory structure and place downloaded files:

```bash
mkdir -p ckpt/crello
cd ckpt/crello

# Move downloaded checkpoint folders here
# Expected structure:
# ckpt/
# └── crello/
#     ├── random/
#     │   └── checkpoints/
#     │       └── last/
#     │           ├── model.pth
#     │           └── codebook.pth
#     ├── optimized/
#     │   └── order_48/
#     │       └── checkpoints/
#     │           └── last/
#     │               ├── model.pth
#     │               └── codebook.pth
#     └── ...
```

## 🚀 Running Evaluation

### Baseline Models Evaluation


```bash
python main.py --multirun \
    +experiment=crello-10k-corrv3 \
    data.elem_order=random \
    trainer.max_epochs=800 \
    data.batch_size=32 \
    hydra.job.chdir=true \
    trainer.learning_rate=5e-5 \
    trainer.render_vis=True \
    trainer.save_svg=True \
    trainer.run_mode=eval \
    "trainer.ckpt_dir=ckpt/crello/\${data.elem_order}/checkpoints" \
    "trainer.codebook_dir=ckpt/crello/\${data.elem_order}/checkpoints/last/codebook.pth" \
    "trainer.model_dir=ckpt/crello/\${data.elem_order}/checkpoints/last/model.pth" \
    trainer.pretrained_mode=model_codebook \
    trainer.top_p=0.95 \
    trainer.eval_fid_steplist='[800]' \
    trainer.split_ratio=1.0 \
    trainer.wandb_mode=disabled \
    trainer.sample_num=32 \
    trainer.compute_fid=False \
    "trainer.exp_name=crello-bs\${data.batch_size}_epochs\${trainer.max_epochs}_5e-5_eval" \
    "hydra.sweep.subdir=crello-bs\${data.batch_size}_epochs\${trainer.max_epochs}-\${data.elem_order}_5e-5_eval"
```

### Our Method Evaluation


```bash
# Set paths
ORDER_PRETRAIN_PATH="$(pwd)/data/crello/order_pretrain"
CKPT_PATH="$(pwd)/ckpt/crello"

python main.py --multirun \
    +experiment=crello-10k-corrv3-optimized \
    data.elem_order=optimized \
    trainer.max_epochs=800 \
    data.batch_size=8 \
    hydra.job.chdir=true \
    trainer.learning_rate=5e-5 \
    trainer.render_vis=True \
    trainer.save_svg=True \
    trainer.run_mode=eval \
    data.order_pretrain_flag=false \
    data.order_pretrain_path=$ORDER_PRETRAIN_PATH \
    data.order_sweep_dir=$CKPT_PATH/optimized \
    data.order_selected='[48]' \
    trainer.pretrained_mode=model_codebook \
    trainer.top_p=0.95 \
    trainer.eval_fid_steplist='[800]' \
    trainer.split_ratio=1.0 \
    trainer.wandb_mode=disabled \
    trainer.sample_num=8 \
    trainer.compute_fid=False \
    "trainer.exp_name=crello-bs\${data.batch_size}_epochs\${trainer.max_epochs}_5e-5_eval" \
    "hydra.sweep.subdir=crello-bs\${data.batch_size}_epochs\${trainer.max_epochs}-\${data.elem_order}_5e-5_eval"
```

## ⚠️ Data Limitation
Due to copyright restrictions of the Crello dataset, we have not uploaded the image patches required for retrieval-based evaluation metrics. This may affect certain evaluation functionalities. You can preprocess from original dataset [Crello-v5](https://huggingface.co/datasets/cyberagent/crello).