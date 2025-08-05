# export CUDA_VISIBLE_DEVICES=1

# Get absolute path of script directory to avoid issues with hydra.job.chdir=true
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$SCRIPT_DIR"

# you can set the ckpt path manually here
# CKPT_PATH="$SCRIPT_DIR/ckpt/crello"
ORDER_PRETRAIN_PATH="$SCRIPT_DIR/data/crello/order_pretrain"


# # TRAIN-BASELINE
# python main.py --multirun \
#     +experiment=crello-10k-corrv3 \
#     data.elem_order=random,raster,saliency,layer,lexical \
#     trainer.max_epochs=800 \
#     data.batch_size=256 \
#     hydra.job.chdir=true \
#     trainer.learning_rate=5e-5 \
#     trainer.save_model=True \
#     trainer.run_mode=full \
#     trainer.top_p=0.95 \
#     trainer.split_ratio=1.0 \
#     trainer.eval_fid_steplist='[800]' \
#     trainer.wandb_mode=disabled \
#     "trainer.exp_name=crello-bs\${data.batch_size}_epochs\${trainer.max_epochs}_5e-5" \
#     hydra.sweep.subdir='crello_bs${data.batch_size}_epochs${trainer.max_epochs}-${data.elem_order}'


# # JOINT-TRAINING
# python main.py --multirun \
#     +experiment=crello-10k-corrv3 \
#     data.elem_order=neural \
#     trainer.max_epochs=800 \
#     data.batch_size=256 \
#     hydra.job.chdir=true \
#     trainer.learning_rate=5e-5 \
#     trainer.save_model=True \
#     trainer.run_mode=full \
#     trainer.top_p=0.95 \
#     trainer.split_ratio=1.0 \
#     trainer.eval_fid_steplist='[800]' \
#     trainer.wandb_mode=disabled \
#     "trainer.exp_name=crello-bs\${data.batch_size}_epochs\${trainer.max_epochs}_5e-5" \
#     hydra.sweep.subdir='crello_bs${data.batch_size}_epochs${trainer.max_epochs}-${data.elem_order}'


# TRAIN-OURS
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