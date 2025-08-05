# export CUDA_VISIBLE_DEVICES=1

# Get absolute path of script directory to avoid issues with hydra.job.chdir=true
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$SCRIPT_DIR"

# you can set the ckpt path manually here
CKPT_PATH="$SCRIPT_DIR/ckpt/crello"
ORDER_PRETRAIN_PATH="$SCRIPT_DIR/data/crello/order_pretrain"

# # EVAL-BASELINE
# python main.py --multirun \
#     +experiment=crello-10k-corrv3 \
#     data.elem_order=random \
#     trainer.max_epochs=800 \
#     data.batch_size=32 \
#     hydra.job.chdir=true \
#     trainer.learning_rate=5e-5 \
#     trainer.render_vis=True \
#     trainer.save_svg=True \
#     trainer.run_mode=eval \
#     "trainer.ckpt_dir=$CKPT_PATH/\${data.elem_order}/checkpoints" \
#     "trainer.codebook_dir=$CKPT_PATH/\${data.elem_order}/checkpoints/last/codebook.pth" \
#     "trainer.model_dir=$CKPT_PATH/\${data.elem_order}/checkpoints/last/model.pth" \
#     trainer.pretrained_mode=model_codebook \
#     trainer.top_p=0.95 \
#     trainer.eval_fid_steplist='[800]' \
#     trainer.split_ratio=1.0 \
#     trainer.wandb_mode=disabled \
#     trainer.sample_num=32 \
#     trainer.compute_fid=False \
#     "trainer.exp_name=crello-bs\${data.batch_size}_epochs\${trainer.max_epochs}_5e-5_eval" \
#     "hydra.sweep.subdir=crello-bs\${data.batch_size}_epochs\${trainer.max_epochs}-\${data.elem_order}_5e-5_eval"


# EVAL-OURS
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