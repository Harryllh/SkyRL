set -x

# GRPO training for safety classification using Nemotron-3-Content-Safety with FSDP2.
#
# Step 1: prepare data
#   uv run --no-project --python 3.12 --with datasets \
#     examples/train/safety/safety_dataset.py \
#     --data_path ~/aq_worktrial/sdg_output/prompt_pairs.jsonl \
#     --output_dir $HOME/data/safety
#
# Step 2: train
#   export WANDB_API_KEY=<your_key_here>
#   bash examples/train/safety/run_nemotron.sh
#
# Override defaults:
#   TRAIN_GPUS=4 ROLLOUT_GPUS=4 bash examples/train/safety/run_nemotron.sh

: "${DATA_DIR:="$HOME/data/safety"}"
: "${NUM_GPUS:=8}"
: "${LOGGER:=wandb}"          # set to "console" to skip wandb
: "${INFERENCE_BACKEND:=vllm}"
: "${MODEL:=nvidia/Nemotron-3-Content-Safety}"
: "${LORA_RANK:=8}"
: "${LORA_ALPHA:=16}"

# trainer.policy.model.lora.rank=$LORA_RANK \
#   trainer.policy.model.lora.alpha=$LORA_ALPHA \
#   trainer.policy.model.lora.target_modules='[q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj]' \

uv run --isolated --python 3.12 --extra fsdp -m skyrl.train.entrypoints.main_base \
  data.train_data="['$DATA_DIR/train.parquet']" \
  data.val_data="['$DATA_DIR/validation.parquet']" \
  trainer.algorithm.advantage_estimator="grpo" \
  trainer.policy.model.path="$MODEL" \
  trainer.placement.colocate_all=true \
  trainer.strategy=fsdp2 \
  trainer.placement.policy_num_gpus_per_node=$NUM_GPUS \
  trainer.placement.critic_num_gpus_per_node=$NUM_GPUS \
  trainer.placement.ref_num_gpus_per_node=$NUM_GPUS \
  generator.inference_engine.num_engines=$NUM_GPUS \
  generator.inference_engine.tensor_parallel_size=1 \
  trainer.epochs=20 \
  trainer.eval_batch_size=50 \
  trainer.eval_before_train=true \
  trainer.eval_interval=1 \
  trainer.update_epochs_per_batch=1 \
  trainer.train_batch_size=512 \
  trainer.policy_mini_batch_size=512 \
  trainer.micro_forward_batch_size_per_gpu=4 \
  trainer.micro_train_batch_size_per_gpu=4 \
  trainer.ckpt_interval=2 \
  trainer.use_sample_packing=false \
  trainer.max_prompt_length=512 \
  generator.sampling_params.max_generate_length=64 \
  generator.chat_template_kwargs.request_categories="/categories" \
  trainer.policy.optimizer_config.lr=1.0e-6 \
  trainer.algorithm.use_kl_loss=true \
  trainer.algorithm.kl_loss_coef=0.01 \
  generator.inference_engine.backend=$INFERENCE_BACKEND \
  generator.inference_engine.run_engines_locally=true \
  generator.inference_engine.weight_sync_backend=nccl \
  generator.inference_engine.async_engine=true \
  generator.batched=false \
  environment.env_class=safety \
  generator.n_samples_per_prompt=8 \
  generator.inference_engine.gpu_memory_utilization=0.8 \
  trainer.logger="$LOGGER" \
  trainer.project_name="nemotron-safety_final" \
  trainer.run_name="nemotron_3_content_safety_grpo_fsdp2" \
  trainer.resume_mode=null \
  trainer.ckpt_path="$HOME/ckpts/nemotron_3_content_safety" \
  trainer.hf_save_interval=2 \
  trainer.export_path="$HOME/hf_ckpt/nemotron_3_content_safety" \
  $@
