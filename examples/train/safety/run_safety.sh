set -x

# GRPO training for safety classification on XSTest using WildGuard (Mistral-7B).
#
# Step 1: prepare data
#   uv run examples/train/safety/safety_dataset.py --output_dir $HOME/data/xstest
#
# Step 2: train
#   export WANDB_API_KEY=<your_key_here>
#   bash examples/train/safety/run_safety.sh
#
# Override defaults:
#   NUM_GPUS=2 MODEL=allenai/wildguard bash examples/train/safety/run_safety.sh

: "${DATA_DIR:="$HOME/data/xstest"}"
: "${NUM_GPUS:=4}"
: "${LOGGER:=wandb}"          # set to "console" to skip wandb
: "${INFERENCE_BACKEND:=vllm}"
: "${MODEL:=allenai/wildguard}"

uv run --isolated --extra fsdp -m skyrl.train.entrypoints.main_base \
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
  trainer.epochs=30 \
  trainer.eval_batch_size=50 \
  trainer.eval_before_train=true \
  trainer.eval_interval=5 \
  trainer.update_epochs_per_batch=1 \
  trainer.train_batch_size=200 \
  trainer.policy_mini_batch_size=50 \
  trainer.micro_forward_batch_size_per_gpu=16 \
  trainer.micro_train_batch_size_per_gpu=16 \
  trainer.ckpt_interval=10 \
  trainer.max_prompt_length=512 \
  generator.sampling_params.max_generate_length=64 \
  trainer.policy.optimizer_config.lr=1.0e-6 \
  trainer.algorithm.use_kl_loss=true \
  generator.inference_engine.backend=$INFERENCE_BACKEND \
  generator.inference_engine.run_engines_locally=true \
  generator.inference_engine.weight_sync_backend=nccl \
  generator.inference_engine.async_engine=true \
  generator.batched=true \
  environment.env_class=safety \
  generator.n_samples_per_prompt=8 \
  generator.inference_engine.gpu_memory_utilization=0.8 \
  trainer.logger="$LOGGER" \
  trainer.project_name="xstest-safety" \
  trainer.run_name="wildguard_grpo_xstest" \
  trainer.resume_mode=null \
  trainer.log_path="/tmp/skyrl-logs" \
  trainer.ckpt_path="$HOME/ckpts/wildguard_safety_ckpt" \
  $@
