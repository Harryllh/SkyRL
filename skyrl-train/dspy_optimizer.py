from examples.dspy.hover.programs import Hover_query_gen
import dspy
import arbor
from arbor import ArborGRPO, ArborProvider
from examples.dspy.hover.data import hover_data
from examples.dspy.hover.utils import hover_query_reward_fn, hover_final_reward_fn

# Start Arbor server
arbor_server_info = arbor.init()
provider = ArborProvider()

local_lm_name = "Qwen/Qwen2.5-1.5B-Instruct"
local_lm = dspy.LM(
    model=f"openai/arbor:{local_lm_name}",
    provider=provider,
    api_base=arbor_server_info["base_url"],
    api_key="arbor",
    max_tokens=4096,
    temperature=0.6,
    top_p=0.95,
    top_k=-1,
    repetition_penalty=1.0,
)

program = Hover_query_gen()
program.set_lm(local_lm)
trainset, devset = hover_data()

train_kwargs = {
    "per_device_train_batch_size": 1,
    "gradient_accumulation_steps": 8,
    "temperature": 0.6,
    "top_k": -1,
    "top_p": 0.95,
    "repetition_penalty": 1.0,
    "beta": 0.00,
    "learning_rate": 1e-6,
    "gradient_checkpointing": True,
    "bf16": True,
    "lr_scheduler_type": "constant_with_warmup",
    "loss_type": "dapo",
    "max_steps": 1000,
    "report_to": "wandb",
    "log_completions": True,
    "logging_steps": 1,
    "max_prompt_length": None,
    "max_completion_length": None,
    "scale_rewards": False,
    "max_grad_norm": 1.0,
    "num_training_gpus": 1,
    "num_inference_gpus": 1,
    "weight_decay": 0.001,
}

train_kwargs["max_seq_len"] = 4096

compiler = ArborGRPO(
    metric=hover_final_reward_fn,
    num_dspy_examples_per_grpo_step=64,
    num_rollouts_per_grpo_step=8,
    exclude_demos=True,
    num_train_steps=1000,
    num_threads=16,
    use_train_as_val=False,
    num_steps_for_val=50,
    train_kwargs=train_kwargs,
    checkpoint="single-best",
)

optimized_program = compiler.compile(
    student=program,
    trainset=trainset,
    valset=trainset,
)

