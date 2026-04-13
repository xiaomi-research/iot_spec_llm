echo $MODEL_PATH
echo $TRAIN_OUTPUT_PATH
echo $DATA_PATH
# echo $MLFLOW_EXP_ID
# echo $MLFLOW_RUN_ID
# echo $MLFLOW_TRACKING_URI


export WANDB_API_KEY="YOUR API KEY"
export WANDB_MODE="online"
export VLLM_USE_DEEP_GEMM=0
export VLLM_ALLREDUCE_USE_SYMM_MEM=0
# NCCL timeout settings for large MOE models
export NCCL_TIMEOUT=1800  # 60 minutes (increased for large MOE models)
export NCCL_DEBUG=WARN
export NCCL_IB_DISABLE=1  # Disable InfiniBand if not available
# NCCL_SOCKET_IFNAME: Leave unset for auto-detection, or set to specific interface if needed
export NCCL_P2P_DISABLE=0  # Enable P2P if available
export NCCL_SHM_DISABLE=0  # Enable shared memory
export NCCL_ASYNC_ERROR_HANDLING=1  # Enable async error handling
MASTER_ADDR=${MASTER_ADDR:-localhost}
MASTER_PORT=${MASTER_PORT:-6555}
GPUS_PER_NODE=${NPROC_PER_NODE:-8}
WORLD_SIZE=${WORLD_SIZE:-1}
RANK=${RANK:-0}
set -ex

ENGINE=${1:-vllm}  # Use vLLM rollout (LoRA disabled due to MoE incompatibility)
BACKEND=${2:-fsdp2}

offload_cpu=True  # Enable CPU offload for MOE model to reduce memory pressure
nodes=1
gpus_per_node=8
total_epochs=1
model_name="Qwen3-MOE"
dataset="skill"
model_path="${MODEL_PATH%/}"  # 去除末尾的/
# model_path="/mnt/hewuchun/workbench_output/checkpoints/finetune_23061_211721"
training_dataset="${DATA_PATH}"
val_dataset="${DATA_PATH}"

project="VeRL_GSPO_Qwen3-MOE"
timestamp=$(date +"%Y%m%d_%H%M%S")
experiment_set="${model_name}_${dataset}_${nodes}node_${gpus_per_node}gpu-${total_epochs}epoch-${timestamp}"
local_dir="$TRAIN_OUTPUT_PATH"

TRAIN_BATCH_SIZE=2  # Minimized for MOE model to avoid OOM

# data.* 参数配置
data_params=(
data.train_files=${training_dataset}
data.val_files=${val_dataset}
data.train_batch_size=${TRAIN_BATCH_SIZE}
data.max_prompt_length=2000
data.max_response_length=3000
data.filter_overlong_prompts=True
data.truncation='error'
data.prompt_key=prompt
# data.image_key=images
)

# model.* 参数配置 (MOE support)
# NOTE: LoRA disabled for vLLM rollout due to MoE incompatibility
# vLLM does not support LoRA with MoE models. We rely on other memory optimizations instead.
model_params=(
actor_rollout_ref.model.path=${model_path}
actor_rollout_ref.model.use_remove_padding=True
actor_rollout_ref.model.lora_rank=0  # Disabled: vLLM LoRA not compatible with MoE
# Note: Model dtype is set via fsdp_config.model_dtype, not here
# MOE router configuration (if using Megatron, uncomment below)
# actor_rollout_ref.model.override_config.moe_config.freeze_moe_router=False
)

# trainer.* 参数配置
trainer_params=(
trainer.logger="['console','wandb']"
trainer.project_name="${project}"
trainer.experiment_name="${experiment_set}"
trainer.default_local_dir=${local_dir}
trainer.n_gpus_per_node=${gpus_per_node}
trainer.nnodes=${nodes}
trainer.save_freq=10
trainer.test_freq=-1
trainer.total_epochs=${total_epochs}
trainer.resume_mode=auto
# Increase NCCL timeout for large MOE models
actor_rollout_ref.nccl_timeout=1800
)

# PPO actor.* 参数配置 (GSPO specific)
# Note: When use_dynamic_bsz=False, micro batch count = ppo_mini_batch_size / ppo_micro_batch_size_per_gpu
# Minimized batch sizes for MOE model to avoid OOM
ppo_actor_params=(
actor_rollout_ref.actor.ppo_mini_batch_size=2
actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1
actor_rollout_ref.actor.optim.lr=1e-5
actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.05
actor_rollout_ref.actor.optim.weight_decay=0.1
actor_rollout_ref.actor.entropy_coeff=0
actor_rollout_ref.actor.grad_clip=1.0
actor_rollout_ref.actor.use_kl_loss=False
actor_rollout_ref.actor.clip_ratio_low=0.0003
actor_rollout_ref.actor.clip_ratio_high=0.0004
actor_rollout_ref.actor.policy_loss.loss_mode=gspo
actor_rollout_ref.actor.loss_agg_mode="seq-mean-token-mean"
actor_rollout_ref.actor.use_dynamic_bsz=False
actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$(((2000 + 3000) * 1))
actor_rollout_ref.actor.fsdp_config.param_offload=${offload_cpu}
actor_rollout_ref.actor.fsdp_config.optimizer_offload=${offload_cpu}
actor_rollout_ref.actor.fsdp_config.model_dtype=fp16  # Use fp16 to save more memory than bfloat16
actor_rollout_ref.actor.fsdp_config.fsdp_size=-1
actor_rollout_ref.actor.strategy=${BACKEND}
actor_rollout_ref.actor.checkpoint.save_contents="['hf_model']"
actor_rollout_ref.model.enable_gradient_checkpointing=True  # Enable gradient checkpointing for LoRA training to save memory
)

# rollout.* 参数配置 (MOE support)
# For MOE models, set expert_parallel_size based on your model's expert configuration
# expert_parallel_size should match: tensor_model_parallel_size * data_parallel_size
# Example: if TP=4, DP=1, then EP=4; if TP=2, DP=2, then EP=4
rollout_expert_parallel_size=4  # Adjust based on your MOE model and GPU setup
rollout_data_parallel_size=1    # Adjust based on your setup

rollout_params=(
actor_rollout_ref.rollout.dtype=half  # Use half (float16) to save more memory - vLLM requires 'half' or 'float16', not 'fp16'
actor_rollout_ref.rollout.n=4  # Must satisfy: train_batch_size (2) * rollout.n % n_gpus (8) == 0, so n=4
actor_rollout_ref.rollout.gpu_memory_utilization=0.4  # Further reduced to save memory
actor_rollout_ref.rollout.tensor_model_parallel_size=4
actor_rollout_ref.rollout.data_parallel_size=${rollout_data_parallel_size}
actor_rollout_ref.rollout.expert_parallel_size=${rollout_expert_parallel_size}
actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2  # Minimized to save memory
actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=False
actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=$(((2000 + 3000) * 1))
actor_rollout_ref.rollout.max_model_len=5000
actor_rollout_ref.rollout.max_num_batched_tokens=5000
actor_rollout_ref.rollout.enable_chunked_prefill=True
actor_rollout_ref.rollout.temperature=1.0
actor_rollout_ref.rollout.top_p=1.0
actor_rollout_ref.rollout.top_k=-1
actor_rollout_ref.rollout.mode=sync
actor_rollout_ref.rollout.val_kwargs.do_sample=True
actor_rollout_ref.rollout.val_kwargs.n=1
actor_rollout_ref.rollout.val_kwargs.top_p=0.7
actor_rollout_ref.rollout.val_kwargs.top_k=-1
actor_rollout_ref.rollout.val_kwargs.temperature=1.0
actor_rollout_ref.rollout.name=${ENGINE}
)

# ref.* 参数配置
ref_model_params=(
actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2  # Minimized to save memory
actor_rollout_ref.ref.log_prob_use_dynamic_bsz=False
actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=$(((2000 + 3000) * 1))
actor_rollout_ref.ref.fsdp_config.param_offload=${offload_cpu}
actor_rollout_ref.ref.fsdp_config.model_dtype=fp16  # Use fp16 to save more memory
actor_rollout_ref.ref.strategy=${BACKEND}
)

# algorithm.* 参数配置
algorithm_params=(
algorithm.adv_estimator=grpo
algorithm.use_kl_in_reward=False
algorithm.kl_ctrl.kl_coef=0.0
)

# reward_model.* 参数配置
reward_model_params=(
reward_model.reward_manager=dapo
+reward_model.reward_kwargs.overlong_buffer_cfg.enable=True
+reward_model.reward_kwargs.overlong_buffer_cfg.len=500
+reward_model.reward_kwargs.overlong_buffer_cfg.penalty_factor=1.0
+reward_model.reward_kwargs.overlong_buffer_cfg.log=False
+reward_model.reward_kwargs.max_resp_len=2000  # Reduced to match max_response_length
)


pip install wandb nltk
if [ $RANK == 0 ]; then
    ray start --head --node-ip-address=0.0.0.0 --num-gpus $GPUS_PER_NODE
    # sleep 60
    ray job submit --address="http://127.0.0.1:8265"  \
        --runtime-env=verl/trainer/runtime_env.yaml \
        -- python3 -m verl.trainer.main_ppo \
            "${data_params[@]}" \
            "${model_params[@]}" \
            "${trainer_params[@]}" \
            "${ppo_actor_params[@]}" \
            "${rollout_params[@]}" \
            "${ref_model_params[@]}" \
            "${algorithm_params[@]}" \
            "${reward_model_params[@]}" \
            "$@"
else
   ray start --address "$MASTER_ADDR:6379" --num-gpus $GPUS_PER_NODE --block 
fi
