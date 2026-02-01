from dataclasses import dataclass, field
from typing import Optional
from trl import GRPOConfig

@dataclass
class AGRPOConfig(GRPOConfig):
    sampling_steps: Optional[int] = field(
        default=128,
        metadata={"help": "Number of sampling steps at generation time."}
    )
    block_length: Optional[int] = field(
        default=16,
        metadata={"help": "Block length for semi-autoregressive generation."}
    )
    unmasking: Optional[str] = field(
        default="low_confidence",
        metadata={"help": "Unmasking strategy for generation."}
    )
    mc_samples: Optional[int] = field(
        default=1,
        metadata={"help": "Number of Monte Carlo samples to use in GRPO objective."}
    )
    low_discrepancy: Optional[bool] = field(
        default=True,
        metadata={"help": "Use low discrepancy sampler to reduce variance of MC estimates."}
    )
    importance_sampling: Optional[str] = field(
        default="uniform",
        metadata={"help": "Importance sampling strategy to use when sampling timesteps."}
    )
    mask_token_id: Optional[int] = field(
        default=126336,
        metadata={"help": "Mask token id; defaults to LLaDA's mask id (126336)."}
    )
    dataset: Optional[str] = field(
        default="gsm_train",
        metadata={"help": "Dataset to use for RL training (see data.py)."}
    )
    filter_max_prompt_length: Optional[int] = field(
        default=None,
        metadata={"help": "Filter out prompts that exceed this length."}
    )
    activation_checkpointing_strategy: Optional[str] = field(
        default=None,
        metadata={
            "help": "LLaDA's gradient checkpointing strategy; see "
            "https://huggingface.co/GSAI-ML/LLaDA-8B-Instruct/blob/main/configuration_llada.py."
        }
    )
