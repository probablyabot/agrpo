import random
import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer
from trl.trainer.utils import get_quantization_config

from data import DATASETS
from rewards import (
    thinking_format_reward,
    answer_format_reward,
    correctness_reward,
    countdown_format_reward,
    countdown_correctness_reward,
    sudoku_format_reward,
    sudoku_correctness_reward,
)


def set_random_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_model_and_tokenizer(model_config, device_map=None):
    tokenizer = AutoTokenizer.from_pretrained(
        model_config.model_name_or_path,
        trust_remote_code=model_config.trust_remote_code,
        padding_side="left"
    )

    use_flash_attn = model_config.attn_implementation in {"flash_attention", "flash_attention_2"}
    model = AutoModel.from_pretrained(
        model_config.model_name_or_path,
        device_map=device_map,
        trust_remote_code=model_config.trust_remote_code,
        dtype=model_config.dtype,
        quantization_config=get_quantization_config(model_config),
        **({"flash_attention": True} if use_flash_attn else {})
    )

    return model, tokenizer


def get_dataset(agrpo_config, tokenizer):
    if agrpo_config.dataset not in DATASETS:
        raise ValueError(f"Dataset {agrpo_config.dataset} not registered. Available datasets: {list(DATASETS.keys())}")

    dataset = DATASETS[agrpo_config.dataset]()
    
    if (max_len := agrpo_config.filter_max_prompt_length) is not None:
        orig_len = len(dataset)
        dataset = dataset.filter(
            lambda x: len(tokenizer.apply_chat_template(x["prompt"])) <= max_len
        )
        assert len(dataset) > 0
        print(f"{len(dataset)} / {orig_len} examples left after filtering for prompt length <= {max_len}.")
    
    if agrpo_config.shuffle_dataset:
        dataset = dataset.shuffle(seed=agrpo_config.seed)
    
    return dataset


def get_reward_functions(agrpo_config):
    if agrpo_config.dataset.startswith("countdown"):
        return [countdown_format_reward, countdown_correctness_reward]
    if agrpo_config.dataset.startswith("sudoku"):
        return [sudoku_format_reward, sudoku_correctness_reward]
    return [thinking_format_reward, answer_format_reward, correctness_reward]
