import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase, TrainerCallback
from datasets import Dataset, IterableDataset
from typing import Optional, Union
import warnings
from trl import GRPOTrainer
from trl.models.utils import unwrap_model_for_generation
from trl.data_utils import is_conversational
from trl.extras.profiling import profiling_context, profiling_decorator
from trl.trainer.utils import selective_log_softmax
from trl.trainer.grpo_trainer import RewardFunc, nanstd, nanmin, nanmax
from accelerate.utils import gather_object
from peft import PeftConfig
from accelerate import logging
import copy

from agrpo_config import AGRPOConfig
from generate import generate

logger = logging.get_logger(__name__)


class AGRPOTrainer(GRPOTrainer):
    def __init__(
        self,
        model: Union[str, PreTrainedModel],
        reward_funcs: Union[RewardFunc, list[RewardFunc]],
        args: Optional[AGRPOConfig] = None,
        train_dataset: Optional[Union[Dataset, IterableDataset]] = None,
        eval_dataset: Optional[Union[Dataset, IterableDataset, dict[str, Union[Dataset, IterableDataset]]]] = None,
        processing_class: Optional[PreTrainedTokenizerBase] = None,
        reward_processing_classes: Optional[Union[PreTrainedTokenizerBase, list[PreTrainedTokenizerBase]]] = None,
        callbacks: Optional[list[TrainerCallback]] = None,
        optimizers: tuple[Optional[torch.optim.Optimizer], Optional[torch.optim.lr_scheduler.LambdaLR]] = (None, None),
        peft_config: Optional[PeftConfig] = None,
    ):
        # This subclass does NOT support eval mode, vllm, FSDP, deepspeed stage 3, VLMs, or pretrained reward models.
        # Also assumes peft is being used (so ref model just disables adapters).
        super().__init__(
            model,
            reward_funcs,
            args,
            train_dataset,
            eval_dataset,
            processing_class,
            reward_processing_classes,
            callbacks,
            optimizers,
            peft_config,
        )
        self.sampling_steps = args.sampling_steps
        self.mc_samples = args.mc_samples

        if args.unmasking not in ["low_confidence", "random"]:
            warnings.warn(f"{args.unmasking} unmasking strategy not recognized. Defaulting to random.")
            args.unmasking = "random"

        num_blocks = args.max_completion_length // args.block_length
        if args.max_completion_length % args.block_length or self.sampling_steps % num_blocks:
            raise ValueError("Sampling steps or completion length doesn't match block length.")

        if self.accelerator.is_main_process:
            print(f"Memory allocated after AGRPOTrainer init: {torch.cuda.memory_allocated()/1e9:.2f} gb")
            self.model.print_trainable_parameters()

    @profiling_decorator
    def _get_per_token_logps_and_entropies(
        self,
        model,
        input_ids,
        attention_mask,
        sampling_masks,
        unmask_positions,
        completion_ids,
    ):
        unmask_logits = model(input_ids, attention_mask=attention_mask).logits[:, -completion_ids.size(1):]
        unmask_logits = unmask_logits[unmask_positions].float()
        unmask_logits /= self.temperature
        unmask_logits[~sampling_masks[unmask_positions]] = -torch.inf
        return selective_log_softmax(unmask_logits, completion_ids[unmask_positions])

    def _generate(self, prompts: list):
        device = self.accelerator.device
        mode = "train" if self.model.training else "eval"

        prompts = copy.deepcopy(prompts)

        if is_conversational({"prompt": prompts[0]}):
            prompt_inputs = self.processing_class.apply_chat_template(
                conversation=prompts,
                add_generation_prompt=True,
                tokenize=True,
                padding=True,
                return_tensors="pt",
                return_dict=True,
                **self.chat_template_kwargs,
            )
        else:
            prompt_inputs = self.processing_class(
                text=prompts, padding=True, padding_side="left", return_tensors="pt"
            )
        prompt_inputs = super(GRPOTrainer, self)._prepare_inputs(prompt_inputs)
        prompt_ids, prompt_mask = prompt_inputs["input_ids"], prompt_inputs["attention_mask"]

        with (
            profiling_context(self, "generate"),
            unwrap_model_for_generation(self.model_wrapped, self.accelerator) as unwrapped_model,
            torch.no_grad(),
        ):
            generation_ids, sampling_masks, unmask_steps, token_logps, token_entropies = generate(
                unwrapped_model,
                prompt_ids,
                self.args,
                attention_mask=prompt_mask,
                mode="default"
            )
        
        completion_ids = generation_ids[:, prompt_ids.size(1):]

        # Mask everything after the last non-EOS token
        is_eos = completion_ids == self.eos_token_id
        sequence_indices = torch.arange(is_eos.size(1), device=device).expand(is_eos.size(0), -1)
        last_non_eos_idx = sequence_indices.masked_fill(is_eos, -1).max(dim=1, keepdim=True)[0]
        completion_mask = sequence_indices <= last_non_eos_idx

        # Log the metrics
        if mode == "train":
            input_tokens = prompt_mask.sum() + completion_mask.sum()
            self.state.num_input_tokens_seen += self.accelerator.gather_for_metrics(input_tokens).sum().item()
        self._metrics[mode]["num_tokens"] = [self.state.num_input_tokens_seen]

        # Log completion lengths, mean, min, max
        agg_completion_lengths = self.accelerator.gather_for_metrics(completion_mask.sum(1))
        self._metrics[mode]["completions/mean_length"].append(agg_completion_lengths.float().mean().item())
        self._metrics[mode]["completions/min_length"].append(agg_completion_lengths.float().min().item())
        self._metrics[mode]["completions/max_length"].append(agg_completion_lengths.float().max().item())

        # Identify sequences that terminated with EOS and log their lengths
        agg_terminated_with_eos = self.accelerator.gather_for_metrics(is_eos.any(dim=1))
        term_completion_lengths = agg_completion_lengths[agg_terminated_with_eos]
        clipped_completions_ratio = 1 - len(term_completion_lengths) / len(agg_completion_lengths)
        self._metrics[mode]["completions/clipped_ratio"].append(clipped_completions_ratio)
        if len(term_completion_lengths) == 0:  # edge case where no completed sequences are found
            term_completion_lengths = torch.zeros(1, device=device)
        self._metrics[mode]["completions/mean_terminated_length"].append(term_completion_lengths.float().mean().item())
        self._metrics[mode]["completions/min_terminated_length"].append(term_completion_lengths.float().min().item())
        self._metrics[mode]["completions/max_terminated_length"].append(term_completion_lengths.float().max().item())

        return (
            prompt_ids,
            prompt_mask,
            completion_ids,
            sampling_masks,
            unmask_steps,
            token_logps,
            token_entropies,
        )
        
    def _generate_and_score_completions(self, inputs):
        device = self.accelerator.device
        mode = "train" if self.model.training else "eval"

        prompts = [x["prompt"] for x in inputs]
        (
            prompt_ids,
            prompt_mask,
            completion_ids,
            sampling_masks,
            unmask_steps,
            logps,
            entropies,
        ) = self._generate(prompts)

        # Decode generated text for reward calculation
        completions = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)
        
        # Looking at special tokens is sometimes useful for debugging
        prompts_text = self.processing_class.batch_decode(prompt_ids, skip_special_tokens=False)
        completions_text = self.processing_class.batch_decode(completion_ids, skip_special_tokens=False)

        rewards_per_func = self._calculate_rewards(inputs, prompts, completions, None)

        # Apply weights to each reward function's output and sum
        rewards = (rewards_per_func * self.reward_weights.to(device).unsqueeze(0)).nansum(dim=1)

        mean_grouped_rewards = rewards.view(-1, self.num_generations).mean(dim=1)
        mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        advantages = rewards - mean_grouped_rewards

        std_rewards = rewards.view(-1, self.num_generations).std(dim=1)
        std_rewards = std_rewards.repeat_interleave(self.num_generations, dim=0)
        is_std_zero = torch.isclose(std_rewards, torch.zeros_like(std_rewards))

        # Slice to keep only the local part of the data
        process_slice = slice(
            self.accelerator.process_index * len(prompts),
            (self.accelerator.process_index + 1) * len(prompts),
        )
        all_process_advantages = advantages.clone()  # keep the aggregated advantages for logging
        advantages = advantages[process_slice]

        # Calculate mean reward per function, but only for samples where the function was applied (non-NaN values)
        for i, reward_func_name in enumerate(self.reward_func_names):
            mean_rewards = torch.nanmean(rewards_per_func[:, i]).item()
            self._metrics[mode][f"rewards/{reward_func_name}/mean"].append(mean_rewards)
            std_func_rewards = nanstd(rewards_per_func[:, i]).item()
            self._metrics[mode][f"rewards/{reward_func_name}/std"].append(std_func_rewards)
        self._metrics[mode]["reward"].append(mean_grouped_rewards.mean().item())
        self._metrics[mode]["reward_std"].append(std_rewards.mean().item())
        self._metrics[mode]["frac_reward_zero_std"].append(is_std_zero.float().mean().item())

        # Log prompt and completion texts
        self._logs["prompt"].extend(gather_object(prompts_text))
        self._logs["completion"].extend(gather_object(completions_text))
        for i, name in enumerate(self.reward_func_names):
            self._logs["rewards"][name].extend(rewards_per_func[:, i].tolist())
        self._logs["advantages"].extend(all_process_advantages.tolist())

        return {
            "prompt_ids": prompt_ids,
            "prompt_mask": prompt_mask,
            "completion_ids": completion_ids,
            "advantages": advantages,
            "sampling_masks": sampling_masks,
            "unmask_steps": unmask_steps,
            "old_per_token_logps": logps,
            "old_per_token_entropies": entropies,
        }

    @profiling_decorator
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if return_outputs:
            raise ValueError("The GRPOTrainer does not support returning outputs")

        prompt_ids = inputs["prompt_ids"]
        completion_ids = inputs["completion_ids"]
        attention_mask = torch.cat((inputs["prompt_mask"], torch.ones_like(completion_ids, dtype=torch.bool)), dim=1)
        advs = inputs["advantages"]
        sampling_masks = inputs["sampling_masks"]
        unmask_steps = inputs["unmask_steps"]
        old_per_token_logps = inputs["old_per_token_logps"]
        old_per_token_entropies = inputs["old_per_token_entropies"]

        device = self.accelerator.device
        B = prompt_ids.size(0)
        C = completion_ids.size(1)
        UM = C // self.sampling_steps

        unmask_advs = advs.repeat_interleave(UM)

        low_clip = torch.zeros(self.mc_samples, device=device)
        high_clip = torch.zeros_like(low_clip)
        clip_ratio = torch.zeros_like(low_clip)
        kls = torch.zeros_like(low_clip)

        non_eos_step = unmask_steps.masked_fill(completion_ids == self.eos_token_id, -1)
        last_step = torch.max(non_eos_step, dim=1, keepdim=True)[0]
        assert (last_step >= 0).all()

        # Low discrepancy sampler from arXiv:2409.02908, appendix G.2
        S = B * self.mc_samples
        u = torch.rand(S, device=device)
        if self.args.low_discrepancy:
            u = (torch.randperm(S, device=device) + u) / S
        u = u.view(B, -1)

        if self.args.importance_sampling == "entropy":
            # Sum token entropies for each timestep
            step_entropies = torch.full((B, self.sampling_steps), 1e-8, device=device)
            step_entropies.scatter_add_(1, unmask_steps, old_per_token_entropies)
            step_indices = torch.arange(self.sampling_steps, device=device).unsqueeze(0)
            step_entropies[step_indices > last_step] = 0.0

            step_probs = step_entropies / step_entropies.sum(dim=1, keepdim=True)
            step_cdf = step_probs.cumsum(dim=1)

            # Inverse CDF transform
            mc_steps = torch.searchsorted(step_cdf, u)
            sampled_probs = step_probs.gather(1, mc_steps)
            importance_weights = (1.0 / (last_step + 1)) / sampled_probs
        else:
            mc_steps = torch.floor(u * (last_step + 1))
            importance_weights = None

        for i in range(self.mc_samples):
            step = mc_steps[:, i].unsqueeze(1)
            unmask_positions = step == unmask_steps
            assert unmask_positions.sum() == B * UM

            completion_ids_at_step = completion_ids.masked_fill(unmask_steps >= step, self.args.mask_token_id)
            input_ids = torch.cat((prompt_ids, completion_ids_at_step), dim=1)
            old_per_token_logps_at_step = old_per_token_logps[unmask_positions]
            
            per_token_logps = self._get_per_token_logps_and_entropies(
                model,
                input_ids,
                attention_mask,
                sampling_masks,
                unmask_positions,
                completion_ids,
            )

            ratio = torch.exp(per_token_logps - old_per_token_logps_at_step)
            
            if self.beta != 0.0:
                with torch.no_grad():
                    with self.accelerator.unwrap_model(self.model).disable_adapter():
                        ref_per_token_logps = self._get_per_token_logps_and_entropies(
                            self.model,
                            input_ids,
                            attention_mask,
                            sampling_masks,
                            unmask_positions,
                            completion_ids,
                        )
                per_token_kl = (
                    torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1
                )
                # DeepSeek-V3.2 importance sampling correction from arXiv:2512.02556
                per_token_kl = per_token_kl * ratio

            ratio_clipped = torch.clip(ratio, 1 - self.epsilon_low, 1 + self.epsilon_high)

            per_token_loss = -torch.min(ratio * unmask_advs, ratio_clipped * unmask_advs)
            if self.beta != 0.0:
                per_token_loss = per_token_loss + self.beta * per_token_kl
                kls[i] = per_token_kl.mean()

            # Compute the clipped probability ratios
            is_low_clipped = (ratio < 1 - self.epsilon_low) & (unmask_advs < 0)
            is_high_clipped = (ratio > 1 + self.epsilon_high) & (unmask_advs > 0)
            low_clip[i] = is_low_clipped.float().mean()
            high_clip[i] = is_high_clipped.float().mean()
            clip_ratio[i] = (is_low_clipped | is_high_clipped).float().mean()
            
            # Apply importance weights if necessary
            if importance_weights is not None:
                iw = importance_weights[:, i].repeat_interleave(UM)
                per_token_loss = per_token_loss * iw

            # Scale loss to the expected magnitude
            loss = per_token_loss.sum() / (self.mc_samples * self.num_generations)
            
            # Leave the last backward for Trainer's built in backward
            if i < self.mc_samples - 1:
                loss = loss / self.current_gradient_accumulation_steps
                self.accelerator.backward(loss, scale_wrt_gas=False)
        
        # Log the metrics
        mode = "train" if self.model.training else "eval"

        if self.beta != 0.0:
            mean_kl = kls.mean()
            self._metrics[mode]["kl"].append(self.accelerator.gather_for_metrics(mean_kl).nanmean().item())

        gathered_low_clip = self.accelerator.gather_for_metrics(low_clip)
        self._metrics[mode]["clip_ratio/low_mean"].append(gathered_low_clip.nanmean().item())
        self._metrics[mode]["clip_ratio/low_min"].append(nanmin(gathered_low_clip).item())
        gathered_high_clip = self.accelerator.gather_for_metrics(high_clip)
        self._metrics[mode]["clip_ratio/high_mean"].append(gathered_high_clip.nanmean().item())
        self._metrics[mode]["clip_ratio/high_max"].append(nanmax(gathered_high_clip).item())
        gathered_clip_ratio = self.accelerator.gather_for_metrics(clip_ratio)
        self._metrics[mode]["clip_ratio/region_mean"].append(gathered_clip_ratio.nanmean().item())

        return loss