import torch
import torch.nn.functional as F
from trl.trainer.utils import selective_log_softmax


def gumbel_sample(logits, config):
    '''
    Samples from logits (B x L x V) using the Gumbel-max trick with float64 for stability.
    Additionally, keeps track of the masks used for top-p/top-k sampling.
    '''
    mask = torch.ones_like(logits, dtype=torch.bool)
    if config.temperature == 0.0:
        return logits.argmax(dim=-1), mask
    logits /= config.temperature
    if config.top_k:
        remove_mask = logits < logits.topk(config.top_k)[0][..., -1, None]
        logits[remove_mask] = -torch.inf
        mask &= ~remove_mask
    if config.top_p < 1.0:
        sorted_logits, sorted_idx = logits.sort(descending=True)
        sorted_remove_mask = sorted_logits.softmax(dim=-1).cumsum(dim=-1) > config.top_p
        sorted_remove_mask[..., 0] = False
        remove_mask = torch.zeros_like(mask).scatter(-1, sorted_idx, sorted_remove_mask)
        logits[remove_mask] = -torch.inf
        mask &= ~remove_mask
    # use float64 for Gumbel-max, as mentioned in arXiv:2409.02908
    gumbel_noise = -torch.rand_like(logits, dtype=torch.float64).log()
    return (logits.exp() / gumbel_noise).argmax(dim=-1), mask


@torch.no_grad()
def generate(model, input_ids, config, attention_mask=None, mode="default"):
    '''
    dLLM generation based on https://github.com/ML-GSAI/LLaDA/blob/main/generate.py, with modifications for online RL
    (keeping track of top-p/top-k masks, unmasking steps, and token logprobs/entropies). Note this does NOT support
    arbitrary prompting, only left-to-right semi-AR generation (for now).

    `mode` can be one of two values, intended for finer memory control in different (train vs eval) settings:
    - "default": return ids along with sampling masks, unmask_steps, token_logps, and token_entropies
    - "ids_only": only return token ids
    '''
    if config.unmasking not in ["low_confidence", "random", "entropy"]:
        raise ValueError(f"{config.unmasking=} not recognized.")
    if mode not in ["default", "ids_only"]:
        raise ValueError(f"{mode=} not a valid mode for generate().")
    device = model.device
    B, P = input_ids.size()
    C = config.max_completion_length
    BL = config.block_length
    MASK_ID = config.mask_token_id
    EOS_ID = model.config.eos_token_id
    UM = C // config.sampling_steps

    x = torch.cat((input_ids, torch.full((B, C), MASK_ID, dtype=torch.long, device=device)), dim=1)

    if attention_mask is not None:
        attention_mask = torch.cat((attention_mask, torch.ones((B, C), dtype=torch.bool, device=device)), dim=1)
    
    unmask_steps = -torch.ones((B, C), dtype=torch.long, device=device)
    if mode == "default":
        sampling_masks = torch.ones((B, C, model.config.vocab_size), dtype=torch.bool, device=device)
        token_logps = torch.zeros_like(unmask_steps, dtype=torch.float)
        token_entropies = torch.zeros_like(token_logps)

    # performs a single unmasking step on the given block for masked rows only
    def unmask(block, step, row_mask):
        i, j = block * BL, (block + 1) * BL
        attn_mask = attention_mask[row_mask] if attention_mask is not None else None
        logits_block = model(x[row_mask], attention_mask=attn_mask).logits[:, P+i:P+j].float()

        # note gumbel_sample modifies logits (temperature, top-k, top-p)
        sampled_tokens, mask = gumbel_sample(logits_block, config)

        if mode == "default" or config.unmasking == "entropy":
            logps = F.log_softmax(logits_block, dim=-1)
            entropies = -torch.where(logps.isfinite(), logps.exp() * logps, 0.0).sum(dim=-1)
        sampled_logps = selective_log_softmax(logits_block, sampled_tokens)
        
        if config.unmasking == "low_confidence":
            scores = sampled_logps
        elif config.unmasking == "random":
            scores = torch.rand_like(sampled_logps)
        else:
            scores = -entropies

        scores[unmask_steps[row_mask, i:j] != -1] = -torch.inf  # ignore already unmasked tokens
        row_idx = torch.arange(B, device=device).unsqueeze(1)[row_mask]
        col_idx = scores.topk(UM, sorted=False)[1]
        
        x[row_idx, P+i+col_idx] = sampled_tokens.gather(-1, col_idx)
        unmask_steps[row_idx, i+col_idx] = step
        if mode == "default":
            row_range = torch.arange(mask.size(0), device=device).unsqueeze(1)
            sampling_masks[row_idx, i+col_idx] = mask[row_range, col_idx]
            token_logps[row_idx, i+col_idx] = sampled_logps.gather(-1, col_idx)
            token_entropies[row_idx, i+col_idx] = entropies.gather(-1, col_idx)

    step = 0
    removed_rows = torch.zeros(B, dtype=torch.bool, device=device)
    for block in range(C // BL):
        # if the previous block is all eos, we can remove that row from the batch
        prev_eos_rows = ~removed_rows & (x[:, P + (block - 1) * BL : P + block * BL] == EOS_ID).all(dim=1)
        removed_rows |= prev_eos_rows

        # set remaining tokens to eos
        x[prev_eos_rows, P + block * BL:] = EOS_ID
        unmask_steps[prev_eos_rows, block * BL:] = step
        
        if removed_rows.all():
            break

        while step < (block + 1) * BL // UM:
            unmask(block, step, ~removed_rows)
            step += 1

    if mode == "default":
        return x, sampling_masks, unmask_steps, token_logps, token_entropies
    else:
        return x
