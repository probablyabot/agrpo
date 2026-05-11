import torch
from tqdm import tqdm
from trl import TrlParser, ModelConfig
import wandb
from peft import PeftModel
from time import perf_counter
from math_verify import parse, verify
from math_verify.errors import TimeoutException
import re
from accelerate import PartialState
from accelerate.utils import gather_object
from transformers.utils.logging import disable_progress_bar

from utils import set_random_seed, get_model_and_tokenizer, get_dataset
from agrpo_config import AGRPOConfig
from generate import generate
from rewards import verify_equation, parse_ans

dist_state = PartialState()


def grade_completion(comp, example, dataset_name):
    if dataset_name.startswith("countdown"):
        eq_matches = re.findall(r"([^=\n]+=.*)$", comp, re.MULTILINE)
        eq_text = eq_matches[-1] if eq_matches else ""
        return verify_equation(eq_text, example["target"], example["nums"])

    if dataset_name.startswith("sudoku"):
        comp = comp.zfill(16)[:16]
        empty = match = 0
        for i in range(16):
            if example["puzzle"][i] == "0":
                empty += 1
                if example["answer"][i] == comp[i]:
                    match += 1
        return match / empty

    if dataset_name.startswith("gsm"):
        if first_num := re.search(r"\d+", comp):
            comp = first_num.group(0)
    parsed_comp = parse(f"${comp}$")
    parsed_gold = parse(f"${example['answer']}$")
    try:
        return verify(parsed_gold, parsed_comp)
    except TimeoutException:
        print(f"Verification timed out for {comp=} and {example['answer']=}")
        return False


def eval_loop(model, dataset, tokenizer, agrpo_config):
    num_correct = num_examples = total_length = 0

    ans_col = "target" if agrpo_config.dataset.startswith("countdown") else "answer"
    wandb_table = wandb.Table(columns=["prompt", "completion", "true_ans", "is_correct"])

    for i, batch in enumerate(tqdm(dataset, desc="Evaluating...", disable=not dist_state.is_main_process)):
        prompts = batch["prompt"].copy()
        if conversational := isinstance(prompts[0], list):
            prompts = [p[-1]["content"] for p in prompts]
        answers = batch[ans_col].copy()
        B = len(prompts)        
        start_time = perf_counter()
        with dist_state.split_between_processes(batch, apply_padding=True) as local_batch:
            if conversational:
                local_inputs = tokenizer.apply_chat_template(
                    local_batch["prompt"],
                    add_generation_prompt=True,
                    tokenize=True,
                    padding=True,
                    return_tensors="pt",
                    return_dict=True,
                )
            else:
                local_inputs = tokenizer(local_batch["prompt"], padding=True, return_tensors="pt")
            input_ids = local_inputs["input_ids"].to(dist_state.device)
            prompt_mask = local_inputs["attention_mask"].to(dist_state.device)
            
            local_generated_ids = generate(model, input_ids, agrpo_config, attention_mask=prompt_mask, mode="ids_only")

            completion_ids = local_generated_ids[:, input_ids.size(1):]
            lengths = (completion_ids != tokenizer.eos_token_id).sum(dim=1).tolist()
            completions = tokenizer.batch_decode(completion_ids, skip_special_tokens=False)

            # ignore padded examples created by split_between_processes
            if (extras := B % dist_state.num_processes) and dist_state.process_index >= extras:
                completions = completions[:-1]
                lengths = lengths[:-1]

            correct = []
            for j in range(len(completions)):
                comp = parse_ans(completions[j])
                example = {k: v[j] for k, v in batch.items()}
                is_correct = grade_completion(comp, example, agrpo_config.dataset)
                correct.append(is_correct)
        
        # gather results from all processes
        completions = gather_object(completions)
        total_length += sum(gather_object(lengths))
        correct = gather_object(correct)
        num_correct += sum(correct)
        num_examples += B
        assert B == len(completions) == len(correct)
        end_time = perf_counter()

        if dist_state.is_main_process:
            for j in range(B):
                wandb_table.add_data(
                    prompts[j],
                    completions[j],
                    answers[j],
                    correct[j]
                )

            wandb.log({
                "accuracy": (sum(correct) / B) * 100,
                "gen_time_per_response": (end_time - start_time) / B,
                **({"completions": wandb_table} if i == 0 else {})
            })
    
    return {
        "final_accuracy": (num_correct / num_examples) * 100,
        "num_correct": num_correct,
        "num_examples": num_examples,
        "avg_response_length": total_length / num_examples,
        "completions": wandb.Table(wandb_table.columns, wandb_table.data),
    }


def main(agrpo_config, model_config):
    if not dist_state.is_main_process:
        disable_progress_bar()

    set_random_seed(agrpo_config.seed)
    model, tokenizer = get_model_and_tokenizer(model_config, device_map=dist_state.device)

    if agrpo_config.resume_from_checkpoint:
        model = PeftModel.from_pretrained(model, agrpo_config.resume_from_checkpoint)
    model.eval()
    if agrpo_config.torch_compile:
        model = torch.compile(model)

    dataset = get_dataset(agrpo_config, tokenizer)
    dataset = dataset.batch(agrpo_config.per_device_eval_batch_size * dist_state.num_processes)

    if dist_state.is_main_process and "wandb" in agrpo_config.report_to:
        wandb.init(
            project="agrpo-eval",
            config={**vars(agrpo_config), **vars(model_config)},
            name=agrpo_config.run_name
        )
        
    res = eval_loop(model, dataset, tokenizer, agrpo_config)

    if dist_state.is_main_process:
        if "wandb" in agrpo_config.report_to:
            wandb.log(res)
            wandb.finish()
        print("-" * 30 + " Evaluation finished " + "-" * 30)
        print(f"Final accuracy: {res['final_accuracy']:.2f}%")
        print(f"Correct predictions: {res['num_correct']}")
        print(f"Total examples evaluated: {res['num_examples']}")
        print(f"Average response length: {res['avg_response_length']:.2f}")
        print("-" * 81)


if __name__ == "__main__":
    parser = TrlParser((AGRPOConfig, ModelConfig))
    agrpo_config, model_config = parser.parse_args_and_config()
    main(agrpo_config=agrpo_config, model_config=model_config)
