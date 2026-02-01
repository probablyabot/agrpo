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
from accelerate.utils import gather

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

    if dist_state.is_main_process:
        wandb.init( 
            project="agrpo-eval",
            config={**vars(agrpo_config), **vars(model_config)},
            name=agrpo_config.run_name
        )
        wandb_table = wandb.Table(columns=["prompt", "completion", "true_ans", "is_correct"])

    for batch in tqdm(dataset, desc="Evaluating...", disable=not dist_state.is_main_process):
        prompts = tokenizer.apply_chat_template(
            batch["prompt"],
            add_generation_prompt=True,
            tokenize=False
        )
        prompt_inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            padding_side="left",
            add_special_tokens=False
        )
        B, P = prompt_inputs["input_ids"].size()
        
        start_time = perf_counter()
        with dist_state.split_between_processes(prompt_inputs, apply_padding=True) as local_inputs:
            input_ids = local_inputs["input_ids"].to(dist_state.device)
            prompt_mask = local_inputs["attention_mask"].to(dist_state.device)
            if model.config.model_type == "Dream":
                local_generated_ids = model.diffusion_generate(
                    input_ids,
                    attention_mask=prompt_mask,
                    max_new_tokens=agrpo_config.max_completion_length,
                    steps=agrpo_config.sampling_steps,
                    temperature=agrpo_config.temperature,
                    alg="entropy"
                )
            else:
                local_generated_ids = generate(model, input_ids, agrpo_config, attention_mask=prompt_mask, mode="ids_only")
        generated_ids = gather(local_generated_ids)[:B]
        end_time = perf_counter()

        if dist_state.is_main_process:
            completion_ids = generated_ids[:, P:]
            response_length = (completion_ids != tokenizer.eos_token_id).sum()
            total_length += response_length
            completions = tokenizer.batch_decode(completion_ids, skip_special_tokens=False)

            correct = 0
            for j in range(B):
                comp = parse_ans(completions[j])
                example = {k: v[j] for k, v in batch.items()}
                is_correct = grade_completion(comp, example, agrpo_config.dataset)
                correct += is_correct

                wandb_table.add_data(
                    prompts[j],
                    completions[j],
                    example["answer"] if not agrpo_config.dataset.startswith("countdown") else example["target"],
                    is_correct
                )

            wandb.log({
                "accuracy": (correct / B) * 100,
                "completions": wandb.Table(columns=wandb_table.columns, data=wandb_table.data),
                "generate_time": end_time - start_time,
            })
            num_correct += correct
            num_examples += B
    
    if dist_state.is_main_process:
        return_dict = {
            "final_accuracy": (num_correct / num_examples) * 100,
            "num_correct": num_correct,
            "num_examples": num_examples,
            "avg_response_length": total_length / num_examples
        }
        wandb.log(return_dict)
        wandb.finish()
        return return_dict


def main(agrpo_config, model_config):
    set_random_seed(agrpo_config.seed)
    model, tokenizer = get_model_and_tokenizer(model_config, device_map=dist_state.device)

    dataset = get_dataset(agrpo_config, tokenizer)
    
    # this is needed so .batch() doesn't use outdated sysprompts
    from datasets import disable_caching
    disable_caching()

    dataset = dataset.batch(agrpo_config.per_device_eval_batch_size * dist_state.num_processes)
    if agrpo_config.resume_from_checkpoint:
        model = PeftModel.from_pretrained(model, agrpo_config.resume_from_checkpoint)
    model.eval()

    if agrpo_config.torch_compile:
        model = torch.compile(model)
        
    res = eval_loop(model, dataset, tokenizer, agrpo_config)

    if dist_state.is_main_process:
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
