from trl import TrlParser, ModelConfig
from trl.trainer.utils import get_peft_config
from peft import prepare_model_for_kbit_training

from agrpo_trainer import AGRPOTrainer
from agrpo_config import AGRPOConfig
from utils import set_random_seed, get_model_and_tokenizer, get_dataset, get_reward_functions


def main(agrpo_config, model_config):
    set_random_seed(agrpo_config.seed)

    model, tokenizer = get_model_and_tokenizer(model_config)

    if agrpo_config.activation_checkpointing_strategy:
        model.model.set_activation_checkpointing(agrpo_config.activation_checkpointing_strategy)
    
    if model_config.load_in_4bit:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)

    dataset = get_dataset(agrpo_config, tokenizer)
    reward_functions = get_reward_functions(agrpo_config)

    trainer = AGRPOTrainer(
        model=model,
        reward_funcs=reward_functions,
        args=agrpo_config,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=get_peft_config(model_config),
    )
    
    resume = agrpo_config.resume_from_checkpoint
    if isinstance(resume, str) and resume.lower() == "true":
        resume = True
    trainer.train(resume_from_checkpoint=resume)


if __name__ == "__main__":
    parser = TrlParser((AGRPOConfig, ModelConfig))
    agrpo_config, model_config = parser.parse_args_and_config()
    main(agrpo_config=agrpo_config, model_config=model_config)
