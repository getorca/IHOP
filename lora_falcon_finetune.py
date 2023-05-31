from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, DataCollatorForLanguageModeling, TrainingArguments, HfArgumentParser
from datasets import load_dataset
from typing import Dict, Optional, Sequence, List
from dataclasses import dataclass, field
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
# import transformers
import torch
import os
from datetime import datetime as dt


@dataclass
class LoraArguments:
    lora_r: int = field(
        default=2,
        metadata={
            "help": "Lora attention dimensions. Corresponds to the number of parameters. " 
            "The paper demostrates a low rank and for a low rank and adapt more weight adapt more weight matrices."
        }
    )
    lora_alpha: int = field(
        default=4,
        metadata={"help": "he alpha parameter for Lora scaling. usually double rank."}
    )
    lora_dropout: float = field(
        default=0.1,
        metadata={"help": "Dropout for probability for the Lora layer."}
    )
    lora_target_modules: List[str] = field(
        default_factory=lambda: ['q_proj','k_proj','v_proj','o_proj'],
        metadata={"help": "which weight matrices to adapt. The paper argues more matricies with lower ranks."}
    )
    bias: str = field(
        default="none",
        metadata={"help": "Bias type for Lora. Can be 'none', 'all' or 'lora_only'"}
    )
    

@dataclass
class ModelArguments:
    base_model: Optional[str] = field(default="tiiuae/falcon-7b")


@dataclass
class DataArguments:
    data_path: str = field(
        default='datasets/compiled.jsonl', 
        metadata={"help": "Path to the training data."}
    )
    max_samples: int = field(
        default=None,
        metadata={"help": "Limit the max number of training examples."}
        # ToDo: impliment
    )


@dataclass
class TrainingArguments(TrainingArguments):
    output_dir: str = field(
        default=f'finetunes/{str(dt.now())}',
        metadata={"help": "The output directory where the model predictions and checkpoints will be written."}
    )            
    optim: str = field(
        default="adamw_torch"
    )
    model_max_length: int = field(
        default=2048,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    bf16: bool = field(
        default=False,
        metadata={"help": "Whether to use bf16 16-bit (mixed) precision training instead of 32-bit training. Requires Ampere or higher NVIDIA architecture"}, 
    )
    fp16: bool = field(
        default=True,
        metadata={"help": "Whether to use fp16 16-bit (mixed) precision training instead of 32-bit training"},
    )
    local_rank: int = field(default=0) # for DDP
    learning_rate: float = field(default=3e-4)
    per_device_train_batch_size: int = 8
    gradient_accumulation_steps: int = 4,
    warmup_ratio: float = 0.05,
    num_train_epochs: int = 3,
    logging_steps: int = 200,
    evaluation_strategy: str = "steps",
    save_strategy: str = "steps",
    eval_steps: int = 200
    save_steps: int = 200,
    save_total_limit: int = 3,
    load_best_model_at_end: bool = True,
    ddp_find_unused_parameters: bool = False,



system_msg = "You are a helpful executive assitant. Your Job is to respond to user inputs with helpful and factual answers. The user may pass context, if context is provided use the context combined with your knowledge to provide answers."

def preprocess_dataset(x):
    prompt = f"<|SYSTEM|>{system_msg}<|END_SYSTEM|>"
    if 'context' in x:
        if len(x['context']) > 0:
            prompt += f"<|CONTEXT|>{x['context']}<|CONTEXT|>"
    if 'message' in x:
        prompt += x['message']
    else:
        prompt += f'<|USER_INPUT|>{x["input"]}<|END_USER_INPUT|>'
        prompt += f'<|RESPONSE|>{x["response"]}<|END_RESPONSE|>'
    return {
        **x,
        'prompt': prompt
    }



def train():
    parser = HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments, LoraArguments)
    )
    (
        model_args,
        data_args,
        training_args,
        lora_args,
    ) = parser.parse_args_into_dataclasses()
    
    model = AutoModelForCausalLM.from_pretrained(
        model_args.base_model,
        load_in_8bit=True,
        torch_dtype=torch.float16,
        trust_remote_code=True,
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}  # this is for DDP to use 1 GPU per process
    )
    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)
    
    lora_config = LoraConfig(
        r=lora_args.lora_r,
        lora_alpha=lora_args.lora_alpha,
        target_modules=lora_args.lora_target_modules,
        lora_dropout=lora_args.lora_dropout,
        bias=lora_args.bias,
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    
    model.print_trainable_parameters()
    
    tokenizer = AutoTokenizer.from_pretrained(
        model, 
        add_eos_token=True,
    )
    tokenizer.pad_token = tokenizer.eos_token

    dataset = load_dataset(
        "json",
        name='ihop_vector_v1',
        data_files=data_args.data_path, 
        split='train[:]'
    ).train_test_split(test_size=0.05)
        
    prompt_ds = dataset.map(preprocess_dataset)
    ds = prompt_ds.map(
        lambda samples: tokenizer(
            samples['prompt'], 
            padding=True,
            truncation=True,
        ), 
        batched=True
    )
    
    trainer = Trainer(
        model=model, 
        train_dataset=ds['train'],
        eval_dataset=ds['test'],
        args=training_args,        
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False)
    )
    model.config.use_cache = False  # silence the warnings. Please re-enable for inference!# silence the warnings. Please re-enable for inference!

    trainer.train()
    trainer.save_state()

    model.save_pretrained(training_args.output_dir)