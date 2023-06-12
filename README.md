# IHOP (Instruct Humanoid Optimising Protocol)

IHOP is a tool for optimizing and compiling opensource instruct datasets to create better finetunings for large language models. 

## The IHOP format


## IHOP Models
| Model | Base Model | Dataset | Description | Training |
|-------|------------|---------| -------- | ---- |
| [IHOPv01_Falcon7b_lora](https://huggingface.co/winddude/IHOPv01_Falcon7b_lora) | Falcon 7b | [IHOPv01](https://huggingface.co/datasets/winddude/IHOPv01) | An experimental general purpose instruct following chat model | This repo |
| [pb_lora_7b_v0.1](https://huggingface.co/winddude/pb_lora_7b_v0.1) | llama 7b | [reddit_finance_43_250k](https://huggingface.co/datasets/winddude/reddit_finance_43_250k) | An experimental model trained to reply to various finance, crypto and investing subreddits | [Training](https://github.com/getorca/ProfitsBot_V0_OLLM/blob/main/training) |

## IHOP Datasets

| Dataset | Size | Source | Description | Recreation |
|---------|------|--------|-------------|------------|
| [IHOPv01](https://huggingface.co/datasets/winddude/IHOPv01) | 67.4k | composite | A collecton of instruct datasets, curated to not reject instructions | [Building](https://github.com/getorca/IHOP/tree/main/notebooks) |

## Run training

Run the training with accelerate and deepspeed <https://huggingface.co/docs/accelerate/main/en/usage_guides/deepspeed#deepspeed>

1 - run `accelerate config` answer the options for your training system
2 - run training with `accelerate launch lora_falcon_finetune.py --output_dir loras/IHOP_lora_falcon_7b_v1`
