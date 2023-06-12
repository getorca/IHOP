# IHOP (Instruct Humanoid Optimising Protocol)

IHOP is a tool for optimizing and compiling opensource instruct datasets to create better finetunings for large language models. 

## The IHOP format

Messages are sent to the model in the following format:
```md
<|SYSTEM|>You are a helpful executive assitant. Your Job is to respond to user inputs with helpful and factual answers. The user may pass context, if context is provided use the context combined with your knowledge to provide answers.<|END_SYSTEM|>  # The system message defaults to 
<|CONTEXT|>News articles or websites, etc<|CONTEXT|>  # Context is optional
<|USER_INPUT|>The user input<|END_USER_INPUT|>
<|RESPONSE|>  # always end in the opening response tag
```
User `<|USER_INPUT|>` and `<|RESPONSE|>` tags can be chained to continue chat converstations. 

Running server.py with `sanic server --workers=1` workers can be increased to increase throughput, but each worker keeps an instance of the model in vram. The sanic server is also compatible with [OpenAI Chat Completions endpoint]([https://platform.openai.com/docs/api-reference/chat](https://platform.openai.com/docs/api-reference/chat/create)) with added support for the context message. Full support is coming in the future. An example api request can be seen here <https://github.com/getorca/IHOP/blob/main/scripts/test_chat_api.sh> 



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
