# IHOP (Instruct Humanoid Optimising Protocol)

IHOP is a tool for optimizing and compiling opensource instruct datasets to create better finetunings for large language models. 


## Run training

Run the training with accelerate and deepspeed <https://huggingface.co/docs/accelerate/main/en/usage_guides/deepspeed#deepspeed>

1 - run `accelerate config` answer the options for your training system
2 - run training with `accelerate launch lora_falcon_finetune.py --output_dir loras/IHOP_lora_falcon_7b_v1`