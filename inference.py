from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig, StoppingCriteria, StoppingCriteriaList, TextStreamer
import transformers
import torch
from peft import PeftModel


base_model = '/home/llmadmin/models/falcon-7b'
lora_tune = '/home/llmadmin/lawrence/IHOP/loras/IHOP_lora_falcon_7b_v1'


EOF_STRINGS = ["<|END_RESPONSE|>"]


class EndOfFunctionCriteria(StoppingCriteria):
    """Custom `StoppingCriteria` which checks if all generated functions in the batch are completed."""

    def __init__(self, start_length, eof_strings, tokenizer):
        self.start_length = start_length
        self.eof_strings = eof_strings
        self.tokenizer = tokenizer

    def __call__(self, input_ids, scores, **kwargs):
        """Returns true if all generated sequences contain any of the end-of-function strings."""
        decoded_generations = self.tokenizer.batch_decode(input_ids[:, self.start_length :])
        done = []
        for decoded_generation in decoded_generations:
            done.append(any([stop_string in decoded_generation for stop_string in self.eof_strings]))
        return all(done)


async def extract_response(output):
    '''
    extracts the text first  response
    '''
    start = output.rfind('<|RESPONSE|>')+12
    end = output.rfind('<|END_RESPONSE|>')
    first_response = output[start:end]
    return first_response

async def generate_prompt(
    x,
    system_msg = "You are a helpful executive assitant. Your Job is to respond to user inputs with helpful and factual answers. The user may pass context, if context is provided use the context combined with your knowledge to provide answers."
):

    prompt = f"<|SYSTEM|>{system_msg}<|END_SYSTEM|>"
    if x['context'] is not None:
        if len(x['context']) > 0:
            prompt += f"<|CONTEXT|>{x['context']}<|CONTEXT|>"
    if 'message' in x:
        prompt += x['message']
    else:
        prompt += f'<|USER_INPUT|>{x["input"]}<|END_USER_INPUT|>'
        prompt += f'<|RESPONSE|>'
    
    return prompt

async def generate_prompt_v2(
    msg_chain,
    system_msg = "You are a helpful executive assitant. Your Job is to respond to user inputs with helpful and factual answers. The user may pass context, if context is provided use the context combined with your knowledge to provide answers."
):
    prompt = ''
    last_idx = len(msg_chain) - 1
    for i, msg in enumerate(msg_chain):
        if i == 0 and msg['role'] == 'system':
            prompt = f"<|SYSTEM|>{msg['content']}<|END_SYSTEM|>"
        elif i == 0 and msg['role'] != 'system':  
            prompt = f"<|SYSTEM|>{system_msg}<|END_SYSTEM|>"
            
        if msg['role'] == 'context':
            prompt += f"<|CONTEXT|>{msg['content']}<|CONTEXT|>"
        elif msg['role'] == 'user':
            prompt += f"<|USER_INPUT|>{msg['content']}<|END_USER_INPUT|>"
        elif msg['role'] == 'assistant':
            prompt += f"<|RESPONSE|>{msg['content']}<|END_RESPONSE|>"
        
        if i == last_idx:
            prompt += f'<|RESPONSE|>'
    
    return prompt

def generate(model, tokenizer, prompt):
    
    prompt = generate_prompt(prompt)
    
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].cuda()


    generation_output = model.generate(
        input_ids=input_ids,
        stopping_criteria=StoppingCriteriaList([EndOfFunctionCriteria(0, EOF_STRINGS, tokenizer)]), 
        # streamer=streamer, 
        generation_config=GenerationConfig(
            temperature=0.7,
            top_p=0.1,
            do_sample=True,
            top_k=40,
            num_beams=1,
            repition_penalty=1.18,
            encoder_repetion_penalty=1,
            max_new_tokens=512,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id  
        )
    )
    
    # return tokenizer.decode(generation_output[0])
    return extract_response(tokenizer.decode(generation_output[0]))