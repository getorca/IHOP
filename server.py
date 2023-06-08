from sanic import Sanic
from sanic.response import text
from inference import  EndOfFunctionCriteria, extract_response, generate_prompt, generate_prompt_v2
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig, StoppingCriteria, StoppingCriteriaList, TextStreamer
import transformers
import torch
from peft import PeftModel
from dataclasses import dataclass, asdict, field
from enum import Enum
from sanic_ext import validate



base_model = '/home/llmadmin/models/falcon-7b'
lora_tune = '/home/llmadmin/lawrence/IHOP/loras/IHOP_lora_falcon_7b_v1'
EOF_STRINGS = ["<|END_RESPONSE|>"]


app = Sanic("IhopAPI")

@dataclass
class MessageRoles(Enum):
    system = 'system'
    user = 'user'
    assistant = 'assistant'
    context = 'context'
    
@dataclass
class Message:
    role: str
    content: str
    
    def __post_init__(self):
        # validate role
        if self.role not in MessageRoles.__members__:
            raise ValueError(f"Invalid role. Must be one of: " + ", ".join(MessageRoles.__members__))
    
@dataclass
class ChatRequest:
    messages: list[Message]
    max_tokens: int = 512


@app.listener("before_server_start")
def load_model(app, loop):
    print('LOADING MODEL')
    
    app.ctx.tokenizer = AutoTokenizer.from_pretrained(
        base_model,
        add_eos_token=True,
    )
    app.ctx.tokenizer.pad_token = app.ctx.tokenizer.eos_token

    app.ctx.model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map = 'auto' 
    )

    app.ctx.model = PeftModel.from_pretrained(
        app.ctx.model,
        lora_tune,
        torch_dtype=torch.bfloat16,
    )
    





# @app.listener("main_process_start")
# async def listener_1(app, loop):
#     app.ctx.model, app.ctx.tokenizer = await load_model()
#     print("listener_0")

# @app.listener("before_server_start")
# async def listener_1(app, loop):
#     print("listener_1")
        
# app.add_task(infer_generate, name='infer_generate')
    
@app.post("/chat")
@validate(json=ChatRequest)
async def chat_request(request, body: ChatRequest):
    
    request_obj = asdict(body)
    
    prompt = await generate_prompt_v2(msg_chain=request_obj['messages'])
    
    inputs = app.ctx.tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].cuda()


    generation_output = app.ctx.model.generate(
        input_ids=input_ids,
        stopping_criteria=StoppingCriteriaList([EndOfFunctionCriteria(len(prompt), EOF_STRINGS, app.ctx.tokenizer)]), 
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
            eos_token_id=app.ctx.tokenizer.eos_token_id,
            pad_token_id=app.ctx.tokenizer.eos_token_id  
        )
    )
    response = await extract_response(app.ctx.tokenizer.decode(generation_output[0]))
    return text(response)