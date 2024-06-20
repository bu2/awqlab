import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer

device = "cuda"

torch.random.manual_seed(0)

model_path = "../AutoAWQ/llama-2-7b-chat-hf-awq-gemv"

model = AutoModelForCausalLM.from_pretrained(model_path).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

prompt = "[INST]How does `A → B` relate to `¬A ∨ B`?[/INST]"

chat = [
    {"role": "user", "content": prompt},
]

terminators = [
    tokenizer.eos_token_id,
]

tokens = tokenizer.apply_chat_template(
    chat,
    return_tensors="pt"
)
tokens = tokens.to(device)

# Generate output
tstart = time.perf_counter()
generation_output = model.generate(
    tokens,
    streamer=streamer,
    max_new_tokens=512,
    eos_token_id=terminators
)
tend = time.perf_counter()
print("Generation: %fs elapsed." % (tend-tstart))
