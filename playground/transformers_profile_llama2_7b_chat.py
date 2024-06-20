import time

import torch
from torch.profiler import profile, record_function, ProfilerActivity
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer

BATCH_SIZE = 16

MODEL = "meta-llama/Llama-2-7b-chat-hf"
# MODEL = "../AutoAWQ/llama-2-7b-chat-hf-awq-gemm"

DEVICE = "cuda"

torch.random.manual_seed(0)


model = AutoModelForCausalLM.from_pretrained(MODEL).to(DEVICE)
model = torch.compile(model)
tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)

prompts = ["[INST]How does `A → B` relate to `¬A ∨ B`?[/INST]"] * BATCH_SIZE

terminators = [
    tokenizer.eos_token_id,
]

tokens = tokenizer(prompts, return_tensors="pt").to(DEVICE)

with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
             on_trace_ready=torch.profiler.tensorboard_trace_handler("./profile"),
             record_shapes=True,
             profile_memory=True,
             with_stack=True) as prof:
    with record_function("model_inference"):
        # Generate output
        tstart = time.perf_counter()
        model.generate(
            **tokens,
            max_new_tokens=512,
            eos_token_id=terminators,
            do_sample=False,
            temperature=None,
            top_p=None,
            num_beams=1,
            use_cache=True,
        )
        tend = time.perf_counter()
        print("Generation: %fs elapsed." % (tend-tstart))
    prof.step()
