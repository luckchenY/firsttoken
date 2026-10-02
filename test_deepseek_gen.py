#!/usr/bin/env python3
"""Check what DeepSeek-R1-Distill actually generates as first tokens.
Using HuggingFace transformers (no ninja needed)."""
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import warnings; warnings.filterwarnings('ignore')

MODEL = '/data/chenyang2/models/DeepSeek-R1-Distill-Qwen-1.5B'

tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL, torch_dtype=torch.bfloat16, device_map='auto'
)
model.eval()

# 1. Prompt 末尾
msgs = [{'role': 'user', 'content': 'What is 1+1?'}]
prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
print("=== Prompt 末尾 80 字符 ===")
print(repr(prompt[-80:]))
print()

# 2. 看 prompt 最后几个 token
prompt_ids = tok(prompt, add_special_tokens=False, return_tensors='pt').input_ids.to(model.device)
print("=== Prompt 最后 6 个 token ===")
for i in range(prompt_ids.shape[1] - 6, prompt_ids.shape[1]):
    tid = prompt_ids[0, i].item()
    print(f"  pos={i}  id={tid}  text={tok.decode([tid])!r}")
print()

# 3. 用模型生成，看前 5 个 token 的 logprobs
with torch.no_grad():
    out = model.generate(
        prompt_ids,
        max_new_tokens=5,
        do_sample=False,
        temperature=0,
        return_dict_in_generate=True,
        output_scores=True,
    )

# 生成的 token ids
gen_ids = out.sequences[0][prompt_ids.shape[1]:]
print("=== 模型生成的前 5 个 token ===")
for i, tid in enumerate(gen_ids.tolist()):
    print(f"  pos={i}  id={tid}  text={tok.decode([tid])!r}")
print()

# 每个位置的 top-5 logprobs
print("=== 每个位置的 top-5 logprobs ===")
for i, scores in enumerate(out.scores):
    if i >= 5:
        break
    logprobs = torch.nn.functional.log_softmax(scores[0], dim=-1)
    topk = logprobs.topk(5)
    print(f"  pos={i}:")
    for j in range(5):
        tid = topk.indices[j].item()
        prob = topk.values[j].exp().item()
        print(f"    {tok.decode([tid])!r:20s}  prob={prob:.4f}")
