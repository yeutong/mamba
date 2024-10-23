import torch
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
model = MambaLMHeadModel.from_pretrained('state-spaces/mamba-130m', device='cuda', dtype=torch.float16)

input_ids = torch.randint(1, 1000, (16, 100), dtype=torch.long, device="cuda")
attn_mask = torch.ones_like(input_ids, dtype=torch.long, device="cuda")

max_length = 100
temperature = 1
topk = 1
topp = 1
repetition_penalty = 1

fn = lambda: model.generate(
        input_ids=input_ids,
        attention_mask=attn_mask,
        max_length=max_length,
        return_dict_in_generate=True,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=True,
        temperature=temperature,
        top_k=topk,
        top_p=topp,
        repetition_penalty=repetition_penalty,
    )

print(fn())
