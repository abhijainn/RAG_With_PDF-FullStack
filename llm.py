from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline
import torch

def build_llm(model_name: str):
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token_id is None: tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.float16,
        quantization_config=bnb,
        resume_download=True,
        low_cpu_mem_usage=True,
    )
    return pipeline(
        "text-generation",
        model=model,
        tokenizer=tok,
        max_new_tokens=200,
        temperature=0.2,
        do_sample=False,
        eos_token_id=tok.eos_token_id,
        pad_token_id=tok.pad_token_id,
    )
