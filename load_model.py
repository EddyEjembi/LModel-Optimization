import torch
import transformers

model_id = "meta-llama/Llama-3.2-1B-Instruct" #Currently using the Llama-3.2-1B-Instruct Model


model = transformers.AutoModelForCausalLM.from_pretrained(model_id,)

tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)

#SAVE MODEL AND TOKENIZER
model.save_pretrained("/LLAMA3.2-1B-QUANT/MODEL", from_pt=True)
tokenizer.save_pretrained("/LLAMA3.2-1B-QUANT/TOKENIZER", from_pt=True)