from transformers import AutoProcessor, SiglipVisionModel
# pip install accelerate
import torch
import requests
from PIL import Image
from transformers import AutoProcessor, LlavaNextForConditionalGeneration, AutoModelForCausalLM, AutoTokenizer
import torch.nn as nn


model_teacher = LlavaNextForConditionalGeneration.from_pretrained("llava-hf/llama3-llava-next-8b-hf", torch_dtype=torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained("llava-hf/llama3-llava-next-8b-hf")




LLM = AutoModelForCausalLM.from_pretrained("adalbertojunior/Llama-3-8B-Dolphin-Portuguese-v0.3", torch_dtype=torch.bfloat16)
# LLM.resize_token_embeddings(len(tokenizer))
LLM.config.image_token = "<image>"
LLM.config.vocab_size = 128320

lm_head = LLM.lm_head
embed_tokens = LLM.model.embed_tokens
lm_head_teacher = model_teacher.language_model.lm_head
embed_tokens_teacher = model_teacher.language_model.model.embed_tokens

model_teacher.language_model = LLM
model_teacher.config.text_config = LLM.config
model_teacher.language_model.model.embed_tokens = embed_tokens_teacher

new_lm_head = nn.Linear(4096, 128320, bias=False).to(torch.bfloat16)

# Copy weights from model_1's lm_head for the first 128256 tokens
new_lm_head.weight.data[:128256, :] = lm_head.weight.data

# Copy weights from model's lm_head for the remaining 64 tokens
new_lm_head.weight.data[128256:, :] = lm_head_teacher.weight.data[128256:, :]

model_teacher.language_model.lm_head = new_lm_head

processor = AutoProcessor.from_pretrained("llava-hf/llama3-llava-next-8b-hf")



model_teacher.save_pretrained("./models/llava_portuguese")
processor.save_pretrained("./models/llava_portuguese")
tokenizer.save_pretrained("./models/llava_portuguese")

print(model_teacher)

