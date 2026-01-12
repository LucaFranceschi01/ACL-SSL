from transformers import AutoTokenizer, CLIPSegForImageSegmentation
import torch

model_name = "CIDAS/clipseg-rd64-refined"

# Downloads model + tokenizer into the Hugging Face cache (~/.cache/huggingface)
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
model = CLIPSegForImageSegmentation.from_pretrained(model_name)

# sd = model.state_dict()
# print(sd)
# Save them into a directory
save_dir = "pretrain/clipseg-rd64-refined-local"
tokenizer.save_pretrained(save_dir)

# model.save_pretrained(save_dir)
torch.save(model.state_dict(), f"{save_dir}/pytorch_model.bin") # save weights as older versions

model.config.save_pretrained(save_dir)