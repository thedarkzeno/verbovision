import torch
import argparse
from peft import PeftModel
from transformers import LlavaNextForConditionalGeneration, AutoProcessor
import os

def parse_args():
    parser = argparse.ArgumentParser(description="Merge LoRA weights with base model")
    parser.add_argument("--base_model", type=str, required=True, help="Path to the base model")
    parser.add_argument("--lora_model", type=str, required=True, help="Path to the LoRA model")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the merged model")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Loading base model from {args.base_model}")
    processor = AutoProcessor.from_pretrained(args.base_model)
    model = LlavaNextForConditionalGeneration.from_pretrained(
        args.base_model, 
        torch_dtype=torch.bfloat16, 
        trust_remote_code=True
    )

    print(f"Loading LoRA weights from {args.lora_model}")
    model = PeftModel.from_pretrained(
        model, 
        args.lora_model, 
        torch_dtype=torch.bfloat16,
    )

    print("Merging LoRA weights with base model...")
    model = model.merge_and_unload()

    print(f"Saving merged model to {args.output_dir}")
    model.save_pretrained(args.output_dir)
    processor.save_pretrained(args.output_dir)
    
    print("Done!")

if __name__ == "__main__":
    main()

