import torch
from datasets import load_dataset
from torch.utils.data import Dataset
from transformers import AutoProcessor, TrainingArguments, Trainer, LlavaNextForConditionalGeneration
from peft import LoraConfig, get_peft_model
import random

def find_all_linear_names(model):
    cls = torch.nn.Linear
    lora_module_names = set()
    multimodal_keywords = ["language_model", "multi_modal_projector"]
    for name, module in model.named_modules():
        if not any(mm_keyword in name for mm_keyword in multimodal_keywords):
            continue
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(name)

    if 'lm_head' in lora_module_names: # needed for 16-bit
        lora_module_names.remove('lm_head')
    print(list(lora_module_names))
    return list(lora_module_names)

# Load the model and processor
def main():
    model = LlavaNextForConditionalGeneration.from_pretrained(
        "VerboVision/llava-portuguese-base-v2",
        attn_implementation="flash_attention_2", #flash_attention_2 
        torch_dtype=torch.bfloat16,
        ignore_mismatched_sizes=True,
    )#.to("cuda")
    processor = AutoProcessor.from_pretrained("VerboVision/llava-portuguese-base-v2")

    model.vision_tower.requires_grad_(False)
    model.language_model.requires_grad_(False)

    # Let's define the LoraConfig
    config = LoraConfig(
        r=64,
        lora_alpha=128,
        lora_dropout=0.05,
        bias="none",
        target_modules=find_all_linear_names(model)#"all-linear"
    )




    print(model)
    # Get our peft model and print the number of trainable parameters

    model = get_peft_model(model, config)

    model.gradient_checkpointing_enable()

    # Load the dataset
    # Load the dataset from Hugging Face Hub
    dataset_name = "adalbertojunior/detailed-image-prompts-portuguese-2"  # Example dataset, replace with the desired dataset
    dataset = load_dataset(dataset_name, split="train", cache_dir="./cache")
    
    output_path = "./models/llava-llama-portuguese-long-lora-3-epochs"



    # Define the ImageCaptioningDataset class
    class ImageCaptioningDataset(Dataset):
        def __init__(self, dataset, processor):
            self.dataset = dataset
            self.processor = processor

        def __len__(self):
            return len(self.dataset)

        def __getitem__(self, idx):
            item = self.dataset[idx]
            try:
                image = ((item["image"]).convert("RGB"))
            except:
                print(f"Error loading image")
                pass

            conversation = [
                {"role": "system", "content": ""},
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": item["instruction"]},
                    ],
                },
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": item["description"]},
                    ],
                }
            ]
            # text_prompt = processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
            text = processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=False)
            
            encoding = self.processor(text=text, images=image, padding=True, truncation=True, max_length=256, return_tensors="pt")
            encoding = {k: v.squeeze() for k, v in encoding.items()}
            encoding["labels"] = encoding["input_ids"]
            return encoding

    # Define the data collator
    def collator(batch):
        processed_batch = {}
        for key in batch[0].keys():
            if key != "text":
                processed_batch[key] = torch.stack([example[key] for example in batch])
            else:
                text_inputs = processor.tokenizer(
                    [example["text"] for example in batch], padding=True, return_tensors="pt"
                )
                processed_batch["input_ids"] = text_inputs["input_ids"]
                processed_batch["labels"] = text_inputs["input_ids"]
                processed_batch["attention_mask"] = text_inputs["attention_mask"]
        return processed_batch

    # Load the dataset and data collator
    train_dataset = ImageCaptioningDataset(dataset, processor)


    # Set training arguments
    training_args = TrainingArguments(
        output_dir=output_path,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=32,
        num_train_epochs=3,
        learning_rate=1e-4,
        logging_dir='./logs',
        logging_steps=10,
        lr_scheduler_type="cosine",
        save_steps=1000,
        save_total_limit=3,
        bf16=True,
        evaluation_strategy="no",
        warmup_ratio=0.1,
        gradient_checkpointing=True,
    )

    # Define the trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
    )

    # Train the model
    trainer.train()

    # Save the model and processor
    model.save_pretrained(output_path)
    processor.save_pretrained(output_path)

if __name__ == "__main__":
    main()