import torch
from datasets import load_dataset
from torch.utils.data import Dataset
from transformers import AutoProcessor, TrainingArguments, Trainer, LlavaNextForConditionalGeneration
from PIL import Image
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
        "./models/llava_portuguese",
        attn_implementation="flash_attention_2", #flash_attention_2 
        torch_dtype=torch.bfloat16,
        ignore_mismatched_sizes=True,

    ).to("cuda")
    processor = AutoProcessor.from_pretrained("./models/llava_portuguese")
    processor.patch_size = 14

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
    train_file = "./data/train.csv"
    output_path = "./models/llava-portuguese-lora"
    dataset = load_dataset("csv", data_files=train_file, cache_dir="./cache")["train"]

    def can_load_images(batch):
        valid_indices = []
        for i, filename in enumerate(batch["filename"]):
            try:
                # Attempt to open the image file
                Image.open(filename).convert("RGB")
                valid_indices.append(True)
            except Exception as e:
                print(f"Error loading image {filename}: {e}")
                valid_indices.append(False)
        return valid_indices

    filtered_dataset = dataset.filter(can_load_images, batched=True, batch_size=32, num_proc=6)
    
    
    prompts_curta = [
        "O que há na imagem?",
        "Descreva brevemente a imagem.",
        "O que é isso?",
        "Resuma o conteúdo da foto.",
        "Diga em poucas palavras o que está na imagem.",
        "Dê uma descrição curta e clara da imagem.",
        "Quais são os elementos principais da imagem?",
        "Forneça uma breve explicação do que se vê.",
        "Resuma a cena da foto.",
        "Apresente uma visão geral rápida da imagem."
    ]

    # Prompts para descrições detalhadas
    prompts_detalhada = [
        "Descreva a imagem",
        "Descreva a imagem em detalhes.",
        "Explique detalhadamente o que está acontecendo na foto.",
        "Forneça uma descrição completa da imagem.",
        "Diga tudo o que você pode perceber na imagem.",
        "Analise os elementos visuais e o contexto da imagem.",
        "Descreva a imagem como se estivesse explicando para alguém que não pode vê-la.",
        "Explique o cenário, os objetos e as ações presentes na foto.",
        "Dê uma interpretação detalhada dos elementos visuais da imagem.",
        "Quais detalhes chamam a atenção na imagem? Descreva-os.",
        "Faça uma descrição rica e aprofundada da imagem."
    ]
    
    

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
                image = (Image.open(item["filename"]).convert("RGB"))
            except:
                print(f"Error loading image: {item['filename']}")
                pass

            
            conversation = [
                {"role": "system", "content": ""},
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": random.choice(prompts_curta) if "flickr" in item['filename'] else random.choice(prompts_detalhada) },
                    ],
                },
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": item["caption"]},
                    ],
                }
            ]
            # text_prompt = processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
            text = processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=False)
            
            encoding = self.processor(text=text, images=image, padding=True, truncation=True, max_length=512, return_tensors="pt")
            encoding = {k: v.squeeze() for k, v in encoding.items()}
            encoding["labels"] = encoding["input_ids"]
            return encoding


    # Load the dataset and data collator
    train_dataset = ImageCaptioningDataset(filtered_dataset, processor)


    # Set training arguments
    training_args = TrainingArguments(
        output_dir=output_path,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=32,
        num_train_epochs=1,
        learning_rate=1e-4,
        logging_dir='./logs',
        logging_steps=10,
        # warmup_ratio=0.03,
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