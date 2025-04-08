# VerboVision - Portuguese Image Captioning Model

VerboVision is a Portuguese image captioning model based on LLaVA (Large Language and Vision Assistant). This project provides tools to train and fine-tune a vision-language model for generating detailed Portuguese captions for images.

## Overview

The training process consists of three main steps:
1. Model initialization using `init_model.py`
2. Initial pretraining using `pretrain.py`
3. Fine-tuning for detailed captions using `train_long.py`

After each training step (pretrain.py and train_long.py), you need to run `merge.py` to apply the LoRA weights to the base model.

## Prerequisites

- Python 3.8+
- CUDA-compatible GPU with at least 16GB VRAM
- PyTorch
- Transformers library
- PEFT (Parameter-Efficient Fine-Tuning)
- Accelerate

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/verbovision.git
cd verbovision
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Training Process

### 1. Model Initialization

First, initialize the base model using `init_model.py`. This script sets up the LLaVA model with Portuguese language capabilities.

```bash
python training/init_model.py
```

This will create the initial model in the `./models/llava_portuguese` directory.

### 2. Initial Pretraining

The second step involves pretraining the model using `pretrain.py`. This phase focuses on basic image captioning capabilities.

```bash
python training/pretrain.py
```

Key features of pretraining:
- Uses LoRA (Low-Rank Adaptation) for efficient training
- Trains on a mix of short and detailed captions
- Freezes the vision tower and language model
- Uses gradient checkpointing for memory efficiency

After pretraining, merge the LoRA weights with the base model:
```bash
python training/merge.py --base_model ./models/llava_portuguese --lora_model ./models/llava-portuguese-lora-3-pochs --output_dir ./models/llava-portuguese-merged
```

### 3. Detailed Caption Training

The final step uses `train_long.py` to fine-tune the model for generating detailed captions.

```bash
python training/train_long.py
```

This phase:
- Uses a specialized dataset for detailed captions
- Maintains the LoRA configuration
- Focuses on generating rich, detailed descriptions
- Uses a cosine learning rate scheduler

After the detailed training, merge the LoRA weights with the base model:
```bash
python training/merge.py --base_model ./models/llava-portuguese-merged --lora_model ./models/llava-llama-portuguese-long-lora-3-epochs --output_dir ./models/llava-portuguese-final
```

## Model Architecture

The model is based on LLaVA and includes:
- Vision encoder (frozen during training)
- Language model (trained with LoRA)
- Multi-modal projector (trained with LoRA)
- Custom Portuguese Language model

## Training Configuration

- Batch size: 1 (pretrain) / 1 (detailed)
- Gradient accumulation steps: 32
- Learning rate: 1e-4
- LoRA rank: 64
- LoRA alpha: 128
- LoRA dropout: 0.05
- Training epochs: 1 (pretrain) / 1 (detailed)

## Output

The trained models are saved in the following directories:
- Base model: `./models/llava_portuguese`
- Pretrained model: `./models/llava-portuguese-lora`
- Detailed caption model: `./models/llava-llama-portuguese-long-lora`

## License

[Add your license information here]

## Citation

If you use this work, please cite:
[Add citation information here]

