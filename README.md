# GPT-2 Dialogue Fine-tuning with LoRA

A project for fine-tuning GPT-2-XL model on dialogue data using LoRA (Low-Rank Adaptation) technique to create a conversational chatbot.

##  Overview

This project fine-tunes a GPT-2-XL model on the Daily Dialog dataset to generate natural conversations. It leverages LoRA for efficient fine-tuning with minimal computational resources and 8-bit quantization for memory optimization.

##  Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Train the Model (RUN THIS FIRST!)

```bash
python Fine_tuningmodel.py
```

 **Important**: You must run `Fine_tuningmodel.py` first to train the model. This process may take several hours depending on your hardware.

### 3. Run Interactive Chatbot

After training completes:

```bash
python main.py
```

##  Project Structure

```
├── Fine_tuningmodel.py    # Model training script (run first!)
├── main.py               # Interactive chatbot interface
├── requirements.txt      # Project dependencies
├── gpt2-dialogue-lora/   # LoRA adapters folder (auto-created)
└── gpt2-dialogue-full/   # Final merged model folder (auto-created)
```

##  Technical Details

### Fine_tuningmodel.py
- Downloads and loads Daily Dialog dataset
- Uses GPT-2-XL as the base model
- Applies 8-bit quantization for memory efficiency
- Configures LoRA adapters for efficient training
- Trains for 3 epochs with gradient checkpointing
- Saves both LoRA adapters and merged full model

### main.py
- Loads the fine-tuned model
- Implements advanced text generation with:
  - Temperature sampling
  - Top-k filtering  
  - Top-p (nucleus) sampling
- Provides interactive chat interface with real-time generation

##  Training Configuration

- **Base Model**: GPT-2-XL (1.5B parameters)
- **LoRA Settings**:
  - r=8 (adapter rank)
  - alpha=32
  - dropout=0.05
  - target_modules: ['c_attn', 'c_proj']
- **Training**: 3 epochs, batch size 8, learning rate 3e-4
- **Quantization**: 8-bit for memory optimization

##  Using the Chatbot

After running `main.py`:

1. Type your message and press Enter
2. The model generates responses in real-time
3. Type `exit` to quit the conversation

Example conversation:
```
USER: Hello, how are you doing today?
BOT: I'm doing great, thank you for asking! How can I help you?

USER: Can we talk about movies?
BOT: Of course! I'd love to discuss movies with you. What's your favorite genre?
```

##  Features

- **Memory Efficient**: 8-bit quantization and LoRA reduce memory requirements
- **High-Quality Generation**: Multiple sampling techniques for natural responses
- **Real-time Interaction**: Live text generation with progress visualization
- **Customizable**: Easy-to-modify generation parameters
- **Production Ready**: Merged model format for deployment

##  Model Outputs

The training process creates two model formats:

1. **LoRA Adapters** (`gpt2-dialogue-lora/`): Lightweight adapters for development
2. **Full Model** (`gpt2-dialogue-full/`): Complete merged model for production use

##  System Requirements

- **GPU**: NVIDIA GPU with minimum 8GB VRAM recommended
- **RAM**: Minimum 16GB system memory
- **Storage**: ~10GB for models and dataset cache
- **Python**: 3.8 or higher
- **CUDA**: Compatible CUDA installation for GPU acceleration

##  Performance

- **Training Time**: ~2-4 hours on RTX 3080/4080
- **Memory Usage**: ~6-8GB VRAM during training
- **Generation Speed**: ~1-2 tokens/second on GPU
- **Model Size**: ~3GB final model

##  Customization

### Modify Generation Parameters

Edit the `generate_text()` function in `main.py`:

```python
def generate_text(prompt, max_t=50, temp=0.8, top_k=40, top_p=0.9):
    # Adjust parameters for different generation styles
```

### Change Training Settings

Modify training arguments in `Fine_tuningmodel.py`:

```python
training_args = TrainingArguments(
    num_train_epochs=5,        # More epochs
    learning_rate=1e-4,        # Different learning rate
    per_device_train_batch_size=4,  # Smaller batch size
)
```

##  Troubleshooting

**CUDA Out of Memory**: Reduce `per_device_train_batch_size` in training script

**Model Not Found Error**: Ensure `Fine_tuningmodel.py` completed successfully and created `gpt2-dialogue-full/` directory

**Slow Generation**: Verify GPU usage with `nvidia-smi` during inference

**Import Errors**: Make sure all dependencies are installed: `pip install -r requirements.txt`

##  Dataset Information

- **Source**: Daily Dialog dataset from Hugging Face
- **Size**: ~13k dialogues with ~100k utterances
- **Format**: Multi-turn conversations on daily topics
- **Preprocessing**: Custom tokenization with special tokens `<USER>`, `<BOT>`, `<END>`

##  Technical Implementation

### LoRA Configuration
```python
lora_config = LoraConfig(
    r=8,                    # Low rank
    lora_alpha=32,         # Scaling parameter
    target_modules=['c_attn', 'c_proj'],  # Target attention layers
    lora_dropout=0.05,     # Regularization
    task_type="CAUSAL_LM"  # Causal language modeling
)
```

### Quantization Setup
```python
quant_config = BitsAndBytesConfig(
    load_in_8bit=True,           # 8-bit quantization
    llm_int8_threshold=6.0       # Threshold for outlier detection
)
```
