from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer,BitsAndBytesConfig
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model, PeftModel
import os
import torch
from transformers import TrainingArguments, Trainer


dataset = load_dataset(
    "daily_dialog",
    download_mode="force_redownload",
    cache_dir=os.path.join(os.getcwd(), "daily_dialog_cache"))
print(dataset['train'][0])


model_name = "gpt2-xl"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

quant_config = BitsAndBytesConfig(
    load_in_8bit=True,
    llm_int8_threshold=6.0
)


base_model_s = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config = quant_config,
    device_map = 'auto'
)
base_model_s = prepare_model_for_kbit_training(base_model_s)

lora_config = LoraConfig(
    r=8,
    lora_alpha = 32,
    target_modules = ['c_attn','c_proj'],
    lora_dropout=0.05,
    bias ='none',
    task_type= "CAUSAL_LM"
)

model = get_peft_model(base_model_s,lora_config)
model.print_trainable_parameters()


def preprocess_dialogue(example):
    dialog = example["dialog"]
    new_dialog =""
    for i, utterance in enumerate(dialog):
        role = "<USER>" if i % 2 == 0 else "<BOT>"
        new_dialog+=f"{role}: {utterance}\n"
    return {"text": new_dialog.strip() + "<END>"}
dataset = dataset.map(preprocess_dialogue)

tokenizer.add_special_tokens({
    'additional_special_tokens': ['<USER>', '<BOT>', '<END>']
})
def tokenize(example):
  tokenized = tokenizer(
        example["text"],
        truncation=True,
        max_length=512,
        padding="max_length",
        return_tensors="pt"
    )
  tokenized["labels"] = tokenized["input_ids"].clone()
  # print(tokenized)
  return tokenized

tokenized_dataset = dataset.map(tokenize, batched=True)
model.resize_token_embeddings(len(tokenizer))

savestep = 10 # here any savestep can be used.
training_args = TrainingArguments(
    output_dir='gpt2-dialogue-lora',
    per_device_train_batch_size=8,
    gradient_accumulation_steps=4,
    num_train_epochs=3,
    learning_rate=3e-4,
    logging_steps=10,
    fp16 = True,
    evaluation_strategy = 'steps',
    eval_steps = savestep,
    save_total_limit=1,
    load_best_model_at_end=True,
    metric_for_best_model='eval_loss',
    greater_is_better=False
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset['validation']
)
model.gradient_checkpointing_enable()
trainer.train()

model.save_pretrained("gpt2-dialogue-lora")
tokenizer.save_pretrained("gpt2-dialogue-lora")


mod_name = 'gpt2-xl'

file_path = 'gpt2-dialogue-lora'
quant_config = BitsAndBytesConfig(
    load_in_8bit = True,
    llm_int8_threshold = 6.0
)

base_model = AutoModelForCausalLM.from_pretrained(
    mod_name,
    quantization_config = quant_config,
    # device_map='Auto'
    )
base_model.resize_token_embeddings(len(tokenizer))

model = PeftModel.from_pretrained(base_model,file_path )
model = model.merge_and_unload()
model.save_pretrained("gpt2-dialogue-full", safe_serialization=True)
tokenizer.save_pretrained("gpt2-dialogue-full")