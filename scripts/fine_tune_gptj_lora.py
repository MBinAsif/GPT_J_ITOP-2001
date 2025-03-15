from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model
from datasets import load_dataset

# Load dataset
dataset = load_dataset("json", data_files="../data/income_tax_ordinance_2001.json", split="train")

# Load GPT-J model
model_name = "EleutherAI/gpt-j-6B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", device_map="auto")

# Apply LoRA
lora_config = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.1, bias="none")
model = get_peft_model(model, lora_config)

# Define training arguments
training_args = TrainingArguments(
    output_dir="../models/fine_tuned_gptj",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    num_train_epochs=3,
    logging_dir="../logs",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    tokenizer=tokenizer,
)

trainer.train()
model.save_pretrained("../models/fine_tuned_gptj")
tokenizer.save_pretrained("../models/fine_tuned_gptj")
print("✅ Fine-tuned GPT-J model saved!")
