
# train.py
import torch, os, json
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from trl import SFTTrainer
from peft import LoraConfig, get_peft_model
import evaluate

# 1. Load base model & tokenizer
MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
base_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16, device_map="auto")

# 2. LoRA config
lora_config = LoraConfig(r=16, lora_alpha=32, target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"], lora_dropout=0.05, bias="none", task_type="CAUSAL_LM")
model = get_peft_model(base_model, lora_config)

# 3. Format dataset
dataset = load_dataset("json", data_files="/kaggle/input/your-dataset-name/tickets.jsonl", split="train")
def format_samples(examples):
    texts = []
    for i in range(len(examples["input"])):
        messages = [{"role":"system","content":"Route this support ticket to the correct team."},
                    {"role":"user","content":examples["input"][i]},
                    {"role":"assistant","content":examples["output"][i]}]
        texts.append(tokenizer.apply_chat_template(messages, tokenize=False))
    return {"text": texts}

dataset = dataset.map(format_samples, batched=True)

# 4. Train
args = TrainingArguments(
    output_dir="./ticket_adapter",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    max_steps=120,
    learning_rate=2e-4,
    fp16=True,
    logging_steps=10,
    save_strategy="no"
)
trainer = SFTTrainer(model=model, train_dataset=dataset, dataset_text_field="text", args=args, tokenizer=tokenizer, max_seq_length=512)
trainer.train()
model.save_pretrained("./ticket_adapter")
tokenizer.save_pretrained("./ticket_adapter")

# 5. Quick evaluation
rouge = evaluate.load("rouge")
preds, refs = [], []
for row in dataset.select(range(5)):
    messages = [{"role":"user","content":row["input"]}]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    out = model.generate(**inputs, max_new_tokens=64)
    pred = tokenizer.decode(out[0], skip_special_tokens=True).split("assistant\n")[-1].strip()
    preds.append(pred)
    refs.append(row["output"])
print("ROUGE-L:", rouge.compute(predictions=preds, references=refs, tokenizer=lambda x: x)["rougeL"])
