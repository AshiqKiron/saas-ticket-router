import gradio as gr
import json
import os
from transformers import AutoTokenizer, AutoModelForCausalLM

# 1. Load model (runs on free CPU)
MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype="auto", device_map="cpu")

# 2. Load few-shot examples from tickets.jsonl
EXAMPLES_FILE = "tickets.jsonl"
if os.path.exists(EXAMPLES_FILE):
    with open(EXAMPLES_FILE, "r", encoding="utf-8") as f:
        EXAMPLES = [json.loads(line) for line in f if line.strip()]
else:
    # Fallback for local testing
    EXAMPLES = [
        {"input": "Subject: Duplicate charge\nBody: Charged twice.", 
         "output": '{"team": "billing", "priority": "high", "summary": "Duplicate charge", "action": "Verify logs & refund."}'}
    ]

# 3. Build prompt using model's native chat template
def build_prompt(user_ticket):
    messages = [{"role": "system", "content": "You are a SaaS support triage assistant. Route tickets to the correct team with priority, summary, and action steps. Output valid JSON only."}]
    for ex in EXAMPLES:
        messages.append({"role": "user", "content": ex["input"]})
        messages.append({"role": "assistant", "content": ex["output"]})
    messages.append({"role": "user", "content": user_ticket})
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

# 4. Inference function
def route_ticket(ticket_text):
    if not ticket_text.strip():
        return "⚠️ Please enter a ticket subject & body."
        
    prompt = build_prompt(ticket_text)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=128, temperature=0.3, do_sample=True)
    full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract only the assistant's response
    if "<|im_start|>assistant\n" in full_text:
        raw = full_text.split("<|im_start|>assistant\n")[-1].replace("<|im_end|>", "").strip()
    else:
        raw = full_text.split("assistant\n")[-1].strip()
        
    try:
        return json.dumps(json.loads(raw), indent=2)
    except json.JSONDecodeError:
        return f"⚠️ Output wasn't valid JSON:\n{raw}"

# 5. Gradio UI
demo = gr.Interface(
    fn=route_ticket,
    inputs=gr.Textbox(lines=4, placeholder="Paste customer ticket here (Subject + Body)..."),
    outputs=gr.Textbox(lines=8, label="Routing Result"),
    title="🎫 SaaS Ticket Router",
    description="Routes raw support tickets using in-context learning. Data loaded dynamically from `tickets.jsonl`."
)

demo.launch()
