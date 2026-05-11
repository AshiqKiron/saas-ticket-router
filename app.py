import gradio as gr
import json
from transformers import AutoTokenizer, AutoModelForCausalLM

# 1. Load model (runs on free CPU)
MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype="auto", device_map="cpu")

# 2. Your ticket JSON examples (add/replace as needed)
EXAMPLES = [
    {"input": "Subject: Duplicate charge\nBody: Charged twice for Pro plan. Receipt #8842.",
     "output": '{"team": "billing", "priority": "high", "summary": "Duplicate subscription charge", "action": "Verify payment logs, issue refund, email confirmation."}'},
    {"input": "Subject: Calendar sync broken\nBody: Google Calendar fails with 403 after OAuth.",
     "output": '{"team": "technical", "priority": "medium", "summary": "OAuth scope mismatch", "action": "Check API permissions, re-auth user, test webhook."}'},
    {"input": "Subject: SSO login failed\nBody: New user gets Invalid SAML assertion on first login.",
     "output": '{"team": "security", "priority": "high", "summary": "SAML assertion validation error", "action": "Verify IdP attribute mapping, check ACS URL, test with staging tenant."}'}
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
    description="Routes raw support tickets using in-context learning. No fine-tuning required."
)

demo.launch()
