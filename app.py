import gradio as gr
import json
import re
import os
from llama_cpp import Llama

# 1. Load fast GGUF model (INT4 quantized ~1.1GB)
MODEL_REPO = "Qwen/Qwen2.5-1.5B-Instruct-GGUF"
MODEL_FILE = "qwen2.5-1.5b-instruct-q4_k_m.gguf"

print("⏳ Loading model... (first run downloads ~1.1GB)")
llm = Llama.from_pretrained(
    repo_id=MODEL_REPO,
    filename=MODEL_FILE,
    n_ctx=768,          # Reduced context = faster
    n_threads=4,        # Match HF vCPU cores
    n_gpu_layers=-1,    # Auto-detect (0 on CPU, >0 if GPU enabled)
    verbose=False
)

# 2. Minimal few-shot examples (less prompt = faster)
EXAMPLES = [
    {"input": "Subject: Duplicate charge\nBody: Charged twice for Pro.", 
     "output": '{"team":"billing","priority":"high","summary":"Duplicate charge","action":"Verify logs & refund."}'},
    {"input": "Subject: API 429 error\nBody: Rate limit hit on webhooks.", 
     "output": '{"team":"technical","priority":"high","summary":"Rate limit exceeded","action":"Check quota & scale consumers."}'}
]

def build_prompt(ticket):
    prompt = "You are a SaaS support router. Output ONLY valid JSON with keys: team, priority, summary, action.\n"
    for ex in EXAMPLES:
        prompt += f"User: {ex['input']}\nAssistant: {ex['output']}\n"
    prompt += f"User: {ticket}\nAssistant: "
    return prompt

def route_ticket(ticket_text):
    if not ticket_text.strip():
        return "⚠️ Please enter a ticket."
        
    prompt = build_prompt(ticket_text)
    
    # Fast C++ inference
    output = llm(
        prompt, 
        max_tokens=60, 
        temperature=0.1, 
        top_p=0.9, 
        stop=["\nUser:"],
        echo=False
    )
    
    raw = output['choices'][0]['text'].strip()
    
    # Robust JSON extraction
    match = re.search(r'({.*})', raw, re.DOTALL)
    if match:
        try:
            return json.dumps(json.loads(match.group(1)), indent=2)
        except json.JSONDecodeError:
            return f"⚠️ JSON parse error:\n{match.group(1)}"
    return f"⚠️ No JSON found. Raw:\n{raw}"

demo = gr.Interface(
    fn=route_ticket,
    inputs=gr.Textbox(lines=3, placeholder="Subject: ...\nBody: ..."),
    outputs=gr.Textbox(lines=5, label="Routing Result"),
    title="🎫 Fast SaaS Ticket Router",
    description="Optimized C++ inference + INT4 quantization. ~3-4s on free CPU."
)

demo.launch()
