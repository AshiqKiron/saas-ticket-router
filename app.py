import gradio as gr
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
ADAPTER_PATH = "./adapter"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
base_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16, device_map="cpu")
model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)

def route_ticket(text):
    messages = [{"role":"user","content":f"Route this support ticket:\n{text}"}]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to("cpu")
    outputs = model.generate(**inputs, max_new_tokens=100, temperature=0.3)
    full = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return full.split("assistant\n")[-1].strip()

demo = gr.Interface(
    fn=route_ticket,
    inputs=gr.Textbox(lines=4, placeholder="Paste customer ticket here..."),
    outputs=gr.Textbox(lines=6),
    title="🎫 SaaS Ticket Router (LoRA Fine-Tuned)",
    description="Routes raw support tickets to teams with priority, summary, and action steps."
)

if __name__ == "__main__":
    demo.launch()
