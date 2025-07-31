from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

MODEL_NAME = "EleutherAI/gpt-neo-1.3B"  


print(f"Loading {MODEL_NAME}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

print("Model loaded! Start chatting (type 'quit' to exit).")

history = ""
while True:
    user_input = input("You: ")
    if user_input.lower() == "quit":
        break

    history += f"\nUser: {user_input}\nAI:"
    inputs = tokenizer(history, return_tensors="pt").to(device)

    output = model.generate(
        **inputs,
        max_new_tokens=100,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        pad_token_id=tokenizer.eos_token_id
    )

    response = tokenizer.decode(output[0], skip_special_tokens=True)
    reply = response.split("AI:")[-1].strip()
    print(f"AI: {reply}")

    history += f" {reply}"
