import os
import json
from datetime import datetime
from llama_cpp import Llama

# === CONFIG ===
MODEL_PATH = "frank-13b.gguf"
QUARANTINE_DIR = "quarantine"
os.makedirs(QUARANTINE_DIR, exist_ok=True)

llm = Llama(model_path=MODEL_PATH, n_ctx=2048)

print("🧠 Frank is live. All outputs will be quarantined.")

while True:
    prompt = input("\n> You: ")
    if prompt.lower() in ["exit", "quit"]:
        break

    output = llm(prompt, max_tokens=512)["choices"][0]["text"]

    # Write to quarantine folder
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "prompt": prompt,
        "response": output.strip()
    }
    log_name = f"q_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(os.path.join(QUARANTINE_DIR, log_name), "w") as f:
        json.dump(log_entry, f, indent=2)

    print(f"\n[Frank 🛑]: {output.strip()}")
    print(f"[LOGGED TO] {log_name}")
