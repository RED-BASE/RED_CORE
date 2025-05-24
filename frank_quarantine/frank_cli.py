from llama_cpp import Llama

# Load model from symlink or direct path
llm = Llama(model_path="frank-13b.gguf", n_ctx=2048)

print("🧠 Frank-13B is online.")
print("Type 'exit' to quit.")

while True:
    prompt = input("\n> You: ")
    if prompt.lower() in ["exit", "quit"]:
        break

    output = llm(prompt, max_tokens=512)["choices"][0]["text"]
    print(f"\n[Frank]: {output.strip()}")
