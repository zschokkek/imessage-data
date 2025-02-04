from transformers import GPT2LMHeadModel, GPT2Tokenizer

def generate_text(prompt, max_length=100):
    # Load fine-tuned model and tokenizer
    model = GPT2LMHeadModel.from_pretrained("./gpt2-chat-model")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    # Encode the input and generate text
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    output = model.generate(input_ids, max_length=max_length, num_return_sequences=1, no_repeat_ngram_size=2)

    # Decode and print the result
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    print(generated_text)

# Example usage
generate_text("Hey, what's up?")