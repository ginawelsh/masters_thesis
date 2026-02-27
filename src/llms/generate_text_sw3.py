from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("AI-Sweden-Models/gpt-sw3-356m")
model = AutoModelForCausalLM.from_pretrained("AI-Sweden-Models/gpt-sw3-356m")

input_ids = tokenizer("Träd är fina för att", return_tensors="pt")["input_ids"]

generated_token_ids = model.generate(inputs=input_ids, max_new_tokens=10, do_sample=True)[0]

print(tokenizer.decode(generated_token_ids))