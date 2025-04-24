import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer
from tqdm import tqdm

# Path to model
model_name = "mistralai/Mistral-7B-Instruct-v0.3"  # or your local path

# Load model and tokenizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
tokenizer.add_special_tokens({'pad_token': '[PAD]'})
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")
model.eval()

# Load input CSV
df = pd.read_csv("classified_prompts.csv")

# Function to format chat prompts for Mistral-style models
def make_chat_prompt(message: str):
    return f"<s>[INST] {message.strip()} [/INST]"

# Inference function
def get_response(prompt: str, max_new_tokens=256):
    encoded = tokenizer(
        make_chat_prompt(prompt),
        return_tensors="pt",
        padding=True,
        truncation=True,
        return_attention_mask=True
    ).to(device)

    output = model.generate(
        input_ids=encoded["input_ids"],
        attention_mask=encoded["attention_mask"],
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=0.7,
        top_p=0.95,
        pad_token_id=tokenizer.eos_token_id,  # ✅ Fix pad token warning
        eos_token_id=tokenizer.eos_token_id,
    )
    decoded = tokenizer.decode(output[0], skip_special_tokens=True)
    return decoded.split("[/INST]")[-1].strip()


# Create new columns for responses
original_responses = []
classified_responses = []
all_guideline_responses = []

for idx, row in tqdm(df.iterrows(), total=len(df), desc="Generating responses"):
    prompt_original = row["Prompt"]
    prompt_classified = row["Prompt + Guidelines by Classification"]
    prompt_all = row["Prompt + All Guidelines"]

    try:
        original_responses.append(get_response(prompt_original))
        classified_responses.append(get_response(prompt_classified))
        all_guideline_responses.append(get_response(prompt_all))
    except Exception as e:
        print("Error:", e)
        original_responses.append("ERROR")
        classified_responses.append("ERROR")
        all_guideline_responses.append("ERROR")

    # Save results after each response
    df.loc[idx, "Response (Original Prompt)"] = original_responses[-1]
    df.loc[idx, "Response (Classified Guidelines)"] = classified_responses[-1]
    df.loc[idx, "Response (All Guidelines)"] = all_guideline_responses[-1]
    df.to_csv("mistral_guideline_responses.csv", index=False)

print("✅ Done. Responses saved to 'mistral_guideline_responses.csv'")
