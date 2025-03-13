import os
import csv
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import login
from tqdm import tqdm

# Log in to Hugging Face
login(token="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")

# Set device to GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load Llama-2-7B-Chat model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-chat-hf",
    device_map="auto"
)

def load_text_files(directory):
    """Load text from all .txt files in a directory."""
    entries = []
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            filepath = os.path.join(directory, filename)
            with open(filepath, "r", encoding="utf-8") as file:
                entries.append(file.read().strip())
    return entries

def generate_response(prompt, model, tokenizer):
    """Generate model response for a given prompt and remove the prompt prefix from it."""
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to(model.device)  # Prevent overly long inputs
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=100,  # Increased for better responses
            do_sample=True,  # Introduce randomness to prevent early stopping
            temperature=0.7  # Adjusts response creativity
        )  
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

    # Remove the prompt from the response
    if response.startswith(prompt):
        response = response[len(prompt):].strip()

    # Handle empty responses
    if not response:
        response = "NO RESPONSE"

    return response

def get_saved_count(output_file):
    """Check how many entries have already been saved to the CSV file."""
    if os.path.exists(output_file):
        with open(output_file, "r", encoding="utf-8") as file:
            return sum(1 for _ in file) - 1  # Subtract header row
    return 0

def process_and_save(entries, output_file, batch_size=30):
    """Process text entries in batches, get model responses, and save to CSV, resuming from last checkpoint."""
    saved_count = get_saved_count(output_file)
    entries = entries[saved_count:]  # Skip already processed entries
    
    with open(output_file, "a", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        if saved_count == 0:
            writer.writerow(["Prompt", "Model Response"])  # Write header if it's a new file
        
        for i in tqdm(range(0, len(entries), batch_size), desc=f"Processing {output_file} in batches"):
            batch = entries[i:i + batch_size]
            results = []
            
            for entry in batch:
                prompt = f"{entry}\n\nIs the person who said this depressed or not? Answer only in a Yes or a No."
                response = generate_response(prompt, model, tokenizer)
                results.append([prompt, response])  # Save the prompt + cleaned response
            
            writer.writerows(results)  # Save batch

# Load entries from both folders
depression_entries = load_text_files("blogs_depression")
non_depression_entries = load_text_files("blogs_non_depression")

# Process and save outputs in batches
process_and_save(depression_entries, "depression_responses.csv", batch_size=30)
process_and_save(non_depression_entries, "non_depression_responses.csv", batch_size=30)

print("Processing complete. Responses saved.")

