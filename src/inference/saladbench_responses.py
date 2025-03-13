import os
import json
import csv
import torch
import re
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

# Ensure CUDA memory allocation is efficient
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

def longest_common_substring(s1, s2):
    """
    Finds the longest common substring between two strings using dynamic programming.
    """
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    length = 0
    end_idx_s2 = 0  

    for i in range(m):
        for j in range(n):
            if s1[i] == s2[j]:
                dp[i + 1][j + 1] = dp[i][j] + 1
                if dp[i + 1][j + 1] > length:
                    length = dp[i + 1][j + 1]
                    end_idx_s2 = j + 1

    return s2[end_idx_s2 - length : end_idx_s2] if length > 5 else ""  

def remove_prompt_from_response(prompt, response):
    """
    Removes the longest common substring between the cleaned prompt and response.
    """
    cleaned_prompt = re.sub(r"<\/?s> ", "", prompt).strip()
    cleaned_prompt = re.sub(r"<\/?s>", "", cleaned_prompt).strip()

    lcs = longest_common_substring(cleaned_prompt, response)
    cleaned_response = response.replace(lcs, "").strip()
    
    return cleaned_response if cleaned_response else "NO RESPONSE"

def generate_response(prompt, model, tokenizer):
    """
    Generate a model response for a given prompt and remove the prompt portion.
    """
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=150,
            do_sample=True,
            temperature=0.7
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
    return remove_prompt_from_response(prompt, response)

# Load LLaMA-2-7B-Chat model
device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf", device_map="auto")

def process_saladbench_data(input_file, output_file):
    """
    Reads SaladBench_attack_enhanced_set.json, extracts queries, generates responses,
    and writes results to a CSV file.
    """
    with open(input_file, "r", encoding="utf-8") as file:
        data = json.load(file)

    with open(output_file, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["Query", "Model Response", "3-Category"])  

        for entry in tqdm(data, desc="Processing SaladBench queries", unit=" queries", dynamic_ncols=True):
            try:
                query = entry["query"]
                category = entry["3-category"]

                model_response = generate_response(query, model, tokenizer)
                writer.writerow([query, model_response, category])  
            
            except Exception as e:
                print(f"Skipping entry due to error: {e}")

# File paths
input_json = "prompts/Psychologically Unsafe Prompts/SaladBench/SaladBench_attack_enhanced_set.json"
output_csv = "saladbench_results.csv"

# Process the dataset
process_saladbench_data(input_json, output_csv)

print("Processing complete. Results saved to saladbench_results.csv.")

