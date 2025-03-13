import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import json
import csv
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import re

def longest_common_substring(s1, s2):
    """
    Finds the longest common substring between two strings using dynamic programming.
    """
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    length = 0
    end_idx_s2 = 0  # Track where in s2 the substring ends

    for i in range(m):
        for j in range(n):
            if s1[i] == s2[j]:
                dp[i + 1][j + 1] = dp[i][j] + 1
                if dp[i + 1][j + 1] > length:
                    length = dp[i + 1][j + 1]
                    end_idx_s2 = j + 1

    return s2[end_idx_s2 - length : end_idx_s2] if length > 5 else ""  # Avoid small noisy matches

def remove_prompt_from_response(prompt, response):
    cleaned_prompt = re.sub(r"<\/?s> ", "", prompt).strip()
    cleaned_prompt = re.sub(r"<\/?s>", "", cleaned_prompt).strip()

    print(cleaned_prompt)
    print("*********")
    print(response)
    print("----------")

    """
    Removes the longest common substring between the cleaned prompt and response.
    """
    lcs = longest_common_substring(cleaned_prompt, response)
    cleaned_response = response.replace(lcs, "").strip()
    
    print(cleaned_response)
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

def format_conversation(conversation):
    """
    Convert the multi-turn conversation into the expected format for LLaMA-2-7B-Chat.
    Ensures that each user message is enclosed in [INST] ... [/INST] and assistant responses follow correctly.
    """
    formatted_prompt = ""
    
    for turn in conversation:
        role = turn["role"]
        content = turn["content"]
        
        if role == "user":
            formatted_prompt += f"<s>[INST] {content} [/INST] "
        elif role == "assistant":
            formatted_prompt += f"{content}</s> "
    
    return formatted_prompt.strip()


from tqdm import tqdm

def process_cosafe_data(input_file, output_file):
    """
    Reads self_harm.jsonl, formats conversation history, generates model responses,
    and writes results to a CSV file.
    """
    # Count total lines first for tqdm progress bar
    with open(input_file, "r", encoding="utf-8") as file:
        total_lines = sum(1 for _ in file)

    with open(input_file, "r", encoding="utf-8") as file, open(output_file, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["Formatted Prompt", "Model Response"])  # CSV header

        # tqdm with total count and better format
        for line in tqdm(file, desc="Processing CoSafe conversations", total=total_lines, unit=" lines", dynamic_ncols=True):
            try:
                conversation = json.loads(line.strip())  # Parse each line as JSON
                if not isinstance(conversation, list):
                    continue  # Skip if the line is not a valid conversation list

                formatted_prompt = format_conversation(conversation)
                model_response = generate_response(formatted_prompt, model, tokenizer)
                writer.writerow([formatted_prompt, model_response])  # Save results
            
            except json.JSONDecodeError:
                print(f"Skipping malformed line: {line}")

# File paths
input_jsonl = "prompts/Psychologically Unsafe Prompts/CoSafe(Multi-Dialogue-Prompts)/self_harm.jsonl"
output_csv = "cosafe_results.csv"

# Process the dataset
process_cosafe_data(input_jsonl, output_csv)

print("Processing complete. Results saved to cosafe_results.csv.")

