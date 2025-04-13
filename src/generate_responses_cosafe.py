import json
import argparse
from typing import List, Dict, Any
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description="Generate responses for CoSAFE dataset using Llama 3")
    parser.add_argument("--input_file", type=str, required=True, help="Path to the CoSAFE dataset JSON file")
    parser.add_argument("--output_file", type=str, required=True, help="Path to save the generated responses")
    parser.add_argument("--model_name", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct", 
                        help="Hugging Face model name to use (default: meta-llama/Meta-Llama-3-8B-Instruct)")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for processing")
    parser.add_argument("--max_new_tokens", type=int, default=512, help="Maximum new tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-p sampling parameter")
    parser.add_argument("--num_samples", type=int, default=None, 
                        help="Number of samples to process (None for all)")
    return parser.parse_args()

def format_prompt(conversation: List[Dict[str, str]]) -> str:
    """Format the conversation for Llama 3 chat completion."""
    prompt = ""
    for message in conversation:
        role = message["role"]
        content = message["content"]
        
        if role == "user":
            prompt += f"<|user|>\n{content}<|endoftext|>\n"
        elif role == "assistant":
            prompt += f"<|assistant|>\n{content}<|endoftext|>\n"
    
    # Add the final assistant prompt for generating a response
    prompt += "<|assistant|>\n"
    return prompt

def load_dataset(file_path: str, num_samples: int = None) -> List[List[Dict[str, str]]]:
    """Load conversations from the CoSAFE dataset."""
    conversations = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if num_samples is not None and i >= num_samples:
                break
            
            try:
                conversation = json.loads(line.strip())
                conversations.append(conversation)
            except json.JSONDecodeError:
                print(f"Error parsing JSON on line {i+1}. Skipping.")
    
    return conversations

def generate_responses(model, tokenizer, conversations: List[List[Dict[str, str]]], 
                      max_new_tokens: int, temperature: float, top_p: float) -> List[Dict[str, Any]]:
    """Generate responses for each conversation using the model."""
    results = []
    
    for conversation in tqdm(conversations, desc="Generating responses"):
        # The conversation might end with a user message that needs a response
        if conversation[-1]["role"] == "user":
            prompt = format_prompt(conversation)
        # Or it might already have the full conversation including assistant responses
        else:
            # For testing purposes, we can remove the last assistant response and regenerate it
            prompt = format_prompt(conversation[:-1])
            
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                pad_token_id=tokenizer.eos_token_id
            )
        
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=False)
        
        # Extract only the newly generated response part
        response_text = generated_text.split("<|assistant|>\n")[-1].split("<|endoftext|>")[0].strip()
        
        results.append({
            "conversation": conversation,
            "generated_response": response_text,
            "original_prompt": prompt
        })
    
    return results

def save_results(results: List[Dict[str, Any]], output_file: str):
    """Save the results to the output file."""
    with open(output_file, 'w', encoding='utf-8') as f:
        for result in results:
            f.write(json.dumps(result) + "\n")

def main():
    args = parse_args()
    
    # Load the model and tokenizer
    print(f"Loading model: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.float16,  # Use half precision for efficiency
        device_map="auto"  # Automatically determine the best device configuration
    )
    
    # Load the dataset
    print(f"Loading dataset from: {args.input_file}")
    conversations = load_dataset(args.input_file, args.num_samples)
    print(f"Loaded {len(conversations)} conversations")
    
    # Generate responses
    results = generate_responses(
        model,
        tokenizer,
        conversations,
        args.max_new_tokens,
        args.temperature,
        args.top_p
    )
    
    # Save the results
    print(f"Saving results to: {args.output_file}")
    save_results(results, args.output_file)
    print("Done!")

if __name__ == "__main__":
    main()