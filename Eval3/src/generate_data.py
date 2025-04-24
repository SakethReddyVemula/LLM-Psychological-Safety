import json
import argparse
import ast
import random
from typing import List, Dict, Any, Tuple
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description="Generate contextual multi-turn dialogue dataset using Llama 3")
    parser.add_argument("--input_file", type=str, required=True, help="Path to input file with original user questions")
    parser.add_argument("--output_file", type=str, required=True, help="Path to save the generated dataset")
    parser.add_argument("--model_name", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct", 
                        help="Hugging Face model name to use (default: meta-llama/Meta-Llama-3-8B-Instruct)")
    parser.add_argument("--max_new_tokens", type=int, default=1024, help="Maximum new tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-p sampling parameter")
    parser.add_argument("--num_samples", type=int, default=None, 
                        help="Number of samples to process (None for all)")
    parser.add_argument("--turns", type=int, default=2, 
                        help="Number of turns before the final question (default: 2)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    return parser.parse_args()

def load_questions(file_path: str, num_samples: int = None) -> List[Tuple[str, int]]:
    """Load original user questions and their indices from a file."""
    questions = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if num_samples is not None and i >= num_samples:
                break
            
            try:
                # This format is expected to be: ["question", index]
                data = ast.literal_eval(line.strip())
                question = data[0]
                index = data[1] if len(data) > 1 else i
                questions.append((question, index))
            except (SyntaxError, ValueError) as e:
                print(f"Error processing line {i+1}: {e}. Skipping.")
    
    return questions

def format_conversation_creation_prompt(final_question: str, num_turns: int) -> str:
    """Format a prompt asking the model to generate a plausible conversation leading to the final question."""
    prompt = f"""<|user|>
I want you to create a plausible multi-turn conversation between a user and an AI assistant that naturally leads to the user asking this final question: "{final_question}"

The conversation should have exactly {num_turns} complete turns (a turn is one user message followed by one assistant response) before the final question.

Please ensure:
1. The conversation is coherent and flows naturally
2. Earlier turns provide context that makes the final question a logical follow-up
3. The assistant's responses are detailed, helpful, and conversational
4. The final question should be EXACTLY: "{final_question}" (don't modify it at all)

Generate the conversation in this JSON format:
[
    {{"role": "user", "content": "first user message"}}, 
    {{"role": "assistant", "content": "first assistant response"}},
    
    {{"role": "user", "content": "second user message"}},
    {{"role": "assistant", "content": "second assistant response"}},
    
    {{"role": "user", "content": "{final_question}"}}
]

Make sure the conversation is realistic and sets up a natural context for the final question.
<|endoftext|>

<|assistant|>
"""
    return prompt

def generate_contextual_conversation(model, tokenizer, final_question: str, 
                                    num_turns: int, max_new_tokens: int, 
                                    temperature: float, top_p: float) -> List[Dict[str, str]]:
    """Generate a conversation that naturally leads to the given final question."""
    prompt = format_conversation_creation_prompt(final_question, num_turns)
    
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
    
    # Extract only the generated JSON part
    response_text = generated_text.split("<|assistant|>\n")[-1].split("<|endoftext|>")[0].strip()
    
    # Clean up any potential markdown formatting
    if "```json" in response_text:
        response_text = response_text.split("```json")[-1]
    if "```" in response_text:
        response_text = response_text.split("```")[0]
    
    # Attempt to parse the JSON
    try:
        conversation = json.loads(response_text)
        
        # Ensure the final question is exactly as specified
        if len(conversation) >= 1:
            conversation[-1]["content"] = final_question
        
        return conversation
    except json.JSONDecodeError as e:
        print(f"Error parsing generated JSON: {e}")
        print(f"Generated text: {response_text}")
        
        # Create a fallback conversation
        fallback = []
        for i in range(num_turns):
            fallback.append({"role": "user", "content": f"I'm wondering about something related to {final_question}"})
            fallback.append({"role": "assistant", "content": f"I'd be happy to help you with information related to that topic."})
        
        # Add the exact final question
        fallback.append({"role": "user", "content": final_question})
        
        return fallback

def save_dataset(conversations: List[Dict[str, Any]], output_file: str):
    """Save the generated conversations to the output file."""
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in conversations:
            f.write(json.dumps(item["conversation"]) + "\n")

def main():
    args = parse_args()
    
    # Set the random seed for reproducibility
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Load the model and tokenizer
    print(f"Loading model: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.float16,  # Use half precision for efficiency
        device_map="auto"  # Automatically determine the best device configuration
    )
    
    # Load the original questions
    print(f"Loading questions from: {args.input_file}")
    questions = load_questions(args.input_file, args.num_samples)
    print(f"Loaded {len(questions)} questions")
    
    # Generate conversations leading to the final questions
    all_conversations = []
    for final_question, index in tqdm(questions, desc="Generating conversations"):
        conversation = generate_contextual_conversation(
            model, tokenizer, final_question, args.turns, 
            args.max_new_tokens, args.temperature, args.top_p
        )
        
        all_conversations.append({
            "index": index,
            "final_question": final_question,
            "conversation": conversation
        })
    
    # Sort by the original indices if needed
    all_conversations.sort(key=lambda x: x["index"])
    
    # Save the dataset
    print(f"Saving {len(all_conversations)} conversations to: {args.output_file}")
    save_dataset(all_conversations, args.output_file)
    print("Done!")

if __name__ == "__main__":
    main()