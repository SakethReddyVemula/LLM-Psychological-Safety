import json
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
import pandas as pd
import re

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
model.eval()

def get_prompt_by_length(query, response, length=3):
    prompt = f"""<|system|>
You are an AI assistant that converts a single-turn query and response into a natural multi-turn dialogue. Follow these instructions carefully:
1. Create exactly {length} back-and-forth exchanges (user and assistant turns)
2. Preserve the complete meaning and information from the original query and response
3. Break down complex questions into simpler ones across multiple turns
4. Make sure the dialogue flows naturally and conversationally
5. Return ONLY valid JSON format with no additional text

Original query: {query}
Original response: {response}
<|user|>
Please convert the above query and response into a natural {length}-turn dialogue. Output only in JSON format as a list of messages with "role" and "content" fields.
<|assistant|>
```json
[
"""

    for i in range(length):
        if i < length - 1:
            prompt += '  {"role": "user", "content": "USER_TURN_' + str(i+1) + '"},\n'
            prompt += '  {"role": "assistant", "content": "ASSISTANT_TURN_' + str(i+1) + '"},\n'
        else:
            prompt += '  {"role": "user", "content": "USER_TURN_' + str(i+1) + '"},\n'
            prompt += '  {"role": "assistant", "content": "ASSISTANT_TURN_' + str(i+1) + '"}\n'
    
    prompt += "]```"
    return prompt

def extract_json(text):
    # Extract JSON content between triple backticks
    json_match = re.search(r'```json\s*([\s\S]*?)\s*```', text)
    
    if json_match:
        json_str = json_match.group(1)
    else:
        # Try to find anything that looks like a JSON array
        json_match = re.search(r'\[\s*{[\s\S]*?}\s*\]', text)
        if json_match:
            json_str = json_match.group(0)
        else:
            return None
    
    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        print("Failed to parse JSON:", json_str)
        return None

def load_json(filename):
    with open(filename, 'r', encoding='utf-8') as file:
        lines = file.readlines()
    data = [json.loads(line) for line in lines] 
    return data

# Load and prepare data
splits = {'train': 'LifeTox_train.jsonl', 'test': 'LifeTox_test.jsonl'}
df_train = pd.read_json("hf://datasets/mbkim/LifeTox/" + splits["train"], lines=True)
df_test = pd.read_json("hf://datasets/mbkim/LifeTox/" + splits["test"], lines=True)

# Filter and sample
df_filtered = df_train[df_train['query'].apply(lambda x: len(tokenizer.tokenize(x)) < 200) & 
                     df_train['response'].apply(lambda x: len(tokenizer.tokenize(x)) < 200)]
df_train_sampled = df_filtered.sample(n=4000, random_state=42)

# Print statistics
print("Train set:")
print("Safe samples:", df_train_sampled[df_train_sampled['is_safe'] == 1].shape[0])
print("Unsafe samples:", df_train_sampled[df_train_sampled['is_safe'] == 0].shape[0])

def get_data(df, lengths=[2, 3, 4, 5], split="train"):
    data = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Processing {split} rows"):
        query = row['query']
        response = row['response']
        is_safe = row['is_safe']
        
        for length in lengths:
            prompt = get_prompt_by_length(query, response, length)
            inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
            
            # Generate with better parameters
            outputs = model.generate(
                **inputs, 
                max_new_tokens=2048,  # More controlled than max_length
                temperature=0.7,      # Add some creativity but not too random
                top_p=0.9,            # Nucleus sampling for better quality
                do_sample=True,       # Enable sampling
                num_return_sequences=1,
                pad_token_id=tokenizer.eos_token_id
            )
            
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            dialogue_json = extract_json(generated_text)
            
            # Handle extraction failure
            if not dialogue_json:
                print(f"Failed to extract JSON for query: {query[:50]}...")
                dialogue_json = "Extraction failed"
            
            data.append({
                "query": query,
                "response": response,
                "is_safe": is_safe,
                "length": length,
                "generated_dialogue": dialogue_json
            })
            
            # Save incrementally in case of crashes
            if len(data) % 100 == 0:
                temp_df = pd.DataFrame(data)
                temp_df.to_csv(f"generated_lifetox_dialogues_{split}_partial.csv", index=False, encoding='utf-8')
    
    df_generated = pd.DataFrame(data)
    df_generated.to_csv(f"generated_lifetox_dialogues_{split}.csv", index=False, encoding='utf-8')
    
    return df_generated

# Run the data generation
get_data(df_train_sampled)
get_data(df_test, split='test')