import json
import csv
import re

def extract_data_from_text(text):
    """Extract JSON data from the provided text format."""
    # Try to find JSON objects in the text
    pattern = r'\{\s*"Prompt":\s*".*?",\s*"Response":\s*".*?",\s*"Safe response":\s*".*?"\s*\}'
    json_matches = re.findall(pattern, text, re.DOTALL)
    
    # If no matches found, try to parse the entire text as JSON
    if not json_matches:
        try:
            # Remove the incomplete part at the end if needed
            if text.strip().endswith(','):
                text = text.strip()[:-1]
            if not text.strip().startswith('['):
                text = '[' + text
            if not text.strip().endswith(']'):
                text = text + ']'
            
            return json.loads(text)
        except json.JSONDecodeError:
            print("Error: Could not parse JSON data from the input text.")
            return []
    
    # If matches found, parse each match
    data = []
    for match in json_matches:
        try:
            # Fix the JSON if it ends with a comma
            if match.strip().endswith(','):
                match = match.strip()[:-1]
            item = json.loads(match)
            data.append(item)
        except json.JSONDecodeError:
            print(f"Error parsing JSON object: {match[:100]}...")
    
    return data

def parse_and_label(data):
    """Parse the data and add labels."""
    result = []
    
    for item in data:
        prompt = item.get("Prompt", "")
        response = item.get("Response", "")
        safe_response = item.get("Safe response", "")
        
        # Add simple labels (you may want to implement a more sophisticated labeling logic)
        label_response = "Unsafe" if response.strip() and response != safe_response else "Safe"
        label_safe_response = "Safe" if safe_response.strip() else "Unknown"
        
        result.append({
            "Prompt": prompt,
            "Response": response,
            "Label Response": label_response,
            "Safe response": safe_response,
            "Label Safe response": label_safe_response
        })
    
    return result

def create_csv(data, output_file="prompt_responses.csv"):
    """Create a CSV file from the parsed data."""
    if not data:
        print("No data to write to CSV.")
        return
    
    fieldnames = ["Prompt", "Response", "Label Response", "Safe response", "Label Safe response"]
    
    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data)
    
    print(f"CSV file '{output_file}' created successfully.")

def main(input_file="/home/saketh/IIITH Sem 6/Responsible-and-Safe-AI/Project/Eval3/saladbench_responses(1).json", output_file="prompt_responses.csv"):
    """Main function to process the input file and create the CSV."""
    try:
        # Read the input file
        with open(input_file, 'r', encoding='utf-8') as f:
            text = f.read()
        
        # Extract and parse the data
        data = extract_data_from_text(text)
        if not data:
            print("No valid data found in the input file.")
            return
        
        # Parse and label the data
        labeled_data = parse_and_label(data)
        
        # Create the CSV file
        create_csv(labeled_data, output_file)
        
    except FileNotFoundError:
        print(f"Error: Input file '{input_file}' not found.")
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()