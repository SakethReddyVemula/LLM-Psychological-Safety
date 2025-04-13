import json
import pandas as pd
import argparse
from pathlib import Path

def parse_jsonl_file(file_path):
    """Parse JSONL file and extract conversation data."""
    data = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                json_obj = json.loads(line.strip())
                
                # Extract conversation
                conversation = json_obj.get('conversation', [])
                
                # Build the conversation history as a formatted string
                conv_history = ""
                for message in conversation:
                    role = message.get('role', '')
                    content = message.get('content', '')
                    conv_history += f"{role.upper()}: {content}\n\n"
                
                # Extract the generated response
                generated_response = json_obj.get('generated_response', '')
                
                # Create a row for the dataframe
                data.append({
                    'Conversation History': conv_history,
                    'Generated Response': generated_response,
                    'Is Toxic? (-1/0/1)': ''  # Empty column for annotation
                })
                
            except json.JSONDecodeError:
                print(f"Warning: Could not parse line: {line[:100]}...")
    
    return data

def main():
    parser = argparse.ArgumentParser(description='Convert JSONL conversations to Excel for annotation')
    parser.add_argument('--input_file', type=str, help='Path to the input JSONL file')
    parser.add_argument('--output_file', type=str, help='Path to the output Excel file (default: output.xlsx)')
    
    args = parser.parse_args()
    input_file = args.input_file
    output_file = args.output_file or 'conversation_annotation.xlsx'
    
    print(f"Processing file: {input_file}")
    
    # Parse the JSONL file
    data = parse_jsonl_file(input_file)
    
    if not data:
        print("No data was extracted from the file.")
        return
    
    # Create a DataFrame
    df = pd.DataFrame(data)
    
    # Calculate row heights based on content length
    max_rows = 40  # Maximum number of Excel rows for each conversation
    conv_lengths = [min(max_rows, len(text.split('\n')) * 2) for text in df['Conversation History']]
    
    # Create Excel writer
    with pd.ExcelWriter(output_file, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='Annotations')
        
        # Get the xlsxwriter workbook and worksheet objects
        workbook = writer.book
        worksheet = writer.sheets['Annotations']
        
        # Add formats
        wrap_format = workbook.add_format({'text_wrap': True, 'valign': 'top'})
        header_format = workbook.add_format({'bold': True, 'bg_color': '#D9D9D9', 'border': 1})
        
        # Format the worksheet
        worksheet.set_column('A:A', 60, wrap_format)  # Conversation history
        worksheet.set_column('B:B', 40, wrap_format)  # Generated response
        worksheet.set_column('C:C', 15)  # Is toxic column
        
        # Set row heights
        for i, height in enumerate(conv_lengths):
            worksheet.set_row(i + 1, height * 15)  # +1 to skip header row
        
        # Format headers
        for col_num, col_name in enumerate(df.columns):
            worksheet.write(0, col_num, col_name, header_format)
        
        # Add data validation for the "Is Toxic?" column
        validation = {
            'validate': 'list',
            'source': ['1', '0', '-1']
        }
        worksheet.data_validation(1, 2, len(data), 2, validation)
    
    print(f"Successfully created Excel file: {output_file}")
    print(f"Total conversations processed: {len(data)}")

if __name__ == "__main__":
    main()