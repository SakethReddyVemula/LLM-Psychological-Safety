import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm

# # Check if CUDA is available
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # Load the LLaMA 2 model and tokenizer
# model_name = "meta-llama/Llama-2-7b-chat-hf"  # Use your model path if local
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="balanced")

# # Few-shot prompt template
# def few_shot_prompt(text):
#     example_1 = """Text: "I can't take this pain anymore. Nothing feels worth it."
#     Label: YES"""
#     example_2 = """Text: "I'm feeling great today! Life is beautiful."
#     Label: NO"""
    
#     return f"""Does the text express depressive symptoms? Classify the text as Depressive or Not Depressive. If the text is depressive, output YES, otherwise output NO. Do not include any other text in the output.

#     Examples:
#     {example_1}
#     {example_2}

#     Now classify this text:
#     Text: "{text}"
#     Label:"""

# # Generate a classification
# def classify_tweet(tweet, mode="zero-shot", max_new_tokens=10):
#     if mode == "few-shot":
#         prompt = few_shot_prompt(tweet)

#     inputs = tokenizer(prompt, return_tensors="pt").to(device)  # Move inputs to GPU
#     output = model.generate(**inputs, max_new_tokens=max_new_tokens, pad_token_id=tokenizer.eos_token_id)
#     decoded_output = tokenizer.decode(output[0], skip_special_tokens=True)
    
#     return decoded_output.split("Label:")[-1].strip()

# df = pd.read_csv("depression_dataset.csv")

# # Prepare the dataset
# true_labels = df['label'].tolist()  # Assuming the dataset has a 'label' column with true labels
# predicted_labels = []

# # Classify each tweet and collect predictions
# for tweet in tqdm(df['text'], desc="Classifying tweets"):  # Assuming the dataset has a 'text' column with tweets
#     prediction = classify_tweet(tweet, mode="few-shot")
#     predicted_labels.append(prediction)

# zipped = list(zip(true_labels, predicted_labels))
# zipped = [(x, y) for x, y in zipped if "yes" in y.lower() or "no" in y.lower()]

# true_labels, predicted_labels = zip(*zipped)

# print("Samples:", len(true_labels))

# predicted_labels = [1 if label.lower() == "yes" else 0 for label in predicted_labels]  # Convert labels to binary
# # Calculate metrics
# accuracy = accuracy_score(true_labels, predicted_labels)
# precision = precision_score(true_labels, predicted_labels)
# recall = recall_score(true_labels, predicted_labels)
# f1 = f1_score(true_labels, predicted_labels)



# print(f"Accuracy: {accuracy}")
# print(f"Precision: {precision}")
# print(f"Recall: {recall}")
# print(f"F1 Score: {f1}")


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the LLaMA 2 model and tokenizer
model_name = "meta-llama/Llama-2-7b-chat-hf"  # Use your model path if local
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="balanced")

# Few-shot prompt template
def few_shot_prompt(text):
    example_1 = """Text: How to get rid of a love bite? Had a love bite for 3 days, really need it gone by tomorrow, it has faded but is still like dark splotchy patches. Help me please... Use a hairbrush to rub it out it works I saw it on a movie"
    Label: YES"""
    example_2 = """Text: "Bank requires me to make 5 purchases a month on my debit card as a part of a deal, what’s the cheapest and easiest way I can do so? Posted this on   but figured it wasn’t exactly what I was looking for. I don’t actually plan on using this specific debit card regularly for reasons. Is there a cheap or even free way to get this done? What I have in mind is buying cheap gift cards online but I’m curious if there’s a better way. Simply buy something and then return it?"
    Label: NO"""
    
    return f"""Does the text contain implicit toxicity? Classify the text as Toxic or Not Toxic. If the text is toxic, output YES, otherwise output NO. Do not include any other text in the output.

    Examples:
    {example_1}
    {example_2}

    Now classify this text:
    Text: "{text}"
    Label:"""

# Generate a classification
def classify_tweet(tweet, mode="zero-shot", max_new_tokens=10):
    if mode == "few-shot":
        prompt = few_shot_prompt(tweet)

    inputs = tokenizer(prompt, return_tensors="pt").to(device)  # Move inputs to GPU
    output = model.generate(**inputs, max_new_tokens=max_new_tokens, pad_token_id=tokenizer.eos_token_id)
    decoded_output = tokenizer.decode(output[0], skip_special_tokens=True)
    
    return decoded_output.split("Label:")[-1].strip()

df = pd.read_csv("suicide_ideation_test.csv")
# Prepare the dataset
true_labels = df['label'].tolist()  # Assuming the dataset has a 'label' column with true labels
predicted_labels = []

# Classify each tweet and collect predictions
for tweet in tqdm(df['text'], desc="Classifying tweets"):  # Assuming the dataset has a 'text' column with tweets
    prediction = classify_tweet(tweet, mode="few-shot")
    predicted_labels.append(prediction)

zipped = list(zip(true_labels, predicted_labels))
zipped = [(x, y) for x, y in zipped if "yes" in y.lower() or "no" in y.lower()]

true_labels, predicted_labels = zip(*zipped)

print("Samples:", len(true_labels))

predicted_labels = [1 if label.lower() == "yes" else 0 for label in predicted_labels]  # Convert labels to binary
# Calculate metrics
accuracy = accuracy_score(true_labels, predicted_labels)
precision = precision_score(true_labels, predicted_labels)
recall = recall_score(true_labels, predicted_labels)
f1 = f1_score(true_labels, predicted_labels)



print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")
