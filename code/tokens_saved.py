import pandas as pd

df = pd.read_csv("classified_prompts.csv")

token_classified = df["New Tokens (By Classification)"].sum()
token_all = df["New Tokens (All Guidelines)"].sum()

token_classified = token_classified / len(df)
token_all = token_all / len(df)

print(f"Average tokens per prompt (by classification): {token_classified}")
print(f"Average tokens per prompt (all guidelines): {token_all}")

tokens_saved = token_all - token_classified

print(f"Tokens saved by classification: {tokens_saved}")
print(f"Percentage of tokens saved by classification: {tokens_saved / token_all * 100:.2f}%")