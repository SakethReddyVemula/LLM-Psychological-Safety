
## Increased context of Pre-Processing Agent

Pre-Processing Agent now uses multiple prompts/answers (basically the conversation) to asses the guidelines.

Ex: If toxic content is first identified, help by LLM.
If repeated toxic content in prompts, prioritize critical resources like suicide hotline etc. 

So, guideline based on the previous context.

**Evidences:**
1. _Toxicity Detection can be Sensitive to the Conversational Context_ (Xenos et al., 2021) - "User posts whose perceived toxicity depends on the conversational context are rare in current toxicity detection datasets. Hence, toxicity detectors trained on existing datasets will also tend to disregard context…" 
2. CoSafe: Evaluating Large Language Model Safety in Multi-Turn Dialogue Coreference - https://aclanthology.org/2024.emnlp-main.968.pdf - "The results indicated that under multi-turn coreference safety attacks, the highest attack successful rate was 56% with the LLaMA2-Chat-7b model, while the lowest was 13.9% with the Mistral-7B-Instruct model. These findings highlight the safety vulnerabilities in LLMs during dialogue coreference interactions."

**Counter-Points:**

Why can't we prompt something like this to large LLM:
		Analyze the past 4 messages and identify for toxic content. If it has repeated > 3 times do this.....

**Reasons:**

1. Context Length - If conditions are all hard coded in the prompt - takes up token, cost increased per token.
2. Time - LLM has to go back and analyze 3 messages, takes a lot of time as this LLM is a large one.

Using light weight classifiers reduces time and computational overhead we are placing on the black box LLM.


**Methodology:**

1. Use multiple conversations (prompts + responses) for classification.
2. Identify if the toxicity is persistent, change the guidelines

**Datasets:**
1. CCC (Civil Comments in Context) (only the parent is available) - need to find more.
2. https://github.com/ErxinYu/CoSafe-Dataset/blob/main/CoSafe%20datasets/self_harm.json 
