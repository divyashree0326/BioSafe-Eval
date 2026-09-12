import os
import time
import pandas as pd
from groq import Groq, RateLimitError
from prompts_v2 import all_prompts

api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise ValueError("GROQ_API_KEY was not found.")

client = Groq(api_key=api_key)

models = [
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b",
    "openai/gpt-oss-safeguard-20b"
]

results = []

for model in models:

    print(f"\nTesting model: {model}")

    for item in all_prompts:

        print(f"  Testing: {item['level']}")

        while True:

            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=[
                        {
                            "role": "user",
                            "content": item["prompt"]
                        }
                    ],
                )

                break

            except RateLimitError:
                print("    Rate limit reached. Waiting 10 seconds...")
                time.sleep(10)

        answer = response.choices[0].message.content

        results.append({
            "model": model,
            "topic": item["topic"],
            "level": item["level"],
            "prompt": item["prompt"],
            "response": answer
        })

        time.sleep(2)

df = pd.DataFrame(results)

df.to_csv("results_v2.csv", index=False)

print("\nExperiment complete!")
print(f"Total responses collected: {len(df)}")
print("Saved to results_v2.csv")