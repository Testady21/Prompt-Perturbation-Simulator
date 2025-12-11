import json
import time
import os
from query_agent import QueryAgent
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

import torch
print(torch.cuda.is_available())  
 
print(torch.cuda.current_device())
print(torch.cuda.get_device_name(torch.cuda.current_device())) 

json_file = r"datasets/prompts.json"
output_file = r"datasets/prompt_response_dataset.jsonl"

BATCH_SIZE = 64   
MAX_WORKERS = 32   

 
llm = QueryAgent(model="llama3.2")
if os.path.exists(output_file):
    with open(output_file, "r", encoding="utf-8") as f:
        completed_data = [json.loads(line) for line in f if line.strip()]
    completed_prompts = {entry['prompt'] for entry in completed_data}
    print(f"⚡ Resuming from existing file: {len(completed_data)} prompts already done.")
else:
    completed_data = []
    completed_prompts = set()
with open(json_file, "r", encoding="utf-8") as f:
    all_data = json.load(f)

pending_data = [entry for entry in all_data if entry['prompt'] not in completed_prompts]
total_pending = len(pending_data)
print(f"🚀 Starting prompt-response generation for {total_pending} prompts\n")

def process_prompt(prompt_entry):
    prompt = prompt_entry["prompt"]
    for attempt in range(3):
        try:
            response = llm.query(prompt, history=False)
            return {"prompt": prompt, "response": response}
        except Exception as e:
            time.sleep(1)
    return {"prompt": prompt, "response": "ERROR: failed after 3 attempts"}

with open(output_file, "a", encoding="utf-8") as f_out:
    for i in tqdm(range(0, total_pending, BATCH_SIZE), desc="Processing batches", unit="batch"):
        batch = pending_data[i:i+BATCH_SIZE]

        results = []
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            future_to_prompt = {executor.submit(process_prompt, entry): entry for entry in batch}

            # Per-prompt progress bar
            for future in tqdm(as_completed(future_to_prompt), total=len(batch), desc="Prompts in batch", unit="prompt", leave=False):
                result = future.result()
                f_out.write(json.dumps(result, ensure_ascii=False) + "\n")
                f_out.flush()
                completed_prompts.add(result["prompt"])

print("\n🎉 DONE! Prompt-response dataset created successfully.")
print(f"📁 Saved to: {output_file}")
print(f"📦 Total entries written: {len(completed_prompts)}")
