import json
import os
import time
import sys
 
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
sys.path.insert(0, parent_dir)

from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed 
from query_agent import QueryAgent 

DATASET_FILE = "datasets/llm-tests/levels_qwen2.json"
LLM_MODEL = "qwen2"
MAX_ATTEMPTS = 5
MAX_CONCURRENT_QUERIES = 64 
SAVE_BATCH_SIZE = 10 
 
def save_data_safely(data, filename):
    """
    Saves the entire JSON data structure to a temporary file first,
    then renames it to the final filename (atomic operation). This prevents 
    data loss if the script is interrupted during the write process.
    """
    temp_filename = filename + ".tmp"
    try:
        with open(temp_filename, "w", encoding="utf-8") as f: 
            json.dump(data, f, indent=2, ensure_ascii=False)
        os.replace(temp_filename, filename)
    except Exception as e:
        print(f"\nFATAL: Failed to save data. Data might be in {temp_filename}. Error: {e}")


def load_data(filename):
    """
    Loads data with safety checks for file existence and JSON validity.
    """
    if not os.path.exists(filename):
        print(f"\nFATAL ERROR: Dataset file not found at '{filename}'")
        sys.exit(1)
    if os.path.getsize(filename) == 0:
        print(f"\nFATAL ERROR: The file '{filename}' is empty. Please restore the original JSON data.")
        sys.exit(1)
        
    try:
        with open(filename, "r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        print(f"\nFATAL ERROR: JSON Decode failed for '{filename}'. The file may be corrupted.")
        print(f"Error details: {e}")
        sys.exit(1)

def process_ground_truth(agent, i, data):
    entry = data[i] 
    if entry.get("ground_truth") and entry["ground_truth"].strip():
        return True  
        
    prompt = entry["original_prompt"]
    response = None

    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            response = agent.query(prompt, history=False) 
            break  
        except Exception as e:
            if attempt == MAX_ATTEMPTS:
                response = f"ERROR: LLM ground_truth query failed after {MAX_ATTEMPTS} attempts: {e}"
                 
    data[i]["ground_truth"] = response.strip() if response else ""
    return False  


def process_single_task(agent, i, j, data):
    test_case = data[i]["test_cases"][j]
     
    if test_case.get("perturbed_output") and test_case["perturbed_output"].strip():
        return True  

    prompt = test_case["perturbed_prompt"]
    response = None
    
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            response = agent.query(prompt, history=False) 
            break 
        except Exception as e:
            if attempt == MAX_ATTEMPTS:
                response = f"ERROR: LLM query failed after {MAX_ATTEMPTS} attempts: {e}"
                
   
    data[i]["test_cases"][j]["perturbed_output"] = response.strip() if response else ""
    
    return False  

 

if __name__ == "__main__":
    
    print(f"Initializing QueryAgent with model: {LLM_MODEL}...")
    try:
        agent = QueryAgent(model=LLM_MODEL, system_prompt="")
    except Exception as e:
        print(f"\nFATAL ERROR: Failed to initialize QueryAgent. Ensure Ollama is running and the model is pulled.")
        print(f"Error details: {e}")
        
    data = load_data(DATASET_FILE)

     
    gt_task_args = []  
    task_args = []     
    total_tasks = 0    
    completed_tasks = 0  
    gt_total = len(data) 
    gt_completed = 0   
    
    for i, entry in enumerate(data):
        
        if not entry.get("ground_truth") or not entry["ground_truth"].strip():
            gt_task_args.append(i) 
        else:
            gt_completed += 1

        total_tasks += len(entry["test_cases"])
        for j, test_case in enumerate(entry["test_cases"]):
            if test_case.get("perturbed_output") and test_case["perturbed_output"].strip():
                completed_tasks += 1
            else:
                task_args.append((i, j)) 

    pending_gt_tasks = gt_total - gt_completed
    pending_tasks = total_tasks - completed_tasks
    
    print(f"\nDataset loaded. Total entries: {gt_total}.")
    print(f" Ground Truth Status: {gt_completed} completed, {pending_gt_tasks} pending.")
    print(f" Perturbed Output Status: {completed_tasks} completed, {pending_tasks} pending.")
    print(f" Using {MAX_CONCURRENT_QUERIES} threads with a batch save size of {SAVE_BATCH_SIZE}.")

    
 
    if pending_gt_tasks > 0:
        print("\n--- Starting Ground Truth Generation Phase ---")
        gt_processed_in_batch = 0
        
        with tqdm(total=pending_gt_tasks, desc="Ground Truth Progress", unit="entry", initial=0) as pbar:
            with ThreadPoolExecutor(max_workers=MAX_CONCURRENT_QUERIES) as executor:
                
                
                gt_futures = [
                    executor.submit(process_ground_truth, agent, i, data) 
                    for i in gt_task_args
                ]

                for future in as_completed(gt_futures):
                    was_skipped = future.result() 
                    
                    if not was_skipped:
                        pbar.update(1)
                        gt_processed_in_batch += 1
                    
                    if gt_processed_in_batch >= SAVE_BATCH_SIZE:
                        print(f"\n[Ground Truth Batch Save]: Saving progress after {gt_processed_in_batch} entries...")
                        save_data_safely(data, DATASET_FILE)
                        gt_processed_in_batch = 0
                        
            if gt_processed_in_batch > 0:
                print(f"\n[Ground Truth Final Save]: Saving remaining {gt_processed_in_batch} entries...")
                save_data_safely(data, DATASET_FILE)
        
        print("--- Ground Truth Generation Phase Completed ---")

    if pending_tasks > 0:
        print("\n--- Starting Perturbed Output Generation Phase ---")
        tasks_processed_in_batch = 0
        
        with tqdm(total=pending_tasks, desc="Perturbed Output Progress", unit="task", initial=0) as pbar:
            
            with ThreadPoolExecutor(max_workers=MAX_CONCURRENT_QUERIES) as executor:
                
                futures = [
                    executor.submit(process_single_task, agent, i, j, data) 
                    for i, j in task_args
                ]
 
                for future in as_completed(futures):
                    
                    was_skipped = future.result() 
                    
                    if not was_skipped:
                        pbar.update(1)
                        tasks_processed_in_batch += 1
                    
                    if tasks_processed_in_batch >= SAVE_BATCH_SIZE:
                        print(f"\n[Perturbed Output Batch Save]: Saving progress after {tasks_processed_in_batch} tasks...")
                        save_data_safely(data, DATASET_FILE)
                        tasks_processed_in_batch = 0
            
            if tasks_processed_in_batch > 0:
                print(f"\n[Perturbed Output Final Save]: Saving remaining {tasks_processed_in_batch} tasks...")
                save_data_safely(data, DATASET_FILE)

        print("\n DONE! All perturbed outputs processed and saved.")
        
    
    if pending_gt_tasks == 0 and pending_tasks == 0:
        print("\n All tasks (Ground Truth and Perturbed Output) were already completed. No processing needed.")
        
    print(f"\n Final data saved to {DATASET_FILE}")