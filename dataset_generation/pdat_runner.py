import json
import os
import time
import sys
from tqdm import tqdm
# 🚨 NEW: Import for concurrent execution
from concurrent.futures import ThreadPoolExecutor, as_completed 

# --- Configuration ---

# The dataset file to be processed (will be modified in place)
DATASET_FILE = "/home/aditya/advait/wordbug_textfooler_levels_dat.json"

# The LLM model name to use (e.g., "llama3.2" as seen in your test.py)
LLM_MODEL = "llama3.2"

# The maximum number of attempts to query the LLM if an error occurs
MAX_ATTEMPTS = 5

# --- PERFORMANCE CONFIGURATION ---

# Max number of concurrent threads/queries. Use a high number to keep the A100 busy.
MAX_CONCURRENT_QUERIES = 64 

# Save the dataset after this many tasks are completed. Reduces disk I/O contention.
SAVE_BATCH_SIZE = 10 
# ---------------------------------

# Assuming QueryAgent is defined in query_agent.py
from query_agent import QueryAgent 

# --- Helper Functions ---

def save_data_safely(data, filename):
    """
    Saves the entire JSON data structure to a temporary file first,
    then renames it to the final filename (atomic operation). This prevents 
    data loss if the script is interrupted during the write process.
    """
    temp_filename = filename + ".tmp"
    try:
        with open(temp_filename, "w", encoding="utf-8") as f:
            # ensure_ascii=False keeps non-ASCII characters readable
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        # Atomically replace the old file with the new one
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
    
    # Check if the file is empty (common cause of the original error)
    if os.path.getsize(filename) == 0:
        print(f"\nFATAL ERROR: The file '{filename}' is empty. Please restore the original JSON data.")
        sys.exit(1)
        
    try:
        with open(filename, "r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        # Catches the 'Expecting value' error and similar corruption issues
        print(f"\nFATAL ERROR: JSON Decode failed for '{filename}'. The file may be corrupted.")
        print(f"Error details: {e}")
        sys.exit(1)

# --- Concurrent Task Function ---

def process_single_task(agent, i, j, data):
    """
    Handles the LLM query logic for a single test case with retries.
    
    i (int): Index of the entry in the main data list.
    j (int): Index of the test_case within the entry's test_cases list.
    data (list): Reference to the main data list (modified in place).
    
    Returns: True if the task was already completed (skipped), False otherwise.
    """
    # Use the indices to access the specific test case
    test_case = data[i]["test_cases"][j]
    
    # 1. Resume Check
    if test_case.get("perturbed_output") and test_case["perturbed_output"].strip():
        return True # Task was skipped

    prompt = test_case["perturbed_prompt"]
    
    # print statements are disabled in the thread function to avoid race condition 
    # and cluttering the console, but can be re-enabled for deep debugging.
    # print(f"\n-> Processing Task {data[i]['id']}, Variant {test_case['variant_id']}:") 

    response = None
    
    # 2. Retry loop for resilience
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            # print(f"   Attempt {attempt}/{MAX_ATTEMPTS}...")
            response = agent.query(prompt, history=False) 
            break # Success, exit retry loop
        except Exception as e:
            # print(f"   Query failed (Attempt {attempt}): {e}")
            if attempt == MAX_ATTEMPTS:
                response = f"ERROR: LLM query failed after {MAX_ATTEMPTS} attempts: {e}"
                # print("   Marking task as failed.")
            else:
                # Reduced wait time for faster retries when GPU is active
                time.sleep(1) 

    # 3. Update the data structure (thread-safe modification of a list element)
    # The reference 'data' is shared, but only one thread updates a specific (i, j) element.
    data[i]["test_cases"][j]["perturbed_output"] = response.strip() if response else ""
    
    return False # Task was processed

# --- Main Script ---

if __name__ == "__main__":
    
    print(f"🚀 Initializing QueryAgent with model: {LLM_MODEL}...")
    try:
        agent = QueryAgent(model=LLM_MODEL, system_prompt="")
    except Exception as e:
        print(f"\nFATAL ERROR: Failed to initialize QueryAgent. Ensure Ollama is running and the model is pulled.")
        print(f"Error details: {e}")
        sys.exit(1)

    # Load the dataset
    data = load_data(DATASET_FILE)

    # 1. Pre-collect all pending task arguments
    task_args = []
    total_tasks = 0
    completed_tasks = 0
    
    for i, entry in enumerate(data):
        total_tasks += len(entry["test_cases"])
        for j, test_case in enumerate(entry["test_cases"]):
            if test_case.get("perturbed_output") and test_case["perturbed_output"].strip():
                completed_tasks += 1
            else:
                # Store the indices (i, j) for pending tasks
                task_args.append((i, j)) 

    pending_tasks = total_tasks - completed_tasks
    
    print(f"Dataset loaded. Total tasks: {total_tasks}.")
    if completed_tasks > 0:
        print(f"⚡ Resuming progress: {completed_tasks} tasks already completed.")
    print(f"⏳ {pending_tasks} tasks pending.")
    print(f"⚙️ Using {MAX_CONCURRENT_QUERIES} threads with a batch save size of {SAVE_BATCH_SIZE}.")

    
    # 2. Execute tasks using ThreadPoolExecutor
    tasks_processed_in_batch = 0
    
    with tqdm(total=pending_tasks, desc="Overall Progress", unit="task", initial=0) as pbar:
        
        # Initialize the ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=MAX_CONCURRENT_QUERIES) as executor:
            
            # Submit tasks to the executor
            futures = [
                executor.submit(process_single_task, agent, i, j, data) 
                for i, j in task_args
            ]

            # 3. Iterate over the results as they complete (as_completed is non-blocking)
            for future in as_completed(futures):
                
                # Check the result (True means task was skipped, shouldn't happen here)
                was_skipped = future.result() 
                
                if not was_skipped:
                    pbar.update(1)
                    tasks_processed_in_batch += 1
                
                # 4. CRITICAL: Batch Saving Logic
                # This significantly reduces the overhead of writing the massive JSON file
                if tasks_processed_in_batch >= SAVE_BATCH_SIZE:
                    print(f"\n[Batch Save]: Saving progress after {tasks_processed_in_batch} tasks...")
                    save_data_safely(data, DATASET_FILE)
                    tasks_processed_in_batch = 0
        
        # 5. Final Save: Ensure any remaining unsaved tasks are written to disk
        if tasks_processed_in_batch > 0:
            print(f"\n[Final Save]: Saving remaining {tasks_processed_in_batch} tasks...")
            save_data_safely(data, DATASET_FILE)

    print("\n🎉 DONE! All perturbed outputs processed and saved.")
    print(f"📁 Final data saved to {DATASET_FILE}")