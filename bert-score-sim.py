import json
import pandas as pd
from bert_score import score
from datetime import datetime
from tqdm import tqdm
import re

 
FILE_NAME = r"datasets/llm-tests/variants_qwen2.json"
 

def process_and_score_dataset(file_path):
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data_json_string = f.read()
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return None, None
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return None, None
    
    data_json_string = data_json_string[:data_json_string.rfind(']')] + "]"
    
   
    data_json_string = re.sub(r'(?<!\\)\\n', r'\\n', data_json_string) 

    data = json.loads(data_json_string)
    records = []

    for task in data:
        ground_truth = task['ground_truth']
        for test_case in task['test_cases']:
            records.append({
                'task_name': task['task_name'],
                'perturbation_type': test_case['perturbation_type'],
                'ground_truth': ground_truth,
                'perturbed_output': test_case['perturbed_output']
            })

    df = pd.DataFrame(records)
    
    if df.empty:
        print("Dataset is empty. Cannot run scoring.")
        return None, None

    candidates = df['perturbed_output'].tolist()
    references = df['ground_truth'].tolist()
 
    print("Calculating BERTScore (F1)... This may take a few minutes for a large dataset.")
    
  
    P, R, F1 = score(
        candidates, 
        references, 
        lang="en", 
        model_type="bert-base-uncased",
        verbose=True   
    )

    df['BERTScore_F1'] = F1.tolist()

    print("BERTScore calculation complete.")

    results = df.groupby(['perturbation_type'])['BERTScore_F1'].mean().reset_index()
    results.columns = ['Perturbation Type', 'Average BERTScore F1']

    overall_avg = df['BERTScore_F1'].mean()

    return results, overall_avg

def format_results_for_file(results_df, overall_avg, title):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    output_lines = [
        f"==={title}===",
        f"Simulation Date and Time (UTC): {now}",
        f"Metric: BERTScore F1 (Reference: Ground Truth, Candidate: Perturbed Output)",
        "-" * 70,
        "## Average BERTScore F1 by Perturbation Type (Relative to Ground Truth)",
        "-" * 70
    ]
    
 
    table_string = results_df.to_string(index=False)
    output_lines.append(table_string)
    
    output_lines.extend([
        "-" * 70,
        f"Overall Average BERTScore F1: {overall_avg:.4f}",
        "===\n\n"
    ])
    
    return "\n".join(output_lines)
 
if __name__ == '__main__':
    
 
    simulation_title = FILE_NAME
 
    scores_df, overall_score = process_and_score_dataset(FILE_NAME)
    
    if scores_df is not None:
 
        formatted_output = format_results_for_file(scores_df, overall_score, simulation_title)
        
 
        file_name = r"results.txt"
        try:
            with open(file_name, 'a') as f:
                f.write(formatted_output)
            print(f"\n✅ Successfully appended results to '{file_name}'.")
            print("\nSummary of Results:")
            print(scores_df.to_string(index=False))
            print(f"\nOverall Average F1: {overall_score:.4f}")
        except Exception as e:
            print(f"\n❌ An error occurred while writing to the file: {e}")