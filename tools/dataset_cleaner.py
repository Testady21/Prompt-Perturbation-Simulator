import json
import os
from typing import List, Dict, Any

DatasetRecord = Dict[str, Any]

def remove_ground_truth_values(input_filename: str, output_filename: str) -> None:
    if not os.path.exists(input_filename):
        print(f"Error: The file '{input_filename}' was not found.")
        return

    try:
        with open(input_filename, 'r', encoding='utf-8') as f:
            data: List[DatasetRecord] = json.load(f)

        print(f"Successfully loaded dataset from '{input_filename}' with {len(data)} records.")

        modified_count = 0
        if isinstance(data, list):
            for record in data:
                if isinstance(record, dict) and 'ground_truth' in record:
                    record['ground_truth'] = ""
                    modified_count += 1
            
            print(f"Successfully modified 'ground_truth' for {modified_count} records.")

            with open(output_filename, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)

            print(f"Modified dataset saved to '{output_filename}'")
            print("Done.")
        else:
            print("Error: The loaded JSON content is not a list (expected dataset format).")

    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from '{input_filename}'. Please check file formatting.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

# ----------------------------------------------------------------------
def main():
    INPUT_FILE = r"NLP\Prompt-Perturbation-Simulator\advait\variants_base_dataset.json"
    OUTPUT_FILE = r"NLP\Prompt-Perturbation-Simulator\advait\clean_variants_dataset.json"
     
    print("--- Starting Dataset Ground Truth Removal Process ---")
    
    remove_ground_truth_values(
        input_filename=INPUT_FILE,
        output_filename=OUTPUT_FILE
    )
    
    print("-----------------------------------------------------")

if __name__ == "__main__":
    main()