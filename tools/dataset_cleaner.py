import json
import os
from typing import List, Dict, Any

# Define the expected structure for type hinting (optional but good practice)
DatasetRecord = Dict[str, Any]

def remove_ground_truth_values(input_filename: str, output_filename: str) -> None:
    """
    Loads a JSON dataset, removes the value of the 'ground_truth' key
    in each record by replacing it with an empty string (""), and saves
    the modified data to a new JSON file.

    Args:
        input_filename: The name of the original JSON file to read.
        output_filename: The name of the JSON file to save the modified data.
    """
    if not os.path.exists(input_filename):
        print(f"Error: The file '{input_filename}' was not found.")
        return

    try:
        # 1. Load the dataset
        with open(input_filename, 'r', encoding='utf-8') as f:
            data: List[DatasetRecord] = json.load(f)

        print(f"✅ Successfully loaded dataset from '{input_filename}' with {len(data)} records.")

        modified_count = 0
        # 2. Iterate through the dataset and modify the 'ground_truth' field
        if isinstance(data, list):
            for record in data:
                # Check for the key and ensure the record is a dictionary
                if isinstance(record, dict) and 'ground_truth' in record:
                    # Replace the value with an empty string ""
                    record['ground_truth'] = ""
                    modified_count += 1
            
            print(f"🔄 Successfully modified 'ground_truth' for {modified_count} records.")

            # 3. Save the modified dataset
            with open(output_filename, 'w', encoding='utf-8') as f:
                # Use indent=2 for human-readable formatting
                json.dump(data, f, indent=2)

            print(f"💾 Modified dataset saved to '{output_filename}'")
            print("Done.")
        else:
            print("❌ Error: The loaded JSON content is not a list (expected dataset format).")

    except json.JSONDecodeError:
        print(f"❌ Error: Could not decode JSON from '{input_filename}'. Please check file formatting.")
    except Exception as e:
        print(f"❌ An unexpected error occurred: {e}")

# ----------------------------------------------------------------------
def main():
    """
    Main function to execute the dataset modification.
    """
    # Define the file names
    INPUT_FILE = r"NLP\Prompt-Perturbation-Simulator\advait\variants_base_dataset.json"
    OUTPUT_FILE = r"NLP\Prompt-Perturbation-Simulator\advait\clean_variants_dataset.json"
    
    # Check if the user wants to overwrite the original file
    # If you want to overwrite, change OUTPUT_FILE to INPUT_FILE
    # OUTPUT_FILE = INPUT_FILE 
    
    print("--- Starting Dataset Ground Truth Removal Process ---")
    
    # Call the core function
    remove_ground_truth_values(
        input_filename=INPUT_FILE,
        output_filename=OUTPUT_FILE
    )
    
    print("-----------------------------------------------------")

if __name__ == "__main__":
    main()