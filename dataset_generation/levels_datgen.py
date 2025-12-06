import random
import nltk
from nltk.corpus import wordnet
import json

# Ensure NLTK resources
nltk.download("punkt", quiet=True)
nltk.download("wordnet", quiet=True)


###########################################################
# WORDBUG (character-level noise)
###########################################################
def random_char_perturb(word):
    if len(word) <= 2:
        return word

    operations = ["delete", "swap", "insert", "replace"]
    op = random.choice(operations)
    idx = random.randint(0, len(word) - 1)

    if op == "delete":
        return word[:idx] + word[idx + 1:]
    elif op == "swap" and idx < len(word) - 1:
        return word[:idx] + word[idx + 1] + word[idx] + word[idx + 2:]
    elif op == "insert":
        return word[:idx] + random.choice("abcdefghijklmnopqrstuvwxyz") + word[idx:]
    elif op == "replace":
        return word[:idx] + random.choice("abcdefghijklmnopqrstuvwxyz") + word[idx + 1:]
    return word


def apply_wordbug(prompt, perturb_ratio):
    words = nltk.word_tokenize(prompt)
    count = max(1, int(len(words) * perturb_ratio))
    indices = random.sample(range(len(words)), count)

    for i in indices:
        words[i] = random_char_perturb(words[i])

    return " ".join(words)


###########################################################
# TEXTFOOLER STYLE (synonym substitutions)
###########################################################
def synonym(word):
    synsets = wordnet.synsets(word)
    if not synsets:
        return None
    candidates = synsets[0].lemma_names()
    replacement = random.choice(candidates).replace("_", " ")
    return replacement if replacement.lower() != word.lower() else None


def apply_textfooler(prompt, num_replacements):
    words = nltk.word_tokenize(prompt)
    indices = list(range(len(words)))
    random.shuffle(indices)

    replaced = 0
    for i in indices:
        if replaced >= num_replacements:
            break
        s = synonym(words[i])
        if s:
            words[i] = s
            replaced += 1

    return " ".join(words)


###########################################################
# GENERATE VARIANTS
###########################################################
def generate_variants(prompt):
    variants = []
    variant_id = 1

    # 5 WordBug variants
    for pct in [0.05, 0.10, 0.15, 0.20, 0.25]:
        variants.append((
            f"{variant_id:02d}",
            f"wordbug{variant_id}",
            apply_wordbug(prompt, pct)
        ))
        variant_id += 1

    # 5 TextFooler variants
    for k in [1, 2, 3, 4, 5]:
        variants.append((
            f"{variant_id:02d}",
            f"textfooler{variant_id-5}",
            apply_textfooler(prompt, k)
        ))
        variant_id += 1

    return variants


###########################################################
# JSON GENERATION FROM DATASET
###########################################################
def generate_json_from_file(input_path, output_path):
    dataset_json = []
    entry_id = 1

    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue

            entry = json.loads(line)
            prompt = entry["prompt"]
            response = entry["response"]

            variants = generate_variants(prompt)
            test_cases = []

            for vid, ptype, text in variants:
                test_cases.append({
                    "variant_id": vid,
                    "perturbation_type": ptype,
                    "perturbed_prompt": text,
                    "perturbed_output": ""
                })

            dataset_json.append({
                "id": f"{entry_id:03d}",
                "task_name": f"Task_{entry_id}",
                "original_prompt": prompt,
                "ground_truth": response,
                "test_cases": test_cases
            })

            entry_id += 1

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(dataset_json, f, indent=2)

    print(f"Saved structured dataset to: {output_path}")


###########################################################
# MAIN
###########################################################
if __name__ == "__main__":
    input_file = r"NLP\Prompt-Perturbation-Simulator\advait\pr_dataset_pro.jsonl"
    output_file = r"NLP\Prompt-Perturbation-Simulator\advait\pdat_wbtf_lvl.json"
    generate_json_from_file(input_file, output_file)
