import random
import nltk
from nltk.corpus import wordnet, stopwords
from nltk.tokenize import sent_tokenize
import json
import re

nltk.download("punkt", quiet=True)
nltk.download("wordnet", quiet=True)
nltk.download("stopwords", quiet=True)

 
WORDBUG_RATIO = 0.25  
TEXTFOOLER_COUNT = 5 


# --- 1. WORDBUG (Character-level noise) ---
def random_char_perturb(word):
    """Applies a single random character-level perturbation (delete, swap, insert, replace)."""
    if len(word) <= 2:
        return word

    operations = ["delete", "swap", "insert", "replace"]
    op = random.choice(operations)
    idx = random.randint(0, len(word) - 1)

    if op == "delete":
        return word[:idx] + word[idx + 1:]
    elif op == "swap" and idx < len(word) - 1:
        # Check if the swap is possible (not the last character)
        return word[:idx] + word[idx + 1] + word[idx] + word[idx + 2:]
    elif op == "insert":
        # Insert a random lowercase letter
        return word[:idx] + random.choice("abcdefghijklmnopqrstuvwxyz") + word[idx:]
    elif op == "replace":
        # Replace with a random lowercase letter
        return word[:idx] + random.choice("abcdefghijklmnopqrstuvwxyz") + word[idx + 1:]
    return word

def perturb_wordbug(prompt, perturb_ratio=WORDBUG_RATIO):
    """Applies character-level perturbations to a percentage of words."""
    words = nltk.word_tokenize(prompt)
    count = max(1, int(len(words) * perturb_ratio))
    
    # Filter out punctuation for perturbation selection, but keep them in the list
    word_indices = [i for i, w in enumerate(words) if w.isalnum()]
    
    if len(word_indices) < count:
        count = len(word_indices) 

    indices_to_perturb = random.sample(word_indices, count)

    for i in indices_to_perturb:
        words[i] = random_char_perturb(words[i])
    return " ".join(words)


# --- 2. TEXTFOOLER (Synonym substitution) ---
def get_synonym(word):
    """Returns a random, different synonym for a word, or None."""
    synsets = wordnet.synsets(word)
    if not synsets:
        return None
    
    # Collect unique candidate synonyms
    candidates = set()
    for synset in synsets:
        for lemma in synset.lemmas():
            replacement = lemma.name().replace("_", " ")
            if replacement.lower() != word.lower():
                 candidates.add(replacement)
    
    return random.choice(list(candidates)) if candidates else None

def perturb_textfooler(prompt, num_replacements=TEXTFOOLER_COUNT):
    """Replaces a fixed number of words with synonyms."""
    words = nltk.word_tokenize(prompt)
    indices = list(range(len(words)))
    random.shuffle(indices)

    replaced = 0
    for i in indices:
        if replaced >= num_replacements:
            break
        # Only try to replace alphanumeric words
        if words[i].isalnum():
            s = get_synonym(words[i])
            if s:
                words[i] = s
                replaced += 1

    # Reconstruct the sentence
    return " ".join(words)


# --- 3. WORDBUG + TEXTFOOLER (Combined) ---
def perturb_combined(prompt):
    """Applies both WORDBUG (Level 5) and TEXTFOOLER (Level 5)."""
    bugged_prompt = perturb_wordbug(prompt, WORDBUG_RATIO)
    fooler_bugged_prompt = perturb_textfooler(bugged_prompt, TEXTFOOLER_COUNT)
    return fooler_bugged_prompt


# --- 4. RANDOM SHUFFLING OF PHRASES ---
def perturb_phrase_shuffling(prompt):
    """Tokenizes into phrases (sentences) and shuffles their order."""
    sentences = sent_tokenize(prompt)
    if len(sentences) <= 1:
        # If there's only one sentence, the wordss in the middle will be shuffled
        words = nltk.word_tokenize(prompt)
        # Select a phrase in the middle  
        start_idx = max(1, int(len(words) * 0.2))
        end_idx = min(len(words) - 1, int(len(words) * 0.8))

        if end_idx - start_idx >= 2:
            middle_words = words[start_idx:end_idx]
            random.shuffle(middle_words)
            shuffled_words = words[:start_idx] + middle_words + words[end_idx:]
            return " ".join(shuffled_words)
        else:
            return prompt
    
    random.shuffle(sentences)
    return " ".join(sentences)


# --- 5. STOPWORD REMOVAL/INSERTION (Randomly remove or insert common stopwords) ---
def perturb_stopword(prompt, remove_ratio=0.10, insert_count=2):
    """Removes some stopwords and inserts a few random ones in random locations."""
    stop_words = set(stopwords.words('english'))
    words = nltk.word_tokenize(prompt)
    
    processed_words = []
    
    # 1. Removal
    for word in words:
        if word.lower() not in stop_words or random.random() > remove_ratio:
            processed_words.append(word)
    
    # 2. Insertion
    # Filter for common short stopwords to insert
    short_stopwords = [w for w in list(stop_words)[:50] if len(w) <= 3]
    
    if not short_stopwords:
        return " ".join(processed_words)

    for _ in range(insert_count):
        # Insert a stopword at a random, non-punctuation position
        if processed_words:
            insert_idx = random.randint(0, len(processed_words))
            random_stopword = random.choice(short_stopwords)
            processed_words.insert(insert_idx, random_stopword)
    
    return " ".join(processed_words)


# --- 6. CASING PERTURBATION ---
def perturb_casing(prompt, word_casing_ratio=0.25):
    """Randomly changes the casing of words and injects Title/UPPER casing."""
    words = nltk.word_tokenize(prompt)
    
    for i in range(len(words)):
        word = words[i]
        # Only perturb casing for alphanumeric words
        if word.isalnum():
            r = random.random()
            if r < word_casing_ratio:
                # Randomly change to UPPERCASE or Title Case
                if random.choice([True, False]):
                    words[i] = word.upper()
                else:
                    words[i] = word.title()
            elif r < 2 * word_casing_ratio:
                # Randomly change to lowercase
                words[i] = word.lower()
    
    # Re-apply Title Case to the very first word for a common casing change
    if words and words[0].isalnum():
        words[0] = words[0].title()

    return " ".join(words)


# --- 7. PARAPHRASING MODELS (Simulated by extensive synonym substitution) ---
def perturb_paraphrase_simulated(prompt, num_replacements_max=10):
    """Simulates paraphrasing by replacing a large number of words with synonyms."""
    words = nltk.word_tokenize(prompt)
    indices = list(range(len(words)))
    random.shuffle(indices)

    replaced = 0
    max_replacements = min(num_replacements_max, len(words) // 2)

    for i in indices:
        if replaced >= max_replacements:
            break
        
        if words[i].isalnum():
            s = get_synonym(words[i])
            if s:
                words[i] = s
                replaced += 1

    return " ".join(words)


# --- 8. NOISE / PUNCTUATION INJECTION ---
def perturb_punctuation_injection(prompt, injection_count=4):
    """Inserts random punctuation/symbols in random locations."""
    words = nltk.word_tokenize(prompt)
    punctuation = list("!@#$%^&*()_+-=[]{}\\|;:'\",.<>/?`~")
    
    for _ in range(injection_count):
        if words:
            # Randomly select a position to insert a word/punctuation
            insert_idx = random.randint(0, len(words))
            # Choose a random punctuation mark
            random_punc = random.choice(punctuation)
            words.insert(insert_idx, random_punc)
    
    return " ".join(words)


# --- 9. HOMOPHONE SUBSTITUTION (Simple rule-based substitution) ---
CONFUSED_WORDS = {
    # Pronoun Errors
    'their': ['there', 'they\'re'], 'there': ['their', 'they\'re'], 'they\'re': ['their', 'there'],
    'to': ['too', 'two'], 'too': ['to', 'two'], 'two': ['to', 'too'],
    'you\'re': ['your'], 'your': ['you\'re'],
    'it\'s': ['its'], 'its': ['it\'s'],

    # Commonly Confused Verb/Noun Pairs
    'then': ['than'], 'than': ['then'],
    'affect': ['effect'], 'effect': ['affect'],
    'accept': ['except'], 'except': ['accept'],
    'principal': ['principle'], 'principle': ['principal'],
    'complement': ['compliment'], 'compliment': ['complement'],
    'lie': ['lay'], 'lay': ['lie'], 
    'advise': ['advice'], 'advice': ['advise'],
    'allot': ['a lot'],
    
    # True Homophones & Near Homophones
    'write': ['right', 'rite'], 'right': ['write', 'rite'],
    'one': ['won'], 'won': ['one'],
    'hear': ['here'], 'here': ['hear'],
    'know': ['no'], 'no': ['know'],
    'whole': ['hole'], 'hole': ['whole'],
    'where': ['wear'], 'wear': ['where'],
    'buy': ['by', 'bye'], 'by': ['buy', 'bye'],
    'for': ['four'], 'four': ['for'],
    'be': ['bee'], 'bee': ['be'],
    'sun': ['son'], 'son': ['sun'],
    'read': ['red'], 'red': ['read'],
    'would': ['wood'], 'wood': ['would'],
    'break': ['brake'], 'brake': ['break'],
    'peace': ['piece'], 'piece': ['peace'],
    'aloud': ['allowed'], 'allowed': ['aloud'],
    'dear': ['deer'], 'deer': ['dear'],
    'wait': ['weight'], 'weight': ['wait'],
    'cent': ['scent', 'sent'], 'scent': ['cent', 'sent'], 'sent': ['cent', 'scent'],
    'sail': ['sale'], 'sale': ['sail'],
    'weather': ['whether'], 'whether': ['weather'],
}

# --- PERTURBATION FUNCTION ---
def perturb_homophone(prompt, max_substitutions=2):
    """
    Substitutes words with commonly confused counterparts (homophones/near-homophones).

    Args:
        prompt (str): The original text prompt.
        max_substitutions (int): The maximum number of substitutions to attempt.

    Returns:
        str: The perturbed prompt.
    """ 
    words = nltk.word_tokenize(prompt)
    
    substitutions_made = 0
    indices = list(range(len(words)))
    random.shuffle(indices)

    for i in indices:
        if substitutions_made >= max_substitutions:
            break

        word_token = words[i]
        word_lower = word_token.lower()
    
        if word_lower in CONFUSED_WORDS:
            substitute = random.choice(CONFUSED_WORDS[word_lower])
            
            
            if word_token.istitle() and substitute not in ['a lot']: 
                words[i] = substitute.title()
            elif word_token.isupper():
                words[i] = substitute.upper()
            else:
                words[i] = substitute
                
            substitutions_made += 1

    perturbed_text = " ".join(words)
    perturbed_text = re.sub(r'\s([?.!,;:\'])', r'\1', perturbed_text)
    
    return perturbed_text


# --- 10. TYPOGRAPHIC SWAP (Simulated "fat finger" keyboard errors) ---
KEYBOARD_ADJACENT = {
    'q': 'wa', 'w': 'qe', 'e': 'wr', 'r': 'et', 't': 'ry', 'y': 'tu', 'u': 'yi', 'i': 'uo', 'o': 'ip', 'p': 'o[',
    'a': 'qs', 's': 'ad', 'd': 'sf', 'f': 'dg', 'g': 'fh', 'h': 'gj', 'j': 'hk', 'k': 'jl', 'l': 'k;',
    'z': 'sx', 'x': 'zc', 'c': 'xv', 'v': 'cb', 'b': 'vn', 'n': 'bm', 'm': 'n,',
}

def perturb_typographic_swap(prompt, perturb_ratio=0.15):
    words = nltk.word_tokenize(prompt)
    count = max(1, int(len(words) * perturb_ratio))
    
    word_indices = [i for i, w in enumerate(words) if w.isalnum()]
    
    if len(word_indices) < count:
        count = len(word_indices)

    indices_to_perturb = random.sample(word_indices, count)

    for i in indices_to_perturb:
        word = words[i]
        if len(word) > 1:
            idx = random.randint(0, len(word) - 1)
            char = word[idx].lower()
            
            if char in KEYBOARD_ADJACENT:
                # Replace the character with a random adjacent character
                adjacent_chars = KEYBOARD_ADJACENT[char]
                new_char = random.choice(adjacent_chars)
                
                # Maintain the original case of the character
                if word[idx].isupper():
                    new_char = new_char.upper()

                words[i] = word[:idx] + new_char + word[idx + 1:]

    return " ".join(words)


###########################################################
# GENERATE VARIANTS
###########################################################
def generate_variants(prompt):
    """
    Generates 10 different types of perturbations, one variant for each type.
    """
    perturbation_types = [
        ("wordbug_lvl5", "Wordbug (25% ratio)", perturb_wordbug),
        ("textfooler_lvl5", "TextFooler (5 replacements)", perturb_textfooler),
        ("wordbug_textfooler_combined", "Wordbug + TextFooler", perturb_combined),
        ("phrase_shuffling", "Random Shuffling of Phrases/Words", perturb_phrase_shuffling),
        ("stopword_removal_insertion", "Stopword Removal/Insertion", perturb_stopword),
        ("casing_perturbation", "Casing Perturbation", perturb_casing),
        ("paraphrasing_simulated", "Paraphrasing Models (Simulated)", perturb_paraphrase_simulated),
        ("noise_punctuation_injection", "Noise / Punctuation Injection", perturb_punctuation_injection),
        ("homophone_substitution", "Homophone Substitution", perturb_homophone),
        ("typographic_swap", "Typographic Swap (Adjacent Keys)", perturb_typographic_swap),
    ]

    variants = []
    variant_id = 1

    for ptype_id, ptype_name, p_func in perturbation_types:
        perturbed_text = p_func(prompt)
        if ptype_id != "phrase_shuffling":
            words = nltk.word_tokenize(perturbed_text)
            final_text = " ".join(words)
        else:
             final_text = perturbed_text  
        final_text = re.sub(r'\s([?.!,;])', r'\1', final_text)

        variants.append({
            "variant_id": f"{variant_id:02d}",
            "perturbation_type": ptype_id,
            "perturbed_prompt": final_text,
            "perturbed_output": ""
        })
        variant_id += 1

    return variants


###########################################################
# JSON GENERATION FROM DATASET
###########################################################
def generate_json_from_file(input_path, output_path):
    dataset_json = []
    entry_id = 1

    print(f"Reading from: {input_path}")
    try:
        with open(input_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue

                entry = json.loads(line)
                prompt = entry["prompt"]
                response = entry["response"]

                variants = generate_variants(prompt)
                
                dataset_json.append({
                    "id": f"{entry_id:03d}",
                    "task_name": f"Task_{entry_id}",
                    "original_prompt": prompt,
                    "ground_truth": response,
                    "test_cases": variants
                })

                entry_id += 1
                if entry_id % 10 == 0:
                    print(f"Processed {entry_id-1} entries...")

    except FileNotFoundError:
        print(f"Error: Input file not found at {input_path}")
        return
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON format in input file: {input_path}")
        return

    print(f"Finished processing {len(dataset_json)} entries.")
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(dataset_json, f, indent=2)

    print(f"Saved structured dataset to: {output_path}")


###########################################################
# MAIN EXECUTION
###########################################################
if __name__ == "__main__":
    input_file = r"datasets/prompt_response_dataset.jsonl"
    output_file = r"datasets/base_variants_dataset.json"
    generate_json_from_file(input_file, output_file)