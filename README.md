# Prompt Perturbation Simulator

Prompt Perturbation Simulator is an NLP research and experimentation framework for generating, modifying, and evaluating prompt–response datasets. The project supports automated dataset creation, perturbation strategies, and model interaction workflows for analyzing LLM robustness.

---

## 🚀 Features

- 🔁 Automatic prompt–response dataset generation
- 🧪 Modular framework for perturbing and transforming prompts
- 🤖 Integration with LLM agents for query/response evaluation
- 📦 JSON / JSONL dataset output formats
- 🔍 Structured simulation and evaluation workflows

---

## 📂 Project Structure

```
Prompt-Perturbation-Simulator/
│   .gitattributes
│   bert-score-sim.py
│   LICENSE
│   possible tests.txt
│   query_agent.py
│   README.md
│   requirements.txt
│   results.txt
│
├───datasets
│       base_dataset.json
│       base_variants_dataset.json
│       clean_levels_dataset.json
│       clean_variants_dataset.json
│       levels_llama3.2.json
│       prompts.json
│       prompt_response_dataset.jsonl
│
├───dataset_generation
│       levels_datgen.py
│       pdat_runner.py
│       pdat_runner_updated.py
│       prompt_response_generator.py
│       variants_datgen.py
│
└───tools
        dataset_cleaner.py
        docker_buildermcd.txt
```

---

## 🧠 How It Works

1. Base prompts and perturbation rules are defined in dataset files
2. \`pdat_runner.py\` orchestrates dataset processing and generation
3. \`prompt_response_generator.py\` communicates with LLMs to collect responses
4. Results are saved into formatted dataset files for benchmarking or training

---

## 📊 Example Usage

### Generate dataset
```bash
python dataset_generation/pdat_runner.py
```

### Run updated generator version
```bash
python dataset_generation/pdat_runner_updated.py
```

### Query an agent directly
```bash
python query_agent.py
```

---

## 📁 Dataset Format

### Sample JSON entry
 
```bash
[
  {
    "id": "001",
    "task_name": "Short_Summary",
    "original_prompt": "Summarize the key findings of the recent Mars rover mission.",
    "ground_truth": "The rover found evidence of ancient liquid water.",
    "test_cases": [
      {
        "variant_id": "001_1",
        "perturbation_type": ["Synonym_Replacement", "Word_bug"],
        "perturbed_prompt": "Please condense the core resulta of the latest Mars rover mission.",
        "perturbed_output": "The probe found signs of ancient liquid water deposits."
      }
    ]
  },
  {
    "id": "002",
    "task_name": "Code",
    "original_prompt": "Write a Python function to compute the Fibonacci sequence up to N.",
    "ground_truth": "def fibonacci(n): ...",
    "test_cases": [
        "... variants for task 002"
    ]
  }
]
```
---

## 🛠 Requirements

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## 🔮 Roadmap

- [ ] Evaluation metrics for perturbation strength & response quality
- [ ] Support for more LLM backends and adapters
- [ ] Dataset visualization + automatic analysis
- [ ] Web UI for experimentation

---

## 🤝 Contributing

Contributions, suggestions, and pull requests are welcome!
Please open an issue for major changes or discussion.

---

## 📜 License

MIT License © 2025

---

## 🧑‍💻 Author

**Testady21**
- GitHub: https://github.com/Testady21
