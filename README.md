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
│
├── dataset_generation/
│   ├── pdat_runner.py
│   ├── pdat_runner_updated.py
│   ├── prompt_response_generator.py
│   └── prompt_response_dataset.jsonl
│
├── datasets/
│   └── levels_llama3.2.json
│
├── query_agent.py
├── README.md
└── requirements.txt
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

### Sample JSONL entry
 
```json
{
  "original_prompt": "Explain quantum computing to a child",
  "perturbed_prompt": "Describe quantum computers to a young student",
  "response": "..."
}
```
'''
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
