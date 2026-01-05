# Roblox Guard 1.0: Advancing Safety for LLMs with Robust Guardrails


<div align="center" style="line-height: 1;">
  <a href="https://huggingface.co/Roblox/Llama-3.1-8B-Instruct-RobloxGuard-1.0" target="_blank"><img alt="Hugging Face" src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-RobloxGuard 1.0-ffc107?color=ffc107&logoColor=white"/></a>
  <a href="https://github.com/Roblox/RobloxGuard-1.0/blob/main/LICENSE"><img src="https://img.shields.io/badge/Model%20License-RAIL_MS-green" alt="Model License"></a>
</div>
<div align="center" style="line-height: 1;">
  <a href="https://huggingface.co/datasets/Roblox/RobloxGuard-Eval" target="_blank"><img alt="Hugging Face" src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-RobloxGuardEval-ffc107?color=1783ff&logoColor=white"/></a>
  <a href="https://creativecommons.org/licenses/by-nc-sa/4.0/"><img src="https://img.shields.io/badge/Data%20License-CC_BY_NC_4.0-blue" alt="Data License"></a>
</div>

<div align="center" style="line-height: 1;">
<a href="https://corp.roblox.com/newsroom/2025/07/roblox-guard-advancing-safety-for-llms-with-robust-guardrails" target="_blank"><img src=https://img.shields.io/badge/Roblox-Blog-000000.svg?logo=Roblox height=22px></a>
<a href="https://arxiv.org/abs/2512.05339" target="_blank"><img src="https://img.shields.io/badge/Paper-2512.05339-b5212f.svg?logo=arxiv" height="22px"></a><sub></sub>
</div>
<br />

Roblox Guard 1.0, a SOTA instruction fine-tuned LLM, is designed to help safeguard our Text Generation API. It performs safety classification at both the prompt and response levels, deciding whether or not each input or output violates our policies. This dual-level assessment is essential for moderating both user queries and the model’s own generated outputs. At the heart of our system is an LLM that’s been fine-tuned from the Llama-3.1-8B-Instruct model. We trained this LLM with a particular focus on high-quality instruction tuning to optimize for safety judgment performance. 

Llama 3.1 is licensed under the Llama 3.1 Community License, Copyright © Meta Platforms, Inc. All Rights Reserved.


## 📦 Installation
To set up the environment, install all required dependencies by running the following commands from the root of the repository:
```
python -m venv venv_robloxguard
source venv_robloxguard/bin/activate 
pip install -r requirements.txt
```

## 🧠 Inference

Roblox Guard 1.0 is a taxonomy-adaptive model, meaning it can generalize to any given taxonomy. For evaluation and benchmarking, we provide dataset-specific prompts in the prompt directory that reflect the taxonomy of each dataset. Users can also use their own datasets and corresponding taxonomies with Roblox Guard.

To evaluate Roblox Guard on a specific evaluation set, run the following command from the root of the repository, using the appropriate dataset configuration file:
```
python inference.py --config configs/RobloxGuardEval.json
```

## ⚙️ Configuration
Multiple configuration files are already prepared and ready to use in the `configs/` folder.

To run an evaluation, each config file (in JSON format) should follow this structure:
```
{
  "name": "RobloxGuardEval",                                        // Eval name

  "model_path": "Roblox/Llama-3.1-8B-Instruct-RobloxGuard-1.0",     // Our model path in huggingface
  "base_model": "meta-llama/Meta-Llama-3.1-8B-Instruct",        // Base model used for fine-tuning
  "max_output_tokens": 100,                                     // Max tokens the model can generate

  "eval_prompt": "prompts/RobloxGuardEval.txt",                     // Prompt template file, with placeholders for {prompt} and {response}
  "llm_output_field": "Response Safety",                        // Field in model output to evaluate for safety
  "llm_flagged_value": "unsafe",                                // Value indicating an unsafe response from the model

  "eval_dataset": "Roblox/RobloxGuard-Eval",                        // Evaluation dataset on Hugging Face
  "eval_label_field": "violation",                              // Field in the dataset that holds the ground truth label
  "eval_flagged_value": "true",                                 // Label value that indicates a violation (unsafe content)

  "output_file": "outputs/RobloxGuardEval.csv"                      // Path to save the evaluation results as a CSV file
}
```


## 📄 Output Files

- **Evaluation Results** (`*.csv`):  
  - `input_prompt`: the original prompt  
  - `input_response`: the model’s generated response  
  - `actual_unsafe`: ground-truth label (if provided)  
  - `predicted_unsafe`: model’s prediction  
  - `correct`: whether the prediction matched the ground truth  

- **Summary Metrics** (`*_summary.csv`):  
  - Count-based metrics:  
    `Total Examples`, `True Positives`, `False Negatives`, `False Positives`, `True Negatives`  
  - Performance metrics (as percentages):  
    `Precision`, `Recall`, `F1 Score`, `False Positive Rate`


## 📁 Directory Structure
```
.
├── configs/                # Evaluation configs for different datasets
│   ├── aegis.json
│   ├── ...
│   └── RobloxGuardEval.json
├── prompts/                # Prompt files for inference or evaluation
│   ├── aegis.json
│   ├── ...
│   └── RobloxGuardEval.txt
├── outputs/                # Output CSVs for results and summaries
│   ├── RobloxGuardEval.csv
│   └── RobloxGuardEval_summary.csv
├── inference.py            # Script for running inference/evaluation
└── requirements.txt        # Python dependencies
```

## 📊 Model Benchmark Results

We benchmark Roblox Guard 1.0 model on a comprehensive set of open-source datasets for both prompt and response, as well as on Roblox Guard-Eval. This allows us to evaluate our model on both in-domain and out-of-domain datasets. We report our results in terms of F-1 score for binary violating/non-violating classification. In the table below, we compare our performance with that of several well-known models. The Roblox Guard 1.0 outperforms other models while generalizing on out-of-domain datasets.
- **Prompt Metrics**: These evaluate how well the model classifies or responds to potentially harmful **user inputs**
- **Response Metrics**: These measure how well the model handles or generates **responses**, ensuring its outputs are safe and aligned.

<img width="2000" height="1486" alt="Newsroom_RoGuard_2508_Hero_Final-v2_Background" src="https://github.com/user-attachments/assets/3e91cd72-fcec-4ec3-96b9-25b7ceb26247" />


## 📚 Citation

If you are using Roblox Guard 1.0, please cite it as:

```bibtex
@article{nandwana2025taxonomy,
  title={Taxonomy-Adaptive Moderation Model with Robust Guardrails for Large Language Models},
  author={Nandwana, Mahesh Kumar and Lim, Youngwan and Liu, Joseph and Yang, Alex and Notibala, Varun and Khanna, Nishchaie},
  journal={arXiv preprint arXiv:2512.05339},
  year={2025}
}

