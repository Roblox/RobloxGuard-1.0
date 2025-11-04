#!/usr/bin/env python3
"""
Model evaluation script for safety assessment using fine-tuned models.
Outputs results to CSV format.
"""

import json
import torch
import torch.nn.functional as F
import argparse
import time
import re
import os
import csv
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import models
import datasets
import random


def pick_dtype() -> torch.dtype:
    """Pick the appropriate dtype based on device capabilities."""
    device = (
        "cuda" if torch.cuda.is_available()
        else ("mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else "cpu")
    )
    
    if device == "cuda":
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16
    elif device == "mps":
        # MPS supports bfloat16 on newer Apple Silicon
        if getattr(torch.backends, "mps", None) and hasattr(torch.backends.mps, "is_bf16_supported") and torch.backends.mps.is_bf16_supported():
            return torch.bfloat16
        return torch.float16
    else:
        return torch.float32


class ModelEvaluator:
    """Handles model loading and evaluation for safety assessment."""
    
    def __init__(self, max_output_tokens: int, base_model: str, model_path: str):
        self.max_output_tokens = max_output_tokens
        
        # Initialize model and tokenizer
        self.tokenizer = None
        self.model = None
        self.base_model = base_model
        self.model_path = model_path
        self._load_model()
    
    def _load_model(self) -> None:
        """Load the model and tokenizer."""
        
        # Determine device and dtype
        device = (
            "cuda" if torch.cuda.is_available()
            else ("mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else "cpu")
        )
        dtype = pick_dtype()
        
        # Prepare model loading kwargs
        kwargs = dict(torch_dtype=dtype)
        if device != "cpu":
            kwargs["device_map"] = "auto"
        
        # Check if model_path is a Hugging Face model ID or local path
        if self.model_path.startswith("https://huggingface.co/"):
            # Extract model ID from URL
            model_id = self.model_path.replace("https://huggingface.co/", "")
            is_hf_model = True
        elif "/" in self.model_path and not os.path.exists(self.model_path):
            # Assume it's a Hugging Face model ID (e.g., "Roblox/RobloxGuard")
            model_id = self.model_path
            is_hf_model = True
        else:
            # Local model path
            is_hf_model = False
        
        if is_hf_model:
            print(f"Loading model from Hugging Face: {model_id}")
            
            # Load tokenizer from Hugging Face with left padding for causal LMs
            self.tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side="left")
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load model directly from Hugging Face
            self.model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
            
        else:
            # Original local model loading logic
            output_path = self.model_path
            adapter_path = os.getcwd() + "/" + output_path
            
            if not os.path.exists(adapter_path):
                raise FileNotFoundError(f"Adapter path not found: {adapter_path}")
            
            # Load tokenizer with left padding for causal LMs
            self.tokenizer = AutoTokenizer.from_pretrained(self.base_model, padding_side="left")
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load base model
            self.model = AutoModelForCausalLM.from_pretrained(self.base_model, **kwargs)
            
            # Load adapter
            self.model = PeftModel.from_pretrained(self.model, adapter_path)
        
        self.model.eval()
        print(f"Model loaded successfully with dtype: {dtype}")
    
    def generate_text(self, prompt: str, max_length: Optional[int] = None) -> str:
        """Generate text using the loaded model with deterministic decoding."""
        if max_length is None:
            max_length = self.max_output_tokens
            
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
        input_length = inputs['input_ids'].shape[1]
        inputs = {key: value.to(self.model.device) for key, value in inputs.items()}

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_length,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id
            )

        generated_output = outputs[0][input_length:]
        return self.tokenizer.decode(generated_output, skip_special_tokens=True)
    
    def score_safe_unsafe(self, prompt: str, candidates: Tuple[str, str] = ("safe", "unsafe")) -> Dict[str, Any]:
        """Compute probabilities for safe vs unsafe tokens directly from logits."""
        enc = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
        enc = {k: v.to(self.model.device) for k, v in enc.items()}
        
        with torch.no_grad():
            logits = self.model(**enc).logits
        
        pad_id = self.tokenizer.pad_token_id
        last_idx = (enc["input_ids"] != pad_id).sum(dim=1) - 1
        last_logits = logits[0, last_idx[0], :]
        probs = F.softmax(last_logits, dim=-1)
        
        # Get token IDs for candidates
        cand_ids = {}
        for c in candidates:
            encoded = self.tokenizer.encode(c, add_special_tokens=False)
            if encoded:
                cand_ids[c] = encoded[0]
        
        cand_probs = {c: float(probs[cid]) for c, cid in cand_ids.items() if cid < len(probs)}
        
        if cand_probs:
            total = sum(cand_probs.values())
            if total > 0:
                cand_probs = {k: v / total for k, v in cand_probs.items()}
            cand_probs["pred"] = max(cand_probs, key=cand_probs.get)
        else:
            cand_probs["pred"] = None
        
        return cand_probs


class EvaluationMetrics:
    """Handles calculation and tracking of evaluation metrics."""
    
    def __init__(self):
        self.tp = 0  # True Positive
        self.fn = 0  # False Negative
        self.tn = 0  # True Negative
        self.fp = 0  # False Positive
    
    def update(self, predicted_unsafe: bool, actual_unsafe: bool) -> None:
        """Update metrics based on prediction and ground truth."""
        if actual_unsafe:
            if predicted_unsafe:
                self.tp += 1
            else:
                self.fn += 1
        else:
            if predicted_unsafe:
                self.fp += 1
            else:
                self.tn += 1
    
    def calculate_metrics(self) -> Dict[str, float]:
        """Calculate precision, recall, F1, and FPR."""
        metrics = {}
        if self.tp + self.fn > 0:
            metrics['recall'] = round((self.tp * 100) / (self.tp + self.fn), 3)
        else:
            metrics['recall'] = 0.0
        if self.tn + self.fp > 0:
            metrics['fpr'] = round((self.fp * 100) / (self.tn + self.fp), 3)  # Fixed: FP/(TN+FP)
        else:
            metrics['fpr'] = 0.0
        if self.tp + self.fp > 0:
            metrics['precision'] = round((self.tp * 100) / (self.tp + self.fp), 3)
        else:
            metrics['precision'] = 0.0
        if metrics['precision'] + metrics['recall'] > 0:
            metrics['f1'] = round((2 * metrics['precision'] * metrics['recall']) / (metrics['precision'] + metrics['recall']), 3)
        else:
            metrics['f1'] = 0.0
        return metrics
    
    def get_counts(self) -> Tuple[int, int, int, int]:
        """Get confusion matrix counts."""
        return self.tp, self.fn, self.fp, self.tn
    
    def get_metrics_dict(self) -> Dict[str, Any]:
        """Get current metrics as dictionary."""
        metrics = self.calculate_metrics()
        return {
            "confusion_matrix": {
                "tp": self.tp,
                "fn": self.fn,
                "fp": self.fp,
                "tn": self.tn
            },
            "metrics": metrics
        }


class SafetyEvaluator:
    """Main class for running safety evaluations."""
    
    def __init__(self, model_evaluator: ModelEvaluator, output_file: str, has_labels: bool = True):
        self.model_evaluator = model_evaluator
        self.metrics = EvaluationMetrics()
        self.output_file = output_file
        self.has_labels = has_labels
        self.csv_writer = None
        self.csv_file = None
    
    def format_prompt(self, prompt_template: str, eval_data: Dict[str, Any]) -> str:
        """Format the prompt template with evaluation data."""
        prompt = eval_data.get('prompt', '')
        response = eval_data.get('response', '')
        return prompt_template.format(prompt=prompt, response=response)
    
    def extract_llm_output(self, llm_output: str, field_name: str) -> str:
        """Extract the target field from LLM output using regex."""
        try:
            pattern = rf'"{field_name}"\s*:\s*"?(\w+)"?,?'
            match = re.search(pattern, llm_output)
            
            if match:
                result = match.group(1)
                return result
            else:
                return ""
        except Exception as e:
            return ""

    
    def write_result(self, result_data: Dict[str, Any]) -> None:
        """Write a single result to CSV file."""
        if self.csv_writer and result_data.get("type") != "metrics_update":
            # Only write regular evaluation results, not metrics updates
            if "index" in result_data:
                input_response = result_data.get('input_response') or ''
                if self.has_labels:
                    # Include all fields when labels are available
                    current_metrics = result_data.get("current_metrics", {})
                    row = {
                        'index': result_data.get('index'),
                        'input_prompt': result_data.get('input_prompt', '').replace('\n', ' ').replace('\r', ''),
                        'input_response': input_response.replace('\n', ' ').replace('\r', ''),
                        'actual_unsafe': result_data.get('actual_unsafe'),
                        'predicted_unsafe': result_data.get('predicted_unsafe'),
                        'correct': result_data.get('correct')
                    }
                else:
                    # Simplified output when no labels are available
                    row = {
                        'index': result_data.get('index'),
                        'input_prompt': result_data.get('input_prompt', '').replace('\n', ' ').replace('\r', ''),
                        'input_response': input_response.replace('\n', ' ').replace('\r', ''),
                        'llm_prediction': result_data.get('llm_prediction', '')
                    }
                
                self.csv_writer.writerow(row)
                self.csv_file.flush()
    
    def write_summary_csv(self, total_examples: int) -> None:
        """Write a summary CSV file with final metrics."""
        summary_file = self.output_file.replace('.csv', '_summary.csv')

        if not self.has_labels:
            return  # Don't write summary when no labels are available

        final_metrics = self.metrics.get_metrics_dict()
        metrics = final_metrics.get("metrics", {})
        confusion_matrix = final_metrics.get("confusion_matrix", {})
        
        with open(summary_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['Metric', 'Value'])
            writer.writerow(['Total Examples', total_examples])
            writer.writerow(['True Positives', confusion_matrix.get('tp', 0)])
            writer.writerow(['False Negatives', confusion_matrix.get('fn', 0)])
            writer.writerow(['False Positives', confusion_matrix.get('fp', 0)])
            writer.writerow(['True Negatives', confusion_matrix.get('tn', 0)])
            writer.writerow(['Precision (%)', metrics.get('precision', 0.0)])
            writer.writerow(['Recall (%)', metrics.get('recall', 0.0)])
            writer.writerow(['F1 Score (%)', metrics.get('f1', 0.0)])
            writer.writerow(['False Positive Rate (%)', metrics.get('fpr', 0.0)])
        
        print(f"Summary metrics written to: {summary_file}")

    def load_evaluation_dataset(self, eval_config: Dict[str, Any]) -> list:
        eval_dataset = []

        if eval_config["name"] == models.AEGIS:
            df = datasets.load_dataset(eval_config["eval_dataset"], split='test')
            df = df.filter(lambda example: example['text_type'] == 'user_message')

            # Create evaluation dataset in the specified format
            for data in df:
                # Calculate label based on safety annotations
                if not data['labels_0'] == 'Safe' and not data['labels_1'] == 'Safe' and not data['labels_2'] == 'Safe':
                    prompt_label = "unsafe"
                else:
                    prompt_label = "safe"

                entry = {
                    "prompt": data['text'],
                    "prompt_label": prompt_label
                }
                eval_dataset.append(entry)
        elif eval_config["name"] == models.AEGIS2:
            df = datasets.load_dataset(eval_config["eval_dataset"], split='test')

            # Create evaluation dataset in the specified format
            for data in df:
                entry = {
                    "prompt": data['prompt'],
                    "prompt_label": data['prompt_label'],
                    "response": data['response'],
                    "response_label": data['response_label'],
                }
                eval_dataset.append(entry)
        elif eval_config["name"] == models.OPENAI_MOD:
            with open(eval_config["eval_dataset"], 'r') as f:
                df = f.readlines()

            for data in df:
                data = json.loads(data)

                label = "safe"
                for value in data.values():
                    if value == 1:
                        label = "unsafe"
                        break

                entry = {
                    "prompt": data['prompt'],
                    "prompt_label": label
                }
                eval_dataset.append(entry)
        elif eval_config["name"] == models.BEAVERTAILS:
            df = datasets.load_dataset(eval_config["eval_dataset"], split='330k_test')
    
            eval_dataset = []
            for data in df:
                entry = {
                    "prompt": data['prompt'],
                    "response": data['response'],
                    "is_safe": str(data["is_safe"]).lower()
                }
                eval_dataset.append(entry)
        elif eval_config["name"] == models.ROGUARD:
            df = datasets.load_dataset(eval_config["eval_dataset"], split='test')

            eval_dataset = []
            for data in df:
                entry = {
                    "prompt": data['prompt'],
                    "response": data['response'],
                    "violation": str(data["violation"]).lower()
                }
                eval_dataset.append(entry)
        elif eval_config["name"] == models.TOXICCHAT:
            df = datasets.load_dataset(eval_config["eval_dataset"], name='toxicchat0124', split='test')
    
            eval_dataset = []
            for data in df:
                entry = {
                    "prompt": data['user_input'],
                    "response": data['model_output'],
                    "toxicity": data["toxicity"]
                }
                eval_dataset.append(entry)
        elif eval_config["name"] == models.WILDGUARD_PROMPT:
            df = datasets.load_dataset(eval_config["eval_dataset"], name='wildguardtest', split='test')
    
            eval_dataset = []
            for data in df:
                entry = {
                    "prompt": data['prompt'],
                    "response": data['response'],
                    "prompt_harm_label": str(data['prompt_harm_label']).lower()
                }
                eval_dataset.append(entry)
        elif eval_config["name"] == models.WILDGUARD_RESPONSE:
            df = datasets.load_dataset(eval_config["eval_dataset"], name='wildguardtest', split='test')
    
            eval_dataset = []
            for data in df:
                entry = {
                    "prompt": data['prompt'],
                    "response": data['response'],
                    "response_harm_label": str(data['response_harm_label']).lower()
                }
                eval_dataset.append(entry)
        elif eval_config["name"] == models.HARMBENCH:
            with open(eval_config["eval_dataset"], 'r') as f:
                raw_data = json.load(f)
            keys = list(raw_data.keys())

            all_data = []
            ds = []
            def preprocess_data(x):
                human_0 = x["human_0"]
                human_1 = x["human_1"]
                human_2 = x["human_2"]

                harmfulness = 1 if sum([human_0, human_1, human_2]) > 1 else 0
                is_adversarial = True if x["experiment"] != "default" else False
                return {"prompt": x["test_case"], "response": x["generation"], "response_harmfulness": harmfulness,
                        "prompt_type": "adversarial" if is_adversarial else "vanilla"}

            for k in keys:
                all_data.extend([preprocess_data(x) for x in raw_data[k]])

            for data in all_data:
                entry = {
                    "prompt": data['prompt'],
                    "response": data['response'],
                    "violation": data["response_harmfulness"]
                }
                eval_dataset.append(entry)
        elif eval_config["name"] == models.XSTEST:
            df = datasets.load_dataset(eval_config["eval_dataset"], split="train")
    
            eval_dataset = []
            for data in df:
                entry = {
                    "prompt": data['prompt'],
                    "label": str(data['label']).lower()
                }
                eval_dataset.append(entry)
        elif eval_config["name"] == models.SAFERLHF:
            dataset = datasets.load_dataset(eval_config["eval_dataset"], revision="v0")
            test_dataset = dataset["test"]
            all_data = []
            ds = []

            prompts = test_dataset["prompt"]
            response_0 = test_dataset["response_0"]
            response_1 = test_dataset["response_1"]
            is_response_0_safe = test_dataset["is_response_0_safe"]
            is_response_1_safe = test_dataset["is_response_1_safe"]

            safe_responses, harmful_responses = [], []
            for p, r0, r1, r0s, r1s in zip(prompts, response_0, response_1, is_response_0_safe, is_response_1_safe):
                # pick only when r0s != r1s
                if r0s == r1s:
                    continue
                if r0s:
                    safe_responses.append({
                        "prompt": p,
                        "response": r0,
                        "response_harmfulness": 0,
                    })
                else:
                    harmful_responses.append({
                        "prompt": p,
                        "response": r0,
                        "response_harmfulness": 1,
                    })

                if r1s:
                    safe_responses.append({
                        "prompt": p,
                        "response": r1,
                        "response_harmfulness": 0,
                    })
                else:
                    harmful_responses.append({
                        "prompt": p,
                        "response": r1,
                        "response_harmfulness": 1,
                    })

            all_data = safe_responses + harmful_responses

            for i, data in enumerate(all_data, start=0):
                entry = {
                    "prompt": data['prompt'],
                    "response": data['response'],
                    "is_safe": "unsafe" if data['response_harmfulness'] else "safe"
                }
                eval_dataset.append(entry)
    
            return eval_dataset

        elif eval_config["name"] == models.SIMPLEST:
            df = datasets.load_dataset(eval_config["eval_dataset"], split="test")
    
            eval_dataset = []
            for data in df:
                entry = {
                    "prompt": data['prompt'],
                    "violation": "true"
                }
                eval_dataset.append(entry)
    
        return eval_dataset

    def evaluate_dataset(self, eval_config: Dict[str, Any]) -> None:
        """Evaluate a single dataset configuration."""
        print(f"Starting evaluation with config")
        print(f"Output will be written to: {self.output_file}")
        
        # Check if we have labels for evaluation
        eval_label_field_key = eval_config.get("eval_label_field")
        self.has_labels = bool(eval_label_field_key)
        
        # Open CSV file and setup writer
        self.csv_file = open(self.output_file, 'w', newline='', encoding='utf-8')
        
        if self.has_labels:
            fieldnames = [
                'index', 'input_prompt', 'input_response', 'actual_unsafe', 
                'predicted_unsafe', 'correct'
            ]
        else:
            fieldnames = [
                'index', 'input_prompt', 'input_response', 'llm_prediction'
            ]
        
        self.csv_writer = csv.DictWriter(self.csv_file, fieldnames=fieldnames)
        self.csv_writer.writeheader()
        
        try:
            # Load prompt template
            with open(eval_config["eval_prompt"], "r", encoding="utf-8") as file:
                prompt_template = file.read().strip()
            
            # Load evaluation dataset
            eval_dataset = self.load_evaluation_dataset(eval_config)

            
            print(f"Loaded {len(eval_dataset)} evaluation examples")
            
            for i, eval_data in enumerate(eval_dataset):
                # Skip if eval_label_field is null (when specified)
                if self.has_labels and eval_data.get(eval_label_field_key) in [None, "null"]:
                    continue
                
                # Format prompt and generate response
                prompt = self.format_prompt(prompt_template, eval_data)
                llm_output = self.model_evaluator.generate_text(prompt)

                llm_prediction = self.extract_llm_output(
                    llm_output, 
                    eval_config['llm_output_field']
                )
                if not llm_prediction:
                    llm_output = {eval_config["llm_output_field"]: llm_output}
                    llm_output = json.dumps(llm_output)
                    print("Converted to json.")

                print("llm_output:", llm_output)
                
                # Extract prediction from LLM output
                llm_prediction = self.extract_llm_output(
                    llm_output, 
                    eval_config['llm_output_field']
                )
                if self.has_labels:
                    # Determine ground truth and prediction
                    actual_unsafe = (
                        str(eval_data.get(eval_label_field_key, "")).lower() == 
                        str(eval_config["eval_flagged_value"]).lower()
                    )
                    predicted_unsafe = (
                        str(eval_config["llm_flagged_value"]).lower() == 
                        llm_prediction.lower()
                    )
                    
                    # Update metrics
                    self.metrics.update(predicted_unsafe, actual_unsafe)
                    
                    # Prepare result data
                    result_data = {
                        "index": i + 1,
                        "input_prompt": eval_data.get('prompt', ''),
                        "input_response": eval_data.get('response', ''),
                        "actual_unsafe": actual_unsafe,
                        "predicted_unsafe": predicted_unsafe,
                        "correct": predicted_unsafe == actual_unsafe,
                        "current_metrics": self.metrics.get_metrics_dict()
                    }
                    print(result_data)
                    print("correct:", predicted_unsafe == actual_unsafe)
                    print()
                else:
                    # Prepare simplified result data when no labels
                    result_data = {
                        "index": i + 1,
                        "input_prompt": eval_data.get('prompt', ''),
                        "input_response": eval_data.get('response', ''),
                        "llm_prediction": llm_prediction
                    }
                
                # Write result to CSV file
                self.write_result(result_data)
                
                # Print progress to console
                if (i + 1) % 10 == 0:
                    print(f"Progress: {i + 1}/{len(eval_dataset)} examples processed")
                
                # Print periodic metrics update to console (only if labels available)
                if self.has_labels and (i + 1) % 50 == 0:
                    current_metrics = self.metrics.get_metrics_dict()
                    metrics = current_metrics.get("metrics", {})
                    print(f"Current metrics at {i + 1}/{len(eval_dataset)}: "
                          f"Precision: {metrics.get('precision', 0.0)}%, "
                          f"Recall: {metrics.get('recall', 0.0)}%, "
                          f"F1: {metrics.get('f1', 0.0)}%, "
                          f"FPR: {metrics.get('fpr', 0.0)}%")

            # Write summary file (only if labels available)
            if self.has_labels:
                self.write_summary_csv(len(eval_dataset))
                
                # Print final metrics to console
                final_metrics = self.metrics.get_metrics_dict()
                print("\n=== Final Evaluation Results ===")
                print(f"Total examples: {len(eval_dataset)}")
                print(f"Confusion Matrix: TP={final_metrics['confusion_matrix']['tp']}, "
                      f"FN={final_metrics['confusion_matrix']['fn']}, "
                      f"FP={final_metrics['confusion_matrix']['fp']}, "
                      f"TN={final_metrics['confusion_matrix']['tn']}")
                print(f"Metrics: Precision={final_metrics['metrics']['precision']}%, "
                      f"Recall={final_metrics['metrics']['recall']}%, "
                      f"F1={final_metrics['metrics']['f1']}%, "
                      f"FPR={final_metrics['metrics']['fpr']}%")
            else:
                print("\n=== Evaluation Complete ===")
                print(f"Total examples processed: {len(eval_dataset)}")
                print("No ground truth labels available - metrics not calculated")
            
            print(f"Results written to: {self.output_file}")
            
        finally:
            if self.csv_file:
                self.csv_file.close()


def load_config(config_path: str) -> Dict[str, Any]:
    """Load evaluation configuration from JSON file."""
    with open(config_path, 'r', encoding='utf-8') as file:
        return json.load(file)


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description="Evaluate fine-tuned model safety")
    parser.add_argument("--config", required=True, 
                       help="Path to evaluation configuration JSON")
    
    args = parser.parse_args()
    
    print("=== Model Safety Evaluation ===")
    print(f"Config file: {args.config}")
    
    start_time = time.time()
    
    # Load evaluation configuration
    eval_config = load_config(args.config)
    
    # Determine output file: command line argument takes precedence over config file
    if "output_file" in eval_config:
        output_file = eval_config["output_file"]
        # Ensure CSV extension
        if not output_file.endswith('.csv'):
            output_file += '.csv'
    else:
        # Generate default filename if neither is provided
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = f"output_{timestamp}.csv"
    
    print(f"Output file: {output_file}")

    # Initialize model evaluator
    model_evaluator = ModelEvaluator(
        max_output_tokens=eval_config["max_output_tokens"],
        base_model=eval_config["base_model"],
        model_path=eval_config["model_path"]
    )
    
    # Create a directory
    output_dir = os.path.dirname(output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Run evaluation
    safety_evaluator = SafetyEvaluator(model_evaluator, output_file)
    safety_evaluator.evaluate_dataset(eval_config)
        
    end_time = time.time()
    elapsed_time = end_time - start_time
    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    print(f"\nEvaluation completed at: {current_time}")
    print(f"Total elapsed time: {elapsed_time:.2f} seconds")
    
    return 0


if __name__ == "__main__":
    exit(main())
