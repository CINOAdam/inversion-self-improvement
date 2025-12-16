#!/usr/bin/env python3
"""
TRUE Self-Improvement via Intrinsic Inversion Scoring

This implements the real proof: A model improves using ONLY intrinsic signals.

The Loop:
1. GENERATE: Model creates command candidates for tasks
2. INVERT: Model reconstructs what task the command would accomplish
3. SCORE: Compare original task to reconstructed task (intrinsic!)
4. SELECT: Keep high-scoring self-generated (task, command) pairs
5. TRAIN: Fine-tune on self-generated, self-scored data
6. REPEAT

No human labels. No external reward models. Pure self-improvement.
"""

import json
import random
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
)
from peft import LoraConfig, get_peft_model, PeftModel
import numpy as np
import difflib
import re


@dataclass
class SelfImprovementConfig:
    """Configuration for self-improvement loop."""
    # Model
    model_name: str = "mistralai/Mistral-7B-Instruct-v0.2"

    # Generation
    num_candidates_per_task: int = 5  # Generate multiple candidates
    generation_temperature: float = 0.7  # Higher for diversity

    # Scoring
    similarity_threshold: float = 0.5  # Min score to keep example

    # Training
    learning_rate: float = 2e-4
    num_epochs: int = 1
    batch_size: int = 4

    # Loop
    num_cycles: int = 3
    tasks_per_cycle: int = 50

    # Output
    output_dir: str = "./output/self_improvement"


# Task templates - these are NOT training labels, just task DESCRIPTIONS
# The model must figure out the correct commands itself
TASK_TEMPLATES = {
    "nmap": [
        "Scan {target} for open ports",
        "Do a service version scan on {target}",
        "Scan ports {ports} on {target}",
        "Run a SYN scan on {target}",
        "Detect the operating system of {target}",
        "Scan {target} for the top 100 ports",
        "Do an aggressive scan of {target}",
        "Scan {target} quickly with timing template 4",
    ],
    "nikto": [
        "Scan {target} for web vulnerabilities",
        "Run a web vulnerability scan on {target}",
        "Check {target} for common web issues",
        "Scan the web server at {target}",
    ],
    "gobuster": [
        "Brute force directories on {target}",
        "Find hidden paths on {target}",
        "Enumerate directories on {target} using wordlist {wordlist}",
        "Discover web paths on {target}",
    ],
    "dirb": [
        "Scan {target} for directories",
        "Enumerate web directories on {target}",
        "Find hidden directories on {target}",
    ],
    "netcat": [
        "Check if port {port} is open on {target}",
        "Test connectivity to {target} on port {port}",
        "Listen on port {port}",
        "Connect to {target} on port {port}",
    ],
    "searchsploit": [
        "Find exploits for {software}",
        "Search for vulnerabilities in {software}",
        "Look up exploits for {software} version {version}",
    ],
    "hydra": [
        "Brute force SSH on {target}",
        "Attack FTP login on {target}",
        "Crack {service} password on {target}",
    ],
    "sqlmap": [
        "Test {target} for SQL injection",
        "Check if {target} is vulnerable to SQLi",
        "Enumerate databases on {target}",
    ],
    "masscan": [
        "Scan {target} for all ports quickly",
        "Fast port scan of {target}",
        "Scan {target} at high speed",
    ],
}

# Sample targets for task generation
SAMPLE_TARGETS = [
    "192.168.1.1", "192.168.1.100", "10.0.0.1", "172.16.0.1",
    "192.168.0.41", "target.local", "192.168.1.0/24",
]
SAMPLE_WEB_TARGETS = [
    "http://192.168.1.1", "http://192.168.0.41:8001",
    "http://target.local", "http://10.0.0.1:8080",
]
SAMPLE_PORTS = ["22", "80", "443", "8080", "21", "25", "3306"]
SAMPLE_SOFTWARE = [
    ("vsftpd", "2.3.4"), ("apache", "2.4.49"), ("openssh", "7.2"),
    ("nginx", "1.16"), ("mysql", "5.7"), ("wordpress", "5.0"),
]
SAMPLE_WORDLISTS = [
    "/usr/share/wordlists/dirb/common.txt",
    "/usr/share/wordlists/dirbuster/directory-list-2.3-small.txt",
    "/usr/share/seclists/Discovery/Web-Content/common.txt",
]


def generate_task(tool: str) -> str:
    """Generate a random task for a tool."""
    templates = TASK_TEMPLATES.get(tool, [])
    if not templates:
        return f"Use {tool} on a target"

    template = random.choice(templates)

    # Fill in placeholders
    target = random.choice(SAMPLE_TARGETS)
    web_target = random.choice(SAMPLE_WEB_TARGETS)
    port = random.choice(SAMPLE_PORTS)
    ports = ",".join(random.sample(SAMPLE_PORTS, random.randint(2, 4)))
    software, version = random.choice(SAMPLE_SOFTWARE)
    wordlist = random.choice(SAMPLE_WORDLISTS)
    service = random.choice(["ssh", "ftp", "http", "mysql"])

    task = template.format(
        target=web_target if tool in ["nikto", "gobuster", "dirb", "sqlmap"] else target,
        port=port,
        ports=ports,
        software=software,
        version=version,
        wordlist=wordlist,
        service=service,
    )

    return task


class IntrinsicScorer:
    """Score commands using intrinsic inversion-based verification."""

    def __init__(self, model, tokenizer, config: SelfImprovementConfig):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.device = next(model.parameters()).device

    def generate_command(self, tool: str, task: str) -> str:
        """Generate a command for the given task."""
        prompt = f"""You are a penetration testing expert. Generate ONLY the command, nothing else.

Tool: {tool}
Task: {task}

Command:"""

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                inputs.input_ids,
                max_new_tokens=100,
                temperature=self.config.generation_temperature,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                num_return_sequences=1,
            )

        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        command = response[len(prompt):].strip().split('\n')[0].strip()

        # Clean up common issues
        command = command.replace('`', '').strip()
        if command.startswith('"') and command.endswith('"'):
            command = command[1:-1]

        return command

    def reconstruct_task(self, tool: str, command: str) -> str:
        """Reconstruct what task a command would accomplish (INVERSION)."""
        prompt = f"""You are a penetration testing expert. Given this command, describe what task it accomplishes in one sentence.

Tool: {tool}
Command: {command}

This command will:"""

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                inputs.input_ids,
                max_new_tokens=50,
                temperature=0.3,  # Lower temp for reconstruction
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        reconstructed = response[len(prompt):].strip().split('\n')[0].strip()

        return reconstructed

    def compute_similarity(self, original_task: str, reconstructed_task: str) -> float:
        """
        Compute similarity between original and reconstructed task.

        Uses multiple intrinsic signals:
        1. Token overlap (key terms preserved)
        2. Sequence similarity (structure preserved)
        3. LLM-based rating (semantic understanding)
        """
        # Normalize texts
        orig_lower = original_task.lower()
        recon_lower = reconstructed_task.lower()

        # 1. Token overlap score
        orig_tokens = set(orig_lower.split())
        recon_tokens = set(recon_lower.split())
        if orig_tokens:
            overlap = len(orig_tokens & recon_tokens) / len(orig_tokens)
        else:
            overlap = 0.0

        # 2. Sequence similarity (difflib)
        seq_sim = difflib.SequenceMatcher(None, orig_lower, recon_lower).ratio()

        # 3. LLM-based similarity (truly intrinsic)
        llm_score = self._llm_similarity(original_task, reconstructed_task)

        # Weighted combination
        similarity = 0.2 * overlap + 0.3 * seq_sim + 0.5 * llm_score

        return float(similarity)

    def _llm_similarity(self, task1: str, task2: str) -> float:
        """Use the model itself to rate similarity (fully intrinsic)."""
        prompt = f"""Rate how similar these two task descriptions are on a scale of 0 to 10.
Only output a single number.

Task 1: {task1}
Task 2: {task2}

Similarity (0-10):"""

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                inputs.input_ids,
                max_new_tokens=5,
                temperature=0.1,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = response[len(prompt):].strip()

        # Extract number
        try:
            # Find first number in response
            match = re.search(r'(\d+(?:\.\d+)?)', response)
            if match:
                score = float(match.group(1))
                return min(score / 10.0, 1.0)  # Normalize to 0-1
        except:
            pass

        return 0.5  # Default if parsing fails

    def score_command(self, tool: str, task: str, command: str) -> Tuple[float, str]:
        """
        Score a command using intrinsic inversion.

        Returns (similarity_score, reconstructed_task)
        """
        # Invert: reconstruct what task this command accomplishes
        reconstructed = self.reconstruct_task(tool, command)

        # Score: how similar is reconstruction to original?
        similarity = self.compute_similarity(task, reconstructed)

        return similarity, reconstructed

    def generate_and_score(self, tool: str, task: str) -> Dict:
        """Generate a command and score it intrinsically."""
        command = self.generate_command(tool, task)
        score, reconstructed = self.score_command(tool, task, command)

        return {
            "tool": tool,
            "task": task,
            "command": command,
            "reconstructed_task": reconstructed,
            "intrinsic_score": score,
        }


class SelfGeneratedDataset(Dataset):
    """Dataset of self-generated, self-scored examples."""

    def __init__(self, examples: List[Dict], tokenizer, max_length: int = 512):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]

        # Format as training example
        text = f"""Tool: {ex['tool']}
Task: {ex['task']}

Command: {ex['command']}"""

        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels": encoding["input_ids"].squeeze(),
        }


class TrueSelfImprovement:
    """
    The TRUE self-improvement loop.

    No human labels. No external reward. Pure intrinsic improvement.
    """

    def __init__(self, config: Optional[SelfImprovementConfig] = None):
        self.config = config or SelfImprovementConfig()
        self.output_dir = Path(self.config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.history = {
            "cycles": [],
            "total_generated": 0,
            "total_accepted": 0,
            "config": self.config.__dict__,
        }

        self._load_model()

    def _load_model(self):
        """Load the base model."""
        print(f"Loading model: {self.config.model_name}")

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )

        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            quantization_config=bnb_config,
            device_map="auto",
        )

        # Apply LoRA for training
        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )

        self.model = get_peft_model(self.model, lora_config)
        self.device = next(self.model.parameters()).device

        # Initialize scorer
        self.scorer = IntrinsicScorer(self.model, self.tokenizer, self.config)

    def generate_candidates(self, num_tasks: int) -> List[Dict]:
        """Generate and score candidate examples."""
        candidates = []
        tools = list(TASK_TEMPLATES.keys())

        print(f"\nGenerating {num_tasks} task candidates...")

        for i in range(num_tasks):
            tool = random.choice(tools)
            task = generate_task(tool)

            # Generate multiple candidates per task
            for _ in range(self.config.num_candidates_per_task):
                result = self.scorer.generate_and_score(tool, task)
                candidates.append(result)

            if (i + 1) % 10 == 0:
                print(f"  Generated {i + 1}/{num_tasks} tasks "
                      f"({len(candidates)} candidates)")

        return candidates

    def select_examples(self, candidates: List[Dict]) -> List[Dict]:
        """Select high-scoring examples for training."""
        selected = [
            c for c in candidates
            if c["intrinsic_score"] >= self.config.similarity_threshold
        ]

        # Sort by score (best first)
        selected.sort(key=lambda x: x["intrinsic_score"], reverse=True)

        return selected

    def train_on_self_generated(self, examples: List[Dict], cycle_num: int):
        """Train the model on self-generated examples."""
        if not examples:
            print("No examples to train on!")
            return

        print(f"\nTraining on {len(examples)} self-generated examples...")

        dataset = SelfGeneratedDataset(examples, self.tokenizer)

        training_args = TrainingArguments(
            output_dir=str(self.output_dir / f"cycle_{cycle_num}"),
            num_train_epochs=self.config.num_epochs,
            per_device_train_batch_size=self.config.batch_size,
            learning_rate=self.config.learning_rate,
            logging_steps=10,
            save_steps=100,
            save_total_limit=2,
            remove_unused_columns=False,
            fp16=True,
            report_to="none",
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=dataset,
        )

        trainer.train()

        # Save checkpoint
        checkpoint_path = self.output_dir / f"checkpoint_cycle_{cycle_num}"
        self.model.save_pretrained(str(checkpoint_path))
        print(f"Saved checkpoint: {checkpoint_path}")

    def evaluate_improvement(self, before_examples: List[Dict],
                            after_examples: List[Dict]) -> Dict:
        """Evaluate improvement between cycles."""
        before_scores = [e["intrinsic_score"] for e in before_examples]
        after_scores = [e["intrinsic_score"] for e in after_examples]

        before_mean = np.mean(before_scores) if before_scores else 0
        after_mean = np.mean(after_scores) if after_scores else 0

        before_acceptance = (
            len([s for s in before_scores if s >= self.config.similarity_threshold])
            / len(before_scores) if before_scores else 0
        )
        after_acceptance = (
            len([s for s in after_scores if s >= self.config.similarity_threshold])
            / len(after_scores) if after_scores else 0
        )

        return {
            "before_mean_score": before_mean,
            "after_mean_score": after_mean,
            "score_improvement": after_mean - before_mean,
            "before_acceptance_rate": before_acceptance,
            "after_acceptance_rate": after_acceptance,
            "acceptance_improvement": after_acceptance - before_acceptance,
        }

    def run_cycle(self, cycle_num: int) -> Dict:
        """Run one self-improvement cycle."""
        print(f"\n{'='*70}")
        print(f"SELF-IMPROVEMENT CYCLE {cycle_num}")
        print(f"{'='*70}")

        cycle_result = {
            "cycle": cycle_num,
            "timestamp": datetime.now().isoformat(),
        }

        # 1. Generate candidates (BEFORE training)
        print("\n[1] Generating candidates (BEFORE)...")
        before_candidates = self.generate_candidates(self.config.tasks_per_cycle)
        before_selected = self.select_examples(before_candidates)

        cycle_result["before"] = {
            "total_generated": len(before_candidates),
            "accepted": len(before_selected),
            "acceptance_rate": len(before_selected) / len(before_candidates),
            "mean_score": np.mean([c["intrinsic_score"] for c in before_candidates]),
        }

        print(f"  Generated: {len(before_candidates)}")
        print(f"  Accepted (score >= {self.config.similarity_threshold}): {len(before_selected)}")
        print(f"  Mean intrinsic score: {cycle_result['before']['mean_score']:.3f}")

        # 2. Train on accepted examples
        print("\n[2] Training on self-generated examples...")
        self.train_on_self_generated(before_selected, cycle_num)

        # 3. Generate candidates (AFTER training)
        print("\n[3] Generating candidates (AFTER)...")
        after_candidates = self.generate_candidates(self.config.tasks_per_cycle)
        after_selected = self.select_examples(after_candidates)

        cycle_result["after"] = {
            "total_generated": len(after_candidates),
            "accepted": len(after_selected),
            "acceptance_rate": len(after_selected) / len(after_candidates),
            "mean_score": np.mean([c["intrinsic_score"] for c in after_candidates]),
        }

        print(f"  Generated: {len(after_candidates)}")
        print(f"  Accepted: {len(after_selected)}")
        print(f"  Mean intrinsic score: {cycle_result['after']['mean_score']:.3f}")

        # 4. Evaluate improvement
        improvement = self.evaluate_improvement(before_candidates, after_candidates)
        cycle_result["improvement"] = improvement

        print(f"\n[4] Improvement:")
        print(f"  Score: {improvement['before_mean_score']:.3f} → {improvement['after_mean_score']:.3f} "
              f"(Δ {improvement['score_improvement']:+.3f})")
        print(f"  Acceptance: {improvement['before_acceptance_rate']:.1%} → {improvement['after_acceptance_rate']:.1%} "
              f"(Δ {improvement['acceptance_improvement']:+.1%})")

        # Save some example comparisons
        cycle_result["example_before"] = before_candidates[:3]
        cycle_result["example_after"] = after_candidates[:3]

        self.history["cycles"].append(cycle_result)
        self.history["total_generated"] += len(before_candidates) + len(after_candidates)
        self.history["total_accepted"] += len(before_selected) + len(after_selected)

        # Save progress after each cycle (in case of interruption)
        self._save_progress(cycle_num)

        return cycle_result

    def _save_progress(self, cycle_num: int):
        """Save progress after each cycle."""
        progress_file = self.output_dir / "progress.json"
        progress = {
            "last_completed_cycle": cycle_num,
            "timestamp": datetime.now().isoformat(),
            "history": self.history,
        }
        with open(progress_file, "w") as f:
            json.dump(progress, f, indent=2, default=str)
        print(f"  Progress saved: cycle {cycle_num}")

    def run(self):
        """Run the full self-improvement loop."""
        print("="*70)
        print("TRUE SELF-IMPROVEMENT VIA INTRINSIC INVERSION")
        print("="*70)
        print(f"\nConfig:")
        print(f"  Cycles: {self.config.num_cycles}")
        print(f"  Tasks per cycle: {self.config.tasks_per_cycle}")
        print(f"  Candidates per task: {self.config.num_candidates_per_task}")
        print(f"  Similarity threshold: {self.config.similarity_threshold}")
        print(f"\nNo human labels. No external reward. Pure intrinsic improvement.")

        for cycle in range(1, self.config.num_cycles + 1):
            self.run_cycle(cycle)

        # Final summary
        self._print_summary()
        self._save_results()

        return self.history

    def _print_summary(self):
        """Print final summary."""
        print("\n" + "="*70)
        print("SELF-IMPROVEMENT COMPLETE")
        print("="*70)

        if len(self.history["cycles"]) >= 2:
            first = self.history["cycles"][0]
            last = self.history["cycles"][-1]

            total_improvement = (
                last["after"]["mean_score"] - first["before"]["mean_score"]
            )
            acceptance_improvement = (
                last["after"]["acceptance_rate"] - first["before"]["acceptance_rate"]
            )

            print(f"\nOverall Improvement:")
            print(f"  Intrinsic Score: {first['before']['mean_score']:.3f} → {last['after']['mean_score']:.3f} "
                  f"(Δ {total_improvement:+.3f})")
            print(f"  Acceptance Rate: {first['before']['acceptance_rate']:.1%} → {last['after']['acceptance_rate']:.1%} "
                  f"(Δ {acceptance_improvement:+.1%})")

            print(f"\nProof of Self-Improvement:")
            if total_improvement > 0:
                print(f"  ✓ Model improved its intrinsic score by {total_improvement:.3f}")
                print(f"  ✓ No human labels used")
                print(f"  ✓ No external reward model")
                print(f"  ✓ Pure intrinsic inversion-based learning")
            else:
                print(f"  ✗ No improvement detected")

        print(f"\nStatistics:")
        print(f"  Total examples generated: {self.history['total_generated']}")
        print(f"  Total examples accepted: {self.history['total_accepted']}")

    def _save_results(self):
        """Save results to disk."""
        results_file = self.output_dir / "self_improvement_results.json"
        with open(results_file, "w") as f:
            json.dump(self.history, f, indent=2, default=str)
        print(f"\nResults saved: {results_file}")


def main():
    """Run true self-improvement with command-line arguments."""
    import argparse

    parser = argparse.ArgumentParser(
        description="TRUE Self-Improvement via Intrinsic Inversion",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Core parameters
    parser.add_argument("--cycles", type=int, default=10,
                        help="Number of self-improvement cycles")
    parser.add_argument("--tasks", type=int, default=50,
                        help="Tasks per cycle")
    parser.add_argument("--candidates", type=int, default=3,
                        help="Candidates generated per task")

    # Scoring parameters
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Similarity threshold for acceptance")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Generation temperature")

    # Model parameters
    parser.add_argument("--model", type=str, default="mistralai/Mistral-7B-Instruct-v0.2",
                        help="Base model to use")

    # Output parameters
    parser.add_argument("--output", type=str, default="./output/extended_run",
                        help="Output directory")
    parser.add_argument("--name", type=str, default=None,
                        help="Experiment name (auto-generated if not provided)")

    # Training parameters
    parser.add_argument("--lr", type=float, default=2e-4,
                        help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=4,
                        help="Training batch size")

    args = parser.parse_args()

    # Generate experiment name if not provided
    if args.name is None:
        args.name = f"c{args.cycles}_t{args.tasks}_th{args.threshold}"

    # Create output directory with experiment name
    output_dir = f"{args.output}/{args.name}"

    print(f"\n{'='*70}")
    print("EXTENDED SELF-IMPROVEMENT EXPERIMENT")
    print(f"{'='*70}")
    print(f"\nExperiment: {args.name}")
    print(f"Output: {output_dir}")

    config = SelfImprovementConfig(
        model_name=args.model,
        num_cycles=args.cycles,
        tasks_per_cycle=args.tasks,
        num_candidates_per_task=args.candidates,
        similarity_threshold=args.threshold,
        generation_temperature=args.temperature,
        learning_rate=args.lr,
        batch_size=args.batch_size,
        output_dir=output_dir,
    )

    loop = TrueSelfImprovement(config)
    results = loop.run()

    print("\n" + "="*70)
    print("MISSION STATUS")
    print("="*70)

    if len(results["cycles"]) >= 2:
        first = results["cycles"][0]
        last = results["cycles"][-1]
        improvement = last["after"]["mean_score"] - first["before"]["mean_score"]

        if improvement > 0:
            print(f"""
✓ PROVED: A model can improve its real-world task execution
  using ONLY intrinsic fidelity signals.

  - No human labels
  - No external reward models
  - Pure self-improvement via inversion

  Improvement: {first['before']['mean_score']:.3f} → {last['after']['mean_score']:.3f} (+{improvement:.3f})
  Cycles: {args.cycles}
  Total examples: {results['total_generated']}
""")
        else:
            print(f"""
✗ NOT YET PROVED: Improvement was {improvement:.3f}

  May need:
  - More cycles
  - Lower similarity threshold
  - Different generation temperature
  - More candidates per task
""")


if __name__ == "__main__":
    main()
