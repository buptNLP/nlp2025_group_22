import os
import time
import random
from tqdm import tqdm
from transformers import AutoTokenizer

from data_loader import load_data
from model_utils import load_hf_lm_and_tokenizer, generate_completions
from utils.utils import set_seed, save_jsonl

# ----------- Agent Definitions -----------
class Planner:
    def __init__(self, model, tokenizer, dataset):
        self.model = model
        self.tokenizer = tokenizer
        self.dataset = dataset

    def run(self, question):
        if self.dataset in ["gsm8k", "math"]:
            prompt = (
                "Decompose the following math problem into a chain of subtasks, including: formula generation, numerical calculation, and unit verification.\n"
                f"Question: {question}\n"
                "Please output the subtask chain, one step per line:"
            )
        elif self.dataset == "hotpotqa":
            prompt = (
                "Decompose the following multi-hop question into a reasoning path, including: entity localization, evidence retrieval, and logical chaining.\n"
                f"Question: {question}\n"
                "Please output the reasoning path, one step per line:"
            )
        else:
            prompt = f"Decompose the question: {question}"
        result = generate_completions(self.model, self.tokenizer, [prompt], max_new_tokens=128, batch_size=1)[0]
        steps = [line.strip() for line in result.strip().split("\n") if line.strip()]
        return steps

class Executor:
    def __init__(self, model, tokenizer, dataset):
        self.model = model
        self.tokenizer = tokenizer
        self.dataset = dataset

    def run(self, question, subtask, context=""):
        if self.dataset in ["gsm8k", "math"]:
            prompt = (
                f"Original question: {question}\n"
                f"Current subtask: {subtask}\n"
                f"{'Known information: ' + context if context else ''}"
                "Please complete this subtask and output the detailed process and result:"
            )
        elif self.dataset == "hotpotqa":
            prompt = (
                f"Original question: {question}\n"
                f"Current reasoning step: {subtask}\n"
                f"{'Known information: ' + context if context else ''}"
                "Please complete this reasoning step and output the reasoning content:"
            )
        else:
            prompt = f"Please complete the subtask: {subtask}"
        result = generate_completions(self.model, self.tokenizer, [prompt], max_new_tokens=128, batch_size=1)[0]
        return result.strip()

class Checker:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def run(self, question, answer, steps):
        prompt = (
            f"Please verify the following math problem solution:\n"
            f"Question: {question}\n"
            f"Solution steps:\n{chr(10).join(steps)}\n"
            f"Final answer: {answer}\n"
            "Check: 1. Is the calculation correct? 2. Is the unit expression standard? 3. Is the output format correct?\n"
            "Please output your conclusion (Correct/Incorrect) and reasons:"
        )
        result = generate_completions(self.model, self.tokenizer, [prompt], max_new_tokens=128, batch_size=1)[0]
        return result.strip()

class Reflector:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def run(self, question, answer, steps):
        prompt = (
            f"Please analyze the credibility of the following multi-hop QA reasoning chain:\n"
            f"Question: {question}\n"
            f"Reasoning path:\n{chr(10).join(steps)}\n"
            f"Final answer: {answer}\n"
            "Is the evidence chain sufficient and is the logic consistent? Please give a credibility rating (High/Medium/Low) and reasons:"
        )
        result = generate_completions(self.model, self.tokenizer, [prompt], max_new_tokens=128, batch_size=1)[0]
        return result.strip()

# ----------- Multi-Agent Pipeline -----------
def multi_agent_pipeline(example, agents, dataset):
    question = example["question"] if "question" in example else example.get("input", "")
    # 1. Planner
    subtasks = agents['planner'].run(question)
    # 2. Executor
    step_results = []
    context = ""
    for subtask in subtasks:
        result = agents['executor'].run(question, subtask, context)
        step_results.append(result)
        context += result + "\n"
    # 3. Checker/Reflector
    if dataset in ["gsm8k", "math"]:
        check_result = agents['checker'].run(question, step_results[-1], step_results)
        return {
            "steps": step_results,
            "final_answer": step_results[-1],
            "check_result": check_result
        }
    elif dataset == "hotpotqa":
        reflect_result = agents['reflector'].run(question, step_results[-1], step_results)
        return {
            "steps": step_results,
            "final_answer": step_results[-1],
            "reflect_result": reflect_result
        }
    else:
        return {
            "steps": step_results,
            "final_answer": step_results[-1]
        }

def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_name", type=str, default="gsm8k")
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--output_dir", type=str, default="./output/multi_agent")
    parser.add_argument("--planner_model", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--executor_model", type=str, default="mistralai/Mistral-7B-Instruct-v0.2")
    parser.add_argument("--checker_model", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--reflector_model", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--num_test_sample", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()

def main():
    args = parse_args()
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    # Load models
    print("Loading Planner model...")
    planner_model, planner_tokenizer = load_hf_lm_and_tokenizer(args.planner_model)
    print("Loading Executor model...")
    executor_model, executor_tokenizer = load_hf_lm_and_tokenizer(args.executor_model)
    print("Loading Checker/Reflector model...")
    checker_model, checker_tokenizer = load_hf_lm_and_tokenizer(args.checker_model)
    reflector_model, reflector_tokenizer = load_hf_lm_and_tokenizer(args.reflector_model)

    # Build agents
    agents = {
        "planner": Planner(planner_model, planner_tokenizer, args.data_name),
        "executor": Executor(executor_model, executor_tokenizer, args.data_name)
    }
    if args.data_name in ["gsm8k", "math"]:
        agents["checker"] = Checker(checker_model, checker_tokenizer)
    elif args.data_name == "hotpotqa":
        agents["reflector"] = Reflector(reflector_model, reflector_tokenizer)

    # Load data
    print("Loading dataset...")
    data = load_data(args.data_name, args.split, args.data_dir)
    if args.num_test_sample > 0:
        data = data[:args.num_test_sample]

    results = []
    time_costs = []
    print("Start multi-agent inference...")
    for example in tqdm(data):
        start = time.time()
        result = multi_agent_pipeline(example, agents, args.data_name)
        elapsed = time.time() - start
        time_costs.append(elapsed)
        results.append({
            "idx": example.get("idx", None),
            "question": example.get("question", ""),
            "result": result,
            "time": elapsed
        })

    # Save results
    out_file = os.path.join(args.output_dir, f"{args.data_name}_multi_agent_results.jsonl")
    save_jsonl(results, out_file)
    print(f"Inference finished. Results saved to {out_file}")

    print(f"Average inference time: {sum(time_costs)/len(time_costs):.2f} seconds")

if __name__ == "__main__":
    main()