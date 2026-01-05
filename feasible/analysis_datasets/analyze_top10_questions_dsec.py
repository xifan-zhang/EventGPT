#!/usr/bin/env python3
"""
Analysis Script for Top 10 Questions in EventGPT DSEC Dataset

Description:
    This script analyzes the top 10 most frequent questions in the EventGPT DSEC split:
    1. Extracts questions from conversations
    2. Counts question frequencies
    3. Analyzes the top 10 questions in detail
    4. Provides statistics and examples for each top question
    
    Results are saved to results_egpt_dsec_split/top10_questions_analysis/
"""

import os
import sys
import json
from collections import Counter
from typing import Dict, List

import numpy as np
import matplotlib.pyplot as plt
from datasets import Dataset, DatasetDict
from tqdm import tqdm


# Ensure project root is on sys.path
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# Load dataset directly from local directory
DATA_DIR = "/mnt/hdd/data/EventGPT-datasets"
JSON_FILE = os.path.join(DATA_DIR, "EventGPT_Instruction_Subset.json")
OUTPUT_DIR = "/home/ps/Documents/code/EventGPT/feasible/analysis_datasets/results_egpt_dsec_split/top10_questions_analysis"
os.makedirs(OUTPUT_DIR, exist_ok=True)

try:
    # Try loading as a Hugging Face dataset from disk
    ds = Dataset.load_from_disk(DATA_DIR)
except (FileNotFoundError, ValueError):
    print("⚠️ Could not load from disk (not a saved Arrow dataset).")
    print(f"🔄 Attempting to load from JSON: {JSON_FILE}")
    if os.path.exists(JSON_FILE):
        # Load from JSON file
        from datasets import load_dataset
        ds = load_dataset("json", data_files=JSON_FILE, split="train")
    else:
        raise FileNotFoundError(f"Could not find dataset at {DATA_DIR} or {JSON_FILE}")

print(f"✅ Loaded {len(ds)} examples")

# Filter to only dsec instances (zurich_city, interlaken, thun)
dsec_locations = ["zurich_city", "interlaken", "thun"]
def is_dsec_instance(example):
    event_data = example.get("event_data", "")
    return any(event_data.startswith(loc + "_") for loc in dsec_locations)

print("🔍 Filtering to dsec instances only...")
original_count = len(ds)
ds = ds.filter(is_dsec_instance)
dsec_count = len(ds)
print(f"✅ Filtered to {dsec_count} dsec instances (from {original_count} total)")


def extract_question(conversation: List[Dict]) -> str:
    """Extract the question from a conversation."""
    for turn in conversation:
        if turn.get("from") == "human":
            value = turn.get("value", "")
            # Remove <event> tag if present
            question = value.replace("<event>\n", "").replace("<event>", "").strip()
            return question
    return ""


def analyze_top10_questions(dataset: Dataset) -> None:
    """Extract and analyze the top 10 questions."""
    print("\n" + "=" * 60)
    print("📊 Analyzing Top 10 Questions")
    print("=" * 60)
    
    # Extract all questions
    print("🔍 Extracting questions from conversations...")
    questions = []
    question_to_examples = {}  # Map question -> list of example entries
    
    for entry in tqdm(dataset, desc="Processing entries", unit="entry"):
        question = extract_question(entry.get("conversations", []))
        if question:
            questions.append(question)
            if question not in question_to_examples:
                question_to_examples[question] = []
            question_to_examples[question].append(entry)
    
    print(f"✅ Extracted {len(questions)} questions from {len(dataset)} entries")
    
    # Count question frequencies
    print("\n📈 Counting question frequencies...")
    question_counter = Counter(questions)
    top10_questions = question_counter.most_common(10)
    
    print(f"\n🏆 Top 10 Questions:")
    print("-" * 60)
    for i, (question, count) in enumerate(top10_questions, 1):
        percentage = (count / len(questions)) * 100
        print(f"{i:2d}. [{count:5d} occurrences ({percentage:5.2f}%)] {question}")
    
    # Save top 10 questions to file
    top10_file = os.path.join(OUTPUT_DIR, "top10_questions.txt")
    with open(top10_file, 'w', encoding='utf-8') as f:
        f.write("# Top 10 Questions in EventGPT DSEC Dataset\n")
        f.write(f"# Total questions analyzed: {len(questions)}\n")
        f.write(f"# Total unique questions: {len(question_counter)}\n\n")
        for i, (question, count) in enumerate(top10_questions, 1):
            percentage = (count / len(questions)) * 100
            f.write(f"{count}\t{question}\n")
    print(f"\n💾 Saved top 10 questions to: {top10_file}")
    
    # Create visualization
    print("\n📊 Creating visualizations...")
    
    # Figure 1: Bar chart of top 10 questions
    plt.figure(figsize=(14, 8))
    questions_text = [q[:60] + "..." if len(q) > 60 else q for q, _ in top10_questions]
    counts = [count for _, count in top10_questions]
    y_pos = np.arange(len(questions_text))
    
    plt.barh(y_pos, counts, align='center')
    plt.yticks(y_pos, [f"{i+1}. {q}" for i, q in enumerate(questions_text)])
    plt.xlabel('Frequency')
    plt.title('Top 10 Questions in EventGPT DSEC Dataset')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    
    fig1_path = os.path.join(OUTPUT_DIR, "top10_questions_bar.png")
    plt.savefig(fig1_path, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"💾 Saved bar chart to: {fig1_path}")
    
    # Figure 2: Pie chart of top 10 questions
    plt.figure(figsize=(12, 10))
    labels = [f"{i+1}. {q[:40]}..." if len(q) > 40 else f"{i+1}. {q}" for i, (q, _) in enumerate(top10_questions)]
    sizes = counts
    colors = plt.cm.Set3(np.linspace(0, 1, len(labels)))
    
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors=colors)
    plt.title('Distribution of Top 10 Questions in EventGPT DSEC Dataset', fontsize=14, pad=20)
    plt.axis('equal')
    
    fig2_path = os.path.join(OUTPUT_DIR, "top10_questions_pie.png")
    plt.savefig(fig2_path, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"💾 Saved pie chart to: {fig2_path}")
    
    # Detailed analysis for each top 10 question
    print("\n📝 Generating detailed analysis for each top 10 question...")
    detailed_analysis = []
    
    for rank, (question, count) in enumerate(top10_questions, 1):
        examples = question_to_examples[question]
        percentage = (count / len(questions)) * 100
        
        # Get sample answers for this question
        sample_answers = []
        sample_event_files = []
        for ex in examples[:5]:  # Get first 5 examples
            for turn in ex.get("conversations", []):
                if turn.get("from") == "gpt":
                    answer = turn.get("value", "")
                    if answer:
                        sample_answers.append(answer[:200] + "..." if len(answer) > 200 else answer)
                    break
            sample_event_files.append(ex.get("event_data", "unknown"))
        
        analysis_entry = {
            "rank": rank,
            "question": question,
            "frequency": count,
            "percentage": round(percentage, 2),
            "sample_answers": sample_answers[:3],  # Keep only 3 samples
            "sample_event_files": sample_event_files[:3]
        }
        detailed_analysis.append(analysis_entry)
        
        print(f"\n  [{rank}] {question}")
        print(f"      Frequency: {count} ({percentage:.2f}%)")
        print(f"      Sample answers: {len(sample_answers)}")
    
    # Save detailed analysis to JSON
    detailed_json_path = os.path.join(OUTPUT_DIR, "top10_questions_detailed.json")
    with open(detailed_json_path, 'w', encoding='utf-8') as f:
        json.dump(detailed_analysis, f, indent=2, ensure_ascii=False)
    print(f"\n💾 Saved detailed analysis to: {detailed_json_path}")
    
    # Create a human-readable summary
    summary_path = os.path.join(OUTPUT_DIR, "top10_questions_summary.txt")
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("Top 10 Questions Analysis - EventGPT DSEC Dataset\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Total questions analyzed: {len(questions)}\n")
        f.write(f"Total unique questions: {len(question_counter)}\n")
        f.write(f"Total dataset entries: {len(dataset)}\n\n")
        
        f.write("-" * 80 + "\n")
        f.write("Top 10 Questions:\n")
        f.write("-" * 80 + "\n\n")
        
        for entry in detailed_analysis:
            f.write(f"\n[{entry['rank']}] {entry['question']}\n")
            f.write(f"    Frequency: {entry['frequency']} ({entry['percentage']}%)\n")
            f.write(f"    Sample event files:\n")
            for event_file in entry['sample_event_files']:
                f.write(f"      - {event_file}\n")
            f.write(f"    Sample answers:\n")
            for i, answer in enumerate(entry['sample_answers'], 1):
                f.write(f"      {i}. {answer}\n")
            f.write("\n")
    
    print(f"💾 Saved summary to: {summary_path}")
    
    # Create a comparison table
    print("\n📊 Creating comparison statistics...")
    stats = {
        "total_questions": len(questions),
        "unique_questions": len(question_counter),
        "top10_total": sum(counts),
        "top10_percentage": (sum(counts) / len(questions)) * 100,
        "top10_coverage": (sum(counts) / len(dataset)) * 100,
    }
    
    stats_path = os.path.join(OUTPUT_DIR, "top10_questions_stats.json")
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2)
    print(f"💾 Saved statistics to: {stats_path}")
    
    print("\n" + "=" * 60)
    print("✅ Top 10 Questions Analysis Complete!")
    print("=" * 60)
    print(f"\n📁 Results saved to: {OUTPUT_DIR}")
    print(f"   - top10_questions.txt")
    print(f"   - top10_questions_bar.png")
    print(f"   - top10_questions_pie.png")
    print(f"   - top10_questions_detailed.json")
    print(f"   - top10_questions_summary.txt")
    print(f"   - top10_questions_stats.json")


def main():
    """Main analysis function."""
    if isinstance(ds, DatasetDict):
        print("⚠️ DatasetDict detected. Analyzing first split...")
        split_name = list(ds.keys())[0]
        analyze_top10_questions(ds[split_name])
    elif isinstance(ds, Dataset):
        analyze_top10_questions(ds)
    else:
        raise TypeError(f"Unexpected dataset type: {type(ds)}")


if __name__ == "__main__":
    main()

