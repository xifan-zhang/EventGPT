#!/usr/bin/env python3
"""
EventGPT Training Script
Train your own EventGPT model from scratch
"""

import os
import torch
import numpy as np
from datasets import Dataset, DatasetDict
from transformers import (
    GPT2Config,
    GPT2LMHeadModel,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback
)
from huggingface_hub import snapshot_download

# ==================== 配置 ====================
DATA_DIR = "./datasets/EventGPT"
OUTPUT_DIR = "./eventgpt_output"
MODEL_SAVE_DIR = "./eventgpt_final"

# 模型配置
MODEL_CONFIG = {
    "vocab_size": 50257,
    "n_positions": 1024,
    "n_embd": 768,
    "n_layer": 12,
    "n_head": 12,
    "n_inner": 3072,
}

# 训练配置
TRAINING_CONFIG = {
    "num_train_epochs": 10,
    "per_device_train_batch_size": 8,
    "per_device_eval_batch_size": 8,
    "gradient_accumulation_steps": 4,
    "learning_rate": 5e-5,
    "warmup_steps": 500,
    "weight_decay": 0.01,
    "fp16": True,
    "save_steps": 1000,
    "eval_steps": 500,
    "logging_steps": 100,
}


# ==================== 数据下载 ====================
def download_dataset():
    """下载 EventGPT 数据集"""
    print("Downloading EventGPT dataset...")
    snapshot_download(
        repo_id="XduSyL/EventGPT-datasets",
        repo_type="dataset",
        local_dir=DATA_DIR,
        local_dir_use_symlinks=False
    )
    print("Dataset downloaded!")


# ==================== 数据加载 ====================
def load_event_data(data_dir, split='train'):
    """加载 event .npy 文件"""
    npy_dir = os.path.join(data_dir, 'event_npy', split)
    
    if not os.path.exists(npy_dir):
        raise FileNotFoundError(f"Directory not found: {npy_dir}")
    
    event_files = [f for f in os.listdir(npy_dir) if f.endswith('.npy')]
    print(f"Loading {len(event_files)} files from {npy_dir}")
    
    data_list = []
    for file in event_files:
        file_path = os.path.join(npy_dir, file)
        events = np.load(file_path)
        data_list.append(events)
    
    data = np.concatenate(data_list, axis=0)
    print(f"Loaded {split} data with shape: {data.shape}")
    return data


def create_datasets(data_dir):
    """创建训练和验证数据集"""
    print("\n" + "="*50)
    print("Loading datasets...")
    
    # 加载数据
    train_events = load_event_data(data_dir, 'train')
    val_events = load_event_data(data_dir, 'val')
    
    # 创建 Dataset
    train_dataset = Dataset.from_dict({
        'events': [event.tolist() for event in train_events]
    })
    
    val_dataset = Dataset.from_dict({
        'events': [event.tolist() for event in val_events]
    })
    
    dataset_dict = DatasetDict({
        'train': train_dataset,
        'validation': val_dataset
    })
    
    print(f"Dataset created: {dataset_dict}")
    return dataset_dict


# ==================== 模型定义 ====================
def create_model():
    """创建 EventGPT 模型"""
    print("\n" + "="*50)
    print("Creating model...")
    
    config = GPT2Config(**MODEL_CONFIG)
    model = GPT2LMHeadModel(config)
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model created with {num_params:,} parameters")
    
    return model


# ==================== 数据预处理 ====================
def preprocess_function(examples):
    """预处理函数"""
    # 这里根据你的实际数据格式调整
    events = examples['events']
    
    # 示例：转换为模型输入格式
    # 你可能需要进行归一化、tokenization等
    
    return {
        'input_ids': events,
        'labels': events
    }


# ==================== 训练 ====================
def train_model(model, datasets):
    """训练模型"""
    print("\n" + "="*50)
    print("Setting up training...")
    
    # 训练参数
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        
        # 训练超参数
        num_train_epochs=TRAINING_CONFIG["num_train_epochs"],
        per_device_train_batch_size=TRAINING_CONFIG["per_device_train_batch_size"],
        per_device_eval_batch_size=TRAINING_CONFIG["per_device_eval_batch_size"],
        gradient_accumulation_steps=TRAINING_CONFIG["gradient_accumulation_steps"],
        
        # 优化器
        learning_rate=TRAINING_CONFIG["learning_rate"],
        warmup_steps=TRAINING_CONFIG["warmup_steps"],
        weight_decay=TRAINING_CONFIG["weight_decay"],
        
        # 评估和保存
        eval_strategy="steps",
        eval_steps=TRAINING_CONFIG["eval_steps"],
        save_strategy="steps",
        save_steps=TRAINING_CONFIG["save_steps"],
        save_total_limit=3,
        
        # 日志
        logging_dir=os.path.join(OUTPUT_DIR, "logs"),
        logging_steps=TRAINING_CONFIG["logging_steps"],
        
        # 最佳模型
        load_best_model_at_end=True,
        metric_for_best_model="loss",
        greater_is_better=False,
        
        # 性能
        fp16=TRAINING_CONFIG["fp16"],
        dataloader_num_workers=4,
        
        # 报告
        report_to="tensorboard",
    )
    
    # 创建 Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=datasets["train"],
        eval_dataset=datasets["validation"],
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )
    
    # 开始训练
    print("\n" + "="*50)
    print("Starting training...")
    print("="*50 + "\n")
    
    train_result = trainer.train()
    
    # 保存模型
    print("\n" + "="*50)
    print("Saving model...")
    trainer.save_model(MODEL_SAVE_DIR)
    
    # 保存训练结果
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    
    return trainer


# ==================== 评估 ====================
def evaluate_model(trainer):
    """评估模型"""
    print("\n" + "="*50)
    print("Evaluating model...")
    
    eval_results = trainer.evaluate()
    
    print("\nEvaluation Results:")
    for key, value in eval_results.items():
        print(f"  {key}: {value}")
    
    # 保存评估结果
    trainer.save_metrics("eval", eval_results)
    
    return eval_results


# ==================== 主函数 ====================
def main():
    """主训练流程"""
    print("="*50)
    print("EventGPT Training Pipeline")
    print("="*50)
    
    # 1. 下载数据集（如果需要）
    if not os.path.exists(DATA_DIR):
        download_dataset()
    
    # 2. 创建数据集
    datasets = create_datasets(DATA_DIR)
    
    # 3. 创建模型
    model = create_model()
    
    # 4. 训练模型
    trainer = train_model(model, datasets)
    
    # 5. 评估模型
    evaluate_model(trainer)
    
    print("\n" + "="*50)
    print("Training completed!")
    print(f"Model saved to: {MODEL_SAVE_DIR}")
    print("="*50)


if __name__ == "__main__":
    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 运行训练
    main()