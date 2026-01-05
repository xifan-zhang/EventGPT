from datasets import Dataset

# 直接从 arrow 文件加载
dataset = Dataset.from_file("data/XduSyL___event_gpt-datasets/default/0.0.0/8244c5bcf35438c7940d6e21bbaec447f396157d/event_gpt-datasets-train.arrow")

# 查看数据集信息
print(dataset)
print(f"Number of rows: {len(dataset)}")
print(f"Column names: {dataset.column_names}")

# 查看前几条数据
print(dataset[:5])

# 访问特定列
# print(dataset['column_names'])

# 迭代数据
for example in dataset:
    print(example)
    break  # 只打印第一条