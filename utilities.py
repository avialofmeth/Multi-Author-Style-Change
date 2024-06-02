
import os
import json
from datasets import Dataset

# 
def read_paragraphs(directory, start_id=1, end_id=50):
    documents = {}
    for problem_id in range(start_id, end_id + 1):
        subdir = os.path.join(directory, f'problem-{problem_id}')
        if os.path.exists(subdir):
            paragraphs = []
            for filename in sorted(os.listdir(subdir)):
                if filename.endswith('.txt'):
                    with open(os.path.join(subdir, filename), 'r') as file:
                        paragraphs.append(file.read().strip())
            documents[f'problem-{problem_id}'] = paragraphs
    return documents

# 截断文本
# truncate input texts to balance the lengths of the sentence 1 and the sentence 2. 
def truncate_text(text, tokenizer, max_length):
    tokens = tokenizer.tokenize(text)
    if len(tokens) > max_length:
        tokens = tokens[:max_length]
    return tokenizer.convert_tokens_to_string(tokens)


# 读取 ground truth 标签
# Read the ground truth
def read_ground_truth(directory, start_id=1, end_id=50):
    labels = {}
    for problem_id in range(start_id, end_id + 1):
        filename = os.path.join(directory, f'truth-problem-{problem_id}.json')
        if os.path.exists(filename):
            with open(filename, 'r') as file:
                data = json.load(file)
                labels[f'problem-{problem_id}'] = data['changes']
        else:
            print(f"File {filename} not existed")
    return labels

# 生成段落对及其嵌入
# Transfer raw documents and labels into huggingface Dataset object
def generate_dataset(documents, golden_labels, tokenizer):
    setence_1 = []
    setence_2 = []
    labels = []
    ids = []
    max_length = tokenizer.model_max_length
    for doc_id, paragraphs in documents.items():
        label_list = golden_labels[doc_id]
        for i in range(len(paragraphs) - 1):
            setence_1.append(truncate_text(paragraphs[i], tokenizer, max_length // 2))
            setence_2.append(truncate_text(paragraphs[i + 1], tokenizer, max_length // 2)) 
            labels.append(label_list[i])
            ids.append(doc_id)
    return Dataset.from_dict({'sentence1': setence_1, 'sentence2': setence_2, 
                              'label': labels, 'idx': ids})
  