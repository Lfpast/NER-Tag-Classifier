import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import pandas as pd
from ast import literal_eval
from transformers import AutoTokenizer, DataCollatorForTokenClassification
from datasets import Dataset
import torch
import json
import nltk
from safetensors import safe_open
from model_v3 import *
from exp_configs import ExperimentConfig
import csv
from tqdm import tqdm

model_checkpoint_path = "D:\\Compiler\\Environment\\Python\\COMP_4211\\Project\\21035474_21041758_submission\\checkpoint-2414"
safetensor_path = os.path.join(model_checkpoint_path, "model.safetensors")
config_path = os.path.join(model_checkpoint_path, "config.json")

with open(config_path, "r") as f:
    config = json.load(f)

id2label = config["id2label"]
label2id = config["label2id"]

exp_config = ExperimentConfig(
    use_char_feature=config['use_char_feature'],
    use_pos_tag=config['use_pos_tag'],
    use_capital_feature=config['use_capital_feature'],
    use_bilstm=config['use_bilstm'],
    lstm_hidden_size=config['lstm_hidden_size'],       # 示例：增大 LSTM
    lstm_num_layers=config['lstm_num_layers'],          # 示例：增加 LSTM 层数
    use_transformer_attn=config['use_transformer_attn'],  # 启用 Transformer Attention
    transformer_attn_n_head=config['transformer_attn_n_head'],  # 示例：增加 Attention 头数
    transformer_attn_num_layers=config['transformer_attn_num_layers'],# 示例：增加 Attention 层数
    classifier_hidden_dim=config['classifier_hidden_dim'],  # 示例：增大分类头
    classifier_dropout=config['classifier_dropout'],     # 示例：增加 Dropout
    # --- 其他配置保持默认或你的原始设置 ---
    feature_proj_dim=config['feature_proj_dim'],
)

config = EnhancedConfig(
    feature_proj_dim=exp_config.feature_proj_dim,
    use_bilstm=exp_config.use_bilstm,
    lstm_hidden_size=exp_config.lstm_hidden_size,
    lstm_num_layers=exp_config.lstm_num_layers,
    use_transformer_attn=exp_config.use_transformer_attn,
    transformer_attn_n_head=exp_config.transformer_attn_n_head,
    transformer_attn_num_layers=exp_config.transformer_attn_num_layers,
    classifier_hidden_dim=exp_config.classifier_hidden_dim,
    classifier_dropout=exp_config.classifier_dropout,
    use_char_feature=exp_config.use_char_feature,
    use_pos_tag=exp_config.use_pos_tag,
    use_capital_feature=exp_config.use_capital_feature,
    num_labels=len(label2id),
    id2label=id2label,
    label2id=label2id,
    output_hidden_states=True,
    ignore_mismatched_sizes=True,
)

base_model_checkpoint = "bert-large-cased-whole-word-masking"
tokenizer = AutoTokenizer.from_pretrained(base_model_checkpoint)
data_collator = DataCollatorForTokenClassification(tokenizer)

class FeatureEngineer:
    @staticmethod
    def align_labels(tokens, tags, word_ids):
        aligned_labels = []
        previous_word_idx = None
        for word_idx in word_ids:
            if word_idx is None:
                aligned_labels.append(-100)
            elif word_idx != previous_word_idx:
                aligned_labels.append(tags[word_idx])
            else:
                aligned_labels.append(-100)
            previous_word_idx = word_idx
        return aligned_labels

    @classmethod
    def tokenize_with_features(cls, examples):
        tokenized = tokenizer(
            examples["Sentence"],
            truncation=True,
            padding='max_length',
            max_length=64,
            is_split_into_words=True,
            return_overflowing_tokens=False
        )
        # 生成并保存word_ids
        tokenized["word_ids"] = []
        for i in range(len(examples["Sentence"])):
            word_ids = tokenized.word_ids(batch_index=i)
            tokenized["word_ids"].append(word_ids)
        # 标签对齐
        # labels = []
        # for i, tags in enumerate(examples["NER Tag"]):
        #     word_ids = tokenized.word_ids(batch_index=i)
        #     aligned = cls.align_labels(
        #         tokens=tokenized["input_ids"][i],
        #         tags=[label2id[tag] for tag in tags],
        #         word_ids=word_ids
        #     )
        #     labels.append(aligned)
        # tokenized["labels"] = labels

        # 特征生成
        if any([exp_config.use_char_feature, exp_config.use_pos_tag, exp_config.use_capital_feature]):
            total_feature_dim = 0
            feature_configs = {
                'capital': exp_config.use_capital_feature,
                'char': exp_config.use_char_feature,
                'pos': exp_config.use_pos_tag
            }
            total_feature_dim = sum([1 if feature_configs['capital'] else 0,
                                    2 if feature_configs['char'] else 0,
                                    1 if feature_configs['pos'] else 0])

            word_features = []
            for i, sentence in enumerate(examples["Sentence"]):
                # 生成POS标签
                pos_tags = nltk.pos_tag(sentence) if exp_config.use_pos_tag else []
                
                # 生成原始单词特征
                features_per_word = []
                for word_idx, word in enumerate(sentence):
                    features = []
                    # 大写特征
                    if exp_config.use_capital_feature:
                        features.append(float(word.isupper()))
                    # 字符特征
                    if exp_config.use_char_feature:
                        features.extend([float(len(word)), float(any(c.isdigit() for c in word))])
                    # POS特征
                    if exp_config.use_pos_tag:
                        pos_tag = pos_tags[word_idx][1] if pos_tags else None
                        features.append(float(pos_tag2id.get(pos_tag, 0)))
                    features_per_word.append(features)

                # 对齐到subword tokens
                aligned = []
                word_ids = tokenized.word_ids(batch_index=i)
                for word_id in word_ids:
                    if word_id is not None:
                        aligned.append(features_per_word[word_id])
                    else:
                        aligned.append([0.0] * total_feature_dim)
                
                # 验证对齐长度
                if len(aligned) != len(tokenized["input_ids"][i]):
                    raise ValueError(f"特征对齐失败于样本{i}")
                
                word_features.append(aligned)
            tokenized["word_features"] = word_features

        return tokenized









df = pd.read_csv("D:\\Compiler\\Environment\\Python\\COMP_4211\\Project\\21035474_21041758_submission\\data\\test.csv", converters={
    "Sentence": literal_eval,
    # "NER Tag": literal_eval
})

# POS标签处理
if exp_config.use_pos_tag:
    all_pos_tags = set()
    for sentence in df["Sentence"]:
        pos_tags = nltk.pos_tag(sentence)
        all_pos_tags.update(tag for _, tag in pos_tags)
    pos_tag2id = {tag: idx for idx, tag in enumerate(sorted(all_pos_tags))}

dataset = Dataset.from_pandas(df)

tokenized_test = dataset.map(
    FeatureEngineer.tokenize_with_features,
    batched=True,
    batch_size=1000,
    remove_columns=["id", "Sentence"]
)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


base_model = BertModel.from_pretrained(
    base_model_checkpoint,
    num_labels=len(label2id),
    id2label=id2label,
    label2id=label2id,
    ignore_mismatched_sizes=True
)

model = EnhancedModel(base_model, config)
with safe_open(safetensor_path, framework="pt") as f:
    state_dict = {k: f.get_tensor(k) for k in f.keys()}

model.load_state_dict(state_dict)

model.to(device)
model.eval()

def predict(model, tokenized_dataset):
    all_preds = []
    for data in tqdm(tokenized_dataset, desc="predicting", total=len(dataset)):
        # 将数据移动到GPU
        data = {k: torch.tensor(v).unsqueeze(0).to(device) for k, v in data.items() if k not in ["token_type_ids", "word_ids"] }
        with torch.no_grad():
            outputs = model.forward(**data)
            logits = outputs['logits']
            predictions = torch.argmax(logits, dim=-1)
            # 将预测结果转换为CPU上的numpy数组
            predictions = predictions.cpu().numpy()
            all_preds.append(predictions)
    return all_preds

predictions = predict(model, tokenized_test)
# 对齐标签到原始单词
final_labels = []
for i in range(len(dataset)):
    word_ids = tokenized_test[i]["word_ids"]  # 现在可以正确获取
    sentence = dataset[i]["Sentence"]
    preds = predictions[i]
    
    aligned = ["O"] * len(sentence)
    processed_word_ids = set()  # 新增：记录已处理的 word_id

    for token_idx, word_id in enumerate(word_ids):
        if word_id is None or word_id >= len(sentence):
            continue
        
        # 关键修改：检查 word_id 是否已被处理
        if word_id not in processed_word_ids:
            try:
                aligned[word_id] = id2label[str(preds[0][token_idx])]
            except:
                breakpoint()
            processed_word_ids.add(word_id)  # 标记为已处理
    final_labels.append(aligned)


output_df = pd.DataFrame({
    "id": df["id"],
    "NER Tag": final_labels
})

# 将列表格式转为字符串格式（与示例格式一致）
output_df["NER Tag"] = output_df["NER Tag"].apply(
    lambda x: str(x).replace("', '", "','")  # 替换元素间的空格
                     .replace("['", "['")     # 保持左括号后无空格（冗余操作，仅为明确逻辑）
                     .replace("']", "']")     # 保持右括号前无空格
)

output_df.to_csv("test_predictions.csv", index=False, quoting=csv.QUOTE_NONNUMERIC)