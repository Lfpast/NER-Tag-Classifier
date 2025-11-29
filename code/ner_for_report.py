import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["WANDB_DISABLED"] = "1"
import pandas as pd
import numpy as np
from ast import literal_eval
from transformers import (
    AutoTokenizer, AutoModelForTokenClassification,
    TrainingArguments, Trainer, DataCollatorForTokenClassification,
    PreTrainedModel, AutoConfig, BertModel, PretrainedConfig
)
from transformers.modeling_outputs import TokenClassifierOutput
from datasets import Dataset
import evaluate

import torch
from torch import nn
from typing import Dict, List
from dataclasses import dataclass
# from torchcrf import CRF
from sklearn.metrics import f1_score, precision_score, recall_score

import nltk
nltk.download('averaged_perceptron_tagger_eng')

def print_model_params(model):
    total = sum(p.numel() for p in model.parameters())
    print(f"模型参数量 Params: {total/1e9:.2f}B")  # 保留两位小数
# ====================
# 实验配置中心
# ====================
@dataclass
class ExperimentConfig:
    # 特征工程配置
    use_char_feature: bool = True  # 保持你的原始设置
    use_pos_tag: bool = True     # 保持你的原始设置
    use_capital_feature: bool = True # 保持你的原始设置

    # 模型架构配置
    feature_proj_dim: int = 128         # <--- 增加特征投影维度 (原 64)
    use_bilstm: bool = True             # 保持你的原始设置
    lstm_hidden_size: int = 768         # <--- 增加LSTM隐藏层大小 (原 512)
    lstm_num_layers: int = 2            # <--- 增加LSTM层数 (原 1)
    use_transformer_attn: bool = True   # <--- 新增开关控制 Transformer Attention
    transformer_attn_n_head: int = 8    # <--- 增加Attention头数 (原 4)
    transformer_attn_num_layers: int = 2 # <--- 增加Attention层数 (原 1)
    # TransformerEncoderLayer 通常有一个内部前馈网络，其维度可以是 d_model 的倍数
    # 我们可以不显式设置，让它使用默认值 (通常是 4 * d_model)，或者添加控制
    # transformer_attn_ff_dim: int = 2048 # <--- (可选) 控制内部前馈维度

    classifier_hidden_dim: int = 512    # <--- 增加分类头中间维度 (原 256)
    classifier_dropout: float = 0.3     # <--- (建议) 增加分类头的Dropout

    use_crf: bool = False # 你标记为垃圾，保持关闭

    # 训练策略配置 (保持不变)
    use_fgm: bool = False
    use_swa: bool = False
    use_gradient_checkpointing: bool = False
    custom_lr_scheduler: bool = False


# --- 实例化配置 ---
# 你可以根据需要调整这些值进行实验
exp_config = ExperimentConfig(
    use_char_feature=True,
    use_pos_tag=True,
    use_capital_feature=True,
    use_bilstm=True,
    lstm_hidden_size=512,       # 示例：增大 LSTM
    lstm_num_layers=1,          # 示例：增加 LSTM 层数
    use_transformer_attn=True,  # 启用 Transformer Attention
    transformer_attn_n_head=4,  # 示例：增加 Attention 头数
    transformer_attn_num_layers=1,# 示例：增加 Attention 层数
    classifier_hidden_dim=256,  # 示例：增大分类头
    classifier_dropout=0.2,     # 示例：增加 Dropout
    # --- 其他配置保持默认或你的原始设置 ---
    feature_proj_dim=128,
)

# ====================
# 数据预处理（整合版）
# ====================
# 读取数据
df = pd.read_csv("train.csv", converters={
    "Sentence": literal_eval,
    "NER Tag": literal_eval
})

# 标签处理
all_labels = set()
for tags in df["NER Tag"]:
    all_labels.update(tags)

label2id = {label: idx for idx, label in enumerate(sorted(all_labels))}
id2label = {idx: label for label, idx in label2id.items()}
# breakpoint()

# POS标签处理
if exp_config.use_pos_tag:
    all_pos_tags = set()
    for sentence in df["Sentence"]:
        pos_tags = nltk.pos_tag(sentence)
        all_pos_tags.update(tag for _, tag in pos_tags)
    pos_tag2id = {tag: idx for idx, tag in enumerate(sorted(all_pos_tags))}
# ====================
# 特征工程模块
# ====================
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

        # 标签对齐
        labels = []
        for i, tags in enumerate(examples["NER Tag"]):
            word_ids = tokenized.word_ids(batch_index=i)
            aligned = cls.align_labels(
                tokens=tokenized["input_ids"][i],
                tags=[label2id[tag] for tag in tags],
                word_ids=word_ids
            )
            labels.append(aligned)
        tokenized["labels"] = labels

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


# ====================
# 增强模型架构
# ====================
class EnhancedConfig(PretrainedConfig):
    # 建议添加 model_type，以便区分
    model_type = "enhanced_ner_model"

    def __init__(self,
                 # 从 ExperimentConfig 提取的参数，并提供默认值
                 # 这些默认值应该与 ExperimentConfig 中的默认值匹配，或设为合理的基础值
                 feature_proj_dim: int = 64,
                 use_bilstm: bool = False,
                 lstm_hidden_size: int = 512,
                 lstm_num_layers: int = 1,
                 use_transformer_attn: bool = False,
                 transformer_attn_n_head: int = 4,
                 transformer_attn_num_layers: int = 1,
                 classifier_hidden_dim: int = 256,
                 classifier_dropout: float = 0.2,
                 use_char_feature: bool = False, # 也存储特征使用标记
                 use_pos_tag: bool = False,
                 use_capital_feature: bool = False,
                 # 基础 BERT/PretrainedConfig 参数 (num_labels 等应该被正确设置)
                 num_labels: int = len(label2id), # 确保 label2id 已定义
                 id2label: Dict[int, str] = id2label, # 确保 id2label 已定义
                 label2id: Dict[str, int] = label2id, # 确保 label2id 已定义
                 hidden_dropout_prob: float = 0.1,
                 # 其他你可能需要的 PretrainedConfig 参数...
                 **kwargs):

        super().__init__(num_labels=num_labels, id2label=id2label, label2id=label2id, **kwargs)

        # 保存自定义参数
        self.feature_proj_dim = feature_proj_dim
        self.use_bilstm = use_bilstm
        self.lstm_hidden_size = lstm_hidden_size
        self.lstm_num_layers = lstm_num_layers
        self.use_transformer_attn = use_transformer_attn
        self.transformer_attn_n_head = transformer_attn_n_head
        self.transformer_attn_num_layers = transformer_attn_num_layers
        self.classifier_hidden_dim = classifier_hidden_dim
        self.classifier_dropout = classifier_dropout
        self.use_char_feature = use_char_feature
        self.use_pos_tag = use_pos_tag
        self.use_capital_feature = use_capital_feature
        self.hidden_dropout_prob = hidden_dropout_prob

class TransformerAttention(nn.Module):
    def __init__(self, input_dim, n_head, num_layers, dropout, dim_feedforward=None): # <--- 添加 dim_feedforward
        super().__init__()
        # 如果未指定，使用 PyTorch 默认值 (通常是 2048 或 4*input_dim)
        if dim_feedforward is None:
            dim_feedforward = input_dim * 4 # 一个常见的默认值

        # 注意：d_model 必须是 n_head 的整数倍
        if input_dim % n_head != 0:
            # 可以调整 n_head 或 input_dim，这里我们简单地报错或调整 n_head
             print(f"Warning: input_dim ({input_dim}) is not divisible by n_head ({n_head}). Adjusting n_head.")
             # 查找一个能整除 input_dim 的 n_head (简单策略)
             for head in range(n_head, 0, -1):
                 if input_dim % head == 0:
                     n_head = head
                     print(f"Adjusted n_head to {n_head}")
                     break
             if input_dim % n_head != 0: # 如果还是找不到
                 raise ValueError(f"Cannot find suitable n_head for input_dim {input_dim}")

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim,
            nhead=n_head,
            dim_feedforward=dim_feedforward, # <--- 使用 dim_feedforward
            dropout=dropout,
            batch_first=True # <--- 关键：设置 batch_first=True，这样就不需要转置了！
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

    def forward(self, x, src_key_padding_mask=None): # <--- 接受 mask
        # x: [batch, seq_len, input_dim] (因为 batch_first=True)
        # src_key_padding_mask: [batch, seq_len], True 表示需要 mask 的位置
        x = self.transformer_encoder(x, src_key_padding_mask=src_key_padding_mask)
        return x

class EnhancedModel(PreTrainedModel):
    def __init__(self, base_model, config: EnhancedConfig): # 明确类型提示 config 是 EnhancedConfig
        super().__init__(config) # 传递 config 给父类
        self.bert = base_model
        self.config = config # <--- 保存 config (PreTrainedModel 默认会做，但显式写出也无妨)

        # 特征融合层
        self.feature_dim = 0
        feature_input_dim = 0
        # 使用 config 中的参数
        if any([self.config.use_char_feature, self.config.use_pos_tag, self.config.use_capital_feature]):
            feature_input_dim = sum([
                1 if self.config.use_capital_feature else 0,
                2 if self.config.use_char_feature else 0,
                1 if self.config.use_pos_tag else 0
            ])
            # 如果 feature_input_dim > 0 才创建 Linear 层
            if feature_input_dim > 0:
                self.feature_proj = nn.Linear(feature_input_dim, self.config.feature_proj_dim)
                self.feature_dim = self.config.feature_proj_dim
            else:
                self.feature_proj = None # 或者不创建
                self.feature_dim = 0


        # 计算 BiLSTM 的输入维度
        bilstm_input_dim = self.bert.config.hidden_size + self.feature_dim

        # BiLSTM层
        self.bilstm = None
        # 使用 config 中的参数
        if self.config.use_bilstm:
            self.bilstm = nn.LSTM(
                input_size=bilstm_input_dim,
                hidden_size=self.config.lstm_hidden_size, # <--- 使用 config
                num_layers=self.config.lstm_num_layers,    # <--- 使用 config
                bidirectional=True,
                batch_first=True,
                # 使用 config 中的 dropout
                # dropout=self.config.classifier_dropout if self.config.lstm_num_layers > 1 else 0
            )
            current_output_dim = self.config.lstm_hidden_size * 2
        else:
            current_output_dim = bilstm_input_dim

        # Transformer Attention 层
        self.transformer_attn = None
        # 使用 config 中的参数
        if self.config.use_transformer_attn:
            self.transformer_attn = TransformerAttention(
                input_dim=current_output_dim,
                n_head=self.config.transformer_attn_n_head, # <--- 使用 config
                num_layers=self.config.transformer_attn_num_layers, # <--- 使用 config
                dropout=self.config.classifier_dropout, # <--- 使用 config
            )

        # CRF层
        self.crf = None
        # if self.config.use_crf: # 假设 use_crf 也在 config 中 (如果需要的话)
        #     self.crf = CRF(num_tags=config.num_labels, batch_first=True)

        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(current_output_dim, self.config.classifier_hidden_dim), # <--- 使用 config
            nn.LayerNorm(self.config.classifier_hidden_dim),
            nn.Dropout(self.config.classifier_dropout), # <--- 使用 config
            nn.Linear(self.config.classifier_hidden_dim, self.config.num_labels) # 使用 config.num_labels
        )

        # (可选) 更好的初始化
        # self._init_weights() # _init_weights 内部也应该使用 self.config

    def _init_weights(self):
        # 初始化新增的 Linear 和 LSTM 层
        for module in self.modules():
            if isinstance(module, nn.Linear):
                module.weight.data.normal_(mean=0.0, std=self.config.initializer_range if hasattr(self.config, 'initializer_range') else 0.02)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.LSTM):
                for name, param in module.named_parameters():
                    if 'weight_ih' in name:
                        torch.nn.init.xavier_uniform_(param.data)
                    elif 'weight_hh' in name:
                        torch.nn.init.orthogonal_(param.data)
                    elif 'bias' in name:
                        param.data.fill_(0)
                        # LSTM bias 技巧：初始化 forget gate bias 为 1
                        n = param.size(0)
                        start, end = n // 4, n // 2
                        param.data[start:end].fill_(1.)
            elif isinstance(module, nn.LayerNorm):
                module.bias.data.zero_()
                module.weight.data.fill_(1.0)
    
    def forward(self, 
               input_ids,
               attention_mask,
               word_features=None,
               labels=None,
               **kwargs):
        # 删除不需要的参数
        _ = kwargs.pop("num_items_in_batch", None)

        # BERT编码
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            # output_hidden_states=True, # 确保输出了 hidden states
            # return_dict=True,
            **kwargs
        )
        sequence_output = outputs.last_hidden_state
        
        # 特征融合
        if word_features is not None and self.feature_dim > 0 and hasattr(self, 'feature_proj'):
            # 确保 word_features 是 FloatTensor
            if word_features.dtype != torch.float32:
                word_features = word_features.float()
            features = self.feature_proj(word_features)
            sequence_output = torch.cat([sequence_output, features], dim=-1)
        
        # BiLSTM处理
        if self.bilstm is not None:
            # LSTM 需要知道序列的真实长度以进行 PackedSequence 优化（可选但推荐）
            # lengths = attention_mask.sum(dim=1).cpu() # 需要在 CPU 上计算长度
            # packed_embeddings = nn.utils.rnn.pack_padded_sequence(sequence_output, lengths, batch_first=True, enforce_sorted=False)
            # packed_output, _ = self.bilstm(packed_embeddings)
            # sequence_output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True, total_length=input_ids.size(1)) # 填充回原始长度

            # 简化版：直接输入，不进行 packing
            sequence_output, _ = self.bilstm(sequence_output) # [batch, seq_len, hidden*2]
        
        # Transformer增强
        if self.transformer_attn is not None:
            # Transformer Attention 需要知道哪些是 padding token
            # attention_mask 是 1 表示有效，0 表示 padding
            # src_key_padding_mask 需要 True 表示 padding，False 表示有效
            transformer_padding_mask = (attention_mask == 0)
            sequence_output = self.transformer_attn(sequence_output, src_key_padding_mask=transformer_padding_mask)
        
        # 分类预测
        logits = self.classifier(sequence_output)
        
        loss = None
        if labels is not None:
            # 交叉熵损失计算
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            active_loss = attention_mask.view(-1) == 1
            active_logits = logits.view(-1, self.config.num_labels)[active_loss]
            active_labels = labels.view(-1)[active_loss]
            loss = loss_fct(active_logits, active_labels)
    
        return {"loss": loss, "logits": logits} if loss is not None else {"logits": logits}


# ====================
# 对抗训练模块
# ====================
class FGMTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fgm = None

    def _init_fgm(self):
        class FGM:
            def __init__(self, model):
                self.model = model
                self.backup = {}

            def attack(self):
                for name, param in self.model.named_parameters():
                    if param.requires_grad and param.grad is not None:
                        self.backup[name] = param.data.clone()
                        norm = torch.norm(param.grad)
                        if norm != 0:
                            r_at = 0.1 * param.grad / norm
                            param.data.add_(r_at)

            def restore(self):
                for name, param in self.model.named_parameters():
                    if name in self.backup:
                        param.data = self.backup[name]
                self.backup = {}

        self.fgm = FGM(self.model)

    def training_step(self, model, inputs, *args, **kwargs):
        loss = super().training_step(model, inputs, *args)  # 传递所有参数
        return loss
    
    
    def create_optimizer(self):
        # return super().create_optimizer()
        named_params = list(self.model.named_parameters())
        no_optim_params = ["bert.pooler.dense.weight", "bert.pooler.dense.bias"]
        for name, param in self.model.named_parameters():
            if name in no_optim_params:
                if param.requires_grad:
                    param.requires_grad = False
        # ---- 定义参数组 ----
        optimizer_grouped_parameters = [
            # BERT参数（带权重衰减）
            {
                "params": [
                    p for n, p in named_params 
                    if "bert" in n 
                    and not any(nd in n for nd in ["bias", "LayerNorm.weight"]) and n not in no_optim_params
                ],
                "weight_decay": self.args.weight_decay,
                "lr": self.args.learning_rate * 1
            },
            # BERT参数（无权重衰减）
            {
                "params": [
                    p for n, p in named_params 
                    if "bert" in n 
                    and any(nd in n for nd in ["bias", "LayerNorm.weight"]) and n not in no_optim_params
                ],
                "weight_decay": 0.0,
                "lr": self.args.learning_rate * 1
            },
            # 新增层参数（带权重衰减）
            {
                "params": [
                    p for n, p in named_params 
                    if "bert" not in n 
                    and not any(nd in n for nd in ["bias", "LayerNorm.weight"]) and n not in no_optim_params
                ],
                "weight_decay": self.args.weight_decay,
                "lr": self.args.learning_rate
            },
            # 新增层参数（无权重衰减）
            {
                "params": [
                    p for n, p in named_params 
                    if "bert" not in n 
                    and any(nd in n for nd in ["bias", "LayerNorm.weight"]) and n not in no_optim_params
                ],
                "weight_decay": 0.0,
                "lr": self.args.learning_rate
            }
        ]
        
        # ---- 关键修复：过滤空参数组 ----
        # 移除空参数组，避免优化器初始化失败
        optimizer_grouped_parameters = [
            group for group in optimizer_grouped_parameters 
            if len(group["params"]) > 0
        ]
        if not optimizer_grouped_parameters:
            raise ValueError(
                "No parameters to optimize. "
                "请检查参数分组条件或模型参数命名是否符合预期。"
                f"当前模型参数列表：{list(self.model.named_parameters())}"
            )
        optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters,
            lr=self.args.learning_rate,
            betas=(self.args.adam_beta1, self.args.adam_beta2),
            eps=self.args.adam_epsilon
        )
        self.optimizer = optimizer 
        
        return optimizer



# ====================
# 训练流程主函数
# ====================
# 初始化组件
model_checkpoint = "dbmdz/bert-large-cased-finetuned-conll03-english"
# model_checkpoint = "bert-base-cased"
# model_checkpoint = "bert-large-cased"
model_checkpoint = "bert-large-cased-whole-word-masking"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

# 数据预处理
dataset = Dataset.from_pandas(df)

split_dataset = dataset.train_test_split(test_size=0.1, shuffle=True, seed=42)

# 特征工程处理
tokenized_train = split_dataset["train"].map(
    FeatureEngineer.tokenize_with_features,
    batched=True,
    batch_size=1000,
    # remove_columns=df.columns.tolist()  # 自动移除原始列
)
tokenized_eval = split_dataset["test"].map(
    FeatureEngineer.tokenize_with_features,
    batched=True,
    batch_size=1000,
    # remove_columns=df.columns.tolist()
)
# tokenized_train = tokenized_train.filter(
#     lambda example: any(tag != 16 for tag in example["labels"]),
#     num_proc=8
# )
# tokenized_eval = tokenized_eval.filter(
#     lambda example: any(tag != 16 for tag in example["labels"]),
#     num_proc=8
# )
# 移除无用列
columns_to_remove = ["id", "Sentence", "NER Tag", "token_type_ids"]
# columns_to_remove = ["id", "Sentence"]  # 修改处，保留 "NER Tag" 用于后续处理
tokenized_train = tokenized_train.remove_columns([col for col in columns_to_remove if col not in {'input_ids', 'attention_mask', 'labels', 'word_features'}])
tokenized_eval = tokenized_eval.remove_columns([col for col in columns_to_remove if col not in {'input_ids', 'attention_mask', 'labels', 'word_features'}])

# # 显式设置保留字段（关键修复）
# columns_to_keep = ["input_ids", "token_type_ids", "attention_mask", "labels", "word_features"]
# tokenized_train.set_format(type="torch", columns=columns_to_keep)
# tokenized_eval.set_format(type="torch", columns=columns_to_keep)


# 修改处，预处理后检查
print("预处理后的样本键:", tokenized_train[0].keys())
# 应输出：dict_keys(['input_ids', 'token_type_ids', 'attention_mask', 'labels', ...])

# 修改处，检查标签是否有效
sample_labels = tokenized_train[0]["labels"]
print("标签示例:", sample_labels)
print("非-100的标签数量:", sum(1 for x in sample_labels if x != -100))

# 自定义数据收集器
class CustomDataCollator(DataCollatorForTokenClassification):
    def __call__(self, features):
        batch = super().__call__(features)
        # 仅在需要特征时处理word_features
        if exp_config.use_char_feature or exp_config.use_pos_tag or exp_config.use_capital_feature:
            # 检查所有特征是否都有word_features
            if "word_features" not in features[0]:
                raise KeyError(
                    "部分样本缺少'word_features'，请检查特征生成逻辑。"
                    "确保当使用字符、POS或大写特征时生成该字段。"
                )
            # 转换为张量并指定类型
            batch["word_features"] = torch.tensor(
                [f["word_features"] for f in features],
                dtype=torch.float32
            )
        return batch


data_collator = CustomDataCollator(tokenizer)

# 模型初始化
config = AutoConfig.from_pretrained(
    model_checkpoint,
    num_labels=len(label2id),
    id2label=id2label,
    label2id=label2id,
    output_hidden_states=True,
    ignore_mismatched_sizes=True,
)

model = BertModel.from_pretrained(
    model_checkpoint,
    num_labels=len(label2id),
    id2label=id2label,
    label2id=label2id,
    ignore_mismatched_sizes=True
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
model = EnhancedModel(model, config)
print_model_params(model)


# 训练参数
training_args = TrainingArguments(
    output_dir="/root/ner-model-v6",
    evaluation_strategy="epoch",
    learning_rate=1e-4,
    warmup_ratio=0.1,
    lr_scheduler_type="cosine",
    gradient_accumulation_steps=4,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    num_train_epochs=100,
    weight_decay=0.01,
    logging_steps=50,
    report_to="none",
    remove_unused_columns=False,
    # gradient_checkpointing=exp_config.use_gradient_checkpointing,
    bf16=True,
    # fp16=True,   # Enable float16 instead (if you have a compatible GPU)
    dataloader_pin_memory=True,
    ddp_find_unused_parameters=True,
    # dataloader_num_workers=8,
    save_strategy="best", # <--- 推荐保存每轮的模型，以便加载最好的
    load_best_model_at_end=True, # <--- 在训练结束时加载最佳模型
    metric_for_best_model="eval_f1", # <--- 根据 F1 选择最佳模型
    greater_is_better=True,        # <--- F1 越大越好
    save_only_model=True,
    # save_safetensors=False,
)

# 评估指标
metric = evaluate.load("seqeval")

def compute_metrics_new(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    # --- 修改后的逻辑 ---
    # 将 ID 转换为标签字符串，只移除 -100 标签
    true_predictions = [
        [id2label[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [id2label[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    # --- 修改结束 ---

    # 添加一个健壮性检查：如果过滤后所有序列都为空，则返回0
    # 这在理论上不应该发生，除非整个评估集都是空的或只有padding
    if not any(true_labels): # 检查 true_labels 是否为空或只包含空列表
         print("Warning: No valid labels found after filtering -100 in compute_metrics.")
         return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "accuracy": 0.0}

    results = metric.compute(predictions=true_predictions, references=true_labels) # 使用推荐的 mode 和 scheme
                                                                                                 # (如果你的标签是IOB2格式)
                                                                                                 # 如果是其他格式如 BILOU，请相应调整 scheme
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }

def compute_metrics_old(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    true_predictions = [
        [id2label[p] for (p, l) in zip(prediction, label) if l != -100 and id2label[l] != 'O']
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [id2label[l] for (p, l) in zip(prediction, label) if l != -100 and id2label[l] != 'O']
        for prediction, label in zip(predictions, labels)
    ]

    results = metric.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }


def compute_metrics(p):
    """
    Compute metrics including standard seqeval metrics and custom token accuracy on non-'O' labels.
    """
    predictions, labels = p
    # predictions 是模型输出的 logits [batch_size, seq_len, num_labels]
    # labels 是真实的标签 IDs [batch_size, seq_len] (-100 表示 padding)

    # 将 logits 转换为预测的标签 ID
    predictions = np.argmax(predictions, axis=2) # [batch_size, seq_len]

    # --- 1. Prepare for seqeval (Standard NER Metrics) ---
    # seqeval 需要的是列表的列表，每个内层列表是序列的标签字符串
    # seqeval 会自动处理 'O' 和实体标签，我们只需要过滤掉 padding token (-100)
    true_predictions_seqeval = [
        [id2label[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels_seqeval = [
        [id2label[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    # --- 2. Manual Calculation for Token Accuracy on Non-'O' True Labels ---
    # 这是你特别关注的指标：在所有真实标签不是 'O' 的 token 中，有多少被正确预测了
    total_entity_tokens = 0 # 真实标签是非 'O' 的 token 总数
    correct_entity_token_predictions = 0 # 真实标签是非 'O' 且预测标签与其相同的 token 总数

    # 遍历每个序列
    for prediction_ids, label_ids in zip(predictions, labels):
        # 遍历序列中的每个 token
        for pred_id, true_id in zip(prediction_ids, label_ids):
            # 忽略 padding token
            if true_id == -100:
                continue

            true_label_str = id2label[true_id]

            # 如果真实标签是非 'O'
            if true_label_str != 'O':
                total_entity_tokens += 1
                pred_label_str = id2label[pred_id]

                # 如果预测标签与真实标签相同
                if pred_label_str == true_label_str:
                    correct_entity_token_predictions += 1

    # 计算该指标
    # 避免除以零：如果评估集恰好没有任何非 'O' 标签（不太可能），则准确率为 0
    entity_token_accuracy = correct_entity_token_predictions / total_entity_tokens if total_entity_tokens > 0 else 0.0


    # --- 3. Calculate seqeval metrics ---
    # 健壮性检查：如果过滤 -100 后没有任何标签，则返回 0
    if not any(true_labels_seqeval):
         print("Warning: No valid labels found after filtering -100 in compute_metrics (for seqeval). Returning default metrics.")
         return {
            "overall_precision": 0.0,
            "overall_recall": 0.0,
            "overall_f1": 0.0,
            "overall_accuracy": 0.0, # seqeval token accuracy (includes O)
            "entity_token_accuracy": entity_token_accuracy # Our custom metric
         }

    # 使用 seqeval 计算标准 NER 指标 (Chunk-based F1, Precision, Recall)
    # 'mode='strict'' 是标准的评估模式，要求实体边界和类型都完全匹配
    # 'scheme='IOB2'' 需要与你的标签格式一致 (B-*, I-*, O)
    results = metric.compute(
        predictions=true_predictions_seqeval,
        references=true_labels_seqeval,
        mode='strict',
        scheme='IOB2' # 确保这里的 scheme 和你的 NER 标签格式匹配
    )

    # 返回所有计算的指标
    return {
        "overall_precision": results["overall_precision"], # seqeval Chunk Precision
        "overall_recall": results["overall_recall"],     # seqeval Chunk Recall
        "overall_f1": results["overall_f1"],             # seqeval Chunk F1 (通常是主要关注的)
        "overall_accuracy": results["overall_accuracy"], # seqeval Token Accuracy (includes O)
        "entity_token_accuracy": entity_token_accuracy   # Custom Token Accuracy (only on non-'O' true labels)
    }


# 开始训练
# trainer = FGMTrainer(
#     model=model,
#     args=training_args,
#     train_dataset=tokenized_train,
#     eval_dataset=tokenized_eval,
#     data_collator=data_collator,
#     compute_metrics=compute_metrics_old
# )
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_eval,
    data_collator=data_collator,
    compute_metrics=compute_metrics_old
)

print('开始训练')
trainer.train()

