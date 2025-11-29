from transformers import (
    AutoTokenizer, AutoModelForTokenClassification,
    TrainingArguments, Trainer, DataCollatorForTokenClassification,
    PreTrainedModel, AutoConfig, BertModel, PretrainedConfig
)
from typing import Dict, List
import torch
from torch import nn


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
                 num_labels: int = 17, # 确保 label2id 已定义
                 id2label: Dict[int, str] = {}, # 确保 id2label 已定义
                 label2id: Dict[str, int] = {}, # 确保 label2id 已定义
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