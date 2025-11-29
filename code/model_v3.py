from transformers import (
    AutoTokenizer, AutoModelForTokenClassification,
    TrainingArguments, Trainer, DataCollatorForTokenClassification,
    PreTrainedModel, AutoConfig, BertModel, PretrainedConfig
)
from typing import Dict, List
import torch
from torch import nn


class EnhancedConfig(PretrainedConfig):
    # Recommend adding model_type to distinguish this config
    model_type = "enhanced_ner_model"

    def __init__(self,
                 # Parameters extracted from ExperimentConfig with default values
                 # These defaults should match values in ExperimentConfig, or be reasonable baseline values
                 feature_proj_dim: int = 64,
                 use_bilstm: bool = False,
                 lstm_hidden_size: int = 512,
                 lstm_num_layers: int = 1,
                 use_transformer_attn: bool = False,
                 transformer_attn_n_head: int = 4,
                 transformer_attn_num_layers: int = 1,
                 classifier_hidden_dim: int = 256,
                 classifier_dropout: float = 0.2,
                 use_char_feature: bool = False, # Also store flags indicating which features are used
                 use_pos_tag: bool = False,
                 use_capital_feature: bool = False,
                 # Base BERT/PretrainedConfig parameters (e.g. num_labels should be set properly)
                 num_labels: int = 17, # Ensure label2id is defined
                 id2label: Dict[int, str] = {}, # Ensure id2label is defined
                 label2id: Dict[str, int] = {}, # Ensure label2id is defined
                 hidden_dropout_prob: float = 0.1,
                 # Other PretrainedConfig parameters you might need...
                 **kwargs):

        super().__init__(num_labels=num_labels, id2label=id2label, label2id=label2id, **kwargs)

        # Store custom parameters
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
    def __init__(self, input_dim, n_head, num_layers, dropout, dim_feedforward=None): # <--- add dim_feedforward
        super().__init__()
        # If not specified, use the PyTorch default (typically 2048 or 4*input_dim)
        if dim_feedforward is None:
            dim_feedforward = input_dim * 4 # A common default

        # Note: d_model must be divisible by n_head
        if input_dim % n_head != 0:
            # You may adjust n_head or input_dim; here we warn and try to adjust n_head
             print(f"Warning: input_dim ({input_dim}) is not divisible by n_head ({n_head}). Adjusting n_head.")
             # Find an n_head value that divides input_dim (simple heuristic)
             for head in range(n_head, 0, -1):
                 if input_dim % head == 0:
                     n_head = head
                     print(f"Adjusted n_head to {n_head}")
                     break
             if input_dim % n_head != 0: # If still not found
                 raise ValueError(f"Cannot find suitable n_head for input_dim {input_dim}")

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim,
            nhead=n_head,
            dim_feedforward=dim_feedforward, # <--- Use dim_feedforward
            dropout=dropout,
            batch_first=True # <--- Important: set batch_first=True so there is no need to transpose
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

    def forward(self, x, src_key_padding_mask=None): # <--- Accepts padding mask
        # x: [batch, seq_len, input_dim] (because batch_first=True)
        # src_key_padding_mask: [batch, seq_len], True indicates positions to mask
        x = self.transformer_encoder(x, src_key_padding_mask=src_key_padding_mask)
        return x

class EnhancedModel(PreTrainedModel):
    def __init__(self, base_model, config: EnhancedConfig): # Explicitly hint config as EnhancedConfig
        super().__init__(config) # Pass config to base class
        self.bert = base_model
        self.config = config # <--- Save config (PreTrainedModel does this but explicit is OK)

        # Feature fusion layer
        self.feature_dim = 0
        feature_input_dim = 0
        # Use parameters from config
        if any([self.config.use_char_feature, self.config.use_pos_tag, self.config.use_capital_feature]):
            feature_input_dim = sum([
                1 if self.config.use_capital_feature else 0,
                2 if self.config.use_char_feature else 0,
                1 if self.config.use_pos_tag else 0
            ])
            # Create Linear projection only when feature_input_dim > 0
            if feature_input_dim > 0:
                self.feature_proj = nn.Linear(feature_input_dim, self.config.feature_proj_dim)
                self.feature_dim = self.config.feature_proj_dim
            else:
                self.feature_proj = None # or do not create the projection
                self.feature_dim = 0


        # Compute BiLSTM input dimension
        bilstm_input_dim = self.bert.config.hidden_size + self.feature_dim

        # BiLSTM layer
        self.bilstm = None
        # Use config parameters
        if self.config.use_bilstm:
            self.bilstm = nn.LSTM(
                input_size=bilstm_input_dim,
                hidden_size=self.config.lstm_hidden_size, # <--- Use config
                num_layers=self.config.lstm_num_layers,    # <--- Use config
                bidirectional=True,
                batch_first=True,
                # Use dropout defined in config
                # dropout=self.config.classifier_dropout if self.config.lstm_num_layers > 1 else 0
            )
            current_output_dim = self.config.lstm_hidden_size * 2
        else:
            current_output_dim = bilstm_input_dim

        # Transformer Attention layer
        self.transformer_attn = None
        # Use parameters from config
        if self.config.use_transformer_attn:
            self.transformer_attn = TransformerAttention(
                input_dim=current_output_dim,
                n_head=self.config.transformer_attn_n_head, # <--- Use config
                num_layers=self.config.transformer_attn_num_layers, # <--- Use config
                dropout=self.config.classifier_dropout, # <--- Use config
            )

        # CRF layer
        self.crf = None
        # if self.config.use_crf: # Assume use_crf is in config (if needed)
        #     self.crf = CRF(num_tags=config.num_labels, batch_first=True)

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(current_output_dim, self.config.classifier_hidden_dim), # <--- Use config
            nn.LayerNorm(self.config.classifier_hidden_dim),
            nn.Dropout(self.config.classifier_dropout), # <--- Use config
            nn.Linear(self.config.classifier_hidden_dim, self.config.num_labels) # Use config.num_labels
        )

        # (Optional) Better initialization
        # self._init_weights() # _init_weights should also use self.config internally

    def _init_weights(self):
        # Initialize newly-added Linear and LSTM layers
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
                        # LSTM bias trick: initialize forget gate bias to 1
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
        # Remove unnecessary parameters
        _ = kwargs.pop("num_items_in_batch", None)

        # BERT encoding
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            # output_hidden_states=True, # Ensure hidden states are output
            # return_dict=True,
            **kwargs
        )
        sequence_output = outputs.last_hidden_state
        
        # Feature fusion
        if word_features is not None and self.feature_dim > 0 and hasattr(self, 'feature_proj'):
            # Ensure word_features is a FloatTensor
            if word_features.dtype != torch.float32:
                word_features = word_features.float()
            features = self.feature_proj(word_features)
            sequence_output = torch.cat([sequence_output, features], dim=-1)
        
        # BiLSTM processing
        if self.bilstm is not None:
            # LSTM can use true sequence lengths for PackedSequence optimization (optional, recommended)
            # lengths = attention_mask.sum(dim=1).cpu() # Need to calculate lengths on CPU
            # packed_embeddings = nn.utils.rnn.pack_padded_sequence(sequence_output, lengths, batch_first=True, enforce_sorted=False)
            # packed_output, _ = self.bilstm(packed_embeddings)
            # sequence_output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True, total_length=input_ids.size(1)) # Pad back to original length

            # Simpler path: feed directly without packing
            sequence_output, _ = self.bilstm(sequence_output) # [batch, seq_len, hidden*2]
        
        # Transformer enhancement
        if self.transformer_attn is not None:
            # Transformer Attention needs to know which tokens are padding
            # attention_mask: 1 indicates valid token, 0 indicates padding
            # src_key_padding_mask: True = padding, False = valid tokens
            transformer_padding_mask = (attention_mask == 0)
            sequence_output = self.transformer_attn(sequence_output, src_key_padding_mask=transformer_padding_mask)
        
        # Classification prediction
        logits = self.classifier(sequence_output)
        
        loss = None
        if labels is not None:
            # Compute cross-entropy loss
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            active_loss = attention_mask.view(-1) == 1
            active_logits = logits.view(-1, self.config.num_labels)[active_loss]
            active_labels = labels.view(-1)[active_loss]
            loss = loss_fct(active_logits, active_labels)
    
        return {"loss": loss, "logits": logits} if loss is not None else {"logits": logits}