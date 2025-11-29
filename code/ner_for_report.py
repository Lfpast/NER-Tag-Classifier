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
nltk.download('averaged_perceptron_tagger')

def print_model_params(model):
    total = sum(p.numel() for p in model.parameters())
    print(f"Model parameter count: {total/1e9:.2f}B")  # Keep two decimal places
# ====================
# Experiment configuration
# ====================
@dataclass
class ExperimentConfig:
    # Feature engineering configuration
    use_char_feature: bool = True  # Keep the original setting
    use_pos_tag: bool = True     # Keep the original setting
    use_capital_feature: bool = True # Keep the original setting

    # Model architecture configuration
    feature_proj_dim: int = 128         # <--- Increase feature projection dimension (was 64)
    use_bilstm: bool = True             # Keep the original setting
    lstm_hidden_size: int = 768         # <--- Increased LSTM hidden size (was 512)
    lstm_num_layers: int = 2            # <--- Increased LSTM num layers (was 1)
    use_transformer_attn: bool = True   # <--- Add switch to enable Transformer Attention
    transformer_attn_n_head: int = 8    # <--- Increase attention heads (was 4)
    transformer_attn_num_layers: int = 2 # <--- Increase transformer attention layers (was 1)
    # TransformerEncoderLayer usually has a feedforward network whose dim is a multiple of d_model.
    # We can leave the default (commonly 4 * d_model) or add a control parameter if needed.
    # transformer_attn_ff_dim: int = 2048 # <--- (optional) control for feedforward dim

    classifier_hidden_dim: int = 512    # <--- Increase classifier hidden dimension (was 256)
    classifier_dropout: float = 0.3     # <--- (recommended) increase dropout for classifier head

    use_crf: bool = False # Marked as unused; keep disabled

    # Training strategy options (unchanged)
    use_fgm: bool = False
    use_swa: bool = False
    use_gradient_checkpointing: bool = False
    custom_lr_scheduler: bool = False


# --- Instantiate configuration ---
# Adjust these values for experiments as needed
exp_config = ExperimentConfig(
    use_char_feature=True,
    use_pos_tag=True,
    use_capital_feature=True,
    use_bilstm=True,
    lstm_hidden_size=512,       # Example: increase LSTM size
    lstm_num_layers=1,          # Example: increase LSTM number of layers
    use_transformer_attn=True,  # Enable Transformer Attention
    transformer_attn_n_head=4,  # Example: increase attention heads
    transformer_attn_num_layers=1,# Example: increase attention layers
    classifier_hidden_dim=256,  # Example: increase classifier hidden dim
    classifier_dropout=0.2,     # Example: increase dropout
    # --- Other config: keep defaults or your original settings ---
    feature_proj_dim=128,
)

# ====================
# Data preprocessing (integrated)
# ====================
# Read data
df = pd.read_csv("train.csv", converters={
    "Sentence": literal_eval,
    "NER Tag": literal_eval
})

# Label processing
all_labels = set()
for tags in df["NER Tag"]:
    all_labels.update(tags)

label2id = {label: idx for idx, label in enumerate(sorted(all_labels))}
id2label = {idx: label for label, idx in label2id.items()}
# breakpoint()

# POS tag processing
if exp_config.use_pos_tag:
    all_pos_tags = set()
    for sentence in df["Sentence"]:
        pos_tags = nltk.pos_tag(sentence)
        all_pos_tags.update(tag for _, tag in pos_tags)
    pos_tag2id = {tag: idx for idx, tag in enumerate(sorted(all_pos_tags))}
# ====================
# Feature engineering module
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

        # Label alignment
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

        # Feature generation
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
                # Generate POS tags
                pos_tags = nltk.pos_tag(sentence) if exp_config.use_pos_tag else []
                
                # Generate raw word features
                features_per_word = []
                for word_idx, word in enumerate(sentence):
                    features = []
                    # Capitalization feature
                    if exp_config.use_capital_feature:
                        features.append(float(word.isupper()))
                    # Character features
                    if exp_config.use_char_feature:
                        features.extend([float(len(word)), float(any(c.isdigit() for c in word))])
                    # POS feature
                    if exp_config.use_pos_tag:
                        pos_tag = pos_tags[word_idx][1] if pos_tags else None
                        features.append(float(pos_tag2id.get(pos_tag, 0)))
                    features_per_word.append(features)

                # Align to subword tokens
                aligned = []
                word_ids = tokenized.word_ids(batch_index=i)
                for word_id in word_ids:
                    if word_id is not None:
                        aligned.append(features_per_word[word_id])
                    else:
                        aligned.append([0.0] * total_feature_dim)
                
                # Verify aligned length
                if len(aligned) != len(tokenized["input_ids"][i]):
                    raise ValueError(f"Feature alignment failed for sample {i}")
                
                word_features.append(aligned)
            tokenized["word_features"] = word_features

        return tokenized


# ====================
# Enhanced model architecture
# ====================
class EnhancedConfig(PretrainedConfig):
    # Recommend adding model_type to distinguish this config
    model_type = "enhanced_ner_model"

    def __init__(self,
                 # Parameters extracted from ExperimentConfig with default values
                 # These defaults should match the values in ExperimentConfig or be reasonable baselines
                 feature_proj_dim: int = 64,
                 use_bilstm: bool = False,
                 lstm_hidden_size: int = 512,
                 lstm_num_layers: int = 1,
                 use_transformer_attn: bool = False,
                 transformer_attn_n_head: int = 4,
                 transformer_attn_num_layers: int = 1,
                 classifier_hidden_dim: int = 256,
                 classifier_dropout: float = 0.2,
                 use_char_feature: bool = False, # Also store flags for features used
                 use_pos_tag: bool = False,
                 use_capital_feature: bool = False,
                 # Base BERT/PretrainedConfig parameters (e.g., num_labels should be set properly)
                 num_labels: int = len(label2id), # Ensure label2id is defined
                 id2label: Dict[int, str] = id2label, # Ensure id2label is defined
                 label2id: Dict[str, int] = label2id, # Ensure label2id is defined
                 hidden_dropout_prob: float = 0.1,
                 # Other PretrainedConfig parameters you might need...
                 **kwargs):

        super().__init__(num_labels=num_labels, id2label=id2label, label2id=label2id, **kwargs)

        # Save custom parameters
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
        # If not specified, use PyTorch default (typically 2048 or 4*input_dim)
        if dim_feedforward is None:
            dim_feedforward = input_dim * 4 # A common default

        # Note: d_model must be divisible by n_head
        if input_dim % n_head != 0:
            # You can adjust n_head or input_dim; here we warn and adjust n_head if possible
             print(f"Warning: input_dim ({input_dim}) is not divisible by n_head ({n_head}). Adjusting n_head.")
             # Find an n_head that divides input_dim (simple heuristic)
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
            batch_first=True # <--- Important: set batch_first=True so transposing is not required
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

    def forward(self, x, src_key_padding_mask=None): # <--- Accepts mask
        # x: [batch, seq_len, input_dim] (because batch_first=True)
        # src_key_padding_mask: [batch, seq_len], True indicates positions that need masking
        x = self.transformer_encoder(x, src_key_padding_mask=src_key_padding_mask)
        return x

class EnhancedModel(PreTrainedModel):
    def __init__(self, base_model, config: EnhancedConfig): # Explicit type hint: config is EnhancedConfig
        super().__init__(config) # Pass config to parent class
        self.bert = base_model
        self.config = config # <--- Save config (PreTrainedModel does this by default, but explicit is fine)

        # Feature fusion layer
        self.feature_dim = 0
        feature_input_dim = 0
        # Use config parameters
        if any([self.config.use_char_feature, self.config.use_pos_tag, self.config.use_capital_feature]):
            feature_input_dim = sum([
                1 if self.config.use_capital_feature else 0,
                2 if self.config.use_char_feature else 0,
                1 if self.config.use_pos_tag else 0
            ])
            # Create Linear projection only if feature_input_dim > 0
            if feature_input_dim > 0:
                self.feature_proj = nn.Linear(feature_input_dim, self.config.feature_proj_dim)
                self.feature_dim = self.config.feature_proj_dim
            else:
                self.feature_proj = None # Or do not create
                self.feature_dim = 0


        # Compute input dimension for BiLSTM
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
                # Use dropout from config
                # dropout=self.config.classifier_dropout if self.config.lstm_num_layers > 1 else 0
            )
            current_output_dim = self.config.lstm_hidden_size * 2
        else:
            current_output_dim = bilstm_input_dim

        # Transformer Attention layer
        self.transformer_attn = None
        # Use config parameters
        if self.config.use_transformer_attn:
            self.transformer_attn = TransformerAttention(
                input_dim=current_output_dim,
                n_head=self.config.transformer_attn_n_head, # <--- Use config
                num_layers=self.config.transformer_attn_num_layers, # <--- Use config
                dropout=self.config.classifier_dropout, # <--- Use config
            )

        # CRF layer
        self.crf = None
        # if self.config.use_crf: # Assume use_crf is present in config (if needed)
        #     self.crf = CRF(num_tags=config.num_labels, batch_first=True)

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(current_output_dim, self.config.classifier_hidden_dim), # <--- Use config
            nn.LayerNorm(self.config.classifier_hidden_dim),
            nn.Dropout(self.config.classifier_dropout), # <--- Use config
            nn.Linear(self.config.classifier_hidden_dim, self.config.num_labels) # Use config.num_labels
        )

        # (Optional) better initialization
        # self._init_weights() # _init_weights should also use self.config internally

    def _init_weights(self):
        # Initialize newly added Linear and LSTM layers
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
        # Remove unnecessary params
        _ = kwargs.pop("num_items_in_batch", None)

        # BERT encoding
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            # output_hidden_states=True, # Ensure hidden states are output if needed
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
            # LSTM can use true sequence lengths for PackedSequence optimization (optional but recommended)
            # lengths = attention_mask.sum(dim=1).cpu() # Need to compute lengths on CPU
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


# ====================
# Adversarial training module
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
        loss = super().training_step(model, inputs, *args)  # Pass through all arguments
        return loss
    
    
    def create_optimizer(self):
        # return super().create_optimizer()
        named_params = list(self.model.named_parameters())
        no_optim_params = ["bert.pooler.dense.weight", "bert.pooler.dense.bias"]
        for name, param in self.model.named_parameters():
            if name in no_optim_params:
                if param.requires_grad:
                    param.requires_grad = False
        # ---- Define parameter groups ----
        optimizer_grouped_parameters = [
            # BERT params (with weight decay)
            {
                "params": [
                    p for n, p in named_params 
                    if "bert" in n 
                    and not any(nd in n for nd in ["bias", "LayerNorm.weight"]) and n not in no_optim_params
                ],
                "weight_decay": self.args.weight_decay,
                "lr": self.args.learning_rate * 1
            },
            # BERT params (no weight decay)
            {
                "params": [
                    p for n, p in named_params 
                    if "bert" in n 
                    and any(nd in n for nd in ["bias", "LayerNorm.weight"]) and n not in no_optim_params
                ],
                "weight_decay": 0.0,
                "lr": self.args.learning_rate * 1
            },
            # New layers params (with weight decay)
            {
                "params": [
                    p for n, p in named_params 
                    if "bert" not in n 
                    and not any(nd in n for nd in ["bias", "LayerNorm.weight"]) and n not in no_optim_params
                ],
                "weight_decay": self.args.weight_decay,
                "lr": self.args.learning_rate
            },
            # New layers params (no weight decay)
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
        
        # ---- Key fix: filter out empty parameter groups ----
        # Remove any empty parameter groups to avoid optimizer initialization failure
        optimizer_grouped_parameters = [
            group for group in optimizer_grouped_parameters 
            if len(group["params"]) > 0
        ]
        if not optimizer_grouped_parameters:
            raise ValueError(
                "No parameters to optimize. "
                "Please check parameter grouping conditions or model parameter naming."
                f"Current model parameter list: {list(self.model.named_parameters())}"
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
# Training workflow main function
# ====================
# Initialize components
model_checkpoint = "dbmdz/bert-large-cased-finetuned-conll03-english"
# model_checkpoint = "bert-base-cased"
# model_checkpoint = "bert-large-cased"
model_checkpoint = "bert-large-cased-whole-word-masking"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

# Data preprocessing
dataset = Dataset.from_pandas(df)

split_dataset = dataset.train_test_split(test_size=0.1, shuffle=True, seed=42)

# Feature engineering processing
tokenized_train = split_dataset["train"].map(
    FeatureEngineer.tokenize_with_features,
    batched=True,
    batch_size=1000,
    # remove_columns=df.columns.tolist()  # Automatic removal of original columns
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
# Remove unused columns
columns_to_remove = ["id", "Sentence", "NER Tag", "token_type_ids"]
# columns_to_remove = ["id", "Sentence"]  # Modified: keep "NER Tag" for downstream processing
tokenized_train = tokenized_train.remove_columns([col for col in columns_to_remove if col not in {'input_ids', 'attention_mask', 'labels', 'word_features'}])
tokenized_eval = tokenized_eval.remove_columns([col for col in columns_to_remove if col not in {'input_ids', 'attention_mask', 'labels', 'word_features'}])

# # Explicitly set kept fields (key fix)
# columns_to_keep = ["input_ids", "token_type_ids", "attention_mask", "labels", "word_features"]
# tokenized_train.set_format(type="torch", columns=columns_to_keep)
# tokenized_eval.set_format(type="torch", columns=columns_to_keep)


# Modified: check after preprocessing
print("Sample keys after preprocessing:", tokenized_train[0].keys())
# Should print: dict_keys(['input_ids', 'token_type_ids', 'attention_mask', 'labels', ...])

# Modified: check that labels are valid
sample_labels = tokenized_train[0]["labels"]
print("Label example:", sample_labels)
print("Number of labels not equal to -100:", sum(1 for x in sample_labels if x != -100))

# Custom data collator
class CustomDataCollator(DataCollatorForTokenClassification):
    def __call__(self, features):
        batch = super().__call__(features)
        # Only process word_features when features are enabled
        if exp_config.use_char_feature or exp_config.use_pos_tag or exp_config.use_capital_feature:
            # Check that all samples have word_features
            if "word_features" not in features[0]:
                raise KeyError(
                    "Some samples are missing 'word_features'; check the feature generation logic."
                    "Ensure the field is generated when using char/POS/capital features."
                )
            # Convert to tensor and set datatype
            batch["word_features"] = torch.tensor(
                [f["word_features"] for f in features],
                dtype=torch.float32
            )
        return batch


data_collator = CustomDataCollator(tokenizer)

# Model initialization
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


# Training arguments
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
    save_strategy="best", # <--- Recommend saving best model each epoch to load the best
    load_best_model_at_end=True, # <--- Load best model at the end of training
    metric_for_best_model="eval_f1", # <--- Select best model based on F1
    greater_is_better=True,        # <--- Higher F1 means better
    save_only_model=True,
    # save_safetensors=False,
)

# Evaluation metrics
metric = evaluate.load("seqeval")

def compute_metrics_new(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    # --- Modified logic ---
    # Convert IDs to label strings, only filter out -100 labels
    true_predictions = [
        [id2label[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [id2label[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    # --- End modifications ---

    # Add a robustness check: if filtering leaves all sequences empty, return defaults
    # This should not occur unless the entire evaluation set is empty or consists only of padding
    if not any(true_labels): # Check if true_labels is empty or contains only empty lists
         print("Warning: No valid labels found after filtering -100 in compute_metrics.")
         return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "accuracy": 0.0}

    results = metric.compute(predictions=true_predictions, references=true_labels) # Use recommended mode and scheme
                                                                                                 # (if your labels are in IOB2 format)
                                                                                                 # If using another format (e.g., BILOU) adjust scheme accordingly
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
    # predictions are the model's logits [batch_size, seq_len, num_labels]
    # labels are the true label IDs [batch_size, seq_len] (-100 indicates padding)

    # Convert logits to predicted label IDs
    predictions = np.argmax(predictions, axis=2) # [batch_size, seq_len]

    # --- 1. Prepare for seqeval (Standard NER Metrics) ---
    # seqeval expects a list of lists, where each inner list contains label strings for a sequence
    # seqeval handles 'O' and entity labels automatically; we only need to filter padding tokens (-100)
    true_predictions_seqeval = [
        [id2label[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels_seqeval = [
        [id2label[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    # --- 2. Manual Calculation for Token Accuracy on Non-'O' True Labels ---
    # This is a custom metric: of tokens with true labels not 'O', how many are predicted correctly
    total_entity_tokens = 0 # Number of tokens whose true label is not 'O'
    correct_entity_token_predictions = 0 # Number of tokens (true label != 'O') correctly predicted

    # Iterate over each sequence
    for prediction_ids, label_ids in zip(predictions, labels):
        # Iterate over tokens in the sequence
        for pred_id, true_id in zip(prediction_ids, label_ids):
            # Ignore padding tokens
            if true_id == -100:
                continue

            true_label_str = id2label[true_id]

            # If the true label is not 'O'
            if true_label_str != 'O':
                total_entity_tokens += 1
                pred_label_str = id2label[pred_id]

                # If predicted label matches the true label
                if pred_label_str == true_label_str:
                    correct_entity_token_predictions += 1

    # Compute the metric
    # Avoid division by zero: if the evaluation set contains no non-'O' labels, set accuracy to 0
    entity_token_accuracy = correct_entity_token_predictions / total_entity_tokens if total_entity_tokens > 0 else 0.0


    # --- 3. Calculate seqeval metrics ---
    # Robustness: if no labels remain after filtering -100, return default metrics 0
    if not any(true_labels_seqeval):
         print("Warning: No valid labels found after filtering -100 in compute_metrics (for seqeval). Returning default metrics.")
         return {
            "overall_precision": 0.0,
            "overall_recall": 0.0,
            "overall_f1": 0.0,
            "overall_accuracy": 0.0, # seqeval token accuracy (includes O)
            "entity_token_accuracy": entity_token_accuracy # Our custom metric
         }

    # Compute standard NER metrics using seqeval (Chunk-based F1, Precision, Recall)
    # 'mode="strict"' is the standard evaluation mode, which requires exact match on entity boundaries and types
    # 'scheme="IOB2"' should match your label format (B-*, I-*, O)
    results = metric.compute(
        predictions=true_predictions_seqeval,
        references=true_labels_seqeval,
        mode='strict',
        scheme='IOB2' # Ensure this scheme matches your NER label format
    )

    # Return all computed metrics
    return {
        "overall_precision": results["overall_precision"], # seqeval Chunk Precision
        "overall_recall": results["overall_recall"],     # seqeval Chunk Recall
        "overall_f1": results["overall_f1"],             # seqeval Chunk F1 (typically the main metric)
        "overall_accuracy": results["overall_accuracy"], # seqeval Token Accuracy (includes O)
        "entity_token_accuracy": entity_token_accuracy   # Custom Token Accuracy (only on non-'O' true labels)
    }


# Start training
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

print('Start Training')
trainer.train()

