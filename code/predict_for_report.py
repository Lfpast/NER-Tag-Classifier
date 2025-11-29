import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import argparse
import json
import pandas as pd
from ast import literal_eval
from transformers import AutoTokenizer, DataCollatorForTokenClassification
from datasets import Dataset
import torch
import nltk
from safetensors import safe_open
from model_v3 import *
from exp_configs import ExperimentConfig
import csv
from tqdm import tqdm


def load_config_and_model(cfg_dir, base_model_checkpoint):
    safetensor_path = os.path.join(cfg_dir, "model.safetensors")
    config_path = os.path.join(cfg_dir, "config.json")

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"config.json not found in checkpoint dir: {cfg_dir}")
    with open(config_path, "r") as f:
        cfg = json.load(f)

    id2label = cfg["id2label"]
    label2id = cfg["label2id"]

    exp_config = ExperimentConfig(
        use_char_feature=cfg.get('use_char_feature', True),
        use_pos_tag=cfg.get('use_pos_tag', True),
        use_capital_feature=cfg.get('use_capital_feature', True),
        use_bilstm=cfg.get('use_bilstm', True),
        lstm_hidden_size=cfg.get('lstm_hidden_size', 512),
        lstm_num_layers=cfg.get('lstm_num_layers', 1),
        use_transformer_attn=cfg.get('use_transformer_attn', True),
        transformer_attn_n_head=cfg.get('transformer_attn_n_head', 4),
        transformer_attn_num_layers=cfg.get('transformer_attn_num_layers', 1),
        classifier_hidden_dim=cfg.get('classifier_hidden_dim', 256),
        classifier_dropout=cfg.get('classifier_dropout', 0.2),
        feature_proj_dim=cfg.get('feature_proj_dim', 64),
    )

    conf = EnhancedConfig(
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

    tokenizer = AutoTokenizer.from_pretrained(base_model_checkpoint)
    data_collator = DataCollatorForTokenClassification(tokenizer)

    base_model = BertModel.from_pretrained(
        base_model_checkpoint,
        num_labels=len(label2id),
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True
    )

    model = EnhancedModel(base_model, conf)

    if not os.path.exists(safetensor_path):
        raise FileNotFoundError(f"model.safetensors not found in checkpoint dir: {cfg_dir}")
    with safe_open(safetensor_path, framework="pt") as f:
        state_dict = {k: f.get_tensor(k) for k in f.keys()}

    model.load_state_dict(state_dict)
    return model, tokenizer, id2label, label2id, exp_config

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
    def tokenize_with_features(cls, examples, tokenizer=None, exp_config=None, pos_tag2id=None, label2id=None):
        # If not passed in, fall back to module-level globals (backwards compatible)
        tokenizer = tokenizer or globals().get("tokenizer")
        exp_config = exp_config or globals().get("exp_config")
        pos_tag2id = pos_tag2id or globals().get("pos_tag2id")
        label2id = label2id or globals().get("label2id")

        tokenized = tokenizer(
            examples["Sentence"],
            truncation=True,
            padding='max_length',
            max_length=64,
            is_split_into_words=True,
            return_overflowing_tokens=False
        )
        # Generate and store word_ids
        tokenized["word_ids"] = []
        for i in range(len(examples["Sentence"])):
            word_ids = tokenized.word_ids(batch_index=i)
            tokenized["word_ids"].append(word_ids)
        # Label alignment
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
                        # pos_tags elements are (word, tag) tuples
                        pos_tag = pos_tags[word_idx][1] if pos_tags else None
                        features.append(float(pos_tag2id.get(pos_tag, 0) if pos_tag2id is not None else 0))
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









def run_predict(model_dir, test_csv, base_model_checkpoint, output_csv):
    """Run prediction and save a submission CSV.

    Args:
        model_dir: path to checkpoint folder (contains model.safetensors and config.json)
        test_csv: path to test.csv
        base_model_checkpoint: base Hugging Face model checkpoint
        output_csv: output csv path
    """
    # 1. Read data
    df = pd.read_csv(test_csv, converters={"Sentence": literal_eval})

    # 2. Compute POS mapping (POS tags expected by FeatureEngineer)
    # Ensure NLTK pos tagger data is available
    try:
        nltk.data.find('taggers/averaged_perceptron_tagger')
    except LookupError:
        nltk.download('averaged_perceptron_tagger')

    all_pos_tags = set()
    for sentence in df["Sentence"]:
        pos_tags = nltk.pos_tag(sentence)
        all_pos_tags.update(tag for _, tag in pos_tags)
    pos_tag2id = {tag: idx for idx, tag in enumerate(sorted(all_pos_tags))}

    dataset = Dataset.from_pandas(df)

    # 3. Load model + tokenizer + configs
    model, tokenizer, id2label, label2id, exp_cfg = load_config_and_model(model_dir, base_model_checkpoint)

    # 4. Make global objects for FeatureEngineer to reference
    globals()["tokenizer"] = tokenizer
    globals()["exp_config"] = exp_cfg
    globals()["pos_tag2id"] = pos_tag2id
    globals()["label2id"] = label2id
    globals()["id2label"] = id2label

    tokenized_test = dataset.map(
        FeatureEngineer.tokenize_with_features,
        batched=True,
        batch_size=1000,
        remove_columns=["id", "Sentence"],
        fn_kwargs={
            'tokenizer': tokenizer,
            'exp_config': exp_cfg,
            'pos_tag2id': pos_tag2id,
            'label2id': label2id
        }
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    # 5. Predict
    def predict_fn(model, tokenized_dataset):
        all_preds = []
        for data in tqdm(tokenized_dataset, desc="predicting", total=len(dataset)):
            data = {k: torch.tensor(v).unsqueeze(0).to(device) for k, v in data.items() if k not in ["token_type_ids", "word_ids"]}
            with torch.no_grad():
                outputs = model.forward(**data)
                logits = outputs['logits']
                predictions = torch.argmax(logits, dim=-1)
                predictions = predictions.cpu().numpy()
                all_preds.append(predictions)
        return all_preds

    predictions = predict_fn(model, tokenized_test)

    # 6. Align predictions back to word-level labels
    final_labels = []
    for i in range(len(dataset)):
        word_ids = tokenized_test[i]["word_ids"]
        sentence = dataset[i]["Sentence"]
        preds = predictions[i]

        aligned = ["O"] * len(sentence)
        processed_word_ids = set()

        for token_idx, word_id in enumerate(word_ids):
            if word_id is None or word_id >= len(sentence):
                continue
            if word_id not in processed_word_ids:
                # predictions shape: (1, seq_len) due to unsqueeze during batching
                try:
                    aligned[word_id] = id2label[str(preds[0][token_idx])]
                except Exception:
                    aligned[word_id] = "O"
                processed_word_ids.add(word_id)
        final_labels.append(aligned)

    output_df = pd.DataFrame({"id": df["id"], "NER Tag": final_labels})
    output_df["NER Tag"] = output_df["NER Tag"].apply(lambda x: str(x).replace("', '", "','"))
    output_df.to_csv(output_csv, index=False, quoting=csv.QUOTE_NONNUMERIC)

def parse_args():
    parser = argparse.ArgumentParser(description="Run prediction and produce a submission csv")
    parser.add_argument("--model-dir", type=str, default=os.path.join("..", "checkpoint-2414"), help="Path to checkpoint folder (contains model.safetensors and config.json)")
    parser.add_argument("--test-csv", type=str, default=os.path.join("..", "data", "test.csv"), help="Path to test.csv file")
    parser.add_argument("--base-model", type=str, default="bert-large-cased-whole-word-masking", help="Base model checkpoint (huggingface) for the encoder")
    parser.add_argument("--output-csv", type=str, default=os.path.join("..", "submission.csv"), help="Output submission file path")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_predict(args.model_dir, args.test_csv, args.base_model, args.output_csv)