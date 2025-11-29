from dataclasses import dataclass


@dataclass
class ExperimentConfig:
    # Feature engineering configuration
    use_char_feature: bool = True  # Keep original settings
    use_pos_tag: bool = True     # Keep original settings
    use_capital_feature: bool = True # Keep original settings

    # Model architecture configuration
    feature_proj_dim: int = 128         # <--- Increase feature projection dimension (was 64)
    use_bilstm: bool = True             # Keep original settings
    lstm_hidden_size: int = 768         # <--- Increased LSTM hidden size (was 512)
    lstm_num_layers: int = 2            # <--- Increased LSTM layers (was 1)
    use_transformer_attn: bool = True   # <--- Add switch to enable Transformer Attention
    transformer_attn_n_head: int = 8    # <--- Increase attention heads (was 4)
    transformer_attn_num_layers: int = 2 # <--- Increase attention layers (was 1)
    # TransformerEncoderLayer usually has a feedforward network whose dim is a multiple of d_model.
    # We can leave the default (normally 4 * d_model), or optionally add control over it.
    # transformer_attn_ff_dim: int = 2048 # <--- (optional) control the feedforward dim

    classifier_hidden_dim: int = 512    # <--- Increase classifier hidden dim (was 256)
    classifier_dropout: float = 0.3     # <--- (recommended) Increase dropout for classifier head

    use_crf: bool = False # Marked unused; keep disabled

    # Training strategy configuration (unchanged)
    use_fgm: bool = False
    use_swa: bool = False
    use_gradient_checkpointing: bool = False
    custom_lr_scheduler: bool = False

