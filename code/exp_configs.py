from dataclasses import dataclass


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

