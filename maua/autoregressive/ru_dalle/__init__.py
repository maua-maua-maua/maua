SURREALIST_XL_DICT = dict(
    hf_version="v3",
    description="Surrealist is 1.3 billion params model from the family GPT3-like, "
    "that was trained on surrealism and Russian.",
    model_params=dict(
        num_layers=24,
        hidden_size=2048,
        num_attention_heads=16,
        embedding_dropout_prob=0.1,
        output_dropout_prob=0.1,
        attention_dropout_prob=0.1,
        image_tokens_per_dim=32,
        text_seq_length=128,
        cogview_sandwich_layernorm=True,
        cogview_pb_relax=True,
        vocab_size=16384 + 128,
        image_vocab_size=8192,
    ),
    repo_id="shonenkov-AI/rudalle-xl-surrealist",
    filename="pytorch_model.bin",
    authors="shonenkovAI",
    full_description="",
)


from . import api, finetune, generate
