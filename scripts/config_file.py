import json

config = {
    "audio": {
        "path": "/lustre/fsmisc/dataset/HuggingFace_Models/openai/whisper-medium"
    },
    "projector": {
        "path": None,
        "rmsnorm_pre": True, #False to discard it
        "conv_kernel": 15,
        "conv_stride": 15,
        "n_transformer_layers": 2, #0 to discard it
        "n_heads": 8,
        "ff_mult": 4,
        "ff_dropout": 0.0,
        "positional_encoding": True, #False to discard it
        "act": "silu",       #None to discard it
        "rmsnorm_pos": True, #False to discard it
        "scale": 0,          #0 to discard it
        "use_bias": False,   #False to discard it
    },
    "llm": {
        "path": "/lustre/fsmisc/dataset/HuggingFace_Models/utter-project/EuroLLM-1.7B-Instruct",
        "audio_token": "<extra_id_0>", #must exist (unused) in vocab
    },
    "embeddings": {
        "path": None,
        "special_tokens": [  #will be added in tokenizer, learned embeddigns
            "<|asr|>", 
            "<|ast|>", 
            "<|stt|>", 
            "<|stt-asr|>", 
            "<|en|>", 
            "<|fr|>", 
            "<|de|>", 
            "<|ca|>", 
            "<|it|>", 
            "<|es|>", 
            "<|pt|>", 
            "<|nl|>", 
            "<|ru|>", 
            "<|ja|>", 
            "<|ko|>", 
            "<|ar|>", 
            "<|zh-CN|>", 
        ], 
    },
    "lora": {
        "path": None,
        "r": 16,
        "lora_alpha": 32,
        "target_modules":  ["q_proj", "k_proj", "v_proj", "o_proj"],
        "lora_dropout": 0.05,
        "bias": "none",
        "task_type": "CAUSAL_LM",
    },
    "optim": {
        'best_metric': "wer",
        'best_score': "min",
        'scores':[
            #'checkpoint': score,
        ]
    },
}

with open(f"config.json", "w", encoding="utf-8") as file:
    json.dump(config, file, indent=4)
