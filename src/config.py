from dataclasses import dataclass, field

@dataclass
class HPARAMS:
    vocab_size = 12500
    max_seq_len = 32
    batch_size = 128

    model_hparams: dict = field(default_factory=lambda: {
    "d_model" : 512,
    "nhead" : 8,
    "num_encoder_layers" : 2,
    "num_decoder_layers" : 2,
    "dim_feedforward" : 2048,
    "dropout" : 0.1,
    "padding_idx" : 0,
    })

    optimizer_hparams: dict = field(default_factory=lambda: {
        "lr": 1e-3,
        "weight_decay": 2e-5
    })


    trainer_hparams: dict = field(default_factory=lambda: {
    "n_epochs": 20,
    "enable_mixed_precision": True,
    "restore_best_model" : False,
    "use_early_stopping" : True,
    "early_stopping_patience" : 3,
    "grad_clip_value" : None
    })
