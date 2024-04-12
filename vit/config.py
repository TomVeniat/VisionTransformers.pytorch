
def get_config():
    config = dict(
        batch_size=4,
        num_epochs=20,
        lr=1e-4,
        img_size=28,
        in_channels=1,
        num_classes=10,
        patch_size=4,
        num_heads=8,
        dropout=0.001,
        hidden_dim=768,
        adam_w_decay=0.01,
        adam_beta1=0.9,
        adam_beta2=0.999,
        activation="gelu",
        encoder_layers=4,

        # seq_len=350,
        # d_model=512,
        # lang_src='en',
        # lang_tgt='it',
        # model_folder='weights',
        # model_basename='tmodel_',
        # preload=None,
        # tokenizer_file='tokenizer_{0}.json',
        # experiment_name='runs/tmodel'
    )
    config['embedding_dim'] = config['patch_size'] ** 2 * config['in_channels']
    return config