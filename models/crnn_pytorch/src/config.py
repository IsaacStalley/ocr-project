
common_config = {
    'gnhk_dir': r'C:\Users\IsaacStalley\Documents\GitHub\GNHKdataset',
    'iam_dir': r'C:\Users\IsaacStalley\Documents\GitHub\IAM-dataset',
    'synth_dir': r'C:\Users\IsaacStalley\Documents\GitHub\synthtiger_v1.1',
    'img_width': 100,
    'img_height': 32,
    'map_to_seq_hidden': 64,
    'rnn_hidden': 256,
    'leaky_relu': False,
    'chars': '!"#$%&\'()*+,-./0123456789:;<=>?@[\]^_`abcdefghijklmnopqrstuvwxyz{|}~'
}

train_config = {
    'epochs': 10000,
    'train_batch_size': 32,
    'eval_batch_size': 512,
    'lr': 0.0005,
    'show_interval': 10,
    'valid_interval': 500,
    'save_interval': 2000,
    'cpu_workers': 4,
    'reload_checkpoint': 'models/trained/crnn/crnn_020000_loss1.9362866965646306.pt',
    'valid_max_iter': 100,
    'decode_method': 'greedy',
    'beam_size': 10,
    'checkpoints_dir': 'models/trained/crnn/'
}
train_config.update(common_config)

evaluate_config = {
    'eval_batch_size': 512,
    'cpu_workers': 4,
    'reload_checkpoint': '../models/trained/crnn_three/crnn_312000_loss3.0251498906743617.pt',
    'decode_method': 'greedy',
    'beam_size': 10,
}
evaluate_config.update(common_config)
