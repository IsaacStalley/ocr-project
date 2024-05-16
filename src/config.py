# Create an instance of the TextRecognitionSystem
ssd_config = {
    'label_path': '../models/trained/vgg_three/iam-labels.txt',
    'model_path': '../models/trained/vgg_three/vgg16-ssd-Epoch-90-Loss-2.0508658090260976.pth',
}

crnn_config = {
    'img_width': 100,
    'img_height': 32,
    'map_to_seq_hidden': 64,
    'rnn_hidden': 256,
    'leaky_relu': False,
    'model_path': '../models/trained/crnn_three/crnn_312000_loss3.0251498906743617.pt',
    'chars': '0123456789abcdefghijklmnopqrstuvwxyz!"#$%&()*+,-./:;<=>?@[\]\'^_`{|}~ ',
    'decode_method': 'greedy',
    'beam_size': 10,
}