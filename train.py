import os
import pickle

import datasets
import preprocessing
import models


MAX_LENGTH = 20


def create_entrypoint(dir='.entrypoint'):
    if not os.path.exists(dir):
        os.makedirs(dir)
    return dir


def load_token_vectors(processor):
    print('[preprocessor] loading index data...')
    dir = create_entrypoint()
    vectors_filename = os.path.join(dir, 'token_vectors.pkl')
    dictionary_filename = os.path.join(dir, 'dictionary.pkl')
    if os.path.isfile(vectors_filename) and os.path.isfile(dictionary_filename):
        with open(vectors_filename, 'rb') as f:
            token_vectors = pickle.load(f)
        with open(dictionary_filename, 'rb') as f:
            dictionary = pickle.load(f)
    else:
        token_vectors, dictionary = processor.generate_token_vectors(data, max_len=MAX_LENGTH)
        with open(vectors_filename, 'wb+') as f:
            pickle.dump(token_vectors, f)
        with open(dictionary_filename, 'wb+') as f:
            pickle.dump(dictionary, f)
    return token_vectors, dictionary


def _main():
    data = datasets.open_subtitles(download=False)
    processor = preprocessing.PreProcessing()
    # s = list(data.values())[0]

    token_vectors, dictionary = load_token_vectors(processor)

    print('[train] creating network...')
    input_dim = len(dictionary['words'])
    input_length = MAX_LENGTH
    output_length = MAX_LENGTH
    output_dim = input_dim
    n_hidden = 10
    depth = 4
    batch_size = 50
    nb_epoch = 10

    dialog = models.Dialog(input_dim=input_dim, input_length=input_length,
                           hidden_dim=n_hidden, output_length=output_length,
                           output_dim=output_dim, depth=depth)
    model = dialog.create_model()
    print('[train] validating...')
    for vectors in token_vectors:
        print('epoch')
        x_train, x_test, y_test, y_train = dialog.get_training_batch(vectors)
        dialog.train(x_train, y_train,
                     batch_size=batch_size, nb_epoch=nb_epoch,
                     validation_data=(x_test, y_test),
                     save_model=True)


if __name__ == '__main__':
    _main()
