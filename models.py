import numpy as np
import seq2seq
from seq2seq.models import AttentionSeq2Seq


class Dialog:
    def __init__(self, input_dim=5, input_length=7, hidden_dim=10,
                 output_length=8, output_dim=20, depth=4,
                 loss='mse', optimizer='rmsprop'):
        self.input_dim = input_dim
        self.input_length = input_length
        self.hidden_dim = hidden_dim
        self.output_length = output_length
        self.output_dim = output_dim
        self.depth = depth
        self.loss = loss
        self.optimizer = optimizer

    def create_model(self):
        self.model = AttentionSeq2Seq(self.output_dim, self.output_length,
                                      input_dim=self.input_dim,
                                      # input_length=self.input_length,
                                      # hidden_dim=self.input_length,
                                      depth=self.depth)
        self.model.compile(loss=self.loss, optimizer=self.optimizer)
        return self.model

    def get_training_batch(self, vectors, rate=0.8):
        mask = np.random.choice([True, False], size=vectors.shape[0], p=[rate, 1-rate])
        x_train, x_test = vectors[mask], vectors[mask==False]
        mask = np.roll(mask, 1)
        y_train, y_test = vectors[mask], vectors[mask==False]
        return x_train, x_test, y_train, y_test

    def save_model(self, mpath='./entrypoint/model.h5'):
        self.model.save_weight(mpath, override=True)

    def train(self, x_train, y_train,
              validation_data=None, batch_size=50,
              nb_epoch=10, save_model=True):
        options = dict(
            batch_size=batch_size,
            nb_epoch=nb_epoch,
            verbose=1,
        )
        if validation_data:
            options['validation_data'] = validation_data
        self.model.fit(x_train, y_train, **options)
        if save_model:
            self.save_model()
