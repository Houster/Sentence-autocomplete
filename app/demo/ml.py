from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences
import json
import os
import pickle


class TFModel:
    def __init__(self, model_dir, sentiment):

        if sentiment:
            model_path = os.path.join(model_dir, 'gru_positive.h5')
            artifacts_path = os.path.join(model_dir, 'model_artifacts_positive.pkl')
        else:

            model_path = os.path.join(model_dir, 'gru_negative.h5')
            artifacts_path = os.path.join(model_dir, 'model_artifacts_negative.pkl')

        self.model = load_model(model_path)
        self.artifacts = pickle.load(open(artifacts_path, 'rb'))
        tokenizer_config = self.artifacts['tokenizer_config']
        self.tokenizer = tokenizer_from_json(json.dumps(tokenizer_config))
        self.sequence_len = self.artifacts['sequence_len']


    def predict(self, data: str, num_words=10):
        # generate num_words
        data_seqs = self.tokenizer.texts_to_sequences([data])
        for i in range(num_words):
            padded_seqs = pad_sequences(data_seqs, maxlen=self.sequence_len,
                                        padding='pre', truncating='pre')
            next_word = self.model.predict(padded_seqs).argmax(axis=1)
            data_seqs[0].append(next_word[0])

        return self.tokenizer.sequences_to_texts(data_seqs)


# tests
if __name__ == '__main__':
    model = TFModel(model_dir='model')
    print(model.tokenizer.texts_to_sequences(['this is a test']))
    print(model.predict('hello'))
