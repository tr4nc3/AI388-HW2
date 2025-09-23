# models.py
# import datetime as dt
import random
# from pathlib import Path

import torch
import torch.nn as nn
# import torch.utils.tensorboard as tb
from torch import optim
from sentiment_data import *

#import logging

#logging.basicConfig(filename='ai388-hw2.log', level=logging.DEBUG,
#                    format='%(asctime)s [%(filename)s:%(lineno)s - %(funcName)10s() ] %(message)s',
#                    datefmt='%Y-%m-%d %H:%M:%S')
#log = logging.getLogger(__name__)
#log.setLevel(logging.DEBUG)

class SentimentClassifier(object):
    """
    Sentiment classifier base type
    """

    def predict(self, ex_words: List[str], has_typos: bool) -> int:
        """
        Makes a prediction on the given sentence
        :param ex_words: words to predict on
        :param has_typos: True if we are evaluating on data that potentially has typos, False otherwise. If you do
        spelling correction, this parameter allows you to only use your method for the appropriate dev eval in Q3
        and not otherwise
        :return: 0 or 1 with the label
        """
        raise Exception("Don't call me, call my subclasses")

    def predict_all(self, all_ex_words: List[List[str]], has_typos: bool) -> List[int]:
        """
        You can leave this method with its default implementation, or you can override it to a batched version of
        prediction if you'd like. Since testing only happens once, this is less critical to optimize than training
        for the purposes of this assignment.
        :param all_ex_words: A list of all exs to do prediction on
        :param has_typos: True if we are evaluating on data that potentially has typos, False otherwise.
        :return:
        """
        return [self.predict(ex_words, has_typos) for ex_words in all_ex_words]


class CharLSTM(torch.nn.Module):
    def __init__(self, vocab_sz, embed_dim, hidden_dim: int) -> None:
        super(CharLSTM, self).__init__()
        self.char_emb = nn.Embedding(vocab_sz, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.char_emb(x)
        _, (x, _) = self.lstm(x.unsqueeze(0))
        return x.squeeze(0)

class DeepAveragingLSTMNetwork(torch.nn.Module):
    def __init__(self, glove_embedding_layer: nn.Embedding, char_lstm :nn.Module, in_features, hidden_features, out_features: int) -> None:
        super(DeepAveragingLSTMNetwork, self).__init__()
        self.glove_embedding_layer = glove_embedding_layer
        self.char_lstm = char_lstm
        self.in_features = in_features
        self.out_features = out_features
        self.hidden_features = hidden_features
        combined_dim = self.glove_embedding_layer.embedding_dim + self.char_lstm.lstm.hidden_size
        self.fc1 = nn.Linear(combined_dim, hidden_features)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.dan2 = nn.Sequential(
            self.fc1,
            nn.ReLU(),
            nn.Dropout(p=0.1, inplace=False),
            self.fc2
        )
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        self.loss = nn.CrossEntropyLoss()

    def forward(self, word_indices, char_indices_per_word: torch.Tensor) -> torch.Tensor :
        hybrid_vectors = []
        for i in range(len(word_indices)):
            word_idx = word_indices[i]
            char_indices = char_indices_per_word[i]
            glove_vec = self.glove_embedding_layer(word_idx)
            char_vec = self.char_lstm(char_indices)
            hybrid_vec = torch.cat((glove_vec.squeeze(), char_vec.squeeze()), dim=0)
            hybrid_vectors.append(hybrid_vec)
        avg_vector = torch.mean(torch.stack(hybrid_vectors), dim=0)
        logits = self.dan2(avg_vector.unsqueeze(0))
        return logits


class TrivialSentimentClassifier(SentimentClassifier):
    def predict(self, ex_words: List[str], has_typos: bool) -> int:
        """
        :param ex:
        :return: 1, always predicts positive class
        """
        return 1

class NeuralSentimentClassifier(SentimentClassifier):
    def __init__(self, model: DeepAveragingLSTMNetwork, word_embeddings: WordEmbeddings, train_exs: List[SentimentExample], char_indexer: Indexer):
        self.model = model
        self.model.eval()
        self.word_indexer = word_embeddings.word_indexer
        self.char_indexer = char_indexer
        self.train_exs = train_exs

    def predict(self, ex_words: List[str], has_typos: bool) -> int:
        """
        """
        word_unk_idx = self.word_indexer.index_of('UNK')
        char_unk_idx = self.char_indexer.index_of('UNK')
        word_indices = [self.word_indexer.index_of(word) for word in ex_words]
        word_indices = [idx if idx != -1 else word_unk_idx for idx in word_indices]
        word_indices_tensor = torch.tensor(word_indices, dtype=torch.long)

        char_indices_per_word = []
        for word in ex_words:
            char_indices = [self.char_indexer.index_of(char) for char in word]
            char_indices = [idx if idx != -1 else char_unk_idx for idx in char_indices]
            char_indices_per_word.append(torch.tensor(char_indices, dtype=torch.long))

        with torch.no_grad():
            logits = self.model(word_indices_tensor, char_indices_per_word)
            prediction = torch.argmax(logits, dim=1).item()
        return prediction

    def predict_all(self, all_ex_words: List[List[str]], has_typos: bool) -> List[int]:
        return [self.predict(ex_words, has_typos) for ex_words in all_ex_words]

def compute_accuracy(outputs: torch.Tensor, labels: torch.Tensor):
    """
    Arguments:
        outputs: torch.Tensor, shape (b, num_classes) either logits or probabilities
        labels: torch.Tensor, shape (b,) with the ground truth class labels

    Returns:
        a single torch.Tensor scalar
    """
    outputs_idx = outputs.max(1)[1].type_as(labels)
    return (outputs_idx == labels).float().mean()

def train_deep_averaging_network(args, train_exs: List[SentimentExample], dev_exs: List[SentimentExample],
                                 word_embeddings: WordEmbeddings, train_model_for_typo_setting: bool) -> NeuralSentimentClassifier:
    """
    :param args: Command-line args so you can access them here
    :param train_exs: training examples
    :param dev_exs: development set, in case you wish to evaluate your model during training
    :param word_embeddings: set of loaded word embeddings
    :param train_model_for_typo_setting: True if we should train the model for the typo setting, False otherwise
    :return: A trained NeuralSentimentClassifier model. Note: you can create an additional subclass of SentimentClassifier
    and return an instance of that for the typo setting if you want; you're allowed to return two different model types
    for the two settings.
    """
    #exp_dir = "logs"
    #model_name = 'DAN2'
    #log_dir = Path(exp_dir) / f'{model_name}_{dt.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}'
    #logger = tb.SummaryWriter(str(log_dir))
    num_epochs = args.num_epochs
    lr = args.lr
    if lr is None:
        lr = 2e-3 # default learning rate
    batch_size = args.batch_size

    char_indexer = Indexer()
    char_indexer.add_and_get_index("UNK")  # Add UNK token
    char_indexer.add_and_get_index("PAD")  # Add PAD token for potential batching
    for ex in train_exs:
        for word in ex.words:
            for char in word:
                char_indexer.add_and_get_index(char)

    # Hyperparameters for the character model
    char_embedding_dim = 50
    lstm_hidden_dim = 100
    input_size = word_embeddings.get_embedding_length()
    glove_layer = word_embeddings.get_initialized_embedding_layer(frozen=True, padding_idx=0)
    char_lstm = CharLSTM(len(char_indexer), char_embedding_dim, lstm_hidden_dim)
    model = DeepAveragingLSTMNetwork(glove_layer, char_lstm, input_size, args.hidden_size, 2)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()

    optimizer.zero_grad()

    for i in range(num_epochs):
        random.shuffle(train_exs)
        model.train()
        total_loss = 0.0
        train_accs = []
        loss_values = []

        for j, ex in enumerate(train_exs):
            model.zero_grad()
            word_unk_idx = word_embeddings.word_indexer.index_of("UNK")
            char_unk_idx = char_indexer.index_of("UNK")

            word_indices = [word_embeddings.word_indexer.index_of(w) for w in ex.words]
            word_indices = [idx if idx != -1 else word_unk_idx for idx in word_indices]
            word_indices_tensor = torch.tensor(word_indices, dtype=torch.long)

            char_indices_per_word = []
            for word in ex.words:
                char_indices = [char_indexer.index_of(c) for c in word]
                char_indices = [idx if idx != -1 else char_unk_idx for idx in char_indices]
                char_indices_per_word.append(torch.tensor(char_indices, dtype=torch.long))

            label = torch.tensor([ex.label], dtype=torch.long)
            logits = model(word_indices_tensor, char_indices_per_word)
            loss = loss_fn(logits, label) / batch_size
            loss.backward()
            total_loss += loss.item() * batch_size
            train_accs.append(compute_accuracy(logits, label))
            loss_values.append(loss.item())
            if (j + 1) % batch_size == 0:
                optimizer.step()
                optimizer.zero_grad()

        if (j + 1) % batch_size != 0:
            optimizer.step()
        # Dev eval
        model.eval()
        val_accs = []

        for dev_ex in dev_exs:
            word_unk_idx = word_embeddings.word_indexer.index_of("UNK")
            char_unk_idx = char_indexer.index_of("UNK")
            indices = [word_embeddings.word_indexer.index_of(word) for word in dev_ex.words]
            indices = [idx if idx != -1 else word_unk_idx for idx in indices]
            indices_tensor = torch.tensor(indices, dtype=torch.long)
            char_indices_per_word = []
            for word in dev_ex.words:
                char_indices = [char_indexer.index_of(char) for char in word]
                char_indices = [idx if idx != -1 else char_unk_idx for idx in char_indices]
                char_indices_per_word.append(torch.tensor(char_indices, dtype=torch.long))
            label = torch.tensor([dev_ex.label], dtype=torch.long)
            with torch.no_grad():
                logits = model(indices_tensor, char_indices_per_word)
            val_accs.append(compute_accuracy(logits, label))

        epoch_train_acc = torch.mean(torch.stack(train_accs))
        epoch_val_acc = torch.mean(torch.stack(val_accs))
        avg_loss = total_loss / (len(train_exs) / batch_size) if len(train_exs) > 0 else 0
        print(f'Epoch {i + 1}/{num_epochs}: loss={avg_loss:.4f}, train_acc={epoch_train_acc:.4f}, dev_acc={epoch_val_acc:.4f}')
        #log.info(f'[+] Epoch {i + 1}/{num_epochs}: loss={avg_loss:.4f}, train_acc={epoch_train_acc:.4f}, dev_acc={epoch_val_acc:.4f}')
        #logger.add_scalar("loss", avg_loss, i)
        #logger.add_scalars("accuracy", {"train": epoch_train_acc, "dev": epoch_val_acc}, i)
        #logger.flush()
        # return
    return NeuralSentimentClassifier(model, word_embeddings, train_exs, char_indexer)
