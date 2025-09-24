# models.py
import datetime as dt
import random
# from pathlib import Path
from collections import defaultdict

import torch
import torch.nn as nn
# import torch.utils.tensorboard as tb
from torch import optim
from sentiment_data import *

# import logging

# logging.basicConfig(filename='ai388-hw2.log', level=logging.DEBUG,
#                    format='%(asctime)s [%(filename)s:%(lineno)s - %(funcName)10s() ] %(message)s',
#                    datefmt='%Y-%m-%d %H:%M:%S')
#log = logging.getLogger(__name__)
#log.setLevel(logging.DEBUG)

device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.mps.is_available() else 'cpu'

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


class PrefixEmbeddings:
    """
    Manages embeddings for 3-character prefixes, initialized by averaging
    the GloVe vectors of words sharing the same prefix.
    """
    def __init__(self, word_embeddings: WordEmbeddings):
        self.word_indexer = Indexer()
        self.vectors = []
        prefix_to_vectors = defaultdict(list)
        for i in range(len(word_embeddings.word_indexer)):
            word = word_embeddings.word_indexer.get_object(i)
            if len(word) >= 3:
                prefix = word[:3]
                prefix_to_vectors[prefix].append(word_embeddings.vectors[i])

        embedding_dim = word_embeddings.get_embedding_length()
        self.word_indexer.add_and_get_index("PAD")
        self.vectors.append(np.zeros(embedding_dim))
        self.word_indexer.add_and_get_index("UNK")
        self.vectors.append(np.zeros(embedding_dim))

        for prefix, vectors_list in prefix_to_vectors.items():
            avg_vector = np.mean(vectors_list, axis=0)
            self.word_indexer.add_and_get_index(prefix)
            self.vectors.append(avg_vector)

        self.vectors = np.array(self.vectors)
        print(f"Created {len(self.vectors)} prefix embeddings.")

    def get_embedding_length(self):
        return len(self.vectors[0])

    def get_initialized_embedding_layer(self, frozen=False, padding_idx=0):
        return nn.Embedding.from_pretrained(
            torch.FloatTensor(self.vectors),
            freeze=frozen,
            padding_idx=padding_idx
        )

class CharLSTM(torch.nn.Module):
    def __init__(self, vocab_sz, embed_dim, hidden_dim: int) -> None:
        super(CharLSTM, self).__init__()
        self.char_emb = nn.Embedding(vocab_sz, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Added for perf
        # x = x.to(device)
        x = self.char_emb(x)
        _, (x, _) = self.lstm(x.unsqueeze(0))
        return x.squeeze(0)

class DeepAveragingNetwork(torch.nn.Module):
    def __init__(self, glove_embedding_layer: nn.Embedding, in_features, hidden_features, out_features: int) -> None:
        super(DeepAveragingNetwork, self).__init__()
        self.in_features = in_features
        self.hidden_features = hidden_features
        self.out_features = out_features
        self.glove_embedding_layer = glove_embedding_layer
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.dan = nn.Sequential(
            self.fc1,
            nn.ReLU(),
            nn.Dropout(p=0.4),
            self.fc2)
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        self.loss = nn.CrossEntropyLoss()

    def forward(self, sentence_indices, sentence_lengths: torch.Tensor) -> torch.Tensor:
        emb = self.glove_embedding_layer(sentence_indices)
        # avg = torch.mean(emb, dim=1)
        avg = torch.sum(emb, dim=1) / sentence_lengths.unsqueeze(1).float()
        logits = self.dan(avg)
        return logits


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
            # nn.ReLU(),
            nn.Sigmoid(),
            # nn.Dropout(p=0.1, inplace=False),
            self.fc2
        )# .to(device)
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        self.loss = nn.CrossEntropyLoss()

    def forward(self, word_indices, char_indices_per_word: torch.Tensor) -> torch.Tensor :
        hybrid_vectors = []
        # Added for perf
        word_indices = word_indices# .to(device)
        char_indices_per_word = [x for x in char_indices_per_word]

        for i in range(len(word_indices)):
            word_idx = word_indices[i].to(device)
            char_indices = char_indices_per_word[i].to(device)
            glove_vec = self.glove_embedding_layer(word_idx).to(device)

            if len(char_indices) < 2:
                # Create a zero tensor of the correct size for the char vector
                char_vec = torch.zeros(self.char_lstm.lstm.hidden_size, device=device)
            else:
                # Only run the CharLSTM for words with 2 or more characters
                char_vec = self.char_lstm(char_indices).to(device)
            #char_vec = self.char_lstm(char_indices).to(device)
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
    # def __init__(self, model: DeepAveragingLSTMNetwork, word_embeddings: WordEmbeddings, train_exs: List[SentimentExample], char_indexer: Indexer):
    def __init__(self, model: DeepAveragingNetwork, word_embeddings: WordEmbeddings, train_model_for_typo_setting: bool):
        self.model = model
        self.model = self.model.to(device)
        self.model.eval()
        # self.word_indexer = word_embeddings.word_indexer
        #self.char_indexer = char_indexer
        #self.train_exs = train_exs
        self.embeddings = word_embeddings
        self.word_indexer = word_embeddings.word_indexer
        self.train_model_for_typo_setting = train_model_for_typo_setting

    # def predict(self, ex_words: List[str], has_typos: bool) -> int:
    #     """
    #     """
    #     word_unk_idx = self.word_indexer.index_of('UNK')
    #     char_unk_idx = self.char_indexer.index_of('UNK')
    #     word_indices = [self.word_indexer.index_of(word) for word in ex_words]
    #     word_indices = [idx if idx != -1 else word_unk_idx for idx in word_indices]
    #     word_indices_tensor = torch.tensor(word_indices, dtype=torch.long, device=device)
    #
    #     char_indices_per_word = []
    #     for word in ex_words:
    #         char_indices = [self.char_indexer.index_of(char) for char in word]
    #         char_indices = [idx if idx != -1 else char_unk_idx for idx in char_indices]
    #         char_indices_per_word.append(torch.tensor(char_indices, dtype=torch.long, device=device))
    #
    #     with torch.no_grad():
    #         logits = self.model(word_indices_tensor, char_indices_per_word)
    #         prediction = torch.argmax(logits, dim=1).item()
    #     return prediction

    def predict(self, ex_words: List[str], has_typos: bool) -> int:
        word_unk_idx = self.word_indexer.index_of('UNK')
        if self.train_model_for_typo_setting:
            tokens_to_lookup = [word[:3] if len(word) >= 3 else "UNK" for word in ex_words]
        else:
            tokens_to_lookup = ex_words
        word_indices = [self.word_indexer.index_of(word) for word in tokens_to_lookup]
        word_indices = [idx if idx != -1 else word_unk_idx for idx in word_indices]
        word_indices_tensor = torch.tensor([word_indices], dtype=torch.long, device=device)
        lengths_tensor = torch.tensor([len(word_indices)], dtype=torch.long, device=device)
        with torch.no_grad():
            logits = self.model(word_indices_tensor, lengths_tensor)
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
    # exp_dir = "logs"
    # model_name = 'DAN2'
    # log_dir = Path(exp_dir) / f'{model_name}_{dt.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}'
    # logger = tb.SummaryWriter(str(log_dir))
    num_epochs = args.num_epochs if args.num_epochs > 1 else 2
    num_epochs = 2
    lr = args.lr if args.lr > 0 else 0.001
    batch_size = args.batch_size if args.batch_size > 1 else 96
    train_model_for_typo_setting = args.use_typo_setting
    # Choose the embedding strategy based on the typo setting
    if train_model_for_typo_setting:
        print("Training with prefix embeddings.")
        embeddings_for_model = PrefixEmbeddings(word_embeddings)
        emb_layer = embeddings_for_model.get_initialized_embedding_layer(frozen=False, padding_idx=0).to(device)
    else:
        print("Training with standard WordEmbeddings.")
        embeddings_for_model = word_embeddings
        emb_layer = embeddings_for_model.get_initialized_embedding_layer(frozen=False, padding_idx=0).to(device)

    model = DeepAveragingNetwork(emb_layer, embeddings_for_model.get_embedding_length(), args.hidden_size, 2)
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()

    optimizer.zero_grad()

    # char_indexer = Indexer()
    # char_indexer.add_and_get_index("UNK")  # Add UNK token
    # char_indexer.add_and_get_index("PAD")  # Add PAD token for potential batching
    # for ex in train_exs:
    #     for word in ex.words:
    #         for char in word:
    #             char_indexer.add_and_get_index(char)
    # # Hyperparameters for the character model
    # char_embedding_dim = 50
    # lstm_hidden_dim = 100
    # input_size = word_embeddings.get_embedding_length()
    # glove_layer = word_embeddings.get_initialized_embedding_layer(frozen=False, padding_idx=0).to(device)#, padding_idx=0).to(device)
    # char_lstm = CharLSTM(len(char_indexer), char_embedding_dim, lstm_hidden_dim).to(device)
    # model = DeepAveragingLSTMNetwork(glove_layer, char_lstm, input_size, args.hidden_size, 2).to(device)
    # optimizer = optim.Adam(model.parameters(), lr=args.lr)# , weight_decay=1e-5)
    # loss_fn = nn.CrossEntropyLoss().to(device)
    #
    # optimizer.zero_grad()

    for i in range(num_epochs):
        random.shuffle(train_exs)
        model.train()
        total_loss = 0.0
        train_accs = []
        loss_values = []
        optimizer.zero_grad()

        for j, ex in enumerate(train_exs):
            # model.zero_grad()
            if train_model_for_typo_setting:
                tokens_to_lookup = [word[:3] if len(word) >= 3 else "UNK" for word in ex.words]
            else:
                tokens_to_lookup = ex.words
            word_unk_idx = embeddings_for_model.word_indexer.index_of("UNK")
            # char_unk_idx = char_indexer.index_of("UNK")

            word_indices = [embeddings_for_model.word_indexer.index_of(w) for w in tokens_to_lookup]
            word_indices = [idx if idx != -1 else word_unk_idx for idx in word_indices]
            # word_indices = [idx for idx in word_indices if idx != -1]
            word_indices_tensor = torch.tensor(word_indices, dtype=torch.long, device=device).unsqueeze(0)
            lengths_tensor = torch.tensor([len(word_indices)], dtype=torch.long, device=device)
            label = torch.tensor([ex.label], dtype=torch.long, device=device)
            logits = model(word_indices_tensor, lengths_tensor)
            loss = loss_fn(logits, label) / batch_size
            loss.backward()
            total_loss += loss.item() * batch_size
            train_accs.append(compute_accuracy(logits, label))
            loss_values.append(loss.item())
            if (j + 1) % batch_size == 0:
                optimizer.step()
                optimizer.zero_grad()

            # char_indices_per_word = []
            # for word in ex.words:
            #     char_indices = [char_indexer.index_of(c) for c in word]
            #     char_indices = [idx if idx != -1 else char_unk_idx for idx in char_indices]
            #     # char_indices = [idx for idx in char_indices if idx != -1]
            #     char_indices_per_word.append(torch.tensor(char_indices, dtype=torch.long))

        if (j + 1) % batch_size != 0:
            optimizer.step()
        # Dev eval
        model.eval()
        val_accs = []

        for dev_ex in dev_exs:
            word_unk_idx = embeddings_for_model.word_indexer.index_of("UNK")
            if train_model_for_typo_setting:
                tokens_to_lookup = [word[:3] if len(word) >= 3 else "UNK" for word in dev_ex.words]
            else:
                tokens_to_lookup = dev_ex.words
            #char_unk_idx = char_indexer.index_of("UNK")
            indices = [embeddings_for_model.word_indexer.index_of(word) for word in tokens_to_lookup]
            indices = [idx if idx != -1 else word_unk_idx for idx in indices]
            #indices = [idx for idx in indices if idx != -1]
            indices_tensor = torch.tensor([indices], dtype=torch.long, device=device) # .unsqueeze(0)
            lengths_tensor = torch.tensor([len(indices)], dtype=torch.long, device=device)
            #char_indices_per_word = []
            #for word in dev_ex.words:
            #    char_indices = [char_indexer.index_of(char) for char in word]
            #    char_indices = [idx if idx != -1 else char_unk_idx for idx in char_indices]
                #char_indices = [ idx for idx in char_indices if idx != -1]
            #    char_indices_per_word.append(torch.tensor(char_indices, dtype=torch.long, device=device))
            label = torch.tensor([dev_ex.label], dtype=torch.long, device=device)
            with torch.no_grad():
                logits = model(indices_tensor, lengths_tensor).to(device)
            val_accs.append(compute_accuracy(logits, label))

        epoch_train_acc = torch.mean(torch.stack(train_accs))
        epoch_val_acc = torch.mean(torch.stack(val_accs))
        avg_loss = total_loss / (len(train_exs) ) if len(train_exs) > 0 else 0
        print(f'Epoch {i + 1}/{num_epochs}: loss={avg_loss:.4f}, train_acc={epoch_train_acc:.4f}, dev_acc={epoch_val_acc:.4f}')
        # log.info(f'[+] Epoch {i + 1}/{num_epochs}: loss={avg_loss:.4f}, train_acc={epoch_train_acc:.4f}, dev_acc={epoch_val_acc:.4f}')
        # logger.add_scalar("loss", avg_loss, i)
        # logger.add_scalars("accuracy", {"train": epoch_train_acc, "dev": epoch_val_acc}, i)
        # logger.flush()
        # return
    # return NeuralSentimentClassifier(model, word_embeddings, train_exs, char_indexer)
    return NeuralSentimentClassifier(model, embeddings_for_model, train_model_for_typo_setting=train_model_for_typo_setting)
