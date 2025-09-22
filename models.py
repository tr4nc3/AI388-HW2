# models.py
#import datetime as dt
import random
#from pathlib import Path

import torch
import torch.nn as nn
# import torch.utils.tensorboard as tb
from torch import optim

from nltk.metrics.distance import edit_distance
from collections import defaultdict

from sentiment_data import *

# import logging

# logging.basicConfig(filename='ai388-hw2.log', level=logging.DEBUG,
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

class DeepAveragingNetwork(torch.nn.Module):
    def __init__(self, embedding_layer: nn.Embedding, in_features, hidden_features, out_features: int) -> None:
        super(DeepAveragingNetwork, self).__init__()
        self.embedding_layer = embedding_layer
        self.in_features = in_features
        self.out_features = out_features
        self.hidden_features = hidden_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.fc2 = nn.Linear(hidden_features, out_features)
        # from hw example code - ffnn_example.py
        # Initialize weights according to a formula due to Xavier Glorot.
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        self.dan = nn.Sequential(
            self.fc1,
            # nn.BatchNorm1d(hidden_features), # TODO: Remove batchnorm
            nn.ReLU(),
            nn.Dropout(p=0.1, inplace=False),
            self.fc2
        )
        self.loss = nn.CrossEntropyLoss()
        # TODO: Try dropout
    def forward(self, sentence_indices, sentence_lengths) -> torch.Tensor :
        # log.info(f'[+] {sentence_indices.shape=} {sentence_lengths.shape=}')
        embeddings = self.embedding_layer(sentence_indices)
        avg = torch.sum(embeddings, dim=1) / sentence_lengths.unsqueeze(1).float() # TODO: User torch.mean()
        logits = self.dan(avg)
        # x = torch.nn.functional.log_softmax(logits, dim=1) # activation sigmoid
        return logits

class TrivialSentimentClassifier(SentimentClassifier):
    def predict(self, ex_words: List[str], has_typos: bool) -> int:
        """
        :param ex:
        :return: 1, always predicts positive class
        """
        return 1


class NeuralSentimentClassifier(SentimentClassifier):
    """
    Implement your NeuralSentimentClassifier here. This should wrap an instance of the network with learned weights
    along with everything needed to run it on new data (word embeddings, etc.). You will need to implement the predict
    method and you can optionally override predict_all if you want to use batching at inference time (not necessary,
    but may make things faster!)
    """
    def __init__(self, model: DeepAveragingNetwork, word_embeddings: WordEmbeddings) -> None:
        self.model = model
        self.model.eval()
        self.word_embeddings = word_embeddings
        # typo corrections
        self.prefix_cache = defaultdict(list)
        for i in range(len(self.word_embeddings.word_indexer)):
            word = self.word_embeddings.word_indexer.get_object(i)
            if len(word) >= 3:
                prefix = word[:3]
                self.prefix_cache[prefix].append(word)

    def predict(self, ex_words: List[str], has_typos :bool = False) -> int:
        """
        """
        indices = []
        unk_idx = self.word_embeddings.word_indexer.index_of("UNK")
        if has_typos:
            word_indices = []
            for word in ex_words:
                idx = self.word_embeddings.word_indexer.index_of(word)
                # If the word is unknown AND we are in typo mode, try to correct it.
                if idx == -1 and len(word) >= 3:
                    prefix = word[:3]
                    candidate_words = self.prefix_cache.get(prefix)
                    if candidate_words:
                        best_match = min(candidate_words, key=lambda c: edit_distance(word, c))
                        idx = self.word_embeddings.word_indexer.index_of(best_match)
                    else:  # If no candidates, fall back to UNK
                        idx = unk_idx
                elif idx == -1: # If not in typo mode or word is too short, just use UNK
                    idx = unk_idx
                word_indices.append(idx)
                lengths = torch.tensor([len(word_indices)], dtype=torch.long)
                with torch.no_grad():
                    indices_tensor = torch.tensor([word_indices], dtype=torch.long)
                    log_probs = self.model(indices_tensor, lengths)
        else:

            indices = [self.word_embeddings.word_indexer.index_of(word) for word in ex_words]
            indices = [idx if idx != -1 else unk_idx for idx in indices]
            lengths = torch.tensor([len(ex_words)], dtype=torch.long)
            with torch.no_grad():
                indices_tensor = torch.tensor([indices], dtype=torch.long)
                log_probs = self.model(indices_tensor, lengths)
            # DEBUG
            # log.info(f'[+] {log_probs=}')
        prediction = torch.argmax(log_probs, dim=1).item()
        return prediction

    def predict_all(self, all_ex_words: List[List[str]], has_typos: bool = False):
        batch_size = 64
        predictions = []

        for i in range(0, len(all_ex_words), batch_size):
            batch = all_ex_words[i:i+batch_size]
            batch_lengths = torch.tensor([len(ex_words) for ex_words in batch], dtype=torch.long)
            max_len = max(len(ex) for ex in batch)
            pad_idx = self.word_embeddings.word_indexer.index_of("PAD")
            unk_idx = self.word_embeddings.word_indexer.index_of("UNK")
            batch_indices_lists = [[self.word_embeddings.word_indexer.index_of(w) for w in ex_words] for ex_words in batch]
            # Map OOVs (-1) to UNK
            batch_indices_lists = [[idx if idx != -1 else unk_idx for idx in indices] for indices in batch_indices_lists]
            padded_batch_indices = [indices + [pad_idx] * (max_len - len(indices)) for indices in batch_indices_lists]
            batch_tensor = torch.tensor(padded_batch_indices, dtype=torch.long)
            with torch.no_grad():
                log_probs = self.model(batch_tensor, batch_lengths)
            batch_predictions = torch.argmax(log_probs, dim=1).tolist()
            predictions.extend(batch_predictions)
        return predictions

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
    # raise NotImplementedError
    #
    exp_dir = "logs"
    model_name = 'DAN'
    # log_dir = Path(exp_dir) / f'{model_name}_{dt.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}'
    # logger = tb.SummaryWriter(str(log_dir))
    num_epochs = args.num_epochs
    lr = args.lr
    batch_size = args.batch_size
    emb_layer = word_embeddings.get_initialized_embedding_layer(frozen=False, padding_idx=0)
    input_size = word_embeddings.get_embedding_length()
    num_classes = 2
    hidden_features = args.hidden_size
    model = DeepAveragingNetwork(emb_layer, input_size, hidden_features, num_classes)
    model.embedding_layer = emb_layer
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    for i in range(num_epochs):
        random.shuffle(train_exs)
        model.train()
        total_loss = 0.0
        train_accs = []
        loss_values = []
        for j in range(0, len(train_exs), batch_size):
            model.zero_grad()
            batch = train_exs[j:j + batch_size]
            if not batch:
                continue
            labels = torch.tensor([ex.label for ex in batch])
            lengths = torch.tensor([len(ex.words) for ex in batch])
            unk_idx = word_embeddings.word_indexer.index_of("UNK")

            batch_indices_lists = [[word_embeddings.word_indexer.index_of(w) for w in ex.words] for ex in batch]
            batch_indices_lists = [[idx if idx != -1 else unk_idx for idx in indices] for indices in batch_indices_lists]

            max_len = max(len(indices) for indices in batch_indices_lists)
            pad_idx = word_embeddings.word_indexer.index_of("PAD")
            unk_idx = word_embeddings.word_indexer.index_of("UNK")
            padded_batch_indices = [indices + [pad_idx] * (max_len - len(indices)) for indices in batch_indices_lists]
            batch_tensor = torch.tensor(padded_batch_indices, dtype=torch.long)
            # TODO: print shapes, lengths, loss, etc.
            # log.info(f'[+] {batch_tensor.shape=} {lengths.shape=} {labels.shape=}')
            out = model(batch_tensor, lengths)
            loss = loss_fn(out, labels) # CE Loss expects logits not softmax
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            train_accs.append(compute_accuracy(out, labels))
        # Dev eval
        model.eval()
        val_accs = []
        unk_idx = word_embeddings.word_indexer.index_of("UNK")
        for dev_ex in dev_exs:
            indices = [word_embeddings.word_indexer.index_of(word) for word in dev_ex.words]
            indices = [idx if idx != -1 else unk_idx for idx in indices]
            length = torch.tensor([len(dev_ex.words)])
            indices_tensor = torch.tensor([indices], dtype=torch.long)
            label = torch.tensor([dev_ex.label])
            with torch.no_grad():
                output = model(indices_tensor, length)
            val_accs.append(compute_accuracy(output, label))

        epoch_train_acc = torch.mean(torch.stack(train_accs))
        epoch_val_acc = torch.mean(torch.stack(val_accs))
        avg_loss = total_loss / (len(train_exs) / batch_size) if len(train_exs) > 0 else 0
        print(f'Epoch {i + 1}/{num_epochs}: loss={avg_loss:.4f}, train_acc={epoch_train_acc:.4f}, dev_acc={epoch_val_acc:.4f}')
        #log.info(f'[+] Epoch {i + 1}/{num_epochs}: loss={avg_loss:.4f}, train_acc={epoch_train_acc:.4f}, dev_acc={epoch_val_acc:.4f}')
        #logger.add_scalar("loss", avg_loss, i)
        #logger.add_scalars("accuracy", {"train": epoch_train_acc, "dev": epoch_val_acc}, i)
        #logger.flush()
        # return
    return NeuralSentimentClassifier(model, word_embeddings)
