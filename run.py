import re
import random
import itertools as it
import time
import math
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pickle
from collections import namedtuple
from collections import Counter, OrderedDict, defaultdict

import utils
from deep_cbow import DeepCBOW
from lstm import LSTMClassifier
from tree_lstm import TreeLSTMClassifier

##############################
## USER CONFIG:
## set the correct directories
##############################

# dir to save model state dicts in
MODELS_DIR = Path("/Users/jeroen/code/uva/msc/nlp1/nlp1-ass2/models")
# dir to save results during training in
RESULTS_DIR = Path("/Users/jeroen/code/uva/msc/nlp1/nlp1-ass2/results")

possible_experiments = {
    "deep_cbow",
    "lstm",
    "lstm_shuffled",
    "lstm_random",
    "tree_lstm",
    "tree_lstm_shuffled",
    "tree_lstm_random",
    "tree_lstm_intermittent_supervision",
}

################################
## INDICATE YOUR EXPERIMENT HERE
## (choose from above)
################################

EXPERIMENT = "tree_lstm_intermittent_supervision"

assert (
    EXPERIMENT in possible_experiments
), "Please choose an experiment from the set of possible experiments"

# FROM HERE ON, YOU DON'T NEED TO CHANGE ANYTHING (feel free if you want to)

##############################
## HYPERPARAMETERS
##############################

# DEFAULT
INTERMITTENT_SUPERVISION = False
INTERMITTENT_SUPERVISION_SKIP_LEAF_NODES = False

BATCH_SIZE = 256
LEARNING_RATE = 2e-4
NUM_ITERATIONS = 3200

# we change if needed
if EXPERIMENT == "tree_lstm_intermittent_supervision":
    INTERMITTENT_SUPERVISION = True
    INTERMITTENT_SUPERVISION_SKIP_LEAF_NODES = True

    BATCH_SIZE = 512
    LEARNING_RATE = 2e-3
    NUM_ITERATIONS = 5000
##############################
##
##############################


# additional static hparams/config

EVAL_EVERY = int(NUM_ITERATIONS // 20)
SHUFFLE_PROB = 0.5
intermittent_supervision_suffix = (
    "_intermittent_supervision" if INTERMITTENT_SUPERVISION else ""
)


# this function reads in a textfile and fixes an issue with "\\"
def filereader(path):
    with open(path, mode="r", encoding="utf-8") as f:
        for line in f:
            yield line.strip().replace("\\", "")


# Let's first make a function that extracts the tokens (the leaves).
def tokens_from_treestring(s):
    """extract the tokens from a sentiment tree"""
    return re.sub(r"\([0-9] |\)", "", s).split()


# We will also need the following function, but you can ignore this for now.
# It is explained later on.
SHIFT = 0
REDUCE = 1


def transitions_from_treestring(s):
    s = re.sub("\([0-5] ([^)]+)\)", "0", s)
    s = re.sub("\)", " )", s)
    s = re.sub("\([0-4] ", "", s)
    s = re.sub("\([0-4] ", "", s)
    s = re.sub("\)", "1", s)
    return list(map(int, s.split()))


# A simple way to define a class is using namedtuple.
Example = namedtuple(
    "Example",
    [
        "tokens",
        "label",
        "transitions",
    ],
)


def examplereader(
    path,
    lower=False,
    shuffle_tree=False,
    shuffle_words=False,
    include_subtrees=INTERMITTENT_SUPERVISION,
):
    """Returns all examples in a file one by one."""
    for line in filereader(path):
        line = line.lower() if lower else line
        if include_subtrees:
            subtrees = utils.parenthetic_contents(line)
        else:
            subtrees = [line]
        for subtree in subtrees:
            label = int(line[1])
            subtree = (
                utils.shuffle_tree(subtree, p=SHUFFLE_PROB)
                if shuffle_tree
                else subtree
            )
            tokens = tokens_from_treestring(subtree)
            if shuffle_words:
                np.random.shuffle(tokens)
            trans = transitions_from_treestring(subtree)
            if len(tokens) > 1 or not (
                INTERMITTENT_SUPERVISION_SKIP_LEAF_NODES and include_subtrees
            ):
                yield Example(
                    tokens=tokens,
                    label=label,
                    transitions=trans,
                )


# Let's load the data into memory.
LOWER = False  # we will keep the original casing
train_data_normal = list(examplereader("trees/train.txt", lower=LOWER))
dev_data_normal = list(
    examplereader("trees/dev.txt", lower=LOWER, include_subtrees=False)
)
test_data_normal = list(
    examplereader("trees/test.txt", lower=LOWER, include_subtrees=False)
)

train_data_shuffled = list(
    examplereader("trees/train.txt", lower=LOWER, shuffle_tree=True)
)
dev_data_shuffled = list(
    examplereader(
        "trees/dev.txt", lower=LOWER, shuffle_tree=True, include_subtrees=False
    )
)
test_data_shuffled = list(
    examplereader(
        "trees/test.txt",
        lower=LOWER,
        shuffle_tree=True,
        include_subtrees=False,
    )
)

train_data_random = list(
    examplereader("trees/train.txt", lower=LOWER, shuffle_words=True)
)
dev_data_random = list(
    examplereader(
        "trees/dev.txt",
        lower=LOWER,
        shuffle_words=True,
        include_subtrees=False,
    )
)
test_data_random = list(
    examplereader(
        "trees/test.txt",
        lower=LOWER,
        shuffle_words=True,
        include_subtrees=False,
    )
)


SENT_LENGTH_BREAKPOINTS = [0, 11, 29, np.inf]


def bin_data_on_sent_len(data, lens=SENT_LENGTH_BREAKPOINTS):
    """
    Split data based on sentence length breakpoints.
    """
    data_sorted = sorted(
        data, key=lambda ex: len(ex.tokens)
    )  # shortest to longest

    def get_data(lower, upper):
        data = []
        for ex in data_sorted:
            if len(ex.tokens) >= upper:
                break
            if len(ex.tokens) >= lower:
                data.append(ex)
        return data

    return [get_data(lens[i - 1], lens[i]) for i in range(1, len(lens))]


train_data_binned = bin_data_on_sent_len(train_data_normal)
dev_data_binned = bin_data_on_sent_len(dev_data_normal)
test_data_binned = bin_data_on_sent_len(test_data_normal)

print("train", len(train_data_normal))
print("dev", len(dev_data_normal))
print("test", len(test_data_normal))
print("train shuffled", len(train_data_shuffled))
print("dev shuffled", len(dev_data_shuffled))
print("test shuffled", len(test_data_shuffled))
print("train random", len(train_data_random))
print("dev random", len(dev_data_random))
print("test random", len(test_data_random))
print("dev binned lengths:", *[len(ds) for ds in dev_data_binned])


# Here we first define a class that can map a word to an ID (w2i)
# and back (i2w).


class OrderedCounter(Counter, OrderedDict):
    """Counter that remembers the order elements are first seen"""

    def __repr__(self):
        return "%s(%r)" % (self.__class__.__name__, OrderedDict(self))

    def __reduce__(self):
        return self.__class__, (OrderedDict(self),)


class Vocabulary:
    """A vocabulary, assigns IDs to tokens"""

    def __init__(self):
        self.freqs = OrderedCounter()
        self.w2i = {}
        self.i2w = []

    def count_token(self, t):
        self.freqs[t] += 1

    def add_token(self, t):
        self.w2i[t] = len(self.w2i)
        self.i2w.append(t)

    def build(self, min_freq=0):
        """
        min_freq: minimum number of occurrences for a word to be included
                  in the vocabulary
        """
        self.add_token("<unk>")  # reserve 0 for <unk> (unknown words)
        self.add_token("<pad>")  # reserve 1 for <pad> (discussed later)
        tok_freq = list(self.freqs.items())
        tok_freq.sort(key=lambda x: x[1], reverse=True)
        for tok, freq in tok_freq:
            if freq >= min_freq:
                self.add_token(tok)


# Now let's map the sentiment labels 0-4 to a more readable form
i2t = ["very negative", "negative", "neutral", "positive", "very positive"]
t2i = OrderedDict({p: i for p, i in zip(i2t, range(len(i2t)))})

import torch
import torch.nn as nn

print("Using torch", torch.__version__)  # should say 1.7.0+cu101

# PyTorch can run on CPU or on Nvidia GPU (video card) using CUDA
# This cell selects the GPU if one is available.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device

# Seed manually to make runs reproducible
# You need to set this again if you do multiple runs of the same model
torch.manual_seed(0)
random.seed(0)

# When running on the CuDNN backend two further options must be set for reproducibility
if torch.cuda.is_available():
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def prepare_example(example, vocab):
    """
    Map tokens to their IDs for a single example
    """
    # vocab returns 0 if the word is not there (i2w[0] = <unk>)
    x = [vocab.w2i.get(t, 0) for t in example.tokens]
    x = torch.LongTensor([x])
    x = x.to(device)
    y = torch.LongTensor([example.label])
    y = y.to(device)
    return x, y


def get_minibatch(data, batch_size=25, shuffle=True):
    """Return minibatches, optional shuffling"""
    if shuffle:
        random.shuffle(data)  # shuffle training data each epoch
    batch = []
    # yield minibatches
    for example in data:
        batch.append(example)
        if len(batch) == batch_size:
            yield batch
            batch = []
    # in case there is something left
    if len(batch) > 0:
        yield batch


import pandas as pd

emb_file = "glove.840B.300d.sst.txt"

word2vec_df = pd.read_csv(emb_file, sep=" ", index_col=0, header=None).T
vector_tokens = set(word2vec_df.columns)
v = Vocabulary()
for data_set in (train_data_normal, test_data_normal, dev_data_normal):
    for ex in data_set:
        for token in ex.tokens:
            if token in vector_tokens:
                v.count_token(token)

v.build()
print("Vocabulary size:", len(v.w2i))
vectors = []
for token in v.i2w:
    if not token in vector_tokens:  # init as zeros for now
        print(token)
        vectors.append(np.zeros(word2vec_df.shape[0]))
        continue
    vectors.append(word2vec_df[token])

vectors = np.stack(vectors, axis=0)


def pad(tokens, length, pad_value=1):
    """add padding 1s to a sequence to that it has the desired length"""
    return tokens + [pad_value] * (length - len(tokens))


def prepare_minibatch(mb, vocab):
    """
    Minibatch is a list of examples.
    This function converts words to IDs and returns
    torch tensors to be used as input/targets.
    """
    batch_size = len(mb)
    maxlen = max([len(ex.tokens) for ex in mb])
    # vocab returns 0 if the word is not there
    x = [pad([vocab.w2i.get(t, 0) for t in ex.tokens], maxlen) for ex in mb]
    x = torch.LongTensor(x)
    x = x.to(device)
    y = [ex.label for ex in mb]
    y = torch.LongTensor(y)
    y = y.to(device)
    return x, y


def prepare_treelstm_minibatch(mb, vocab):
    """
    Returns sentences reversed (last word first)
    Returns transitions together with the sentences.
    """
    batch_size = len(mb)
    maxlen = max([len(ex.tokens) for ex in mb])

    # vocab returns 0 if the word is not there
    # NOTE: reversed sequence!
    x = [
        pad([vocab.w2i.get(t, 0) for t in ex.tokens], maxlen)[::-1]
        for ex in mb
    ]

    x = torch.LongTensor(x)
    x = x.to(device)

    y = [ex.label for ex in mb]
    y = torch.LongTensor(y)
    y = y.to(device)

    maxlen_t = max([len(ex.transitions) for ex in mb])
    transitions = [pad(ex.transitions, maxlen_t, pad_value=2) for ex in mb]
    transitions = np.array(transitions)
    transitions = transitions.T  # time-major

    return (x, transitions), y


def evaluate(
    model,
    data,
    loss_fn,
    batch_fn=get_minibatch,
    prep_fn=prepare_minibatch,
    batch_size=16,
):
    """Accuracy of a model on given data set (using mini-batches)"""
    correct = 0
    total = 0
    model.eval()  # disable dropout
    total_loss = 0
    total_nof_targets = 0
    for mb in batch_fn(data, batch_size=batch_size, shuffle=False):
        x, targets = prep_fn(mb, model.vocab)
        with torch.no_grad():
            logits = model(x)
            total_loss += loss_fn(logits, targets).item()
            total_nof_targets += len(targets)
        predictions = logits.argmax(dim=-1).view(-1)
        # add the number of correct predictions to the total correct
        correct += (predictions == targets.view(-1)).sum().item()
        total += targets.size(0)
    return (
        total_loss / total_nof_targets,
        correct,
        total,
        correct / float(total),
    )


def get_examples(data, shuffle=True, **kwargs):
    """Shuffle data set and return 1 example at a time (until nothing left)"""
    if shuffle:
        random.shuffle(data)  # shuffle training data each epoch
    for example in data:
        yield example


from torch import optim


class EarlyStopping:
    def __init__(self, patience):
        self.patience = patience
        self.best_score = np.inf
        self.counter = 0

    def __call__(self, val_loss):
        if self.patience is None:
            return False
        self.counter += 1
        if val_loss < self.best_score:
            self.best_score = val_loss
            self.counter = 0
        return self.counter >= self.patience


def train_model_(
    seed,
    name,
    model_init_fn,
    optimizer_fn,
    num_iterations,
    train_data,
    dev_data,
    test_data,
    patience=None,
    print_every=EVAL_EVERY,
    eval_every=EVAL_EVERY,
    batch_fn=get_examples,
    prep_fn=prepare_example,
    eval_fn=evaluate,
    batch_size=1,
    eval_batch_size=None,
):
    """Train a model."""
    model = model_init_fn()
    model = model.to(device)
    optimizer = optimizer_fn(model)
    iter_i = 0
    train_loss = 0.0
    print_num = 0
    start = time.time()
    criterion = nn.CrossEntropyLoss()  # loss function
    best_eval = 0.0
    best_iter = 0
    results = defaultdict(lambda: defaultdict(list))

    if INTERMITTENT_SUPERVISION:
        dev_data = random.sample(dev_data, int(2500))

    train_data_eval = random.sample(train_data, int(len(dev_data)))

    early_stopping = EarlyStopping(patience)
    lap = time.time()

    path = str(MODELS_DIR / "{}_{}.pt".format(name, seed))

    if eval_batch_size is None:
        eval_batch_size = batch_size
    while True:  # when we run out of examples, shuffle and continue
        for batch in batch_fn(train_data, batch_size=batch_size):
            # forward pass
            optimizer.zero_grad()
            model.train()
            x, targets = prep_fn(batch, model.vocab)
            logits = model(x)
            B = targets.size(0)  # later we will use B examples per update

            loss = criterion(logits.view([B, -1]), targets.view(-1))
            loss.backward()

            optimizer.step()
            iter_i += 1

            # evaluate
            should_stop_early = False
            if iter_i % eval_every == 0:
                dev_loss, _, _, dev_accuracy = eval_fn(
                    model,
                    dev_data,
                    loss_fn=criterion,
                    batch_size=eval_batch_size,
                    batch_fn=batch_fn,
                    prep_fn=prep_fn,
                )
                train_loss2, _, _, train_accuracy = eval_fn(
                    model,
                    train_data_eval,
                    loss_fn=criterion,
                    batch_size=eval_batch_size,
                    batch_fn=batch_fn,
                    prep_fn=prep_fn,
                )
                print(
                    round(iter_i / float(num_iterations), 2),
                    "progress",
                    round(time.time() - lap, 3),
                    "secs",
                )
                lap = time.time()
                print("dev_accuracy", dev_accuracy)
                print("train_accuracy", train_accuracy)
                results["train"]["loss"].append(train_loss2)
                results["train"]["accuracy"].append(train_accuracy)
                results["valid"]["loss"].append(dev_loss)
                results["valid"]["accuracy"].append(dev_accuracy)
                # save best model parameters
                if dev_accuracy > best_eval:
                    best_eval = dev_accuracy
                    best_iter = iter_i
                    ckpt = {
                        "state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "best_eval": best_eval,
                        "best_iter": best_iter,
                    }
                    torch.save(ckpt, path)
                should_stop_early = early_stopping(dev_loss)
            # done training
            if iter_i == num_iterations or should_stop_early:
                print("Done training")
                # evaluate on train, dev, and test with best model
                ckpt = torch.load(path)
                model.load_state_dict(ckpt["state_dict"])
                train_loss, _, _, train_acc = eval_fn(
                    model,
                    train_data,
                    loss_fn=criterion,
                    batch_size=eval_batch_size,
                    batch_fn=batch_fn,
                    prep_fn=prep_fn,
                )
                dev_loss, _, _, dev_acc = eval_fn(
                    model,
                    dev_data,
                    loss_fn=criterion,
                    batch_size=eval_batch_size,
                    batch_fn=batch_fn,
                    prep_fn=prep_fn,
                )
                test_loss, _, _, test_acc = eval_fn(
                    model,
                    test_data,
                    loss_fn=criterion,
                    batch_size=eval_batch_size,
                    batch_fn=batch_fn,
                    prep_fn=prep_fn,
                )
                results["test"]["loss"].append(test_loss)
                results["test"]["accuracy"].append(test_acc)
                print(
                    "best model iter {:d}/{:d}: "
                    "train acc={:.4f}, dev acc={:.4f}, test acc={:.4f}".format(
                        best_iter, iter_i, train_acc, dev_acc, test_acc
                    )
                )
                return results


def train_model(*args, **kwargs):
    total_results = {}
    for seed in [0, 42, 420]:
        print("training with seed", str(seed))
        torch.manual_seed(seed)
        random.seed(seed)
        results = train_model_(seed, *args, **kwargs)
        total_results[str(seed)] = utils.defaultdict2dict(results)
    return total_results


def pt_deep_cbow_model_init_fn():
    pt_deep_cbow_model = DeepCBOW(
        vocab_size=len(v.w2i),
        emb_dim=300,
        hidden_dim=100,
        vocab=v,
    )
    pt_deep_cbow_model.embed.weight.data.copy_(torch.from_numpy(vectors))
    pt_deep_cbow_model.embed.weight.requires_grad = False
    return pt_deep_cbow_model


def lstm_model_init_fn():
    lstm_model = LSTMClassifier(len(v.w2i), 300, 168, len(t2i), v)
    # copy pre-trained vectors into embeddings table
    with torch.no_grad():
        lstm_model.embed.weight.data.copy_(torch.from_numpy(vectors))
        lstm_model.embed.weight.requires_grad = False
    return lstm_model


def tree_model_init_fn():
    tree_model = TreeLSTMClassifier(len(v.w2i), 300, 150, len(t2i), v)
    with torch.no_grad():
        tree_model.embed.weight.data.copy_(torch.from_numpy(vectors))
        tree_model.embed.weight.requires_grad = False
    return tree_model


optimizer_fn = lambda model: optim.Adam(model.parameters(), lr=LEARNING_RATE)


def train(
    name,
    model_init_fn,
    train_data,
    dev_data,
    test_data,
    prep_fn=prepare_minibatch,
):
    orig_name = name

    print("----------------------")
    print(f"TRAINING: {name}")
    print("----------------------")
    total_results = train_model(
        name,
        model_init_fn,
        optimizer_fn,
        num_iterations=NUM_ITERATIONS,
        patience=None,
        eval_every=EVAL_EVERY,
        prep_fn=prep_fn,
        eval_fn=evaluate,
        batch_fn=get_minibatch,
        batch_size=BATCH_SIZE,
        eval_batch_size=BATCH_SIZE,
        train_data=train_data,
        dev_data=dev_data,
        test_data=test_data,
    )
    utils.save_pickle(f"{name}_results.pkl", total_results)


name2model_init_and_datasets = {
    "deep_cbow": (
        pt_deep_cbow_model_init_fn,
        "normal",
    ),
    "lstm": (lstm_model_init_fn, "normal"),
    "tree_lstm": (tree_model_init_fn, "normal"),
    "lstm_shuffled": (
        lstm_model_init_fn,
        "tree-shuffled",
    ),
    "tree_lstm_shuffled": (
        tree_model_init_fn,
        "tree-shuffled",
    ),
    "lstm_random": (
        lstm_model_init_fn,
        "randomly-shuffled",
    ),
    "tree_lstm_random": (
        tree_model_init_fn,
        "randomly-shuffled",
    ),
    "tree_lstm_intermittent_supervision": (tree_model_init_fn, "normal"),
}

dataset_name2data = {
    "normal": (train_data_normal, dev_data_normal, test_data_normal),
    "tree-shuffled": (
        train_data_shuffled,
        dev_data_shuffled,
        test_data_shuffled,
    ),
    "randomly-shuffled": (
        train_data_random,
        dev_data_random,
        test_data_random,
    ),
}

if __name__ == "__main__":
    init_fn, dataset_name = name2model_init_and_datasets[EXPERIMENT]
    train_data, dev_data, test_data = dataset_name2data[dataset_name]
    train(
        EXPERIMENT,
        init_fn,
        train_data=train_data,
        dev_data=dev_data,
        test_data=test_data,
        prep_fn=prepare_treelstm_minibatch,
    )
