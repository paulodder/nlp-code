import torch.nn as nn


class DeepCBOW(nn.Module):
    """A simple bag-of-words model"""

    def __init__(self, vocab_size, vocab, emb_dim=300, hidden_dim=100):
        super(DeepCBOW, self).__init__()
        self.vocab = vocab
        self.emb_dim = 300
        self.embed = nn.Embedding(vocab_size, self.emb_dim)
        # this is a trainable look-up table with word embeddings
        self.output_layer = nn.Sequential(
            nn.Linear(self.emb_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 5),
        )

    def forward(self, inputs):
        # this is the forward pass of the neural network
        # it applies a function to the input and returns the output

        # this looks up the embeddings for each word ID in inputs
        # the result is a sequence of word embeddings
        embeds = self.embed(inputs)
        # the output is the sum across the time dimension (1)
        # with the bias term added
        logits = self.output_layer(embeds.sum(1))
        return logits
