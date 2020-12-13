import torch.nn as nn
import torch
import math


class Gate(nn.Module):
    def __init__(self, input_dim, hidden_dim, nonlinear):
        super(Gate, self).__init__()
        self.Wx = nn.Parameter(
            nn.init.kaiming_normal_(torch.empty(hidden_dim, input_dim))
        )
        self.Wh = nn.Parameter(
            nn.init.kaiming_normal_(torch.empty(hidden_dim, hidden_dim))
        )
        self.bx = nn.Parameter(torch.empty(hidden_dim))
        self.bh = nn.Parameter(torch.empty(hidden_dim))
        self.nonlinear_layer = nonlinear()

    def forward(self, x, prev_hidden):
        # prev_hidden = bs x hidden_dim
        # x = bs x input_dim
        # Wx = hidden x input
        # (Wx @ x) = bs x hidden
        out = self.nonlinear_layer(
            (x @ self.Wx.T) + (prev_hidden @ self.Wh.T) + self.bx + self.bh
        )
        # print('OUT.SHAPE', out.shape)
        return out


class MyLSTMCell(nn.Module):
    """Our own LSTM cell"""

    def __init__(self, input_size, hidden_size, bias=True):
        """Creates the weights for this LSTM"""
        super(MyLSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.g = Gate(input_size, hidden_size, nn.Tanh)
        self.i = Gate(input_size, hidden_size, nn.Sigmoid)
        self.f = Gate(input_size, hidden_size, nn.Sigmoid)
        self.o = Gate(input_size, hidden_size, nn.Sigmoid)
        self.tan_layer = nn.Tanh()
        self.reset_parameters()

    def reset_parameters(self):
        """This is PyTorch's default initialization method"""
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, input_, hx, mask=None):
        """
        input is (batch, input_size)
        hx is ((batch, hidden_size), (batch, hidden_size))
        """
        prev_h, prev_c = hx
        c = (self.g(input_, prev_h) * self.i(input_, prev_h)) + (
            prev_c * self.f(input_, prev_h)
        )
        h = self.tan_layer(c) * self.o(input_, prev_h)
        # print('h', h.shape)
        # print('c', c.shape)
        return h, c

    def __repr__(self):
        return "{}({:d}, {:d})".format(
            self.__class__.__name__, self.input_size, self.hidden_size
        )


class LSTMClassifier(nn.Module):
    """Encodes sentence with an LSTM and projects final hidden state"""

    def __init__(
        self, vocab_size, embedding_dim, hidden_dim, output_dim, vocab
    ):
        super(LSTMClassifier, self).__init__()
        self.vocab = vocab
        self.hidden_dim = hidden_dim
        self.embed = nn.Embedding(vocab_size, embedding_dim, padding_idx=1)
        self.rnn = MyLSTMCell(embedding_dim, hidden_dim)
        self.output_layer = nn.Sequential(
            nn.Dropout(p=0.5),  # explained later
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):

        B = x.size(0)  # batch size (this is 1 for now, i.e. 1 single example)
        T = x.size(1)  # timesteps (the number of words in the sentence)

        input_ = self.embed(x)

        # here we create initial hidden states containing zeros
        # we use a trick here so that, if input is on the GPU, then so are hx and cx
        hx = input_.new_zeros(B, self.rnn.hidden_size)
        cx = input_.new_zeros(B, self.rnn.hidden_size)

        # process input sentences one word/timestep at a time
        # input is batch-major (i.e., batch size is the first dimension)
        # so the first word(s) is (are) input_[:, 0]
        outputs = []
        for i in range(T):
            # print('i', i)
            hx, cx = self.rnn(input_[:, i], (hx, cx))
            # print('hx.shape', hx.shape)
            # print('cx.shape', cx.shape)
            outputs.append(hx)

        # if we have a single example, our final LSTM state is the last hx
        if B == 1:
            final = hx
        else:
            #
            # This part is explained in next section, ignore this else-block for now.
            #
            # We processed sentences with different lengths, so some of the sentences
            # had already finished and we have been adding padding inputs to hx.
            # We select the final state based on the length of each sentence.

            # two lines below not needed if using LSTM from pytorch
            outputs = torch.stack(outputs, dim=0)  # [T, B, D]
            outputs = outputs.transpose(0, 1).contiguous()  # [B, T, D]

            # to be super-sure we're not accidentally indexing the wrong state
            # we zero out positions that are invalid
            pad_positions = (x == 1).unsqueeze(-1)

            outputs = outputs.contiguous()
            outputs = outputs.masked_fill_(pad_positions, 0.0)

            mask = x != 1  # true for valid positions [B, T]
            lengths = mask.sum(dim=1)  # [B, 1]

            indexes = (lengths - 1) + torch.arange(
                B, device=x.device, dtype=x.dtype
            ) * T
            final = outputs.view(-1, self.hidden_dim)[indexes]  # [B, D]

        # we use the last hidden state to classify the sentence
        logits = self.output_layer(final)
        return logits
