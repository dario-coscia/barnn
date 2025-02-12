import torch
import torch.nn as nn


class AlphaFunction(nn.Module):
    def forward(self, x):
        p = torch.sigmoid(x)
        return p / (1.-p)
   
class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size = 128, device=None, dtype=None):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_features=input_size, out_features=hidden_size, device=device, dtype=dtype),
            nn.LeakyReLU(),
            nn.Linear(in_features=hidden_size, out_features=hidden_size, device=device, dtype=dtype),
            nn.LeakyReLU(),
            nn.Linear(in_features=hidden_size, out_features=input_size, device=device, dtype=dtype),
            AlphaFunction()
            )
    def forward(self, x):
        return self.encoder(x)

class OneHotEncoding(torch.nn.Module):
    def __init__(self, vocab_size, hidden_size):
        super().__init__()
        self.vocab_size = vocab_size
        self.linear = torch.nn.Linear(vocab_size, hidden_size)

    def forward(self, seq):
        code = nn.functional.one_hot(seq, self.vocab_size).float()
        return self.linear(code)
    
class BARNNLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, bias=True, device=None, dtype=None):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, bias=bias, batch_first=True, device=device, dtype=dtype)
        self.posterior_h = Encoder(hidden_size)
        self.posterior_y = Encoder(input_size)

    def init_hidden(self, batch_size, dtype, device):
        return [
                torch.zeros(self.lstm.num_layers, batch_size,
                            self.lstm.hidden_size, dtype=dtype,
                            device=device),
                torch.zeros(self.lstm.num_layers, batch_size,
                            self.lstm.hidden_size, dtype=dtype,
                            device=device)
        ]
                
    def forward(self, input, hidden):
        self.kl_log = 0
        self.seq_len = input.size(1)
        outputs = torch.tensor([], device=input.device, dtype=input.dtype)
        inputs = input.unbind(1)
        if hidden is None:
            hidden = self.init_hidden(batch_size=input.size(0), dtype=input.dtype, device=input.device)
        h_n, c_n = hidden
        # Loop over time steps
        for t in range(len(inputs)):
            input_t = inputs[t].unsqueeze(1)
            alpha_y = self.posterior_y(input_t)
            alpha_h = self.posterior_h(h_n)
            self._kl(alpha_h)
            self._kl(alpha_y)
            noise = 1. + torch.randn_like(h_n) if self.training else 1.
            h_n = h_n * alpha_h * noise
            noise = 1. + torch.randn_like(input_t) if self.training else 1.
            input_t = input_t * alpha_y * noise
            output_t, (h_n, c_n) = self.lstm(input_t, (h_n, c_n))
            outputs = torch.cat((outputs, output_t), 1)
        return outputs, (h_n, c_n)

    def _kl(self, alpha): # alpha = [# layers, batch, hidden_size]
        scale = alpha.size(-1) ** 2
        mean_alpha = alpha
        var_alpha = alpha.pow(2)
        mean_beta = mean_alpha.mean(1, keepdim=True)
        var_beta = var_alpha.mean(1, keepdim=True)
        self.kl_log += (scale * 0.5 * ((mean_alpha-mean_beta).pow(2)/var_beta + (var_alpha/var_beta) - 1 - torch.log(var_alpha/var_beta))).mean((0, 2))
    
    def kl(self):
        return self.kl_log / self.seq_len

class RNN(nn.Module):
    def __init__(
            self,
            vocab_size,
            hidden_size=1024,
            num_layers=3,
            dropout=0.,
            layer_type='lstm'):
        super().__init__()
        # encoder and decoder
        self.encoder = OneHotEncoding(vocab_size, hidden_size)
        self.decoder = nn.Linear(hidden_size, vocab_size)
        # rnn layers (by default hidden initialize with zeros vector)
        assert isinstance(num_layers, int), "num_layers must be int."
        assert num_layers >= 1, "num_layers must be greater or equal than one"
        if layer_type == 'dropout_lstm':
            dropout = 0.2
        if layer_type == 'lstm':
            dropout == 0.
        # initialize layer type
        if layer_type == 'barnn':
            self.rnn = BARNNLSTM(input_size=hidden_size,
                                 hidden_size=hidden_size,
                                 num_layers=num_layers)
        else:
            self.rnn = torch.nn.LSTM(input_size=hidden_size,
                                  hidden_size=hidden_size,
                                  num_layers=num_layers,
                                  dropout=dropout,
                                  batch_first=True)
        # saving vocabulary size, num_layers, n_hidden
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.layer_type = layer_type
        self.hidden_size = hidden_size
    
    def forward(self, input, hidden=None):
        """
        Forward pass for LSTM model. Returing log probabilities with shape
        sequence batch size X length X vocabulary size.

        :param torch.Tensor input: Input sequence of size
            sequence batch size X length x input size.
        :param torch.Tensor hidden: Hidden variables, defaults to None. If None
            they are initialized with zero.
        """
        emb = self.encoder(input)
        output, hidden = self.rnn(emb, hidden)
        logits = self.decoder(output)
        return logits, hidden
    
    @property
    def device(self):
        return next(self.parameters()).device