import torch
import torch.nn as nn
import torch.nn.functional as F
from .model import Model


class EncoderDecoder(Model):

    def __init__(self, encoder, decoder, activation, n_classes=None):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        if n_classes is not None:
            self.linear = nn.Linear(self.encoder.out_shapes[0], n_classes, bias=True)

        if callable(activation) or activation is None:
            self.activation = activation
        elif activation == 'softmax':
            self.activation = nn.Softmax(dim=1)
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        else:
            raise ValueError('Activation should be "sigmoid"/"softmax"/callable/None')

    def forward(self, x):
        """Sequentially pass `x` trough model`s `encoder` and `decoder` (return logits!)"""
        bottleneck = self.encoder(x)
        x = self.decoder(bottleneck)
        if hasattr(self, 'linear'):
            enc_out = F.adaptive_avg_pool2d(bottleneck[0], (1, 1)).view(bottleneck[0].size(0), -1)
            enc_out = self.linear(enc_out)
            return x, enc_out

        return x

    def predict(self, x):
        """Inference method. Switch model to `eval` mode, call `.forward(x)`
        and apply activation function (if activation is not `None`) with `torch.no_grad()`

        Args:
            x: 4D torch tensor with shape (batch_size, channels, height, width)

        Return:
            prediction: 4D torch tensor with shape (batch_size, classes, height, width)

        """
        if self.training:
            self.eval()

        with torch.no_grad():
            x = self.forward(x)
            if self.activation:
                x = self.activation(x)

        return x
