import torch
import snntorch as snn

class SpikingConv2d(torch.nn.Module):
    """
    Represents a spiking convolutional layer for 2D data.

    This class implements a spiking convolutional operation, handling input and output
    spike trains, and weight updates.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        kernel_size (int or tuple): Size of the convolutional kernel.
        stride (int or tuple, optional): Stride of the convolution (default: 1).
        padding (int or tuple, optional): Padding added to both sides of the input (default: 0).
        dilation (int or tuple, optional): Spacing between kernel elements (default: 1).
        groups (int, optional): Number of blocked connections from input channels to output channels (default: 1).
        bias (bool, optional): If True, adds a learnable bias to the output (default: True).
        spike_grad (str, optional): Surrogate gradient for the spike function (default: 'Heaviside').
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, spike_grad='Heaviside'):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias
        self.spike_grad = spike_grad

        # Initialize the spiking convolutional layer using snnTorch
        self.conv = snn.Convolution(self.in_channels, self.out_channels, self.kernel_size, self.stride, self.padding, self.dilation, self.groups, self.bias, self.spike_grad)

    def forward(self, x):
        """
        Performs the forward pass of the spiking convolutional layer.

        This method applies the spiking convolutional operation to the input spike trains
        and returns the output spike trains.

        Args:
            x (torch.Tensor): Input spike trains tensor of shape (batch_size, in_channels, height, width, time_steps).

        Returns:
            torch.Tensor: Output spike trains tensor of shape (batch_size, out_channels, out_height, out_width, time_steps).
        """
        return self.conv(x)

class SpikingRecurrent(torch.nn.Module):
    """
    Represents a spiking recurrent layer.

    This class implements a spiking recurrent operation, handling input and output
    spike trains, and state updates.

    Args:
        input_size (int): Size of the input features.
        hidden_size (int): Number of features in the hidden state.
        num_layers (int, optional): Number of recurrent layers (default: 1).
        bias (bool, optional): If True, adds a learnable bias to the output (default: True).
        dropout (float, optional): If non-zero, introduces a dropout layer on the outputs of each recurrent layer except the last layer (default: 0.0).
        bidirectional (bool, optional): If True, becomes a bidirectional recurrent layer (default: False).
        spike_grad (str, optional): Surrogate gradient for the spike function (default: 'Heaviside').
    """

    def __init__(self, input_size, hidden_size, num_layers=1, bias=True, dropout=0.0, bidirectional=False, spike_grad='Heaviside'):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.spike_grad = spike_grad

        # Initialize the spiking recurrent layer using snnTorch
        self.rnn = snn.Recurrent(self.input_size, self.hidden_size, self.num_layers, self.bias, self.dropout, self.bidirectional, self.spike_grad)

    def forward(self, x, hidden=None):
        """
        Performs the forward pass of the spiking recurrent layer.

        This method applies the spiking recurrent operation to the input spike trains
        and returns the output spike trains and the updated hidden state.

        Args:
            x (torch.Tensor): Input spike trains tensor of shape (batch_size, input_size, time_steps).
            hidden (torch.Tensor, optional): Initial hidden state tensor of shape (num_layers * num_directions, batch_size, hidden_size).

        Returns:
            tuple: A tuple containing:
                - output (torch.Tensor): Output spike trains tensor of shape (batch_size, hidden_size * num_directions, time_steps).
                - hidden (torch.Tensor): Updated hidden state tensor of shape (num_layers * num_directions, batch_size, hidden_size).
        """
        return self.rnn(x, hidden)