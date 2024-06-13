import torch
import torch.nn as nn

class MambaBackbone(nn.Module):
    """
    Represents the backbone network for the proposed architecture.

    This class implements the MambaBackbone, which serves as the foundation
    for the proposed architecture. It utilizes the basic implementation of the
    Selective State-Space Model (SSM) to process input data and produce the
    desired output representations or features.

    Args:
        input_size (int): Size of the input features.
        hidden_size (int): Number of features in the hidden state.
        output_size (int): Size of the output features.
        num_layers (int, optional): Number of SSM layers (default: 1).
        dropout (float, optional): Dropout probability (default: 0.0).
    """

    def __init__(self, input_size, hidden_size, output_size, num_layers=1, dropout=0.0):
        super(MambaBackbone, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.dropout = dropout

        # Initialize the SSM layers
        self.ssm_layers = nn.ModuleList([
            SSMLayer(input_size, hidden_size, dropout) for _ in range(num_layers)
        ])

        # Output projection layer
        self.output_layer = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        """
        Performs the forward pass of the MambaBackbone network.

        This method accepts input data and produces the desired output representations
        or features by passing the input through the SSM layers and the output
        projection layer.

        Args:
            x (torch.Tensor): Input data tensor of shape (batch_size, sequence_length, input_size).

        Returns:
            torch.Tensor: Output representations or features tensor of shape (batch_size, output_size).
        """
        batch_size, sequence_length, _ = x.size()

        # Initialize hidden state
        hidden = torch.zeros(batch_size, self.hidden_size).to(x.device)

        # Pass the input through the SSM layers
        for layer in self.ssm_layers:
            x, hidden = layer(x, hidden)

        # Apply the output projection layer
        output = self.output_layer(hidden)

        return output

class SSMLayer(nn.Module):
    """
    Represents a single layer of the Selective State-Space Model (SSM).

    This class implements a basic building block of the SSM, which performs
    state transition dynamics, selective attention mechanisms, and feature extraction.

    Args:
        input_size (int): Size of the input features.
        hidden_size (int): Number of features in the hidden state.
        dropout (float, optional): Dropout probability (default: 0.0).
    """

    def __init__(self, input_size, hidden_size, dropout=0.0):
        super(SSMLayer, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.dropout = dropout

        # State transition layer
        self.state_transition = nn.Linear(hidden_size, hidden_size)

        # Input projection layer
        self.input_projection = nn.Linear(input_size, hidden_size)

        # Selective attention layer
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()
        )

        # Dropout layer
        self.dropout_layer = nn.Dropout(dropout)

    def forward(self, x, hidden):
        """
        Performs the forward pass of the SSMLayer.

        This method applies state transition dynamics, selective attention mechanisms,
        and feature extraction to the input tensor and hidden state.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, sequence_length, input_size).
            hidden (torch.Tensor): Hidden state tensor of shape (batch_size, hidden_size).

        Returns:
            tuple: A tuple containing the updated hidden state and output tensor.
                - hidden (torch.Tensor): Updated hidden state tensor of shape (batch_size, hidden_size).
                - output (torch.Tensor): Output tensor of shape (batch_size, sequence_length, hidden_size).
        """
        # State transition
        hidden = self.state_transition(hidden)

        # Input projection
        input_proj = self.input_projection(x)

        # Selective attention
        attention_scores = self.attention(hidden)
        attention_scores = attention_scores.unsqueeze(1)
        attention_weights = torch.softmax(attention_scores, dim=1)

        # Attended input
        attended_input = attention_weights * input_proj
        attended_input = attended_input.sum(dim=1)

        # Update hidden state
        hidden = hidden + attended_input

        # Dropout
        hidden = self.dropout_layer(hidden)

        # Output projection
        output = self.input_projection(hidden)

        return output, hidden