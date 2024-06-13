import torch
import torch.nn as nn

class InterfaceLayer(nn.Module):
    """
    Represents an interface layer that bridges different components or modules.

    This class implements an interface layer that accepts input data or representations
    from one component and transforms or processes them to be compatible with the
    requirements of another component. It ensures seamless integration and information
    flow between different components of the proposed architecture.

    Args:
        input_size (int): Size of the input features.
        output_size (int): Size of the output features.
        hidden_size (int, optional): Number of features in the hidden layers (default: None).
            If None, no hidden layers are used.
        num_layers (int, optional): Number of hidden layers (default: 1).
        activation (str, optional): Activation function to use in the hidden layers (default: 'relu').
            Supported options: 'relu', 'tanh', 'sigmoid'.
        dropout (float, optional): Dropout probability (default: 0.0).
    """

    def __init__(self, input_size, output_size, hidden_size=None, num_layers=1,
                 activation='relu', dropout=0.0):
        super(InterfaceLayer, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.activation = activation
        self.dropout = dropout

        # Define the layers
        layers = []
        if hidden_size is None:
            # No hidden layers, direct mapping from input to output
            layers.append(nn.Linear(input_size, output_size))
        else:
            # Hidden layers with specified activation and dropout
            input_dim = input_size
            for _ in range(num_layers):
                layers.append(nn.Linear(input_dim, hidden_size))
                layers.append(self._get_activation())
                layers.append(nn.Dropout(dropout))
                input_dim = hidden_size
            layers.append(nn.Linear(hidden_size, output_size))

        # Create the sequential model
        self.model = nn.Sequential(*layers)

    def _get_activation(self):
        """
        Returns the activation function based on the specified activation type.

        Returns:
            nn.Module: Activation function module.
        """
        if self.activation == 'relu':
            return nn.ReLU()
        elif self.activation == 'tanh':
            return nn.Tanh()
        elif self.activation == 'sigmoid':
            return nn.Sigmoid()
        else:
            raise ValueError(f"Unsupported activation type: {self.activation}")

    def forward(self, x):
        """
        Performs the forward pass of the InterfaceLayer.

        This method accepts input data or representations from one component and
        transforms or processes them to be compatible with the requirements of another
        component. It applies the necessary data transformations, reshaping, or
        preprocessing operations to ensure seamless integration and information flow.

        Args:
            x (torch.Tensor): Input data or representations.

        Returns:
            torch.Tensor: Transformed or processed output data or representations.
        """
        # Perform preprocessing on the input data
        x = self._preprocess_input(x)

        # Pass the preprocessed input through the interface layers
        output = self.model(x)

        # Perform postprocessing on the output data
        output = self._postprocess_output(output)

        return output

    def _preprocess_input(self, x):
        """
        Preprocesses the input data or representations.

        This method performs the necessary preprocessing steps on the input data
        or representations before passing them to the next component or module.
        It handles tasks such as converting spike trains to activation values and
        reshaping the input to the required format.

        Args:
            x (torch.Tensor): Input data or representations.

        Returns:
            torch.Tensor: Preprocessed input data or representations.
        """
        # Convert spike trains to activation values
        x = self._spike_to_activation(x)

        # Reshape the input to the required format
        x = x.view(x.size(0), -1)

        return x

    def _postprocess_output(self, x):
        """
        Postprocesses the output data or representations.

        This method performs the necessary postprocessing steps on the output data
        or representations after receiving them from the previous component or module.
        It handles tasks such as converting activation values back to spike trains and
        reshaping the output to the required format.

        Args:
            x (torch.Tensor): Output data or representations.

        Returns:
            torch.Tensor: Postprocessed output data or representations.
        """
        # Reshape the output to the required format
        x = x.view(x.size(0), self.output_size)

        # Convert activation values back to spike trains
        x = self._activation_to_spike(x)

        return x

    def _spike_to_activation(self, x):
        """
        Converts spike trains to activation values.

        This method converts the input spike trains to activation values by
        accumulating the spikes over a specified time window and normalizing
        the accumulated values.

        Args:
            x (torch.Tensor): Input spike trains.

        Returns:
            torch.Tensor: Activation values.
        """
        # Accumulate the spikes over a specified time window
        x = x.sum(dim=-1)

        # Normalize the accumulated values
        x = x / x.max()

        return x

    def _activation_to_spike(self, x):
        """
        Converts activation values back to spike trains.

        This method converts the activation values back to spike trains by
        applying a threshold and generating spikes based on the activation values.

        Args:
            x (torch.Tensor): Activation values.

        Returns:
            torch.Tensor: Spike trains.
        """
        # Apply a threshold to generate spikes
        threshold = 0.5
        x = (x > threshold).float()

        return x