import torch
import torch.nn as nn
from spikingjelly.clock_driven import functional
from modules.layers import SpikingConv2d, SpikingRecurrent
from modules.mambabackbone import MambaBackbone
from modules.interfacelayer import InterfaceLayer

class MambaSpike(nn.Module):
    """
    Represents the MambaSpike component of the proposed architecture.

    This class implements the MambaSpike component, which integrates the spiking front-end
    with the MambaBackbone to perform spike-based computations and achieve efficient
    temporal data processing.

    Args:
        input_size (int): Size of the input features.
        hidden_size (int): Number of features in the hidden state.
        output_size (int): Size of the output features.
        num_layers (int, optional): Number of MambaBackbone layers (default: 1).
        spike_grad (str, optional): Surrogate gradient for the spike function (default: 'Heaviside').
        tau_mem (float, optional): Membrane time constant for the spiking neuron (default: 10.0).
        v_thresh (float, optional): Firing threshold voltage for the spiking neuron (default: 1.0).
        v_reset (float, optional): Reset voltage after firing for the spiking neuron (default: 0.0).
    """

    def __init__(self, input_size, hidden_size, output_size, num_layers=1, spike_grad='Heaviside',
                 tau_mem=10.0, v_thresh=1.0, v_reset=0.0):
        super(MambaSpike, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.spike_grad = spike_grad
        self.tau_mem = tau_mem
        self.v_thresh = v_thresh
        self.v_reset = v_reset

        # Spiking front-end components
        self.spiking_conv = SpikingConv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.spiking_recurrent = SpikingRecurrent(input_size=16*28*28, hidden_size=256)
        self.interface_layer = InterfaceLayer(input_size=256, output_size=hidden_size)

        # MambaBackbone
        self.mamba_backbone = MambaBackbone(input_size=hidden_size, hidden_size=hidden_size,
                                            output_size=output_size, num_layers=num_layers)

    def forward(self, x):
        """
        Performs the forward pass of the MambaSpike component.

        This method accepts input data and performs the necessary computations and
        transformations using the spiking front-end and MambaBackbone components to
        generate the output spike trains.

        Args:
            x (torch.Tensor): Input data tensor.

        Returns:
            torch.Tensor: Output spike trains.
        """
        # Spiking front-end processing
        x = self.spiking_conv(x)
        x = x.view(x.size(0), -1)
        x, _ = self.spiking_recurrent(x)
        x = self.interface_layer(x)

        # MambaBackbone processing
        x = self.mamba_backbone(x)

        # Generate output spike trains
        output_spikes = functional.heaviside(x)

        return output_spikes

    def train(self, dataset, num_epochs, batch_size, learning_rate, device):
        """
        Trains the MambaSpike model on the given dataset.

        Args:
            dataset (torch.utils.data.Dataset): Training dataset.
            num_epochs (int): Number of training epochs.
            batch_size (int): Batch size for training.
            learning_rate (float): Learning rate for optimization.
            device (str): Device to use for training (e.g., 'cuda' or 'cpu').

        Returns:
            list: Training loss history.
        """
        self.to(device)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)

        train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        train_loss_history = []

        for epoch in range(num_epochs):
            running_loss = 0.0
            for images, labels in train_loader:
                images = images.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                # Forward pass
                outputs = self(images)
                loss = criterion(outputs, labels)

                # Backward pass and optimization
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

            epoch_loss = running_loss / len(train_loader)
            train_loss_history.append(epoch_loss)

            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}")

        return train_loss_history

    def evaluate(self, dataset, batch_size, device):
        """
        Evaluates the MambaSpike model on the given dataset.

        Args:
            dataset (torch.utils.data.Dataset): Evaluation dataset.
            batch_size (int): Batch size for evaluation.
            device (str): Device to use for evaluation (e.g., 'cuda' or 'cpu').

        Returns:
            tuple: A tuple containing the evaluation loss and accuracy.
        """
        self.to(device)
        criterion = nn.MSELoss()

        eval_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)

        running_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in eval_loader:
                images = images.to(device)
                labels = labels.to(device)

                # Forward pass
                outputs = self(images)
                loss = criterion(outputs, labels)

                running_loss += loss.item()

                # Compute accuracy
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        eval_loss = running_loss / len(eval_loader)
        eval_accuracy = correct / total

        return eval_loss, eval_accuracy