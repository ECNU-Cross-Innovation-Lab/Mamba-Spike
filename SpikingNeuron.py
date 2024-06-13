import torch
import snntorch as snn

class SpikingNeuron(torch.nn.Module):
    """
    Represents a spiking neuron using the Leaky Integrate-and-Fire (LIF) model.

    This class implements the dynamics and behavior of an LIF neuron, including the
    leaky integration of input currents and the generation of output spikes based on
    the membrane potential.

    Args:
        input_size (int): Size of the input features.
        tau_mem (float): Membrane time constant (default: 10.0).
        v_thresh (float): Firing threshold voltage (default: 1.0).
        v_reset (float): Reset voltage after firing (default: 0.0).
        tau_syn (float): Synaptic time constant (default: 5.0).
        spike_grad (str): Surrogate gradient for the spike function (default: 'Heaviside').
    """

    def __init__(self, input_size, tau_mem=10.0, v_thresh=1.0, v_reset=0.0, tau_syn=5.0, spike_grad='Heaviside'):
        super().__init__()
        self.input_size = input_size
        self.tau_mem = tau_mem
        self.v_thresh = v_thresh
        self.v_reset = v_reset
        self.tau_syn = tau_syn

        # Initialize the spiking neuron using snnTorch
        self.lif_neuron = snn.Leaky(self.input_size, self.tau_mem, self.v_thresh, self.v_reset, self.tau_syn, spike_grad)

    def forward(self, x, mem=None):
        """
        Performs the forward pass of the LIF neuron.

        This method updates the neuron's membrane potential based on the input current
        and generates output spikes if the membrane potential exceeds the threshold.

        Args:
            x (torch.Tensor): Input current tensor of shape (batch_size, input_size).
            mem (torch.Tensor, optional): Initial membrane potential tensor of shape (batch_size, 1).

        Returns:
            tuple: A tuple containing the output spikes and the updated membrane potential.
                - spikes (torch.Tensor): Output spikes tensor of shape (batch_size, 1).
                - mem (torch.Tensor): Updated membrane potential tensor of shape (batch_size, 1).
        """
        spikes, mem = self.lif_neuron(x, mem)
        return spikes, mem

    def reset(self):
        """
        Resets the neuron's internal state variables.

        This method is typically called at the beginning of a new input sequence or
        when the neuron needs to be reset to its initial state.
        """
        self.lif_neuron.reset()
        
    
    '''
    neuron = SpikingNeuron(input_size=10, tau_mem=10.0, v_thresh=1.0)
    input_current = torch.randn(batch_size, 10)
    spikes, mem = neuron(input_current)
    
    '''