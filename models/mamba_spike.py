"""
Mamba-Spike: Neuromorphic Computing for High-Resolution Temporal Data with Selective State Spaces.
Implementation of the Mamba-Spike architecture.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from einops import rearrange, repeat
import snntorch as snn
from snntorch import surrogate
from typing import Optional, Tuple


class SelectiveSSM(nn.Module):
    """
    Selective State Space Model (S4) layer implementation.
    Since mamba-ssm is not available, implementing core SSM functionality.
    """
    
    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dt_rank: str = "auto",
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        dt_init: str = "random",
        dt_scale: float = 1.0,
        bias: bool = False,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        
        # Linear projections
        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias)
        
        # Convolution
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=bias,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
        )
        
        # SSM parameters
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + self.d_state * 2, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)
        
        # Initialize dt bias
        dt = torch.exp(
            torch.rand(self.d_inner) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_min)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        
        # S4D real initialization
        A = repeat(
            torch.arange(1, self.d_state + 1, dtype=torch.float32),
            "n -> d n",
            d=self.d_inner,
        ).contiguous()
        A_log = torch.log(A)
        self.A_log = nn.Parameter(A_log)
        self.A_log._no_weight_decay = True
        
        # D parameter
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.D._no_weight_decay = True
        
        # Output projection
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, L, D) tensor
        Returns:
            output: (B, L, D) tensor
        """
        batch, seqlen, dim = x.shape
        
        # Linear projection and split
        x_and_res = self.in_proj(x)  # (B, L, 2 * d_inner)
        x, res = x_and_res.split(self.d_inner, dim=-1)
        
        # Convolution
        x = rearrange(x, "b l d -> b d l")
        x = self.conv1d(x)[:, :, :seqlen]
        x = rearrange(x, "b d l -> b l d")
        
        # Activation
        x = F.silu(x)
        
        # SSM
        y = self.ssm(x)
        
        # Gated MLP
        y = y * F.silu(res)
        
        # Output projection
        output = self.out_proj(y)
        
        return output
    
    def ssm(self, x: torch.Tensor) -> torch.Tensor:
        """Runs the SSM."""
        batch, seqlen, dim = x.shape
        
        # Compute SSM parameters
        A = -torch.exp(self.A_log.float())  # (d_inner, d_state)
        D = self.D.float()
        
        # Compute dt, B, C
        x_dbl = self.x_proj(x)  # (B, L, dt_rank + 2 * d_state)
        
        dt, B, C = x_dbl.split([self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = self.dt_proj(dt)  # (B, L, d_inner)
        dt = F.softplus(dt)
        
        # Discretize
        deltaA = torch.exp(dt.unsqueeze(-1) * A)  # (B, L, d_inner, d_state)
        deltaB = dt.unsqueeze(-1) * B.unsqueeze(2)  # (B, L, d_inner, d_state)
        
        # Perform SSM operation
        y = self.selective_scan(x, deltaA, deltaB, C, D)
        
        return y
    
    def selective_scan(self, u, deltaA, deltaB, C, D):
        """Performs the selective scan operation."""
        batch, seqlen, dim = u.shape
        
        # Initialize hidden state
        x = torch.zeros((batch, dim, self.d_state), device=u.device, dtype=u.dtype)
        ys = []
        
        # Sequential scan (could be optimized with parallel scan)
        for i in range(seqlen):
            x = deltaA[:, i] * x + deltaB[:, i] * u[:, i:i+1].transpose(1, 2)
            y = torch.einsum("bdn,bn->bd", x, C[:, i])
            ys.append(y)
        
        y = torch.stack(ys, dim=1)  # (B, L, d_inner)
        y = y + u * D
        
        return y


import math


class MambaBlock(nn.Module):
    """A single Mamba block."""
    
    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
    ):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.mamba = SelectiveSSM(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Residual connection
        return x + self.mamba(self.norm(x))


class SpikingFrontEnd(nn.Module):
    """
    Spiking front-end for encoding event data into spikes.
    Uses Leaky Integrate-and-Fire (LIF) neurons.
    """
    
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        beta: float = 0.9,
        spike_grad: Optional[object] = None
    ):
        super().__init__()
        
        if spike_grad is None:
            spike_grad = surrogate.fast_sigmoid(slope=25)
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels, hidden_channels, kernel_size, padding=kernel_size//2)
        self.lif1 = snn.Leaky(beta=beta, spike_grad=spike_grad)
        
        self.conv2 = nn.Conv2d(hidden_channels, hidden_channels, kernel_size, padding=kernel_size//2)
        self.lif2 = snn.Leaky(beta=beta, spike_grad=spike_grad)
        
        self.conv3 = nn.Conv2d(hidden_channels, out_channels, kernel_size, padding=kernel_size//2)
        self.lif3 = snn.Leaky(beta=beta, spike_grad=spike_grad)
        
        # Pooling
        self.pool = nn.MaxPool2d(2)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, T, C, H, W) tensor of event frames
        Returns:
            spikes: (B, T, C', H', W') tensor of output spikes
            membrane: final membrane potential
        """
        batch_size, time_steps, _, _, _ = x.shape
        
        # Initialize membrane potentials
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        mem3 = self.lif3.init_leaky()
        
        spk_rec = []
        
        # Process each time step
        for t in range(time_steps):
            x_t = x[:, t]
            
            # Layer 1
            cur1 = self.pool(self.conv1(x_t))
            spk1, mem1 = self.lif1(cur1, mem1)
            
            # Layer 2
            cur2 = self.pool(self.conv2(spk1))
            spk2, mem2 = self.lif2(cur2, mem2)
            
            # Layer 3
            cur3 = self.conv3(spk2)
            spk3, mem3 = self.lif3(cur3, mem3)
            
            spk_rec.append(spk3)
        
        # Stack spikes
        spikes = torch.stack(spk_rec, dim=1)  # (B, T, C, H, W)
        
        return spikes, mem3


class SpikeToActivation(nn.Module):
    """
    Interface layer to convert spikes to continuous activations.
    """
    
    def __init__(self, method: str = "rate"):
        super().__init__()
        self.method = method
        
    def forward(self, spikes: torch.Tensor) -> torch.Tensor:
        """
        Args:
            spikes: (B, T, C, H, W) tensor of spikes
        Returns:
            activations: (B, T, D) tensor of activations
        """
        if self.method == "rate":
            # Rate coding: average spike count over spatial dimensions
            batch_size, time_steps, channels, height, width = spikes.shape
            # Flatten spatial dimensions
            activations = spikes.view(batch_size, time_steps, -1)
            # Apply temporal smoothing
            # Ensure we have the right dimensions for conv1d
            # activations shape: (batch, time, features)
            batch_size, time_steps, num_features = activations.shape
            
            # Create kernel for each feature channel
            kernel = torch.ones(num_features, 1, 5, device=spikes.device) / 5
            
            # Transpose for conv1d: (batch, features, time)
            activations = activations.transpose(1, 2)
            
            # Apply convolution with groups=num_features
            activations = F.conv1d(
                activations,
                kernel,
                padding=2,
                groups=num_features
            )
            
            # Transpose back: (batch, time, features)
            activations = activations.transpose(1, 2)
        else:
            raise ValueError(f"Unknown method: {self.method}")
        
        return activations


class MambaSpike(nn.Module):
    """
    Complete Mamba-Spike architecture.
    """
    
    def __init__(
        self,
        input_channels: int = 2,  # DVS has 2 channels (ON/OFF events)
        input_size: Tuple[int, int] = (34, 34),  # N-MNIST size
        num_classes: int = 10,
        spiking_channels: int = 64,
        d_model: int = 128,
        n_layers: int = 4,
        d_state: int = 16,
        beta: float = 0.9,
    ):
        super().__init__()
        
        # Spiking front-end
        self.spiking_frontend = SpikingFrontEnd(
            in_channels=input_channels,
            hidden_channels=32,
            out_channels=spiking_channels,
            beta=beta
        )
        
        # Calculate output size after pooling
        h_out = input_size[0] // 4  # Two pooling layers
        w_out = input_size[1] // 4
        spike_features = spiking_channels * h_out * w_out
        
        # Interface layer
        self.spike_to_activation = SpikeToActivation(method="rate")
        
        # Project to model dimension
        self.input_proj = nn.Linear(spike_features, d_model)
        
        # Mamba backbone
        self.mamba_blocks = nn.ModuleList([
            MambaBlock(d_model=d_model, d_state=d_state)
            for _ in range(n_layers)
        ])
        
        # Classification head
        self.norm = nn.LayerNorm(d_model)
        self.classifier = nn.Linear(d_model, num_classes)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, C, H, W) tensor of event frames
        Returns:
            logits: (B, num_classes) tensor
        """
        # Spiking front-end
        spikes, _ = self.spiking_frontend(x)
        
        # Convert to activations
        activations = self.spike_to_activation(spikes)
        
        # Project to model dimension
        x = self.input_proj(activations)
        
        # Process through Mamba blocks
        for block in self.mamba_blocks:
            x = block(x)
        
        # Global average pooling over time
        x = x.mean(dim=1)
        
        # Classification
        x = self.norm(x)
        logits = self.classifier(x)
        
        return logits


def create_mamba_spike_nmnist(num_classes: int = 10) -> MambaSpike:
    """Create Mamba-Spike model for N-MNIST dataset."""
    return MambaSpike(
        input_channels=2,
        input_size=(34, 34),
        num_classes=num_classes,
        spiking_channels=64,
        d_model=128,
        n_layers=4,
        d_state=16,
        beta=0.9,
    )


def create_mamba_spike_dvsgesture(num_classes: int = 11) -> MambaSpike:
    """Create Mamba-Spike model for DVS Gesture dataset."""
    return MambaSpike(
        input_channels=2,
        input_size=(128, 128),
        num_classes=num_classes,
        spiking_channels=128,
        d_model=256,
        n_layers=6,
        d_state=16,
        beta=0.9,
    )


def create_mamba_spike_cifar10dvs(num_classes: int = 10) -> MambaSpike:
    """Create Mamba-Spike model for CIFAR10-DVS dataset."""
    return MambaSpike(
        input_channels=2,
        input_size=(128, 128),
        num_classes=num_classes,
        spiking_channels=128,
        d_model=256,
        n_layers=6,
        d_state=16,
        beta=0.9,
    )


if __name__ == "__main__":
    # Test model creation
    model = create_mamba_spike_nmnist()
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Test forward pass
    batch_size = 2
    time_steps = 100
    x = torch.randn(batch_size, time_steps, 2, 34, 34)
    logits = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {logits.shape}")