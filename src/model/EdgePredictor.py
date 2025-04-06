#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
Edge Predictor Model

This module contains neural network models for edge prediction in graphs.
It includes an EdgePredictor class that uses an MLP for prediction.

Author: Zhongjian Zhang
Date: 2024/3/28
"""
import torch.nn as nn
import torch.nn.functional as F


class EdgePredictor(nn.Module):
    """
    Neural network model for predicting edges in graphs.
    
    Uses a multi-layer perceptron followed by a sigmoid activation
    to output edge existence probability.
    """
    def __init__(self, embedding_dim, hidden_dim, num_classes=1, num_layers=3, dropout=0.5):
        """
        Initialize the EdgePredictor model.
        
        Args:
            embedding_dim (int): Dimension of input node embeddings
            hidden_dim (int): Dimension of hidden layers
            num_classes (int): Number of output classes (default: 1 for binary prediction)
            num_layers (int): Number of layers in the MLP
            dropout (float): Dropout probability
        """
        super(EdgePredictor, self).__init__()
        self.mlp = MLP(
            in_dim=embedding_dim, 
            hidden_dim=hidden_dim, 
            out_dim=num_classes, 
            num_layers=num_layers, 
            dropout=dropout
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x (Tensor): Input tensor containing node features
            
        Returns:
            Tensor: Edge prediction probabilities
        """
        x = self.mlp(x)
        x = self.sigmoid(x)
        return x


class MLP(nn.Module):
    """
    Multi-Layer Perceptron implementation with batch normalization and dropout.
    """
    def __init__(self, in_dim, hidden_dim, out_dim, num_layers, dropout):
        """
        Initialize the MLP.
        
        Args:
            in_dim (int): Input dimension
            hidden_dim (int): Hidden layer dimension
            out_dim (int): Output dimension
            num_layers (int): Number of layers
            dropout (float): Dropout probability
        """
        super(MLP, self).__init__()
        self.dropout_rate = dropout
        
        # Create layers
        self.layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        self.activations = nn.ModuleList()
        
        # Input layer
        self.layers.append(nn.Linear(in_dim, hidden_dim))
        self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
        self.activations.append(nn.RReLU())
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
            self.activations.append(nn.RReLU())
        
        # Output layer
        self.layers.append(nn.Linear(hidden_dim, out_dim))
        
        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize model weights using Xavier uniform initialization."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)

    def forward(self, x):
        """
        Forward pass through the MLP.
        
        Args:
            x (Tensor): Input tensor
            
        Returns:
            Tensor: Output of the MLP
        """
        # Process through all layers except the last
        for i, layer in enumerate(self.layers[:-1]):
            x = layer(x)
            x = self.batch_norms[i](x)
            x = self.activations[i](x)
            x = F.dropout(x, p=self.dropout_rate, training=self.training)
        
        # Final layer (without activation and batch norm)
        x = self.layers[-1](x)
        return x
