import torch
from torch.hub import load_state_dict_from_url
from torchvision.models.resnet import ResNet, Bottleneck

# Dictionary storing URLs to pre-trained ResNeXt model weights
model_urls = {
    'resnext101_32x8d': 'https://download.pytorch.org/models/ig_resnext101_32x8-c38310e5.pth',
    'resnext101_32x16d': 'https://download.pytorch.org/models/ig_resnext101_32x16-c6f796b0.pth',
    'resnext101_32x32d': 'https://download.pytorch.org/models/ig_resnext101_32x32-e4b90b00.pth',
    'resnext101_32x48d': 'https://download.pytorch.org/models/ig_resnext101_32x48-3e41cc8a.pth',
}

def _resnext(arch, block, layers, pretrained, progress, **kwargs):
    """
    Helper function to create a ResNeXt model.

    Args:
        arch (str): Model architecture name (e.g., 'resnext101_32x8d').
        block (torchvision.models.resnet.Bottleneck): Type of block used in ResNet architecture.
        layers (list): Number of layers for each block.
        pretrained (bool): Whether to load pretrained weights.
        progress (bool): If True, displays a progress bar during model download.

    Returns:
        model (ResNet): ResNeXt model with the specified architecture and pretrained weights.
    """
     # Initialize ResNet with the specified block and layers
    model = ResNet(block, layers, **kwargs)
    
    # Load pretrained weights from the URL
    state_dict = load_state_dict_from_url(model_urls[arch], progress=progress)
    
    # Load the state dictionary (weights) into the model
    model.load_state_dict(state_dict)
    return model

def resnext101_32x8d_wsl(progress=True, **kwargs):
    """Constructs a ResNeXt-101 32x8 model pre-trained on weakly-supervised data
    and finetuned on ImageNet from Figure 5 in
    `"Exploring the Limits of Weakly Supervised Pretraining" <https://arxiv.org/abs/1805.00932>`_
    Args:
        progress (bool): If True, displays a progress bar of the download to stderr.
    """
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 8
    return _resnext('resnext101_32x8d', Bottleneck, [3, 4, 23, 3], True, progress, **kwargs)

class ResNeXtModel(torch.nn.Module):
    """
    Custom model class that modifies the pre-trained ResNeXt model for a specific task.
    
    The model removes the final fully connected layer of ResNeXt and adds a custom classification layer.

    Methods:
        forward(input): Defines the forward pass of the model.
    """
    def __init__(self):
        """
        Initializes the ResNeXtModel class.
        
        - Loads the pre-trained ResNeXt-101 32x8 model.
        - Removes the final fully connected layer.
        - Adds a custom fully connected layer for classification.
        """ 
        super(ResNeXtModel, self).__init__()
        # Load the pre-trained ResNeXt model
        resnext = resnext101_32x8d_wsl()
        # Keep all layers except the final fully connected layer
        self.base = torch.nn.Sequential(*list(resnext.children())[:-1])
        # Add a custom fully connected layer for classification (6 output classes)
        self.fc = torch.nn.Sequential(torch.nn.Linear(2048,6))
    
    def forward(self, input):
        """
        Defines the forward pass of the model.

        Args:
            input (torch.Tensor): Input tensor of shape (batch_size, channels, height, width).

        Returns:
            out (torch.Tensor): Output tensor of shape (batch_size, 6) representing class logits.
            features (torch.Tensor): Flattened feature tensor of shape (batch_size, 2048) from the base ResNeXt model.
        """
        # Pass the input through the base model to extract features
        features = self.base(input).reshape(-1, 2048)
        # Pass the features through the custom fully connected layer to get predictions
        out = self.fc(features)
        return out, features