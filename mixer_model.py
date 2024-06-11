from faster_kan.efficient_kan import KAN
import torch.nn as nn
import torch

class KANMixer(nn.Module):
    def __init__(self, input_dim, num_patches, hidden_dim, tokens_kan_dim, channels_kan_dim, num_classes):
        super(KANMixer, self).__init__()

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.num_patches = num_patches
        self.patch_dim = input_dim // num_patches
        
        self.tokens_kan = KAN([self.patch_dim, tokens_kan_dim, self.patch_dim], grid_size=5, spline_order=3).to(device)
        self.channels_kan = KAN([num_patches, channels_kan_dim, num_patches], grid_size=5, spline_order=3).to(device)
        
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes)
        )
        
    def forward(self, x):
        batch_size = x.shape[0]
        
        # Reshape input to (batch_size * num_patches, patch_dim)
        x = x.view(batch_size * self.num_patches, self.patch_dim)
        
        # Apply token KAN
        x = self.tokens_kan(x)
        
        # Reshape back to (batch_size, num_patches, patch_dim)
        x = x.view(batch_size, self.num_patches, self.patch_dim)
        
        # Transpose and apply channel KAN
        x = x.transpose(1, 2).contiguous()
        x = x.view(batch_size * self.patch_dim, self.num_patches)
        x = self.channels_kan(x)
        
        # Flatten and classify
        x = x.view(batch_size, -1)
        x = self.classifier(x)
        
        return x

class MLPMixer(nn.Module):
    def __init__(self, input_dim, num_patches, hidden_dim, tokens_mlp_dim, channels_mlp_dim, num_classes):
        super(MLPMixer, self).__init__()
        self.num_patches = num_patches
        self.patch_dim = input_dim // num_patches

        self.tokens_mlp = nn.Sequential(
            nn.Linear(self.patch_dim, tokens_mlp_dim),
            nn.GELU(),
            nn.Linear(tokens_mlp_dim, self.patch_dim)
        )
        self.channels_mlp = nn.Sequential(
            nn.Linear(num_patches, channels_mlp_dim),
            nn.GELU(),
            nn.Linear(channels_mlp_dim, num_patches)
        )

        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        batch_size = x.shape[0]

        # Reshape input to (batch_size * num_patches, patch_dim)
        x = x.view(batch_size * self.num_patches, self.patch_dim)

        # Apply token MLP
        x = self.tokens_mlp(x)

        # Reshape back to (batch_size, num_patches, patch_dim)
        x = x.view(batch_size, self.num_patches, self.patch_dim)

        # Transpose and apply channel MLP
        x = x.transpose(1, 2).contiguous()
        x = x.view(batch_size * self.patch_dim, self.num_patches)
        x = self.channels_mlp(x)

        # Flatten and classify
        x = x.view(batch_size, -1)
        x = self.classifier(x)

        return x
