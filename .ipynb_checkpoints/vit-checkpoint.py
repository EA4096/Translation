import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

class PatchEmbedding(nn.Module):
    """Convert image into patches and embed them."""
    
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        
        self.proj = nn.Conv2d(
            in_chans, embed_dim, 
            kernel_size=patch_size, stride=patch_size
        )
        
    def forward(self, x):
        B, C, H, W = x.shape
        assert H % self.patch_size == 0 and W % self.patch_size == 0, \
            f"Input image size ({H}*{W}) must be divisible by patch size ({self.patch_size})"
        
        x = self.proj(x)  # (B, embed_dim, H//patch_size, W//patch_size)
        x = x.flatten(2)  # (B, embed_dim, n_patches)
        x = x.transpose(1, 2)  # (B, n_patches, embed_dim)
        
        return x

class MultiHeadSelfAttention(nn.Module):
    """Multi-head Self Attention module."""
    
    def __init__(self, dim, n_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.n_heads = n_heads
        self.dim = dim
        self.head_dim = dim // n_heads
        self.scale = self.head_dim ** -0.5
        
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class MLP(nn.Module):
    """Multi-Layer Perceptron module."""
    
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class TransformerBlock(nn.Module):
    """Transformer block with pre-normalization."""
    
    def __init__(self, dim, n_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MultiHeadSelfAttention(
            dim, n_heads=n_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop
        )
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            drop=drop
        )
        
    def forward(self, x):
        # Pre-norm architecture
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class VisionTransformerEncoder(nn.Module):
    """Vision Transformer Encoder for image-to-image tasks."""
    
    def __init__(
        self, 
        img_size=224, 
        patch_size=16, 
        in_chans=3, 
        embed_dim=768, 
        depth=12, 
        n_heads=12, 
        mlp_ratio=4., 
        qkv_bias=True, 
        drop_rate=0., 
        attn_drop_rate=0.
    ):
        super().__init__()
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_chans, embed_dim)
        self.n_patches = self.patch_embed.n_patches
        
        # Positional embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, self.n_patches, embed_dim))
        self.pos_drop = nn.Dropout(drop_rate)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(
                dim=embed_dim,
                n_heads=n_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
            )
            for _ in range(depth)
        ])
        
        self.norm = nn.LayerNorm(embed_dim)
        
        # Initialize weights
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
            
    def forward(self, x):
        B, C, H, W = x.shape
        
        # Patch embedding
        x = self.patch_embed(x)  # (B, n_patches, embed_dim)
        
        # Add positional embedding
        x = x + self.pos_embed
        x = self.pos_drop(x)
        
        # Transformer blocks
        for block in self.blocks:
            x = block(x)
            
        x = self.norm(x)
        
        return x

class DeconvDecoder(nn.Module):
    """Deconvolution-based decoder for image reconstruction."""
    
    def __init__(
        self, 
        img_size=224, 
        patch_size=16, 
        embed_dim=768, 
        out_chans=3,
        deconv_channels=[512, 256, 128, 64]
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.out_chans = out_chans
        
        # Calculate spatial dimensions after patch embedding
        self.grid_size = img_size // patch_size
        
        # Initial projection to increase channels
        self.init_proj = nn.Conv2d(embed_dim, deconv_channels[0], 1)
        
        # Deconvolution blocks
        deconv_blocks = []
        in_channels = deconv_channels[0]
        
        for i, out_channels in enumerate(deconv_channels[1:]):
            deconv_blocks.extend([
                nn.ConvTranspose2d(
                    in_channels, out_channels, 
                    kernel_size=4, stride=2, padding=1,
                    output_padding=0, bias=False
                ),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            ])
            in_channels = out_channels
        
        self.deconv_blocks = nn.Sequential(*deconv_blocks)
        
        # Final convolution to get desired output channels
        self.final_conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, out_chans, 3, padding=1),
            nn.Tanh()  # Assuming normalized input in [-1, 1]
        )
        
        # Calculate the upsampling factor needed
        total_upsample_factor = 2 ** (len(deconv_channels) - 1)
        self.required_upsample = patch_size / total_upsample_factor
        
        if self.required_upsample != 1:
            # Add adaptive upsampling if needed
            self.adaptive_upsample = nn.Upsample(
                scale_factor=self.required_upsample, 
                mode='bilinear', 
                align_corners=False
            )
        else:
            self.adaptive_upsample = nn.Identity()
    
    def forward(self, x):
        B, N, C = x.shape
        grid_size = int(math.sqrt(N))
        
        # Reshape to spatial format (B, C, H, W)
        x = x.transpose(1, 2).view(B, C, grid_size, grid_size)
        
        # Initial projection
        x = self.init_proj(x)
        
        # Deconvolution blocks
        x = self.deconv_blocks(x)
        
        # Adaptive upsampling to match patch size if needed
        x = self.adaptive_upsample(x)
        
        # Final convolution
        x = self.final_conv(x)
        
        return x

class FlexiblePatchEmbedding(nn.Module):
    """Patch embedding that handles variable input sizes."""
    
    def __init__(self, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        
        self.proj = nn.Conv2d(
            in_chans, embed_dim, 
            kernel_size=patch_size, stride=patch_size
        )
        
    def forward(self, x):
        B, C, H, W = x.shape
        
        # Calculate dynamic positional embedding
        grid_h = H // self.patch_size
        grid_w = W // self.patch_size
        n_patches = grid_h * grid_w
        
        x = self.proj(x)  # (B, embed_dim, H//patch_size, W//patch_size)
        x = x.flatten(2)  # (B, embed_dim, n_patches)
        x = x.transpose(1, 2)  # (B, n_patches, embed_dim)
        
        return x, grid_h, grid_w

class ViT(nn.Module):
    """Complete Vision Transformer for image-to-image tasks with variable input sizes."""
    
    def __init__(
        self, 
        patch_size=16,
        in_chans=3, 
        out_chans=3,
        embed_dim=768, 
        depth=12, 
        n_heads=12, 
        mlp_ratio=4., 
        qkv_bias=True, 
        drop_rate=0., 
        attn_drop_rate=0.,
        deconv_channels=[512, 256, 128, 64]
    ):
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        
        # Flexible patch embedding
        self.patch_embed = FlexiblePatchEmbedding(patch_size, in_chans, embed_dim)
        
        # Learnable positional embedding (will be interpolated for different sizes)
        self.base_pos_embed = nn.Parameter(torch.zeros(1, 196, embed_dim))  # Base for 224x224
        
        self.pos_drop = nn.Dropout(drop_rate)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(
                dim=embed_dim,
                n_heads=n_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
            )
            for _ in range(depth)
        ])
        
        self.norm = nn.LayerNorm(embed_dim)
        
        # Flexible decoder
        self.decoder = FlexibleDeconvDecoder(
            patch_size=patch_size,
            embed_dim=embed_dim,
            out_chans=out_chans,
            deconv_channels=deconv_channels
        )
        
        # Initialize weights
        nn.init.trunc_normal_(self.base_pos_embed, std=0.02)
        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
            
    def interpolate_pos_encoding(self, x, grid_h, grid_w):
        """Interpolate positional encoding to match the current spatial dimensions."""
        base_seq_len = self.base_pos_embed.shape[1]
        base_grid_size = int(math.sqrt(base_seq_len))
        
        if grid_h == base_grid_size and grid_w == base_grid_size:
            return self.base_pos_embed
            
        pos_embed = self.base_pos_embed.transpose(1, 2).view(
            1, self.embed_dim, base_grid_size, base_grid_size
        )
        
        pos_embed = F.interpolate(
            pos_embed,
            size=(grid_h, grid_w),
            mode='bicubic',
            align_corners=False
        )
        
        pos_embed = pos_embed.flatten(2).transpose(1, 2)
        return pos_embed
            
    def forward(self, x):
        B, C, H, W = x.shape
        assert H % self.patch_size == 0 and W % self.patch_size == 0, \
            f"Input image size ({H}*{W}) must be divisible by patch size ({self.patch_size})"
        
        # Patch embedding
        x, grid_h, grid_w = self.patch_embed(x)  # (B, n_patches, embed_dim)
        
        # Interpolate positional embedding
        pos_embed = self.interpolate_pos_encoding(x, grid_h, grid_w)
        x = x + pos_embed
        x = self.pos_drop(x)
        
        # Transformer blocks
        for block in self.blocks:
            x = block(x)
            
        x = self.norm(x)
        
        # Decoder
        x = self.decoder(x, grid_h, grid_w)
        
        return x

class FlexibleDeconvDecoder(nn.Module):
    """Flexible deconvolution decoder that handles variable input sizes."""
    
    def __init__(
        self, 
        patch_size=16,
        embed_dim=768, 
        out_chans=3,
        deconv_channels=[512, 256, 128, 64]
    ):
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.out_chans = out_chans
        
        # Initial projection
        self.init_proj = nn.Conv2d(embed_dim, deconv_channels[0], 1)
        
        # Deconvolution blocks
        deconv_blocks = []
        in_channels = deconv_channels[0]
        
        for i, out_channels in enumerate(deconv_channels[1:]):
            deconv_blocks.extend([
                nn.ConvTranspose2d(
                    in_channels, out_channels, 
                    kernel_size=4, stride=2, padding=1,
                    bias=False
                ),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            ])
            in_channels = out_channels
        
        self.deconv_blocks = nn.Sequential(*deconv_blocks)
        
        # Calculate total upsampling from deconv layers
        self.deconv_upsample_factor = 2 ** (len(deconv_channels) - 1)
        
        # Final convolution
        self.final_conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, out_chans, 3, padding=1),
            nn.Tanh()
        )
    
    def forward(self, x, grid_h, grid_w):
        B, N, C = x.shape
        assert N == grid_h * grid_w, "Grid dimensions don't match sequence length"
        
        # Reshape to spatial format
        x = x.transpose(1, 2).view(B, C, grid_h, grid_w)
        
        # Initial projection
        x = self.init_proj(x)
        
        # Deconvolution blocks
        x = self.deconv_blocks(x)
        
        # Current size after deconv
        current_h, current_w = x.shape[2], x.shape[3]
        target_h, target_w = grid_h * self.patch_size, grid_w * self.patch_size
        
        # Adaptive upsampling to reach target size
        if current_h != target_h or current_w != target_w:
            x = F.interpolate(
                x, size=(target_h, target_w), 
                mode='bilinear', align_corners=False
            )
        
        # Final convolution
        x = self.final_conv(x)
        
        return x