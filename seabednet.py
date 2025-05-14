'''
Initial Pytorch Implementation: Panagiotis Agrafiotis (https://github.com/pagraf/Seabed-Net)
Email: agrafiotis.panagiotis@gmail.com

Description: Seabed-Net is a novel deep learning architecture that jointly estimates
bathymetric depth and performs pixel-level seabed classification from satellite and aerial remote
sensing imagery. Unlike prior models that treat these tasks independently or use one as an auxiliary,
Seabed-Net employs a multi-task framework, where both outputs are supervised and contribute to
shared representation learning. The architecture integrates spatially adaptive attention (FAA) and
Vision Transformer (ViT) components, enabling it to capture both local and global features across
diverse sensing modalities.

If you use this code please cite our paper: " "



Attribution-NonCommercial-ShareAlike 4.0 International License

Copyright (c) 2025 Panagiotis Agrafiotis

This license requires that reusers give credit to the creator. It allows reusers 
to distribute, remix, adapt, and build upon the material in any medium or format,
for noncommercial purposes only. If others modify or adapt the material, they 
must license the modified material under identical terms.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.


This work is part of MagicBathy project funded by the European Union’s HORIZON Europe research and innovation 
programme under the Marie Skłodowska-Curie GA 101063294. Work has been carried out at the Remote Sensing Image 
Analysis group. For more information about the project visit https://www.magicbathy.eu/.
'''


import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

class DoubleConv(nn.Module):
    """(conv => BN? => ReLU) * 2"""
    def __init__(self, in_channels, out_channels, batch_norm=True):
        super(DoubleConv, self).__init__()
        if batch_norm:
            self.double_conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
        else:
            self.double_conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.ReLU(inplace=True)
            )
    
    def forward(self, x):
        return self.double_conv(x)

class Encoder(nn.Module):
    """Encoder with optional batch norm"""
    def __init__(self, in_channels, features, batch_norm=True):
        super(Encoder, self).__init__()
        self.enc1 = DoubleConv(in_channels, features, batch_norm=batch_norm)
        self.enc2 = DoubleConv(features, features * 2, batch_norm=batch_norm)
        self.enc3 = DoubleConv(features * 2, features * 4, batch_norm=batch_norm)
        self.enc4 = DoubleConv(features * 4, features * 8, batch_norm=batch_norm)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        return e1, e2, e3, e4




class Up(nn.Module):
    def __init__(self, in_channels, out_channels, batch_norm=True):
        super(Up, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels, batch_norm=batch_norm)

    def forward(self, x1, x2):
        #Upsample x1
        x1 = self.up(x1)
        
        #Calculate the difference in shape and pad x1 if needed to match x2
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        
        #Concatenate along the channel dimension
        x = torch.cat([x2, x1], dim=1)
        
        #Apply double convolution
        return self.conv(x)

class Decoder(nn.Module):
    """New decoder structure with sequential Up blocks and optional batch normalization"""
    def __init__(self, features, out_channels, batch_norm=True):
        super(Decoder, self).__init__()
        
        #Define the Up blocks with batch_norm flag passed down
        self.decoder = nn.ModuleList([
            Up(features * 8, features * 4, batch_norm=batch_norm),
            Up(features * 4, features * 2, batch_norm=batch_norm),
            Up(features * 2, features, batch_norm=batch_norm)
        ])
        
        #Final output convolution
        self.out_conv = nn.Conv2d(features, out_channels, kernel_size=1)

    def forward(self, e1, e2, e3, e4):
        #Sequentially apply each Up block
        x5 = self.decoder[0](e4, e3)
        x6 = self.decoder[1](x5, e2)
        x7 = self.decoder[2](x6, e1)
        
        #Apply the final 1x1 convolution for output
        return self.out_conv(x7)


class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(SEBlock, self).__init__()
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(in_channels // reduction, in_channels, kernel_size=1, bias=False)
        self.sigmoid = nn.Sigmoid()
        

    def forward(self, x):
        se = self.global_avg_pool(x)  # Squeeze: Global Average Pooling
        se = self.fc1(se)  # Reduction
        se = self.relu(se)
        se = self.fc2(se)  # Re-weighting
        se = self.sigmoid(se)
        return x * se  # Reweighting input feature maps by channel-wise attention



class AttentionFeatureFusion(nn.Module):
    def __init__(self, features, reduction=16):
        super(AttentionFeatureFusion, self).__init__()
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(features * 2, 1, kernel_size=7, padding=3),
            nn.Sigmoid()
        )
        self.channel_attention = SEBlock(features * 2, reduction=reduction)
        self.conv = nn.Conv2d(features * 2, features, kernel_size=1)
        

    def forward(self, x1, x2):
    	fusion = torch.cat([x1, x2], dim=1)
    	residual = fusion.clone()  # Clone for residual
    	spatial_att = self.spatial_attention(fusion)
    	fusion = fusion * spatial_att
    	channel_att = self.channel_attention(fusion)
    	fusion = fusion * channel_att
    	fusion = fusion + residual  # Add residual connection
    	return self.conv(fusion)

        


class PatchEmbedding(nn.Module):
    def __init__(self, in_channels, embed_dim, patch_size):
        super(PatchEmbedding, self).__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)
        x = rearrange(x, 'b c h w -> b (h w) c')
        return x


class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads, dropout=0.1):
        super(CrossAttention, self).__init__()
        self.num_heads = num_heads
        self.dim = dim
        
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.proj = nn.Linear(dim, dim)
        
        self.attn_drop = nn.Dropout(dropout)
        self.proj_drop = nn.Dropout(dropout)
        self.scale = nn.Parameter(torch.ones(1) * (dim // num_heads) ** -0.5)

    def forward(self, query, key, value):
        B, N, C = query.shape
        q = self.q(query).reshape(B, N, self.num_heads, C // self.num_heads).transpose(1, 2)
        k = self.k(key).reshape(B, -1, self.num_heads, C // self.num_heads).transpose(1, 2)
        v = self.v(value).reshape(B, -1, self.num_heads, C // self.num_heads).transpose(1, 2)
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class WindowAttention(nn.Module):
    def __init__(self, dim, num_heads, window_size, dropout=0.1):
        super(WindowAttention, self).__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size

        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(dropout)
        self.proj_drop = nn.Dropout(dropout)

        self.scale = nn.Parameter(torch.ones(1) * (dim // num_heads) ** -0.5)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SwinTransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, window_size=16, shift_size=0, mlp_ratio=4., dropout=0.1):
        super(SwinTransformerBlock, self).__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        
        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, num_heads, window_size, dropout=dropout)
        self.cross_attn = CrossAttention(dim, num_heads, dropout=dropout)
        self.norm2 = nn.LayerNorm(dim)
        
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, H, W, cross_input=None):
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)
        attn_windows = self.attn(x_windows)
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)

        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x

        x = x.view(B, H * W, C)
        x = shortcut + x

        if cross_input is not None:
            x = self.cross_attn(query=x, key=cross_input, value=cross_input)

        x = self.norm2(x)
        mlp_output = self.mlp(x)
        x = x + mlp_output

        return x


def window_partition(x, window_size):
    B, H, W, C = x.shape
    pad_h = (window_size - H % window_size) % window_size
    pad_w = (window_size - W % window_size) % window_size
    x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    pad_h = (window_size - H % window_size) % window_size
    pad_w = (window_size - W % window_size) % window_size
    H_padded = H + pad_h
    W_padded = W + pad_w

    B = int(windows.shape[0] / (H_padded * W_padded / window_size / window_size))
    x = windows.view(B, H_padded // window_size, W_padded // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H_padded, W_padded, -1)
    x = x[:, :H, :W, :].contiguous()
    return x



class VisionTransformer(nn.Module):
    def __init__(self, in_channels, embed_dim, depth, num_heads, window_size=32, mlp_ratio=4., dropout=0.1): 
        super(VisionTransformer, self).__init__()
        self.patch_embed = PatchEmbedding(in_channels, embed_dim, patch_size=32) 
        self.pos_embed = nn.Parameter(torch.zeros(1, 1024, embed_dim)) 
        self.pos_drop = nn.Dropout(p=dropout)
        self.transformer_blocks = nn.ModuleList([
            SwinTransformerBlock(embed_dim, num_heads, window_size=window_size, shift_size=0 if (i % 2 == 0) else window_size // 2, mlp_ratio=mlp_ratio, dropout=dropout) for i in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.patch_embed(x)

        num_patches = (H // self.patch_embed.patch_size) * (W // self.patch_embed.patch_size)

        #Ensure positional embedding size matches number of patches
        if self.pos_embed.shape[1] != num_patches:
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, x.shape[-1]).to(x.device))

        x = x + self.pos_embed
        x = self.pos_drop(x)

        for block in self.transformer_blocks:
            x = block(x, H // self.patch_embed.patch_size, W // self.patch_embed.patch_size)

        x = self.norm(x)
        x = rearrange(x, 'b (h w) c -> b c h w', h=H // self.patch_embed.patch_size, w=W // self.patch_embed.patch_size)
        return x


class ViTFusion(nn.Module):
    def __init__(self, in_features, embed_dim, depth, num_heads, window_size=32, mlp_ratio=4., dropout=0.1):
        super(ViTFusion, self).__init__()
        self.conv = nn.Conv2d(in_features, embed_dim, kernel_size=1)
        self.vit = VisionTransformer(
            in_channels=embed_dim,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            window_size=window_size,
            mlp_ratio=mlp_ratio,
            dropout=dropout
        )
        self.final_conv = nn.Conv2d(embed_dim, in_features // 2, kernel_size=1)

    def forward(self, x1, x2):
        #Concatenate features along the channel dimension
        fusion = torch.cat([x1, x2], dim=1)
        
        #Reduce dimension before ViT
        fusion = self.conv(fusion)
        
        #Pass through ViT
        fusion = self.vit(fusion)
        
        #Restore channel dimensions
        fusion = self.final_conv(fusion)
        return fusion


              
class SeabedNet(nn.Module):
    def __init__(self, in_channels, features, bathymetry_out_channels, segmentation_out_channels, batch_norm=True):
        super(SeabedNet, self).__init__()

        #Encoder for bathymetry (without batch norm)
        self.bathy_encoder = Encoder(in_channels, features, batch_norm=False)

        #Encoder for segmentation (with batch norm)
        self.seg_encoder = Encoder(in_channels, features, batch_norm=True)

        #Attention-based feature fusion
        self.fusion1b = AttentionFeatureFusion(features)
        self.fusion2b = AttentionFeatureFusion(features * 2)
        self.fusion3b = AttentionFeatureFusion(features * 4)
        self.fusion4b = AttentionFeatureFusion(features * 8)
        
        #ViT-based feature fusion
        self.fusion1 = ViTFusion(features * 2, embed_dim=features, depth=1, num_heads=4)
        self.fusion2 = ViTFusion(features * 4, embed_dim=features * 2, depth=1, num_heads=4)
        self.fusion3 = ViTFusion(features * 8, embed_dim=features * 4, depth=1, num_heads=4)
        self.fusion4 = ViTFusion(features * 16, embed_dim=features * 8, depth=1, num_heads=4)

        #Decoders for both tasks
        self.bathy_decoder = Decoder(features, bathymetry_out_channels, batch_norm=False)
        self.seg_decoder = Decoder(features, segmentation_out_channels, batch_norm=True)

    def forward(self, x):
        #Bathymetry encoding
        bathy_e1, bathy_e2, bathy_e3, bathy_e4 = self.bathy_encoder(x)
        
        #Segmentation encoding
        seg_e1, seg_e2, seg_e3, seg_e4 = self.seg_encoder(x)

        #Feature fusion between tasks
        f1 = self.fusion1(bathy_e1, seg_e1)
        f2 = self.fusion2(bathy_e2, seg_e2)
        f3 = self.fusion3(bathy_e3, seg_e3)
        f4 = self.fusion4(bathy_e4, seg_e4)
        
        f1b = self.fusion1b(bathy_e1, seg_e1)
        f2b = self.fusion2b(bathy_e2, seg_e2)
        f3b = self.fusion3b(bathy_e3, seg_e3)
        f4b = self.fusion4b(bathy_e4, seg_e4)
       
        
        f1 = F.interpolate(f1, size=seg_e1.shape[2:])
        f2 = F.interpolate(f2, size=seg_e2.shape[2:])
        f3 = F.interpolate(f3, size=seg_e3.shape[2:])
        f4 = F.interpolate(f4, size=seg_e4.shape[2:])
        
        f1b = F.interpolate(f1b, size=bathy_e1.shape[2:])
        f2b = F.interpolate(f2b, size=bathy_e2.shape[2:])
        f3b = F.interpolate(f3b, size=bathy_e3.shape[2:])
        f4b = F.interpolate(f4b, size=bathy_e4.shape[2:])
        

        bathy_out = self.bathy_decoder(f1b + bathy_e1, f2b + bathy_e2, f3b + bathy_e3, f4b + bathy_e4) 
        seg_out = self.seg_decoder(f1 + seg_e1, f2 + seg_e2, f3 + seg_e3, f4 + seg_e4)



        return bathy_out, seg_out
        