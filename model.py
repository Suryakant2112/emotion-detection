import torch
import torch.nn as nn
import torch.nn.functional as F

class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim, head_size, num_heads, ff_dim, dropout=0.3):
        super().__init__()
        bottleneck_dim = head_size * num_heads 
        self.norm1 = nn.LayerNorm(embed_dim, eps=1e-6)
        
        self.mha_proj_in = nn.Linear(embed_dim, bottleneck_dim)
        self.mha = nn.MultiheadAttention(embed_dim=bottleneck_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.mha_proj_out = nn.Linear(bottleneck_dim, embed_dim)
        
        self.norm2 = nn.LayerNorm(embed_dim, eps=1e-6)
        self.ff1 = nn.Linear(embed_dim, ff_dim)
        self.dropout = nn.Dropout(dropout)
        self.ff2 = nn.Linear(ff_dim, embed_dim)

    def forward(self, x):
        normed_x = self.norm1(x)
        qkv = self.mha_proj_in(normed_x)
        attn_out, _ = self.mha(qkv, qkv, qkv)
        x = x + self.mha_proj_out(attn_out)

        normed_y = self.norm2(x)
        y = F.gelu(self.ff1(normed_y))
        y = self.dropout(y)
        y = self.ff2(y)
        return x + y

class CrossAttentionFusion(nn.Module):
    def __init__(self, embed_dim, num_heads=4, dropout=0.3):
        super().__init__()
        self.cross_mha = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)
        
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, embed_dim)
        )
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, query, key_value):
        attn_out, _ = self.cross_mha(query, key_value, key_value)
        x = self.norm1(query + attn_out)
        ff_out = self.ff(x)
        x = self.norm2(x + ff_out)
        return x

class Branch(nn.Module):
    def __init__(self, is_eye=False, shared_embed_dim=256):
        super().__init__()
        in_channels = 1 if is_eye else 5
        spatial_dim = 31 if is_eye else 62 
        
        # Calculate the flattened dimension size for each timestep
        self.raw_feature_dim = in_channels * spatial_dim
        
        # Linearly project the raw flattened features directly to the embedding dimension
        self.proj = nn.Linear(self.raw_feature_dim, shared_embed_dim)

        self.tf1 = TransformerEncoder(shared_embed_dim, head_size=64, num_heads=4, ff_dim=512, dropout=0.3)
        self.tf2 = TransformerEncoder(shared_embed_dim, head_size=64, num_heads=4, ff_dim=512, dropout=0.3)

    def forward(self, x):
        # Input shape expected: (Batch, Channels, Spatial, Temporal)
        b, c, s, t = x.size()
        
        # 1. Permute to make Temporal the sequence dimension: (Batch, Temporal, Channels, Spatial)
        x = x.permute(0, 3, 1, 2).contiguous()
        
        # 2. Flatten Channels and Spatial dims: (Batch, Temporal, Channels * Spatial)
        x = x.view(b, t, c * s)
        
        # 3. Project to the target embedding dimension
        x = self.proj(x)
        
        # 4. Feed perfectly formatted sequences into the Transformers
        x = self.tf1(x)
        x = self.tf2(x)
        
        return x

class MultimodalModel(nn.Module):
    def __init__(self, embed_dim=256):
        super().__init__()
        self.eeg_branch = Branch(is_eye=False, shared_embed_dim=embed_dim)
        self.eye_branch = Branch(is_eye=True, shared_embed_dim=embed_dim)

        self.eeg_cross_eye = CrossAttentionFusion(embed_dim)
        self.eye_cross_eeg = CrossAttentionFusion(embed_dim)

        self.fc1 = nn.Linear(embed_dim * 2, 128)
        self.dropout = nn.Dropout(0.4)
        self.fc2 = nn.Linear(128, 64)
        self.out = nn.Linear(64, 4)

    def forward(self, eeg, eye):
        eeg_seq = self.eeg_branch(eeg) 
        eye_seq = self.eye_branch(eye) 

        eeg_fused = self.eeg_cross_eye(query=eeg_seq, key_value=eye_seq)
        eye_fused = self.eye_cross_eeg(query=eye_seq, key_value=eeg_seq)

        eeg_pooled = torch.mean(eeg_fused, dim=1)
        eye_pooled = torch.mean(eye_fused, dim=1)

        fused = torch.cat((eeg_pooled, eye_pooled), dim=1)
        
        x = F.relu(self.fc1(fused))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        return self.out(x)