# Created By LORD 
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.Wq = nn.Linear(embed_dim, embed_dim)
        self.Wk = nn.Linear(embed_dim, embed_dim)
        self.Wv = nn.Linear(embed_dim, embed_dim)
        self.fc_out = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        batch_size, seq_len, embed_dim = x.size()

        Q = self.Wq(x)   
        K = self.Wk(x)
        V = self.Wv(x)

        # Split into heads
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        weights = F.softmax(scores, dim=-1)

        context = torch.matmul(weights, V)


        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, embed_dim)


        out = self.fc_out(context)
        return out


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_hidden_dim, dropout=0.1):
        super().__init__()
        self.attn = MultiHeadSelfAttention(embed_dim, num_heads)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, ff_hidden_dim),
            nn.ReLU(),
            nn.Linear(ff_hidden_dim, embed_dim),
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
 
        attn_out = self.attn(x)
        print(attn_out.shape)

        x = self.norm1(x + self.dropout(attn_out))


        ff_out = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_out))
        return x

class TransformerTextClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, num_heads=4, ff_hidden_dim=256,
                 num_classes=3, max_len=20, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.pos_embedding = nn.Embedding(max_len, embed_dim)
        self.transformer_block = TransformerBlock(embed_dim, num_heads, ff_hidden_dim, dropout)
        self.fc_out = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )
        self.softmax = nn.Softmax()

    def forward(self, x):
        batch_size, seq_len = x.size()
        positions = torch.arange(0, seq_len).unsqueeze(0).to(x.device)
        x = self.embedding(x) + self.pos_embedding(positions)
        x = self.transformer_block(x)
        x = x.mean(dim=1)  
        out = self.fc_out(x)
        # out = self.softmax(out)
        return out

 
