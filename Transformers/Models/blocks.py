from typing import Optional
import torch
import torch.nn as nn

PAD_ID = 0  


class SimpleTransformerClassifier(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        num_classes: int,
        max_len: int = 128,
        embed_dim: int = 128,
        num_heads: int = 4,
        depth: int = 2,
        ff_dim: int = 256,
        dropout: float = 0.1,
        pooling: str = "cls",  # cls or mean
    ) -> None:
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        assert pooling in {"cls", "mean"}

        self.max_len = max_len
        self.pooling = pooling

        self.token_embed = nn.Embedding(vocab_size, embed_dim, padding_idx=PAD_ID)
        self.pos_embed = nn.Embedding(max_len, embed_dim)
        self.dropout = nn.Dropout(dropout)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=depth)

        self.norm = nn.LayerNorm(embed_dim)
        self.cls_head = nn.Linear(embed_dim, num_classes)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        nn.init.trunc_normal_(self.token_embed.weight, std=0.02)
        nn.init.trunc_normal_(self.pos_embed.weight, std=0.02)

    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None):

        B, L = input_ids.shape
        if L > self.max_len:
            input_ids = input_ids[:, : self.max_len]
            if attention_mask is not None:
                attention_mask = attention_mask[:, : self.max_len]
            L = self.max_len


        pos = torch.arange(L, device=input_ids.device).unsqueeze(0).expand(B, L)
        x = self.token_embed(input_ids) + self.pos_embed(pos)

  
        if self.pooling == "cls":
            cls_tok = self.cls_token.expand(B, -1, -1) 
            x = torch.cat([cls_tok, x], dim=1)  
            if attention_mask is not None:
                attention_mask = torch.cat([torch.ones(B, 1, device=input_ids.device, dtype=attention_mask.dtype), attention_mask], dim=1)

        x = self.dropout(x)

        if attention_mask is None:

            pad_mask = (input_ids == PAD_ID)
            if self.pooling == "cls":
                pad_mask = torch.cat([torch.zeros(B, 1, device=input_ids.device, dtype=torch.bool), pad_mask], dim=1)
        else:
            pad_mask = (attention_mask == 0)

        x = self.encoder(x, src_key_padding_mask=pad_mask)

        if self.pooling == "cls":
            h = x[:, 0, :]
        else:

            if attention_mask is None:
                mask = ~pad_mask  
            else:
                mask = attention_mask.bool()
            if self.pooling == "mean" and x.size(1) != mask.size(1):
                x = x[:, 1:, :]
                mask = mask[:, 1:]
            denom = mask.sum(dim=1, keepdim=True).clamp(min=1)
            h = (x * mask.unsqueeze(-1)).sum(dim=1) / denom

        h = self.norm(h)
        logits = self.cls_head(h)
        return logits


