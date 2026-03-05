from typing import Optional
import torch
import torch.nn as nn
from transformers import AutoModel

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

class Robust_SentimentModel(nn.Module):
    def __init__(
        self,
        model_name="xlm-roberta-base",
        num_sentiment_classes=3,
        num_aspect_labels=5,
        alpha=1.0,
        beta=0.4,
        gamma=0.6,
        pretrained=True
    ):
        super().__init__()


        if pretrained:
            try:
                self.encoder = AutoModel.from_pretrained(model_name)
            except OSError as e:
                if "1455" in str(e) or "paging file" in str(e).lower():
                    print("WARN: Paging file too small (OS Error 1455). Loading base model without safetensors...")
                    self.encoder = AutoModel.from_pretrained(model_name, use_safetensors=False)
                else:
                    raise e
        else:
            from transformers import AutoConfig
            config = AutoConfig.from_pretrained(model_name)
            self.encoder = AutoModel.from_config(config)
            
        hidden = self.encoder.config.hidden_size

        self.aspect_projection = nn.Linear(hidden, hidden)


        self.sentiment_head = nn.Linear(hidden * 2, num_sentiment_classes)

  
        self.sarcasm_head = nn.Linear(hidden, 1)


        self.aspect_head = nn.Linear(hidden, num_aspect_labels)


        self.sentiment_loss = nn.CrossEntropyLoss()
        self.sarcasm_loss = nn.BCEWithLogitsLoss()
        self.aspect_loss = nn.CrossEntropyLoss(ignore_index=-100)

        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

        self.dropout = nn.Dropout(0.2)

    def forward(self, batch):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]

        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )


        token_embeddings = outputs.last_hidden_state      
        cls_embedding = token_embeddings[:, 0, :]         


        absa_token_features = self.aspect_projection(token_embeddings)  
        aspect_logits = self.aspect_head(absa_token_features)       


        absa_pooled = absa_token_features.mean(dim=1)    


        sentiment_input = torch.cat(
            [cls_embedding, absa_pooled],
            dim=1
        )                                           

        sentiment_logits = self.sentiment_head(
            self.dropout(sentiment_input)
        )


        sarcasm_logits = self.sarcasm_head(
            self.dropout(cls_embedding)
        ).squeeze(-1)

 
        loss = None
        if "sentiment_labels" in batch:
            loss_sent = self.sentiment_loss(
                sentiment_logits,
                batch["sentiment_labels"]
            )

            loss_sarc = torch.tensor(0.0, device=loss_sent.device)
            if batch["sarcasm_labels"] is not None:
                loss_sarc = self.sarcasm_loss(
                    sarcasm_logits,
                    batch["sarcasm_labels"].float()
                )

            loss_asp = torch.tensor(0.0, device=loss_sent.device)
            if (batch["aspect_labels"] != -100).any():
                loss_asp = self.aspect_loss(
                    aspect_logits.view(-1, aspect_logits.size(-1)),
                    batch["aspect_labels"].view(-1)
                )

            loss = (
                self.alpha * loss_sent +
                self.beta * loss_sarc +
                self.gamma * loss_asp
            )

        return {
            "loss": loss,
            "sentiment_logits": sentiment_logits,  
            "sarcasm_logits": sarcasm_logits,     
            "aspect_logits": aspect_logits        
        }




