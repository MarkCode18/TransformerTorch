import torch
import torch.nn as nn

class PositionalEmbedding(nn.Module):
    """
    Positional Embedding

    shapes:
        N: batch size
        L: seq len (max seq len of batch)
        E: embedding dim
        max_seq_len: max seq len across all samples

    forward args:
        X: batch of semantic embeddings (N, L, E)
    """
    def __init__(self, emb_dim, max_seq_len, dropout_p=0.1):
        super().__init__()

        # full embedding matrix with shape (maximum_sample_lenght, embedding_dim)
        self.pos_embedding = nn.Parameter(torch.randn(max_seq_len, emb_dim) * 0.01)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, X):

        # sliced for current batch max sequence lenght
        emb_matrix = self.pos_embedding[:X.size(1)].unsqueeze(0)  # (1, L, E)         
        return self.dropout(X + emb_matrix) # (N, L, E) 
class TransformerNMT(nn.Module):
    """
    forward args:
        src_ids: (N, S) token ids
        tgt_ids: (N, L) token ids
        src_key_padding_mask: (N, S) bool, True=PAD (ignored)
        tgt_key_padding_mask: (N, L) bool, True=PAD (ignored)
    """
    def __init__(self, vocab_size, max_seq_len, d_model=512, nhead=4,
                 num_encoder_layers=2, num_decoder_layers=2,
                 dim_feedforward=2048, dropout=0.1, padding_idx=0):
        super().__init__()

        self.shared_embedding = nn.Embedding(vocab_size, d_model, padding_idx = padding_idx)
        self.positional_embedding = PositionalEmbedding(d_model, max_seq_len)

        self.transformer = nn.Transformer(d_model, nhead,
                                          num_encoder_layers, num_decoder_layers,
                                          dim_feedforward, dropout,
                                          activation="relu", batch_first=True, 
                                          norm_first=False, bias=True)

        self.output = nn.Linear(d_model, vocab_size, bias=False)

        # weight tying
        self.output.weight = self.shared_embedding.weight

    def forward(self, src_ids, tgt_ids, src_key_padding_mask, tgt_key_padding_mask):

        src = self.positional_embedding(self.shared_embedding(src_ids)) # (N, S, E)
        tgt = self.positional_embedding(self.shared_embedding(tgt_ids)) # (N, L, E)

        # create target causal mask
        L = tgt.size(1)
        causal_mask = nn.Transformer.generate_square_subsequent_mask(L, dtype=torch.bool, device = tgt.device)

        out = self.transformer(src = src , tgt = tgt,
                               src_key_padding_mask = src_key_padding_mask,
                               tgt_key_padding_mask = tgt_key_padding_mask,
                               memory_key_padding_mask = src_key_padding_mask,
                               tgt_mask = causal_mask
                              ) # (N, L, E)

        return self.output(out).transpose(-2,-1) # (N, vocab_size, L)
