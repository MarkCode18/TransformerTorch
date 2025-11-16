import torch, tokenizers
import torch.nn as nn
from .config import HPARAMS
from .model import TransformerNMT

@torch.no_grad()
def greedy_decode(model, src_ids, pad_id, bos_id, eos_id, max_len, device):
    """
    Greedy decoding for Transformer model: computes encoder memory once, then
    iteratively generates target tokens using prior decoder outputs and memory.
    Supports batched inference, stops at EOS or max_len, and builds its own
    padding and causal masks.
    """
    batch_size = src_ids.size(0)
    model.eval()
    src_ids = src_ids.to(device)
    src_key_padding_mask = (src_ids == pad_id).to(device) # (N, S)

    # compute encoder memory
    src_emb = model.positional_embedding(model.shared_embedding(src_ids)) # (N, S, E)
    memory = model.transformer.encoder(src = src_emb,
                                       src_key_padding_mask = src_key_padding_mask) # (N, S, E)

    # prepare initial decoder input
    current_tokens = torch.full((batch_size, 1), bos_id, dtype=torch.long).to(device) # (N, 1)
    finished = torch.zeros(batch_size, dtype=torch.bool).to(device)
    outputs = [[] for _ in range(batch_size)]

    # decoding
    for step in range(max_len):
        # target embedding & masks (causal/padding)
        tgt_emb = model.positional_embedding(model.shared_embedding(current_tokens)).to(device) # (N, L, E)
        tgt_key_padding_mask = (current_tokens == pad_id).to(device) # usually false (N ,L)
        causal_mask = nn.Transformer.generate_square_subsequent_mask(tgt_emb.size(1), dtype=torch.bool).to(device) # (L, L)

        # decoder outputs
        decoder_outputs = model.transformer.decoder(tgt = tgt_emb, memory = memory, tgt_mask = causal_mask, 
                                                    tgt_key_padding_mask = tgt_key_padding_mask,
                                                    memory_key_padding_mask = src_key_padding_mask) # (N, L, E)

        next_logits = model.output(decoder_outputs)[:, -1, :] # (N, vocab_size)
        next_tokens = next_logits.argmax(dim=-1) # (N,)

        # update current decoded tokens
        current_tokens = torch.cat([current_tokens, next_tokens.unsqueeze(1)], dim=1) # (N, L+1)

        # store output tokens & stop if EOS token found
        for i in range(batch_size):
            if not finished[i]:
                outputs[i].append(int(next_tokens[i].item()))
                if next_tokens[i] == eos_id:
                    finished[i] = True

        if finished.all():
            break

    return outputs

def translate(model, tokenizer, src_list, max_len=64, device=None):
    """
    args:
        src_list (List[str]): Source sentences to translate.
        max_len (int): maximum length of generated output sequence.
        device (torch.device, optional)
    returns:
        List[str]: translated target sentences.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    pad_id, bos_id, eos_id = [tokenizer.token_to_id(i) for i in ["[PAD]", "[BOS]", "[EOS]"]]    
    src_ids = torch.tensor([enc.ids for enc in tokenizer.encode_batch(src_list)],
                               dtype=torch.long) # (N, S)

    outputs = greedy_decode(model, src_ids, pad_id, bos_id, eos_id, max_len, device)
    return tokenizer.decode_batch(outputs)

def load_model_and_tokenizer(tokenizer_path, model_checkpoint_path, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Torch Device: {device}")
    hp = HPARAMS()

    try:
        tokenizer = tokenizers.Tokenizer.from_file(tokenizer_path)
        tokenizer.enable_truncation(hp.max_seq_len)
        tokenizer.enable_padding(pad_id = 0, pad_token = "[PAD]")
        
        model = TransformerNMT(tokenizer.get_vocab_size(), hp.max_seq_len, **hp.model_hparams).to(device)
        state_dict = torch.load(model_checkpoint_path, map_location=device, weights_only=True)
        model.load_state_dict(state_dict)
        return model, tokenizer
    except Exception as e:
        print(f"Error loading model/tokenizer: {e}")
        return None, None