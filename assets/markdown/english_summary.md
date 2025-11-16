**🌐 TransformerTorch** is a Transformer‑based Neural Machine Translation system trained on **220K English–Spanish sentence pairs**. It leverages advanced techniques to improve efficiency, including **Mixed Precision training, Weight Tying, a shared vocabulary and embedding space, and BPE tokenization**. The model uses a custom **greedy decoder** that computes the encoder memory once and then decodes autoregressively with causal and padding masks, reusing that memory at each step for efficient inference.

Originally, for this project I implemented the Transformer architecture from scratch with PyTorch, which you can explore here: [GitHub – TransformerTorch](https://github.com/HooM4N/TransformerTorch)  

**✍️ Improve Translation Quality**  
To improve translation quality, include proper punctuation in the English source text:  
- End **declarative sentences** with a period (`.`)  
- End **questions** with a question mark (`?`)  
- Use **exclamation marks** (`!`) where appropriate  