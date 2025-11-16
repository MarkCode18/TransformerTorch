import gradio as gr
from src.inference import load_model_and_tokenizer, translate
from src.ui import build_demo

tokenizer_path = "tokenizer/bpe_tokenizer.json"
model_checkpoint_path = "model/transformer_nmt_model_params.pt"

model, tokenizer = load_model_and_tokenizer(tokenizer_path , model_checkpoint_path)

def translate_fn(src_text, max_len):
    return translate(model, tokenizer, [src_text], max_len=max_len, device=None)[0]

inputs = [
    gr.Textbox(label="📝 English Text", lines=3),
    gr.Slider(10, 100, value=50, step=5, label="📏 Max Translated Length"),
]

outputs = [gr.Textbox(label="🌎 Spanish Translation", lines=5, interactive=False)]

demo = build_demo(
    translate_fn,
    inputs,
    outputs,
    english_title = "# 🌐✨ TransformerTorch: Transformer-Based Neural Machine Translation 🚀",
    persian_title = "# 🌐✨ مترجم هوشمند انگلیسی به اسپانیایی مبتنی بر معماری ترنسفورمر 🚀",
    assets_dir = "assets",
    app_title = "🌐 TransformerTorch 🌟"
)

demo.launch()
