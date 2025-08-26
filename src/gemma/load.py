import torch
import transformers
from transformers import AutoProcessor, Gemma3ForConditionalGeneration


def load_model(model_name: str):
    transformers.utils.logging.set_verbosity_error()
    torch.set_float32_matmul_precision('high') 
    # Set float32 matmul precision for better performance with Gemma models

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = Gemma3ForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.float32,  # Requires float32 for loss computation
        attn_implementation="eager" # to get attention weights
    ).to(device)

    processor = AutoProcessor.from_pretrained(
        model_name,
        padding_side="left"
    )
    
    return model, processor, device