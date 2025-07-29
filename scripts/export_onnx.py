# export_onnx.py
from transformers import AutoModelForTokenClassification
import torch
from pathlib import Path

def main():
    model_dir = Path("model")
    model = AutoModelForTokenClassification.from_pretrained(model_dir)
    dummy = torch.randint(0, 1000, (1,16), dtype=torch.int64)
    torch.onnx.export(
        model, (dummy,),
        str(model_dir / "model.onnx"),
        input_names = ["input_ids"],
        output_names = ["logits"],
        opset_version = 14,  # повысили номер opset
        dynamic_axes = {
            'input_ids': {1: 'seq_len'},
            'logits': {1: 'seq_len'}
        }
    )
    print("✔ ONNX model saved to", model_dir/"model.onnx")

if __name__ == "__main__":
    main()
