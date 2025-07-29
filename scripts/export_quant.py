# export_quant.py
from transformers import AutoModelForTokenClassification
import torch
from pathlib import Path

def main():
    model_dir = Path("model")
    # Загружаем float32-модель
    model = AutoModelForTokenClassification.from_pretrained(model_dir)
    # Динамическое квантование Linear-слоёв
    from torch.quantization import quantize_dynamic
    model = quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
    # Сохраняем только state_dict квантованной модели
    torch.save(model, model_dir/"model_quantized.pt")
    print("Quantized weights saved to", model_dir / "model_quantized.pt")

if __name__ == "__main__":
    main()
