"""
Quick Start Example for XLM-RoBERTa-CRF-VotIE Model

This example shows how to extract voting information from Portuguese text.
Simply copy-paste this code and replace the text with your own!
"""

from transformers import AutoTokenizer, AutoModel

# Load model and tokenizer
model_name = "Anonymous3445/XLM-RoBERTa-CRF-VotIE"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModel.from_pretrained(model_name, trust_remote_code=True)

# Example text - replace this with your own!
text = """A CÂMARA EM FACE DO AUTO DE VISTORIA ELABORADO PELA COMISSÃO DE VISTORIAS, DELIBEROU, POR TODOS OS MEMBROS PRESENTES, QUE O REFERIDO PRÉDIO E AS SUAS FRAÇÕES AUTÓNOMAS, AS QUAIS SÃO DISTINTAS E ISOLADAS ENTRE SI E COM SAÍDA PRÓPRIA PARA A VIA PÚBLICA, REÚNEM OS REQUISITOS LEGAIS PARA NELE SER INSTITUÍDO O REGIME DE PROPRIEDADE HORIZONTAL."""

# Tokenize
inputs = tokenizer(text, return_tensors="pt")

# Get predictions (automatically returns word-level results!)
predictions = model.decode(**inputs, tokenizer=tokenizer, text=text)

# Print results
print(f"Text: '{text}'\n")
print(f"{'Word':<30} Label")
print("-" * 50)
for pred in predictions:
    print(f"{pred['word']:<30} {pred['label']}")

# Optional: Get predictions with character positions
print("\n" + "="*50)
print("With character offsets:")
print("="*50 + "\n")
predictions_with_offsets = model.decode(**inputs, tokenizer=tokenizer, text=text, return_offsets=True)
for pred in predictions_with_offsets:
    print(f"{pred['word']:<20} {pred['label']:<25} (chars {pred['start']}-{pred['end']})")
