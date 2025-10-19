"""
Quick Start Example for DeBERTa-CRF-VotIE Model

This example shows how to extract voting information from Portuguese text.
Simply copy-paste this code and replace the text with your own!
"""

from transformers import AutoTokenizer, AutoModel

# Load model and tokenizer
model_name = "Anonymous3445/DeBERTa-CRF-VotIE"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModel.from_pretrained(model_name, trust_remote_code=True)

# Example text - replace this with your own!
text = "3. APROVAÇÃO DE ATA\nPelo Senhor Presidente foram presentes a reunião a ata n.º 7, de 22.12.2021. Ponderado e analisado o assunto o Executivo Municipal deliberou por unanimidade aprovar a ata n.º 7, de 22.12.2021. 4. APROVAÇÃO DE RELATÓRIO FINAL DE ANÁLISE DE PROPOSTAS APRESENTADAS AO CONCURSO PÚBLICO PARA ADJUDICAÇÃO DE \"AQUISIÇÃO DE SERVIÇOS DE SEGUROS\" (RATIFICAÇÃO)\nPelo Senhor Presidente foi presente a reunião o Relatório final de análise de propostas apresentadas ao concurso público para adjudicação de \"Aquisição de serviços de seguros\" que se anexa à presente ata. O Sr. Presidente explicou que houve cinco concorrentes, dos quais três foram excluídos e das duas propostas consideradas ficou em primeiro lugar a Generali Seguros S.A., com um valor proposto de 164.761.88€. Está a falar-se da contratação de todos os seguros da Câmara por dois anos. Como gestora do contrato nomeia-se a funcionária Florbela Galhetas. Ponderado e analisado o assunto o Executivo Municipal deliberou por unanimidade ratificar o relatório final de análise de propostas apresentadas ao concurso público para adjudicação de \"aquisição de serviços de seguros\""

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
