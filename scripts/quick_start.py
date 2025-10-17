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
text = "\n\nConsolidação de Mobilidade\n\nPresente proposta do Senhor Presidente da Câmara constante da plataforma de gestão documental SigmaDoc Web/NIPG com o n.º ***********************, que se transcreve:\n\n“O artigo 99.º-A da Lei Geral de Trabalho em Funções Públicas aprovada em anexo à Lei n.º 35/2014, de 20 de junho, na sua atual redação a seguir designada por (LTFP), estabelece o regime da consolidação definitiva da mobilidade intercarreiras.\n\nEm conformidade e nos termos da competência prevista no n.º 5 do citado artigo 99.º-A, em conjugação com a competência dada pela alínea a) n.º 2 do artigo 35.º do Regime Jurídico das Autarquias Locais, aprovado pela Lei n.º 75/2013, de 12 de setembro na atual redação em matéria de recursos humanos:\n\npropõe-se a consolidação da mobilidade na carreira/categoria ************************* entre órgãos/serviços de ******************** - **********************, funções de grau de complexidade funcional 1, no Serviço *****************, ficando posicionado na 1.ª posição remuneratória e nível remuneratório 5 – 821,23 €, atualizada nos termos do decreto-lei n.º 108/2023, de 22 de novembro a partir de 1 de abril de 2024, com base na seguinte fundamentação:\n\nAcordo prévio do trabalhador.\n\nAnuência da entidade de origem.\n\nObservância dos requisitos gerais de recrutamento para a carreira/categoria, *************************.\n\nExistência de posto de trabalho disponível no mapa de pessoal em vigor.\n\nDuração das funções desempenhadas na situação de mobilidade na categoria *************************, para além dos 90 dias previstos para na alínea d) do n.º 1 do artigo 99.º-A da LTFP.\n\nDisponibilidade orçamental.\n\nRemeter ao órgão executivo para deliberação nos termos da parte final do n.º 5 do citado artigo 99.º-A da LTFP.\n\nCovilhã e Paços do Concelho, 6 de março de 2024”\n\nDocumentos que se dão como inteiramente reproduzidos na presente ata e ficam, para todos os efeitos legais, arquivados em pasta própria existente para o efeito.\n\nA Câmara deliberou, com a abstenção dos Senhores Vereadores Pedro Miguel Santos Farromba, Ricardo Miguel Correia Leitão Ferreira da Silva e Marta Maria Tomaz Gomes Morais Alçada Bom Jesus, aprovar a proposta de consolidação da mobilidade na carreira/categoria ************************* entre órgãos/serviços de ******************** - **********************, funções de grau de complexidade funcional 1, no Serviço *****************, ficando posicionado na 1.ª posição remuneratória e nível remuneratório 5 – 821,23 €, atualizada nos termos do decreto-lei n.º 108/2023, de 22 de novembro a partir de 1 de abril de 2024."

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
