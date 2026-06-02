"""
Fixed few-shot examples for AMALIA extraction via langextract (VotIE v5 scheme).

Five short examples covering all 11 entity types plus the empty-extraction case:
  Ex 1 — majority + against + abstention (SUBJECT, VOTING, COUNTING-MAJORITY,
          VOTER-AGAINST, VOTER-ABSTENTION)
  Ex 2 — unanimity + absent voter (VOTER-FAVOR, VOTING, COUNTING-UNANIMITY,
          VOTER-ABSENT, SUBJECT)
  Ex 3 — secret ballot (SUBJECT, VOTING-METHOD, VOTING, COUNT-FAVOR, COUNT-BLANK)
  Ex 4 — multi-vote segment (VOTER-FAVOR, VOTING ×2, SUBJECT ×2)
  Ex 5 — non-voting agenda item (empty extraction; suppresses spurious SUBJECT
          generation on informational items without a VOTING anchor)

All examples are drawn from citilink-votie training data.
"""

import langextract as lx


FIXED_EXAMPLES = [
    # Example 1 — majority with named against + abstention voters (217 chars)
    # Source: Porto_cm_016_2022-06-13_13
    {
        "id": "Porto_cm_016_2022-06-13_13",
        "text": (
            "13. Reconhecimento da isenção de IMI e de IMT para os prédios"
            " cuja descrição consta do quadro no Anexo I.\n"
            "Aprovada, com 1 voto contra do Senhor Vereador do BE e com"
            " 3 abstenções dos Senhores Vereadores do PS e da CDU."
        ),
        "entities": [
            {
                "text": "Reconhecimento da isenção de IMI e de IMT para os prédios",
                "type": "SUBJECT",
            },
            {"text": "Aprovada", "type": "VOTING"},
            {
                "text": (
                    "1 voto contra do Senhor Vereador do BE e com"
                    " 3 abstenções dos Senhores Vereadores do PS e da CDU"
                ),
                "type": "COUNTING-MAJORITY",
            },
            {"text": "Vereador do BE", "type": "VOTER-AGAINST"},
            {"text": "Vereadores do PS", "type": "VOTER-ABSTENTION"},
            {"text": "a CDU", "type": "VOTER-ABSTENTION"},
        ],
    },
    # Example 2 — unanimity with absent voter (292 chars)
    # Source: M01_cm_003_2023-02-01_seg009
    {
        "id": "M01_cm_003_2023-02-01_seg009",
        "text": (
            " \n4. APROVAÇÃO DE ATA\nPelo Senhor Presidente foi presente"
            " a reunião a ata n.º 02, de 18.01.2023. \nPonderado e"
            " analisado o assunto o Executivo Municipal deliberou por"
            " unanimidade, sem a participação da eleita pelo Nós,"
            " Cidadãos por não ter estado presente, aprovar a ata"
            " n.º 02, de 18.01.2023."
        ),
        "entities": [
            {"text": "o Executivo Municipal", "type": "VOTER-FAVOR"},
            {"text": "deliberou", "type": "VOTING"},
            {"text": "por unanimidade", "type": "COUNTING-UNANIMITY"},
            {"text": "a eleita pelo Nós, Cidadãos", "type": "VOTER-ABSENT"},
            {"text": "ata n.º 02, de 18.01.2023", "type": "SUBJECT"},
        ],
    },
    # Example 3 — secret ballot with aggregate counts (172 chars)
    # Source: Porto_cm_007_2022-01-31_14
    {
        "id": "Porto_cm_007_2022-01-31_14",
        "text": (
            "14. Designação do Presidente do Conselho de Administração"
            " da Associação Porto Digital.\n"
            "Em votação por escrutínio secreto, aprovada, com 6 votos"
            " a favor e 7 votos em branco."
        ),
        "entities": [
            {
                "text": (
                    "Designação do Presidente do Conselho de Administração"
                    " da Associação Porto Digital"
                ),
                "type": "SUBJECT",
            },
            {"text": "escrutínio secreto", "type": "VOTING-METHOD"},
            {"text": "aprovada", "type": "VOTING"},
            {"text": "6 votos a favor", "type": "COUNT-FAVOR"},
            {"text": "7 votos em branco", "type": "COUNT-BLANK"},
        ],
    },
    # Example 4 — multi-vote segment: two distinct voting events (~230 chars)
    # Source: Covilha_cm_004-A_2021-12-03_11 (trimmed: legal boilerplate removed)
    {
        "id": "Covilha_cm_004-A_2021-12-03_11",
        "text": (
            'Aceitação de Doação de 74 desenhos de "Debuxo".\n\n'
            "A Câmara deliberou aceitar a doação de 74 desenhos de"
            ' "Debuxo", manifestada por Micael Matias.\n\n'
            "Mais deliberou aprovar e celebrar o respetivo auto de doação."
        ),
        "entities": [
            # First voting event
            {
                "text": 'Aceitação de Doação de 74 desenhos de "Debuxo"',
                "type": "SUBJECT",
            },
            {"text": "A Câmara", "type": "VOTER-FAVOR"},
            {"text": "deliberou", "type": "VOTING"},
            # Second voting event ("Mais deliberou...")
            {"text": "deliberou", "type": "VOTING"},
            {"text": "auto de doação", "type": "SUBJECT"},
        ],
    },
    # Example 5 — non-voting agenda item: empty extraction (102 chars)
    # Pure section heading with no extractable-looking noun phrases or voting
    # verbs — explicitly states no items were scheduled. Teaches the voting-only
    # contract (no VOTING anchor in source → no extracted entities) without
    # giving the model trap-like content it could echo back as a SUBJECT.
    # Source: Covilha_cm_005_2022-03-21_19
    {
        "id": "Covilha_cm_005_2022-03-21_19",
        "text": (
            "5.2. DEPARTAMENTO DE FINANÇAS E MODERNIZAÇÃO ADMINISTRATIVA\n\n"
            "Não foram agendados assuntos neste ponto."
        ),
        "entities": [],
    },
]


def get_fixed_examples(num_examples: int = 5) -> list:
    """Convert fixed examples to langextract ExampleData objects."""
    examples = []
    for ex in FIXED_EXAMPLES[:num_examples]:
        extractions = [
            lx.data.Extraction(
                extraction_class=ent["type"],
                extraction_text=ent["text"],
            )
            for ent in ex["entities"]
        ]
        examples.append(
            lx.data.ExampleData(
                text=ex["text"],
                extractions=extractions,
            )
        )
    return examples
