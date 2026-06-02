"""
Prompt description for AMALIA span extraction via langextract (VotIE v5 annotation scheme).

langextract wraps this description with its own input/output framing and example
formatting, so this module provides only the task description and annotation rules.
Kept in Portuguese to match AMALIA's training language.
"""


PROMPT_DESCRIPTION = """\
Extrai todas as entidades de votação do seguinte texto de atas de câmara municipal portuguesa.

Tipos de entidade (11 tipos):

Papéis de votante (pessoa ou grupo nomeado):
  VOTER-FAVOR      — votante que votou a favor (ex: "o Executivo Municipal", "os eleitos pelo PS")
  VOTER-AGAINST    — votante que votou contra (ex: "Vereador do BE")
  VOTER-ABSTENTION — votante que se absteve (ex: "a eleita pelo Nós, Cidadãos")
  VOTER-ABSENT     — participante ausente da votação

Evento de votação:
  VOTING           — verbo ou expressão verbal que indica o ato de votar, em qualquer caixa
                     (ex: "deliberou", "DELIBEROU", "aprovou", "APROVOU", "Aprovada", "APROVADA")
  SUBJECT          — frase nominal com a matéria votada; sem títulos de secção nem verbos de ação
  COUNTING-UNANIMITY — expressão textual de unanimidade (ex: "por unanimidade", "unanimidade")
  COUNTING-MAJORITY  — expressão textual de maioria, incluindo o desdobramento quando num único span
                       (ex: "por maioria", "com 1 voto contra e 3 abstenções")

Contagens de escrutínio secreto (valores numéricos agregados):
  COUNT-FAVOR      — contagem de votos a favor (ex: "6 votos a favor")
  COUNT-BLANK      — contagem de votos em branco (ex: "7 votos em branco")

Procedimento:
  VOTING-METHOD    — modalidade de votação (ex: "escrutínio secreto", "voto secreto")

Regras de anotação:
1. Extrai o texto EXATO — não modifiques, resumas nem parafraseies nenhuma palavra.
2. SUBJECT: extrai apenas a frase nominal da matéria; remove verbos de ação iniciais
   (ex: "aprovar", "ratificar"); não extraias títulos de secção.
3. COUNTING-MAJORITY vs COUNT-*: usa COUNTING-MAJORITY para expressões em linguagem natural;
   usa COUNT-FAVOR / COUNT-BLANK apenas para contagens numéricas explícitas
   em escrutínio secreto.
4. Votantes dentro de COUNTING-MAJORITY: votantes nomeados que aparecem dentro de uma
   expressão de maioria são marcados também como VOTER-AGAINST / VOTER-ABSTENTION.
5. MÚLTIPLAS VOTAÇÕES: um segmento pode conter vários eventos de votação (vários verbos VOTING).
   Extrai as entidades de TODOS os eventos — cada verbo VOTING define um evento independente
   com o seu SUBJECT, votantes e resultado associados.
6. MAIÚSCULAS/MINÚSCULAS: os verbos de votação são reconhecidos em qualquer caixa.
   "DELIBEROU", "deliberou" e "Deliberou" são todos verbos VOTING válidos.
   "POR UNANIMIDADE" e "por unanimidade" são igualmente COUNTING-UNANIMITY.
   Da mesma forma para "POR MAIORIA", "APROVADA", "APROVOU", "RATIFICOU", etc.
7. SEM VOTAÇÃO: se o texto não contiver nenhum verbo de votação (ex: "deliberou",
   "DELIBEROU", "aprovou", "APROVADA", "ratificou"), devolve lista vazia. Não extraias
   entidades de títulos de secção, balancetes, relatos informativos ou assuntos
   retirados da ordem de trabalhos.
8. Extrai exclusivamente do texto-alvo. Nunca copies texto dos exemplos.
9. Devolve EXCLUSIVAMENTE JSON dentro de uma cerca ```json ... ```, com a
   chave "extractions" no topo. Cada item é um objeto com o tipo da entidade
   como chave e o texto exato como valor, seguido de um campo
   "<TIPO>_attributes" com objeto vazio.

Formato de saída (esquema — não copies o conteúdo):
```json
{
  "extractions": [
    {"SUBJECT": "<texto>", "SUBJECT_attributes": {}},
    {"VOTING": "<texto>", "VOTING_attributes": {}}
  ]
}
```"""


def get_prompt_description() -> str:
    """Return the task description for langextract's prompt_description parameter."""
    return PROMPT_DESCRIPTION
