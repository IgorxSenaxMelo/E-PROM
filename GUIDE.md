# E-PROM — Guide

## 1. Objetivo

O E-PROM (Express Fuzzy Preference Ranking with PROMETHEE) combina:

**FAHP-Express → pesos fuzzy dos critérios → Fuzzy PROMETHEE**

O método mantém os pesos fuzzy durante o Fuzzy PROMETHEE e realiza a
defuzzificação dos fluxos no final.

---

## 2. Arquivo de avaliações

A planilha de avaliações deve possuir:

- uma aba por especialista;
- uma última aba chamada `Guide`, que não é processada;
- coluna A: ID da alternativa;
- coluna B: descrição;
- coluna C em diante: critérios.

Todos os especialistas devem usar os mesmos IDs, descrições e nomes de critérios.

### Exemplo

| ID | Descrição | Impacto | Custo | Prazo |
|---|---|---:|---:|---:|
| R1 | Alternativa 1 | 7 | 300000 | 12 |
| R2 | Alternativa 2 | 5 | 250000 | 18 |

---

## 3. Tipo do critério

### Ordinal

Use quando a avaliação é uma escala ordenada, por exemplo:

- 1–5;
- 1–7;
- 1–9.

Funções permitidas:

- **Usual**
- **Level**

O valor ordinal é convertido para um número fuzzy trapezoidal.

### Contínuo

Use para grandezas como:

- custo em R$;
- prazo em meses;
- distância;
- tempo;
- consumo;
- desempenho físico.

Funções permitidas:

- **U-shape**
- **V-shape**
- **Linear**

O valor permanece na unidade original. Não é convertido para 1–7.

O valor contínuo x é representado como um trapezoide fuzzy cuja largura depende da precisão selecionada na planilha FAHP-Express.

---

## 4. Limiar q e p

### U-shape

`q` representa o limiar de indiferença.

### V-shape

`p` representa o limiar de preferência.

### Linear

Usa os dois:

- `q`: limiar de indiferença;
- `p`: limiar de preferência.

Sempre use a unidade original do critério.

Exemplo:

Custo em reais:

`q = 10000`

`p = 50000`

Prazo em meses:

`q = 1`

`p = 4`

---

## 5. Precisão dos atributos

Na planilha FAHP-Express, a segunda linha é a **Precisão do atributo**.
Use apenas:

- **Alta** — menor incerteza;
- **Média** — representação fuzzy padrão;
- **Baixa** — maior incerteza.

A precisão é aplicável tanto a critérios ordinais quanto contínuos. Os parâmetros numéricos internos são padronizados pelo método e não precisam ser informados pelo usuário.

Para critérios ordinais, a representação fuzzy linguística de precisão média é contraída ou dilatada em torno do valor observado. Para critérios contínuos, o trapezoide é construído diretamente na unidade original do atributo.

Para um atributo ordinal v, a representação de precisão média é usada como base e as distâncias em relação a v são multiplicadas por um fator de precisão: Alta = 0,5; Média = 1,0; Baixa = 1,5. O trapezoide é então limitado aos extremos da escala ordinal. Para atributos contínuos, são usados os parâmetros internos: Alta = suporte ±5% e núcleo ±2%; Média = suporte ±10% e núcleo ±4%; Baixa = suporte ±16,6667% e núcleo ±6,6667%.

## 6. FAHP-Express

Na planilha FAHP-Express, cada especialista informa uma linha de referência
a partir da célula B4.

A linha deve possuir exatamente o mesmo número de critérios da planilha
de avaliações.

O E-PROM reconstrói as comparações par a par e calcula os pesos fuzzy pelo
procedimento de Buckley.

---

## 7. Pesos fuzzy no Fuzzy PROMETHEE

O E-PROM não defuzzifica os pesos antes do PROMETHEE.

Para cada critério:

`peso fuzzy × preferência fuzzy`

Os termos são agregados para obter os fluxos fuzzy:

- Φ+;
- Φ−;
- Φ líquido.

A defuzzificação pelo centro de área é aplicada aos fluxos finais para
obter o ranking.

---

## 8. Monte Carlo / CPP

O módulo de Monte Carlo avalia a estabilidade do resultado.

### Critérios ordinais

A distribuição é **empírica**, construída diretamente a partir das
respostas dos especialistas.

Para cada alternativa e critério, uma resposta observada entre os
especialistas é sorteada de acordo com sua frequência.

Assim, se 10 especialistas responderam:

`7, 7, 6, 5, 7, 6, 7, 4, 7, 6`

a distribuição usada na simulação preserva essas frequências.

### Critérios contínuos

O usuário informa uma variação percentual.

Exemplo:

`20%`

O valor observado `x` é simulado em:

`[0,80x ; 1,20x]`

por amostragem uniforme.

A variação é configurada individualmente para cada critério contínuo.

### Pesos

Em cada iteração, os valores de referência do FAHP-Express também são
reamostrados empiricamente a partir dos especialistas e um novo conjunto
de pesos fuzzy é calculado.

---

## 9. Principais resultados do Monte Carlo

O E-PROM calcula:

- posição média;
- desvio-padrão da posição;
- Φ líquido médio;
- desvio-padrão de Φ líquido;
- frequência de 1º lugar;
- matriz de aceitabilidade de ranking;
- matriz de aceitabilidade de sobreclassificação PROMETHEE I;
- distribuição dos pesos dos critérios.

O módulo não possui classificação `KEY` nem grupos específicos do domínio
militar.

---

## 10. Interpretação

A análise probabilística deve ser usada como análise de estabilidade,
não como substituição automática do ranking determinístico.

Resultados úteis incluem:

- frequência com que uma alternativa ocupa cada posição;
- frequência de 1º lugar;
- dispersão de Φ;
- sensibilidade dos pesos;
- probabilidade de sobreclassificação entre pares.

---

## 11. Fluxo recomendado

1. Preparar as avaliações dos especialistas.
2. Preparar o FAHP-Express.
3. Configurar tipo e direção dos critérios.
4. Selecionar a função de preferência.
5. Definir q e/ou p.
6. Executar o E-PROM determinístico.
7. Verificar o ranking e as relações PROMETHEE I.
8. Executar o Monte Carlo/CPP.
9. Avaliar estabilidade e aceitabilidade das posições.
10. Exportar os resultados.
