"""Leitura das planilhas do E-PROM.

Formato esperado da planilha de avaliações:
- uma aba por especialista;
- a última aba é reservada para instruções/apoio;
- colunas A e B: ID e descrição da alternativa;
- colunas C em diante: critérios;
- todos os especialistas devem possuir os mesmos IDs e critérios.
"""

from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd


def _to_number(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        raise ValueError("Valor vazio encontrado na planilha.")

    if isinstance(x, (int, float, np.integer, np.floating)):
        value = float(x)
        if not np.isfinite(value):
            raise ValueError("Valor NaN/inf encontrado na planilha.")
        return value

    s = str(x).strip().replace(",", ".")
    try:
        value = float(s)
    except ValueError:
        raise ValueError(
            f"Valor não numérico encontrado na matriz de avaliações: {x!r}"
        )
    if not np.isfinite(value):
        raise ValueError("Valor NaN/inf encontrado na planilha.")
    return value


def read_expert_workbook(path):
    path = Path(path)
    xls = pd.ExcelFile(path)
    sheets = xls.sheet_names

    if len(sheets) < 2:
        raise ValueError(
            "A planilha deve possuir pelo menos uma aba de especialista "
            "e uma aba final de instruções/apoio."
        )

    expert_sheets = sheets[:-1]
    if not expert_sheets:
        raise ValueError("Nenhuma aba de especialista foi encontrada.")

    evaluations = {}
    ids = None
    descriptions = None
    criteria = None

    for sheet in expert_sheets:
        df = pd.read_excel(path, sheet_name=sheet, header=0)

        if df.shape[1] < 3:
            raise ValueError(
                f"A aba {sheet!r} precisa ter ID, descrição e pelo menos um critério."
            )

        current_ids = df.iloc[:, 0].tolist()
        current_desc = df.iloc[:, 1].tolist()
        current_criteria = [str(c).strip() for c in df.columns[2:]]

        if any(c.lower().startswith("unnamed") for c in current_criteria):
            raise ValueError(
                f"Aba {sheet!r}: há coluna de critério sem nome."
            )

        if ids is None:
            ids = current_ids
            descriptions = current_desc
            criteria = current_criteria
        else:
            if current_ids != ids:
                raise ValueError(
                    f"Aba {sheet!r} possui IDs diferentes dos demais especialistas."
                )
            if current_criteria != criteria:
                raise ValueError(
                    f"Aba {sheet!r} possui critérios diferentes dos demais especialistas."
                )

        raw = df.iloc[:, 2:]
        numeric = np.empty(raw.shape, dtype=float)

        for i in range(raw.shape[0]):
            for k in range(raw.shape[1]):
                numeric[i, k] = _to_number(raw.iat[i, k])

        evaluations[sheet] = numeric

    n_criteria = len(criteria)

    return {
        "sheets": expert_sheets,
        "expert_names": expert_sheets,
        "ids": ids,
        "descriptions": descriptions,
        "criteria": criteria,
        "evaluations": evaluations,
        "n_experts": len(expert_sheets),
        "n_requirements": len(ids),
        "n_criteria": n_criteria,
    }


def read_fahp_workbook(path, n_criteria=None):
    """Lê a planilha FAHP-Express.

    Formato simplificado para o usuário:
      linha 4: comparação relativa ao critério de referência;
      linha 5: precisão do atributo (Alta, Média ou Baixa).

    O primeiro critério é sempre o critério de referência e deve ser o mais
    importante. Portanto, sua comparação consigo mesmo é fixada em 1.
    Os demais valores são fatores positivos na escala de razão, indicando
    quantas vezes o critério de referência é mais importante que o critério.

    Os parâmetros do trapezoide fuzzy dos atributos contínuos são padronizados
    internamente pelo E-PROM conforme o nível de precisão; o usuário não
    precisa informar suporte e núcleo separadamente.
    """
    path = Path(path)
    xls = pd.ExcelFile(path)
    sheets = xls.sheet_names

    if len(sheets) < 2:
        raise ValueError(
            "A planilha FAHP-Express deve possuir abas de especialistas "
            "e uma aba final de apoio/instruções."
        )

    expert_sheets = sheets[:-1]
    rows = []
    precision_configs = []

    for sheet in expert_sheets:
        df = pd.read_excel(path, sheet_name=sheet, header=None)

        if df.shape[0] < 4:
            raise ValueError(f"Aba FAHP {sheet!r} não possui a linha 4.")

        available = df.iloc[3, 1:].tolist()
        precision = df.iloc[4, 1:].tolist() if df.shape[0] >= 5 else []

        while available and (
            available[-1] is None
            or (isinstance(available[-1], float) and np.isnan(available[-1]))
        ):
            available.pop()

        if n_criteria is not None:
            if len(available) < n_criteria:
                raise ValueError(
                    f"Aba FAHP {sheet!r}: esperados {n_criteria} valores na linha de referência."
                )
            available = available[:n_criteria]

        from .fahp import saaty_linguistic2num
        numeric_row = [saaty_linguistic2num(x) for x in available]

        if len(numeric_row) < 2:
            raise ValueError("O FAHP-Express requer pelo menos 2 critérios.")

        # O primeiro critério é a âncora do FAHP-Express.
        if not np.isclose(numeric_row[0], 1.0, atol=1e-9):
            raise ValueError(
                f"Aba FAHP {sheet!r}: o primeiro critério é a referência e "
                "deve possuir valor 1."
            )

        # Como o primeiro critério é definido como o mais importante, os
        # fatores dos demais critérios devem ser >= 1: 1 = referência; 3 =
        # referência três vezes mais importante que o critério em questão.
        if any(x < 1.0 - 1e-9 for x in numeric_row):
            raise ValueError(
                f"Aba FAHP {sheet!r}: como o primeiro critério é a referência "
                "mais importante, os demais fatores devem ser >= 1."
            )

        if len(precision) < len(numeric_row):
            precision = ["Alta"] * len(numeric_row)
        else:
            precision = precision[:len(numeric_row)]

        allowed = {"Alta", "Média", "Baixa"}
        normalized_precision = []
        for k, value in enumerate(precision):
            if value is None or (isinstance(value, float) and np.isnan(value)):
                value = "Alta"
            value = str(value).strip()
            if value not in allowed:
                raise ValueError(
                    f"Aba FAHP {sheet!r}, critério {k+1}: precisão inválida {value!r}. "
                    "Use Alta, Média ou Baixa."
                )
            normalized_precision.append(value)

        precision_configs.append(normalized_precision)
        rows.append(numeric_row)

    precision_levels = precision_configs[0]
    for cfg in precision_configs[1:]:
        if cfg != precision_levels:
            raise ValueError(
                "A configuração de precisão dos atributos deve ser igual em todas as abas do FAHP-Express."
            )

    # A precisão é uma propriedade geral do atributo e se aplica tanto a
    # critérios ordinais quanto contínuos. A transformação fuzzy específica
    # de cada tipo é realizada no núcleo do E-PROM.

    from .fahp import group_weights_from_reference_rows
    result = group_weights_from_reference_rows(rows)
    result["reference_rows"] = rows
    result["precision_levels"] = precision_levels
    result["sheets"] = expert_sheets
    result["expert_names"] = expert_sheets
    return result
