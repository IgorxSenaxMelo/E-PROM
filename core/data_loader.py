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
    """Lê B4 em diante de cada aba de especialista.

    Cada aba deve conter, na linha 4, os valores de referência do FAHP-Express
    a partir de B4. A última aba é considerada apoio/instruções.
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

    for sheet in expert_sheets:
        df = pd.read_excel(path, sheet_name=sheet, header=None)

        if df.shape[0] < 4:
            raise ValueError(f"Aba FAHP {sheet!r} não possui a linha 4.")

        available = df.iloc[3, 1:].tolist()

        # Remove células vazias somente no final.
        while available and (
            available[-1] is None
            or (isinstance(available[-1], float) and np.isnan(available[-1]))
        ):
            available.pop()

        if n_criteria is not None:
            if len(available) < n_criteria:
                raise ValueError(
                    f"Aba FAHP {sheet!r}: esperados {n_criteria} valores em B4 em diante."
                )
            available = available[:n_criteria]

        from .fahp import saaty_linguistic2num
        numeric_row = [
            saaty_linguistic2num(x)
            for x in available
        ]
        rows.append(numeric_row)

    from .fahp import group_weights_from_reference_rows

    result = group_weights_from_reference_rows(rows)
    result["reference_rows"] = rows
    result["sheets"] = expert_sheets
    result["expert_names"] = expert_sheets
    return result
