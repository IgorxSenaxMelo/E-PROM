"""Leitura genérica de matrizes de decisão para o E-PROM."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from .fahp import group_weights_from_reference_rows


def _to_float(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        raise ValueError("Valor vazio ou missing encontrado na planilha.")
    if isinstance(x, (int, float, np.integer, np.floating)):
        if not np.isfinite(float(x)):
            raise ValueError("Valor NaN/inf encontrado na planilha.")
        return float(x)
    s = str(x).strip().lower()
    mapping = {"muito alto":7,"alto":6,"moderadamente alto":5,"médio":4,"medio":4,
               "moderadamente baixo":3,"baixo":2,"muito baixo":1}
    if s in mapping: return float(mapping[s])
    try: return float(str(x).replace(',', '.'))
    except ValueError: raise ValueError(f"Valor não numérico/linguístico desconhecido: {x!r}")


def read_expert_workbook(path):
    path = Path(path); xls = pd.ExcelFile(path); sheets = xls.sheet_names
    expert_sheets = sheets[:-1] if len(sheets) > 1 else sheets
    if not expert_sheets: raise ValueError("Nenhuma aba de especialista encontrada.")
    evaluations, ids, descriptions, criteria = {}, None, None, None
    for sheet in expert_sheets:
        df = pd.read_excel(path, sheet_name=sheet, header=0)
        if df.shape[1] < 3: raise ValueError(f"A aba {sheet!r} precisa ter ID, descrição e pelo menos um critério.")
        cur_ids = df.iloc[:,0].tolist(); cur_desc = df.iloc[:,1].tolist(); cur_criteria = [str(x) for x in df.columns[2:]]
        if ids is None: ids, descriptions, criteria = cur_ids, cur_desc, cur_criteria
        if cur_ids != ids: raise ValueError(f"Aba {sheet!r} possui IDs diferentes.")
        if cur_criteria != criteria: raise ValueError(f"Aba {sheet!r} possui critérios/colunas diferentes.")
        raw = df.iloc[:,2:]
        numeric = np.empty(raw.shape, dtype=float)
        for i in range(raw.shape[0]):
            for k in range(raw.shape[1]): numeric[i,k] = _to_float(raw.iat[i,k])
        evaluations[sheet] = numeric
    return {"sheets":expert_sheets,"ids":ids,"descriptions":descriptions,"criteria":criteria,
            "evaluations":evaluations,"n_experts":len(expert_sheets),"n_requirements":len(ids),
            "n_criteria":len(criteria)}


def read_fahp_workbook(path, n_criteria):
    path = Path(path); xls = pd.ExcelFile(path); sheets = xls.sheet_names
    expert_sheets = sheets[:-1] if len(sheets) > 1 else sheets
    rows=[]
    for sheet in expert_sheets:
        df=pd.read_excel(path,sheet_name=sheet,header=None)
        row=df.iloc[3,1:1+n_criteria].tolist()
        if len(row)!=n_criteria: raise ValueError(f"Planilha FAHP da aba {sheet!r} não contém {n_criteria} pesos de referência.")
        rows.append(row)
    result=group_weights_from_reference_rows(rows)
    result["sheets"]=expert_sheets
    return result
