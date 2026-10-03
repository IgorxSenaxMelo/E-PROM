"""Núcleo matemático do E-PROM: FAHP-Express + Fuzzy PROMETHEE, sem KEY."""
from __future__ import annotations
import numpy as np
from .promethee import run_fuzzy_promethee_trap, rank_from_net_flow
from .fuzzy_numbers import defuzz_coa

ORDINAL_TFNS = {
    5: np.array([[1,1,1.5,2],[1.5,2,2,2.5],[2,2.5,3,3.5],[3,3.5,4,4.5],[4,4.5,5,5]],float),
    7: np.array([[1,1,1.5,2],[1,1.5,2.5,3],[2,2.5,3.5,4],[3,3.5,4.5,5],[4,4.5,5.5,6],[5,5.5,6.5,7],[6,6.5,7,7]],float),
    9: np.array([[1,1,1.5,2],[1,1.5,2.5,3],[2,2.5,3.5,4],[3,3.5,4.5,5],[4,4.5,5,5.5],[5,5.5,6.5,7],[6,6.5,7.5,8],[7,7.5,8.5,9],[8,8.5,9,9]],float)
}

def fuzzify_ordinal(value, scale_max):
    table=ORDINAL_TFNS.get(int(scale_max))
    if table is None: raise ValueError("Escala ordinal suportada: 5, 7 ou 9.")
    v=int(round(float(value)))
    if not 1<=v<=scale_max: raise ValueError(f"Valor ordinal {v} fora de 1..{scale_max}.")
    return table[v-1].copy()

def fuzzify_continuous(value):
    x=float(value); return np.array([x,x,x,x],float)

def build_fuzzy_matrix(data_numeric, criterion_types, scales):
    data_numeric=np.asarray(data_numeric,float); n_alt,n_crit=data_numeric.shape
    F=np.zeros((n_alt,n_crit,4),float)
    for i in range(n_alt):
        for k in range(n_crit):
            F[i,k]=fuzzify_ordinal(data_numeric[i,k],scales[k]) if criterion_types[k]=="Ordinal" else fuzzify_continuous(data_numeric[i,k])
    return F

def run_e_prom(evaluations, weights_fuzzy_individual, criterion_types, directions, pref_types, q_vals, p_vals, scales):
    names=list(evaluations.keys()); n_exp=len(names); n_alt=next(iter(evaluations.values())).shape[0]
    n_crit=len(directions)
    if weights_fuzzy_individual.shape[0]!=n_crit: raise ValueError("Número de pesos FAHP diferente do número de critérios.")
    plus_e=np.zeros((n_alt,n_exp)); minus_e=np.zeros_like(plus_e); net_e=np.zeros_like(plus_e)
    plus_f=np.zeros((n_alt,4,n_exp)); minus_f=np.zeros_like(plus_f); net_f=np.zeros_like(plus_f)
    for e,name in enumerate(names):
        F=build_fuzzy_matrix(evaluations[name],criterion_types,scales)
        out=run_fuzzy_promethee_trap(F,weights_fuzzy_individual[:,:,e],np.asarray(directions),np.asarray(pref_types),np.asarray(q_vals),np.asarray(p_vals))
        pp,pm,pn,ppf,pmf,pnf=out
        plus_e[:,e],minus_e[:,e],net_e[:,e]=pp,pm,pn
        plus_f[:,:,e],minus_f[:,:,e],net_f[:,:,e]=ppf,pmf,pnf
    plus_fuzzy=plus_f.mean(axis=2); minus_fuzzy=minus_f.mean(axis=2); net_fuzzy=net_f.mean(axis=2)
    plus=np.array([defuzz_coa(x) for x in plus_fuzzy]); minus=np.array([defuzz_coa(x) for x in minus_fuzzy]); net=np.array([defuzz_coa(x) for x in net_fuzzy])
    return {"expert_names":names,"phi_plus_experts":plus_e,"phi_minus_experts":minus_e,"phi_net_experts":net_e,
            "phi_plus_fuzzy_experts":plus_f,"phi_minus_fuzzy_experts":minus_f,"phi_net_fuzzy_experts":net_f,
            "phi_plus_fuzzy":plus_fuzzy,"phi_minus_fuzzy":minus_fuzzy,"phi_net_fuzzy":net_fuzzy,
            "phi_plus":plus,"phi_minus":minus,"phi_net":net,
            "rank_experts":np.column_stack([rank_from_net_flow(net_e[:,e]) for e in range(n_exp)]),
            "rank_e_prom":rank_from_net_flow(net),"q_vals":np.asarray(q_vals),"p_vals":np.asarray(p_vals)}
