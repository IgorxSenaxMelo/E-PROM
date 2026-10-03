"""Módulo CPP (Composition of Probabilistic Preferences) para dados contínuos.
Implementa a composição probabilística por simulação Monte Carlo usando Normal(mu,sigma).
"""
from __future__ import annotations
import numpy as np

def cpp_normal(values, stds, directions, n_sim=10000, seed=0, weights=None):
    X=np.asarray(values,float); S=np.asarray(stds,float); dirs=np.asarray(directions,int)
    n_alt,n_crit=X.shape
    if S.shape!=X.shape: raise ValueError("A matriz de desvios-padrão deve ter a mesma dimensão dos dados.")
    rng=np.random.default_rng(seed)
    samples=rng.normal(X[None,:,:],S[None,:,:],size=(n_sim,n_alt,n_crit))
    best=np.zeros((n_alt,n_crit),float)
    for k in range(n_crit):
        z=samples[:,:,k]
        winner=np.argmax(z,axis=1) if dirs[k]==1 else np.argmin(z,axis=1)
        for i in range(n_alt): best[i,k]=np.mean(winner==i)
    # Composição multiplicativa clássica do CPP; pesos opcionais permitem refletir a importância do FAHP.
    if weights is None:
        score=np.prod(best,axis=1)
    else:
        w=np.asarray(weights,float); w=w/w.sum(); score=np.exp(np.sum(w[None,:]*np.log(np.maximum(best,1e-15)),axis=1))
    order=np.argsort(-score,kind="mergesort")
    rank=np.empty(n_alt,int); rank[order]=np.arange(1,n_alt+1)
    return {"prob_best":best,"score":score,"rank":rank,"n_sim":n_sim}
