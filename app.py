from pathlib import Path
import io, sys, tempfile
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

ROOT=Path(__file__).resolve().parent; sys.path.insert(0,str(ROOT))
from core.data_loader import read_expert_workbook, read_fahp_workbook
from core.e_prom_core import run_e_prom
from core.promethee import pref_trap
from core.fuzzy_numbers import defuzz_coa
from core.cpp import cpp_normal

st.set_page_config(page_title="E-PROM — Fuzzy Preference Ranking", page_icon="⚖️", layout="wide")

def save_uploaded(f):
    tmp=tempfile.NamedTemporaryFile(delete=False,suffix=Path(f.name).suffix or '.xlsx'); tmp.write(f.getvalue()); tmp.close(); return Path(tmp.name)

def relation_matrix(result, ids):
    plus=np.asarray(result['phi_plus']); minus=np.asarray(result['phi_minus']); n=len(ids); rel=np.empty((n,n),object); eps=1e-10
    for i in range(n):
        for j in range(n):
            if i==j: rel[i,j]='I'; continue
            a=plus[i]>plus[j]+eps and minus[i]<minus[j]-eps
            b=plus[i]<plus[j]-eps and minus[i]>minus[j]+eps
            eq=abs(plus[i]-plus[j])<=eps and abs(minus[i]-minus[j])<=eps
            rel[i,j]='P' if a else 'P-' if b else 'I' if eq else 'R'
    return rel

def result_df(data,result):
    return pd.DataFrame({'Rank':result['rank_e_prom'].astype(int),'Alternativa':data['ids'],'Descricao':data['descriptions'],
                         'Phi+':result['phi_plus'],'Phi-':result['phi_minus'],'Phi líquido':result['phi_net']}).sort_values(['Rank','Alternativa']).reset_index(drop=True)

def excel_bytes(data,result,config):
    out=io.BytesIO(); ranking=result_df(data,result)
    with pd.ExcelWriter(out,engine='openpyxl') as w:
        ranking.to_excel(w,'Ranking',index=False)
        pd.DataFrame(result['phi_net_experts'],index=data['ids'],columns=result['expert_names']).to_excel(w,'Phi_por_Especialista')
        pd.DataFrame(config).to_excel(w,'Configuracao',index=False)
    out.seek(0); return out.getvalue()

def plot_weights(fahp,criteria):
    wf=np.asarray(fahp['w_fuzzy_individual']); crisp=np.array([[defuzz_coa(wf[k,:,e]) for e in range(wf.shape[2])] for k in range(wf.shape[0])]); mean=crisp.mean(1)
    fig=go.Figure(go.Bar(x=criteria,y=mean,name='Peso médio'))
    for e,name in enumerate(fahp.get('sheets',[f'E{i+1}' for i in range(crisp.shape[1])])): fig.add_trace(go.Scatter(x=criteria,y=crisp[:,e],mode='markers',name=name))
    fig.update_layout(title='Pesos dos critérios — FAHP-Express',height=500,xaxis_title='Critério',yaxis_title='Peso'); return fig

def plot_rank_heatmap(data,result):
    labels=[str(x) for x in data['ids']]; df=pd.DataFrame(result['rank_experts'],index=labels,columns=result['expert_names']); df['E-PROM']=result['rank_e_prom']; df=df.sort_values('E-PROM')
    fig=go.Figure(go.Heatmap(z=df.values,x=list(df.columns),y=list(df.index),text=np.vectorize(lambda x:f'{x:g}')(df.values),texttemplate='%{text}',colorscale='Blues_r',zmin=1,zmax=len(labels),xgap=1,ygap=1,colorbar=dict(title='Posição')))
    fig.update_layout(title='Posições: especialistas vs E-PROM',height=max(600,len(labels)*25)); return fig

def plot_relation(data,result):
    rel=relation_matrix(result,data['ids']); mp={'P-':0,'I':1,'P':2,'R':3}; z=np.vectorize(mp.get)(rel); labels=[str(x) for x in data['ids']]
    fig=go.Figure(go.Heatmap(z=z,x=labels,y=labels,text=rel,texttemplate='%{text}',zmin=0,zmax=3,colorscale=[[0,'#3333cc'],[.249,'#3333cc'],[.25,'#aaa'],[.499,'#aaa'],[.5,'#2a2'],[.749,'#2a2'],[.75,'#e8cc28'],[1,'#e8cc28']],colorbar=dict(tickvals=[0,1,2,3],ticktext=['P-','I','P','R'],title='Relação')))
    fig.update_layout(title='PROMETHEE I — relações de preferência',height=800); return fig

st.title('E-PROM')
st.subheader('Express Fuzzy Preference Ranking with PROMETHEE')
st.caption('FAHP-Express + Fuzzy PROMETHEE — versão generalizada, sem classificação KEY.')

with st.sidebar:
    st.header('1. Dados')
    req_file=st.file_uploader('Planilha das avaliações',type=['xlsx'])
    fahp_file=st.file_uploader('Planilha FAHP-Express',type=['xlsx'])
    if req_file is not None:
        tmp_req=save_uploaded(req_file)
        try:
            preview=read_expert_workbook(tmp_req)
            criteria=preview['criteria']; ncrit=len(criteria)
            st.success(f'{ncrit} critérios detectados.')
            st.divider(); st.header('2. Configuração dos critérios')
            dirs=[]; types=[]; prefs=[]; qs=[]; ps=[]; scales=[]; rows=[]
            for k,c in enumerate(criteria):
                st.markdown(f'**{c}**')
                typ=st.selectbox('Tipo de dado', ['Ordinal','Contínuo'], key=f'typ{k}')
                direction=st.selectbox('Direção',['Maximizar','Minimizar'],key=f'dir{k}')
                if typ=='Ordinal':
                    scale=st.selectbox('Escala',['5','7','9'],index=1,key=f'scale{k}')
                    pref_name=st.selectbox('Função de preferência',['Usual','Level'],key=f'pref{k}')
                    if pref_name=='Level':
                        q=st.number_input('q — indiferença',min_value=0.0,value=0.0,step=1.0,key=f'q{k}')
                        p=st.number_input('p — preferência',min_value=float(q+1e-9),value=max(float(q+1),1.0),step=1.0,key=f'p{k}')
                    else: q=0.0; p=0.0
                else:
                    scale=1
                    pref_name=st.selectbox('Função de preferência',['Level','Linear'],key=f'pref{k}')
                    q=st.number_input('q — limiar de indiferença',min_value=0.0,value=0.0,step=0.1,key=f'q{k}')
                    p=st.number_input('p — limiar de preferência',min_value=float(q+1e-9),value=max(float(q+1),1.0),step=0.1,key=f'p{k}')
                pref_code={'Usual':1,'Level':4,'Linear':5}[pref_name]
                dirs.append(1 if direction=='Maximizar' else -1); types.append(typ); prefs.append(pref_code); qs.append(q); ps.append(p); scales.append(int(scale))
                rows.append({'Criterio':c,'Tipo':typ,'Direção':direction,'Função':pref_name,'q':q,'p':p,'Escala':scale})
            config=pd.DataFrame(rows)
            st.divider(); st.header('3. Execução')
            run=st.button('▶ Executar E-PROM',type='primary',use_container_width=True)
        except Exception as e: st.error(str(e)); run=False
    else: run=False

if run:
    if fahp_file is None: st.error('Envie a planilha FAHP-Express.'); st.stop()
    try:
        with st.spinner('Executando FAHP-Express + Fuzzy PROMETHEE...'):
            data=preview; fahp=read_fahp_workbook(save_uploaded(fahp_file),ncrit)
            result=run_e_prom(data['evaluations'],fahp['w_fuzzy_individual'],types,dirs,prefs,qs,ps,scales)
            ranking=result_df(data,result); xlsx=excel_bytes(data,result,config)
        st.session_state.update(data=data,fahp=fahp,result=result,config=config,ranking=ranking,xlsx=xlsx)
        st.success('E-PROM executado com sucesso.')
    except Exception as e: st.exception(e); st.stop()

if 'result' in st.session_state:
    data=st.session_state['data']; fahp=st.session_state['fahp']; result=st.session_state['result']; config=st.session_state['config']; ranking=st.session_state['ranking']
    st.divider(); a,b,c=st.columns(3); a.metric('Especialistas',data['n_experts']); b.metric('Alternativas',data['n_requirements']); c.metric('Critérios',data['n_criteria'])
    t1,t2,t3,t4=st.tabs(['🏆 Ranking','📈 PROMETHEE','👥 Especialistas','🎲 CPP'])
    with t1:
        st.dataframe(ranking,use_container_width=True,hide_index=True)
        st.download_button('⬇️ Baixar resultado em Excel',data=st.session_state['xlsx'],file_name='Resultado_E_PROM.xlsx',mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
    with t2:
        st.plotly_chart(plot_weights(fahp,data['criteria']),use_container_width=True)
        st.plotly_chart(plot_relation(data,result),use_container_width=True)
        st.plotly_chart(plot_rank_heatmap(data,result),use_container_width=True)
    with t3:
        st.dataframe(pd.DataFrame(result['phi_net_experts'],index=data['ids'],columns=result['expert_names']),use_container_width=True)
        st.dataframe(config,use_container_width=True,hide_index=True)
    with t4:
        st.info('CPP é executado para critérios contínuos. Para cada alternativa/critério, informe o desvio-padrão na tabela abaixo. A distribuição normal é usada para gerar as preferências probabilísticas por simulação Monte Carlo.')
        cont_idx=[i for i,t in enumerate(config['Tipo']) if t=='Contínuo']
        if not cont_idx: st.warning('Nenhum critério contínuo foi configurado.')
        else:
            std_df=pd.DataFrame(0.0,index=data['ids'],columns=[data['criteria'][i] for i in cont_idx])
            edited=st.data_editor(std_df,use_container_width=True,key='cpp_std')
            nsim=st.number_input('Simulações Monte Carlo',min_value=1000,max_value=500000,value=10000,step=1000)
            seed=st.number_input('Seed',min_value=0,value=0,step=1)
            if st.button('▶ Executar CPP'):
                vals=np.column_stack([data['evaluations'][e][:,cont_idx] for e in []]) if False else np.mean(np.stack([data['evaluations'][e] for e in data['expert_names']]),axis=0)
                X=vals[:,cont_idx]; S=edited.to_numpy(float); W=np.array([defuzz_coa(fahp['w_fuzzy_individual'][i].mean(axis=1)) if False else defuzz_coa(fahp['w_fuzzy_group'][i]) for i in cont_idx])
                cpp=cpp_normal(X,S,np.array(dirs)[cont_idx],int(nsim),int(seed),weights=W)
                cpp_df=pd.DataFrame({'Alternativa':data['ids'],'Score_CPP':cpp['score'],'Rank_CPP':cpp['rank']}).sort_values('Rank_CPP')
                st.dataframe(cpp_df,use_container_width=True,hide_index=True)
                st.dataframe(pd.DataFrame(cpp['prob_best'],index=data['ids'],columns=[data['criteria'][i] for i in cont_idx]),use_container_width=True)
