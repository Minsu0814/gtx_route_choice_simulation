# -*- coding: utf-8 -*-
"""MNL에 EB 피처(거리, 노선 수) 추가해서 돌리기"""
import sys; sys.stdout.reconfigure(encoding='utf-8')
import pickle, sqlite3
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.optimize import minimize, approx_fprime
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent.parent.parent.parent
DATA_DIR = ROOT / 'data' / 'training_set'
EB_DIR = ROOT / 'data' / 'eb'

# === 데이터 ===
df = pd.read_parquet(DATA_DIR / 'route_choice_training.parquet')
if 'alt_idx' not in df.columns:
    df['alt_idx'] = df.groupby('od_pair').cumcount()

ods = set(df['od_pair'].unique())
conn = sqlite3.connect(str(DATA_DIR / 'otp_cache.db'))
ivt = []
for od, blob in conn.execute('SELECT od_pair, data FROM otp_cache'):
    if od not in ods: continue
    data = pickle.loads(blob)
    for i, p in enumerate(data['otp_parsed']):
        b=t=g=0.0
        for leg in p.get('transit_legs', []):
            m, d = leg.get('mode',''), float(leg.get('duration',0))
            if m=='BUS': b+=d
            elif m=='GTX': g+=d
            else: t+=d
        ivt.append((od, i, b/60, t/60, g/60))
conn.close()
ivt_df = pd.DataFrame(ivt, columns=['od_pair','alt_idx','bus_ivt_min','train_ivt_min','gtx_ivt_min'])
df = df.merge(ivt_df, on=['od_pair','alt_idx'], how='left')
for c in ['bus_ivt_min','train_ivt_min','gtx_ivt_min']: df[c]=df[c].fillna(0)
del ivt, ivt_df

for col in ['access_time','egress_time','transfer_walk_time']:
    df[col+'_min'] = df[col]/60
df['ln_access'] = np.log1p(df['access_time_min'])
df['ln_egress'] = np.log1p(df['egress_time_min'])
df['fare_1000won'] = df['fare']/1000
df['total_ivt_min'] = df['bus_ivt_min'] + df['train_ivt_min'] + df['gtx_ivt_min']

# EB 피처
mapping = pd.read_csv(EB_DIR / 'stop_h3_mapping.csv')
stop_to_h3 = dict(zip(mapping['stop_id'].astype(str), mapping['h3_res8']))
df['o_stop'] = df['od_pair'].str.split('_').str[0]
df['d_stop'] = df['od_pair'].str.split('_').str[1]
df['o_h3'] = df['o_stop'].map(stop_to_h3)
df['d_h3'] = df['d_stop'].map(stop_to_h3)

dist = pd.read_csv(EB_DIR / 'h3_stop_distance.csv')
dist = dist[dist['h3_resolution']==8]
dist['stop_id'] = dist['stop_id'].astype(str)
dist_lookup = dict(zip(zip(dist['h3_cell'], dist['stop_id']), dist['distance_m']))
df['eb_access_dist'] = df.apply(lambda r: dist_lookup.get((r['o_h3'],r['o_stop']),0), axis=1)
df['eb_egress_dist'] = df.apply(lambda r: dist_lookup.get((r['d_h3'],r['d_stop']),0), axis=1)
df['ln_eb_access'] = np.log1p(df['eb_access_dist']/80)
df['ln_eb_egress'] = np.log1p(df['eb_egress_dist']/80)

gtfs_match = pd.read_csv(EB_DIR / 'otp_gtfs_stop_matching.csv')
gtfs_match['otp_stop_id'] = gtfs_match['otp_stop_id'].astype(str)
s2r = dict(zip(gtfs_match['otp_stop_id'], gtfs_match['n_routes']))
df['ln_o_n_routes'] = np.log1p(df['o_stop'].map(s2r).fillna(1))
df['ln_d_n_routes'] = np.log1p(df['d_stop'].map(s2r).fillna(1))

# Split
od_dom = df.loc[df.groupby('od_pair')['choice_prob'].idxmax(),
                ['od_pair','transport_category']].set_index('od_pair')['transport_category']
od_s = od_dom.map(lambda c: 'gtx_related' if 'gtx' in c else c)
tr, te = train_test_split(od_s.index.to_numpy(), test_size=0.2, random_state=42, stratify=od_s.values)
trs, tes = set(tr), set(te)
for od in ['9007_9008','9008_9007']: trs.discard(od); tes.add(od)
train_df = df[df['od_pair'].isin(trs)]
test_df = df[df['od_pair'].isin(tes)]

# MNL functions
def prep(data, features):
    X_l,y_l,w_l,g_l=[],[],[],[]
    gid=0
    for od,grp in data.groupby('od_pair'):
        y=grp['choice_prob'].values.astype(np.float64)
        if abs(y.sum()-1)>0.01: continue
        X_l.append(grp[features].values.astype(np.float64))
        y_l.append(y); w_l.append(np.full(len(grp),float(grp['n_total'].iloc[0])))
        g_l.append(np.full(len(grp),gid,dtype=np.int64)); gid+=1
    return {'X':np.vstack(X_l),'y':np.concatenate(y_l),'w':np.concatenate(w_l),
            'gid':np.concatenate(g_l),'n_groups':gid}

def mnl_nll(b,f):
    V=f['X']@b;Vm=np.full(f['n_groups'],-np.inf);np.maximum.at(Vm,f['gid'],V)
    Vs=V-Vm[f['gid']];eV=np.exp(Vs);se=np.bincount(f['gid'],weights=eV,minlength=f['n_groups'])
    return -np.sum(f['w']*f['y']*(Vs-np.log(se[f['gid']])))

def mnl_grad(b,f):
    V=f['X']@b;Vm=np.full(f['n_groups'],-np.inf);np.maximum.at(Vm,f['gid'],V)
    Vs=V-Vm[f['gid']];eV=np.exp(Vs);se=np.bincount(f['gid'],weights=eV,minlength=f['n_groups'])
    return f['X'].T@(f['w']*(eV/se[f['gid']]-f['y']))

def run(name, feat, neg):
    bounds = [(None,0) if f in neg else (None,None) for f in feat]
    trf = prep(train_df, feat)
    tef = prep(test_df, feat)
    r = minimize(mnl_nll, np.zeros(len(feat)), args=(trf,), jac=mnl_grad,
                 method='L-BFGS-B', bounds=bounds, options={'maxiter':2000,'ftol':1e-12})
    beta = r.x

    gid,w,ng=trf['gid'],trf['w'],trf['n_groups']
    gsz=np.bincount(gid,minlength=ng);wpg=np.bincount(gid,weights=w,minlength=ng)/gsz
    ll0=-np.sum(wpg*np.log(gsz));llb=-r.fun;rho_tr=1-llb/ll0

    tll=-mnl_nll(beta,tef)
    tgid,tng=tef['gid'],tef['n_groups']
    tgsz=np.bincount(tgid,minlength=tng);twpg=np.bincount(tgid,weights=tef['w'],minlength=tng)/tgsz
    tll0=-np.sum(twpg*np.log(tgsz));rho_te=1-tll/tll0

    V=tef['X']@beta;Vm=np.full(tng,-np.inf);np.maximum.at(Vm,tgid,V)
    eV=np.exp(V-Vm[tgid]);se=np.bincount(tgid,weights=eV,minlength=tng);pp=eV/se[tgid]
    top1=sum(np.argmax(pp[tgid==g])==np.argmax(tef['y'][tgid==g]) for g in range(tng))/tng

    H=np.zeros((len(beta),len(beta)))
    for i in range(len(beta)):
        H[i,:]=approx_fprime(beta, lambda b,_i=i: mnl_grad(b,trf)[_i], 1e-5)
    H=(H+H.T)/2
    try: ses=np.sqrt(np.abs(np.diag(np.linalg.inv(H))))
    except: ses=np.sqrt(np.abs(np.diag(np.linalg.pinv(H))))
    ts=beta/np.where(ses>0,ses,1)

    lbl = {'total_ivt_min':'Total IVT','ln_access':'ln(OTP Access)','ln_egress':'ln(OTP Egress)',
           'transfer_walk_time_min':'Transfer walk','ln_eb_access':'ln(EB Access)',
           'ln_eb_egress':'ln(EB Egress)','ln_o_n_routes':'ln(O routes)',
           'ln_d_n_routes':'ln(D routes)','num_transfers':'Transfers',
           'fare_1000won':'Fare','has_bus':'Has bus','has_train':'Has train','has_gtx':'Has GTX'}

    print(f'\n{"="*60}')
    print(f'{name} ({len(feat)}개)')
    print(f'{"="*60}')
    print(f'{"Feature":<22} {"beta":>12} {"t-stat":>10}')
    print('-'*46)
    for f,b,t in zip(feat,beta,ts):
        sig='***' if abs(t)>2.576 else ''
        print(f'{lbl.get(f,f):<22} {b:>12.6f} {t:>10.2f} {sig}')
    print('-'*46)
    print(f'Train rho2: {rho_tr:.4f}')
    print(f'Test rho2:  {rho_te:.4f}')
    print(f'Top-1:      {top1*100:.1f}%')

NEG_BASE = {'total_ivt_min','ln_access','ln_egress','transfer_walk_time_min',
            'num_transfers','fare_1000won'}

run('A. K3 기존',
    ['total_ivt_min','ln_access','ln_egress','transfer_walk_time_min',
     'num_transfers','fare_1000won','has_bus','has_train','has_gtx'],
    NEG_BASE)

run('B. K3 + EB거리',
    ['total_ivt_min','ln_access','ln_egress','transfer_walk_time_min',
     'ln_eb_access','ln_eb_egress',
     'num_transfers','fare_1000won','has_bus','has_train','has_gtx'],
    NEG_BASE | {'ln_eb_access','ln_eb_egress'})

run('C. K3 + 노선수',
    ['total_ivt_min','ln_access','ln_egress','transfer_walk_time_min',
     'ln_o_n_routes','ln_d_n_routes',
     'num_transfers','fare_1000won','has_bus','has_train','has_gtx'],
    NEG_BASE)

run('D. K3 + EB거리 + 노선수',
    ['total_ivt_min','ln_access','ln_egress','transfer_walk_time_min',
     'ln_eb_access','ln_eb_egress','ln_o_n_routes','ln_d_n_routes',
     'num_transfers','fare_1000won','has_bus','has_train','has_gtx'],
    NEG_BASE | {'ln_eb_access','ln_eb_egress'})
