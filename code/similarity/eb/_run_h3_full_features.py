# -*- coding: utf-8 -*-
"""H3 집계 데이터 + EB 피처(거리, 노선수) 전부 넣고 MNL"""
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

# === H3 집계 데이터 로드 ===
print("[1/3] 데이터 로드...")
h3_df = pd.read_parquet(EB_DIR / 'h3_choice_prob.parquet')

# IVT 추출
stop_df = pd.read_parquet(DATA_DIR / 'route_choice_training.parquet')
if 'alt_idx' not in stop_df.columns:
    stop_df['alt_idx'] = stop_df.groupby('od_pair').cumcount()

ods = set(stop_df['od_pair'].unique())
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
del ivt

# 피처 생성 (stop_df에서)
stop_df = stop_df.merge(ivt_df, on=['od_pair','alt_idx'], how='left')
for c in ['bus_ivt_min','train_ivt_min','gtx_ivt_min']: stop_df[c]=stop_df[c].fillna(0)
for col in ['access_time','egress_time','transfer_walk_time']:
    stop_df[col+'_min'] = stop_df[col]/60
stop_df['ln_access'] = np.log1p(stop_df['access_time_min'])
stop_df['ln_egress'] = np.log1p(stop_df['egress_time_min'])
stop_df['fare_1000won'] = stop_df['fare']/1000
stop_df['total_ivt_min'] = stop_df['bus_ivt_min'] + stop_df['train_ivt_min'] + stop_df['gtx_ivt_min']

# h3_df에 피처 병합
merge_cols = ['od_pair','alt_idx','bus_ivt_min','train_ivt_min','gtx_ivt_min',
              'total_ivt_min','ln_access','ln_egress','fare_1000won',
              'access_time_min','egress_time_min','transfer_walk_time_min']
h3_df = h3_df.merge(stop_df[merge_cols], on=['od_pair','alt_idx'], how='left', suffixes=('','_dup'))
for c in list(h3_df.columns):
    if c.endswith('_dup'): h3_df.drop(columns=c, inplace=True)
del stop_df, ivt_df

# EB 피처 확인/생성
if 'ln_eb_access' not in h3_df.columns:
    h3_df['ln_eb_access'] = np.log1p(h3_df['eb_access_time_min'])
    h3_df['ln_eb_egress'] = np.log1p(h3_df['eb_egress_time_min'])

# 노선 수
gtfs_match = pd.read_csv(EB_DIR / 'otp_gtfs_stop_matching.csv')
gtfs_match['otp_stop_id'] = gtfs_match['otp_stop_id'].astype(str)
s2r = dict(zip(gtfs_match['otp_stop_id'], gtfs_match['n_routes']))
h3_df['ln_o_n_routes'] = np.log1p(h3_df['o_stop'].map(s2r).fillna(1))
h3_df['ln_d_n_routes'] = np.log1p(h3_df['d_stop'].map(s2r).fillna(1))

print(f"  H3 OD: {h3_df['h3_od'].nunique():,}, rows: {len(h3_df):,}")

# === Split ===
print("[2/3] Split...")
# 정류장 OD 기준 split
od_pairs = h3_df['od_pair'].unique()
np.random.seed(42)
# od_pair 기준으로 80/20 split
from sklearn.model_selection import train_test_split as tts

# transport_category 기반 층화
od_dom = h3_df.loc[h3_df.groupby('od_pair')['choice_prob'].idxmax(),
                   ['od_pair','transport_category']].set_index('od_pair')['transport_category']
od_s = od_dom.map(lambda c: 'gtx_related' if 'gtx' in c else c)
tr, te = tts(od_s.index.to_numpy(), test_size=0.2, random_state=42, stratify=od_s.values)
trs, tes = set(tr), set(te)
for od in ['9007_9008','9008_9007']: trs.discard(od); tes.add(od)

h3_train = h3_df[h3_df['od_pair'].isin(trs)]
h3_test = h3_df[h3_df['od_pair'].isin(tes)]
print(f"  Train H3 OD: {h3_train['h3_od'].nunique():,}, Test: {h3_test['h3_od'].nunique():,}")

# === MNL ===
def prep(data, features, od_col='h3_od'):
    X_l,y_l,w_l,g_l=[],[],[],[]
    gid=0
    for od,grp in data.groupby(od_col):
        y=grp['choice_prob'].values.astype(np.float64)
        if abs(y.sum()-1)>0.01 or y.sum()==0: continue
        X_l.append(grp[features].values.astype(np.float64))
        y_l.append(y)
        w = grp['n_matched'].iloc[0] if 'n_matched' in grp.columns else 1.0
        w_l.append(np.full(len(grp), float(w)))
        g_l.append(np.full(len(grp),gid,dtype=np.int64)); gid+=1
    if not X_l: return None
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
    trf = prep(h3_train, feat)
    tef = prep(h3_test, feat)
    if trf is None or tef is None:
        print(f"  데이터 부족"); return

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
    print(f'  Train: {trf["n_groups"]:,} groups, Test: {tef["n_groups"]:,} groups')
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

# === 실행 ===
print("\n[3/3] MNL 추정...")

NEG = {'total_ivt_min','ln_access','ln_egress','transfer_walk_time_min',
       'ln_eb_access','ln_eb_egress','num_transfers','fare_1000won'}

run('A. H3 기존 (OTP walk)',
    ['total_ivt_min','ln_access','ln_egress','transfer_walk_time_min',
     'num_transfers','fare_1000won','has_bus','has_train','has_gtx'],
    NEG)

run('B. H3 + EB거리',
    ['total_ivt_min','ln_access','ln_egress','transfer_walk_time_min',
     'ln_eb_access','ln_eb_egress',
     'num_transfers','fare_1000won','has_bus','has_train','has_gtx'],
    NEG)

run('C. H3 + 노선수',
    ['total_ivt_min','ln_access','ln_egress','transfer_walk_time_min',
     'ln_o_n_routes','ln_d_n_routes',
     'num_transfers','fare_1000won','has_bus','has_train','has_gtx'],
    NEG)

run('D. H3 + EB거리 + 노선수 (전부)',
    ['total_ivt_min','ln_access','ln_egress','transfer_walk_time_min',
     'ln_eb_access','ln_eb_egress','ln_o_n_routes','ln_d_n_routes',
     'num_transfers','fare_1000won','has_bus','has_train','has_gtx'],
    NEG)

run('E. H3 - OTP walk + EB거리 + 노선수',
    ['total_ivt_min',
     'ln_eb_access','ln_eb_egress','ln_o_n_routes','ln_d_n_routes',
     'num_transfers','fare_1000won','has_bus','has_train','has_gtx'],
    NEG)
