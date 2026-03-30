# -*- coding: utf-8 -*-
"""H3 OD + OTP walk + EB walk 둘 다 넣은 모형 테스트"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
import json, pickle, sqlite3
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.optimize import minimize, approx_fprime
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent.parent.parent.parent
DATA_DIR = ROOT / 'data' / 'training_set'
EB_DIR = ROOT / 'data' / 'eb'

# 데이터 로드
stop_df = pd.read_parquet(DATA_DIR / 'route_choice_training.parquet')
if 'alt_idx' not in stop_df.columns:
    stop_df['alt_idx'] = stop_df.groupby('od_pair').cumcount()

training_ods = set(stop_df['od_pair'].unique())
conn = sqlite3.connect(str(DATA_DIR / 'otp_cache.db'))
ivt_records = []
for od_pair, blob in conn.execute('SELECT od_pair, data FROM otp_cache'):
    if od_pair not in training_ods: continue
    data = pickle.loads(blob)
    for i, p in enumerate(data['otp_parsed']):
        bus_ivt = train_ivt = gtx_ivt = 0.0
        for leg in p.get('transit_legs', []):
            mode = leg.get('mode', '')
            dur = float(leg.get('duration', 0))
            if mode == 'BUS': bus_ivt += dur
            elif mode == 'GTX': gtx_ivt += dur
            else: train_ivt += dur
        ivt_records.append((od_pair, i, bus_ivt/60, train_ivt/60, gtx_ivt/60))
conn.close()

ivt_df = pd.DataFrame(ivt_records, columns=['od_pair','alt_idx','bus_ivt_min','train_ivt_min','gtx_ivt_min'])
stop_df = stop_df.merge(ivt_df, on=['od_pair','alt_idx'], how='left')
for c in ['bus_ivt_min','train_ivt_min','gtx_ivt_min']:
    stop_df[c] = stop_df[c].fillna(0)
del ivt_records, ivt_df

for col in ['access_time','egress_time','transfer_walk_time']:
    stop_df[col+'_min'] = stop_df[col] / 60
stop_df['ln_access'] = np.log1p(stop_df['access_time_min'])
stop_df['ln_egress'] = np.log1p(stop_df['egress_time_min'])
stop_df['fare_1000won'] = stop_df['fare'] / 1000
stop_df['total_ivt_min'] = stop_df['bus_ivt_min'] + stop_df['train_ivt_min'] + stop_df['gtx_ivt_min']

# H3 데이터
h3_df = pd.read_parquet(EB_DIR / 'h3_choice_prob.parquet')
merge_cols = ['od_pair','alt_idx','bus_ivt_min','train_ivt_min','gtx_ivt_min',
              'total_ivt_min','ln_access','ln_egress','fare_1000won',
              'access_time_min','egress_time_min','transfer_walk_time_min']
h3_df = h3_df.merge(stop_df[merge_cols], on=['od_pair','alt_idx'], how='left', suffixes=('','_dup'))
for c in list(h3_df.columns):
    if c.endswith('_dup'): h3_df.drop(columns=c, inplace=True)
if 'ln_eb_access' not in h3_df.columns:
    h3_df['ln_eb_access'] = np.log1p(h3_df['eb_access_time_min'])
    h3_df['ln_eb_egress'] = np.log1p(h3_df['eb_egress_time_min'])

# Split
od_dominant = stop_df.loc[stop_df.groupby('od_pair')['choice_prob'].idxmax(),
                          ['od_pair','transport_category']].set_index('od_pair')['transport_category']
od_strat = od_dominant.map(lambda c: 'gtx_related' if 'gtx' in c else c)
train_ods_arr, test_ods_arr = train_test_split(
    od_strat.index.to_numpy(), test_size=0.2, random_state=42, stratify=od_strat.values)
train_od_set, test_od_set = set(train_ods_arr), set(test_ods_arr)
for od in ['9007_9008','9008_9007']:
    train_od_set.discard(od); test_od_set.add(od)

h3_train = h3_df[h3_df['od_pair'].isin(train_od_set)]
h3_test = h3_df[h3_df['od_pair'].isin(test_od_set)]

# MNL
def prepare_flat(data, features, od_col, prob_col, weight_col):
    X_l, y_l, w_l, g_l = [], [], [], []
    gid = 0
    for od, grp in data.groupby(od_col):
        y = grp[prob_col].values.astype(np.float64)
        if abs(y.sum()-1.0)>0.01 or y.sum()==0: continue
        X_l.append(grp[features].values.astype(np.float64))
        y_l.append(y)
        w_l.append(np.full(len(grp), float(grp[weight_col].iloc[0]) if weight_col in grp.columns else 1.0))
        g_l.append(np.full(len(grp), gid, dtype=np.int64))
        gid += 1
    return {'X':np.vstack(X_l),'y':np.concatenate(y_l),'w':np.concatenate(w_l),'gid':np.concatenate(g_l),'n_groups':gid}

def mnl_nll(beta, f):
    V = f['X']@beta; Vm = np.full(f['n_groups'],-np.inf); np.maximum.at(Vm,f['gid'],V)
    Vs = V-Vm[f['gid']]; eV = np.exp(Vs); se = np.bincount(f['gid'],weights=eV,minlength=f['n_groups'])
    return -np.sum(f['w']*f['y']*(Vs-np.log(se[f['gid']])))

def mnl_grad(beta, f):
    V = f['X']@beta; Vm = np.full(f['n_groups'],-np.inf); np.maximum.at(Vm,f['gid'],V)
    Vs = V-Vm[f['gid']]; eV = np.exp(Vs); se = np.bincount(f['gid'],weights=eV,minlength=f['n_groups'])
    return f['X'].T@(f['w']*(eV/se[f['gid']]-f['y']))

# === 6번: H3 OD + OTP walk + EB walk ===
BOTH_FEATURES = ['total_ivt_min','ln_access','ln_egress','transfer_walk_time_min',
                  'ln_eb_access','ln_eb_egress',
                  'num_transfers','fare_1000won','has_bus','has_train','has_gtx']
BOTH_LABELS = ['Total IVT','ln(1+OTP Access)','ln(1+OTP Egress)','Transfer walk',
               'ln(1+EB Access)','ln(1+EB Egress)',
               'Transfers','Fare','Has bus','Has train','Has GTX']
BOTH_NEG = {'total_ivt_min','ln_access','ln_egress','transfer_walk_time_min',
            'ln_eb_access','ln_eb_egress','num_transfers','fare_1000won'}

print('=' * 60)
print('6. H3 OD + OTP walk + EB walk (둘 다)')
print('=' * 60)
train_f = prepare_flat(h3_train, BOTH_FEATURES, 'h3_od', 'choice_prob', 'n_matched')
test_f = prepare_flat(h3_test, BOTH_FEATURES, 'h3_od', 'choice_prob', 'n_matched')
print(f'Train: {train_f["n_groups"]:,}, Test: {test_f["n_groups"]:,}')

bounds = [(None,0) if f in BOTH_NEG else (None,None) for f in BOTH_FEATURES]
r = minimize(mnl_nll, np.zeros(len(BOTH_FEATURES)), args=(train_f,),
             jac=mnl_grad, method='L-BFGS-B', bounds=bounds, options={'maxiter':2000,'ftol':1e-12})
beta = r.x

# rho
gid,w,ng = train_f['gid'],train_f['w'],train_f['n_groups']
gsz = np.bincount(gid,minlength=ng); wpg = np.bincount(gid,weights=w,minlength=ng)/gsz
ll0 = -np.sum(wpg*np.log(gsz)); llb = -r.fun; rho_tr = 1-llb/ll0

tll = -mnl_nll(beta,test_f)
tgid,tng = test_f['gid'],test_f['n_groups']
tgsz = np.bincount(tgid,minlength=tng); twpg = np.bincount(tgid,weights=test_f['w'],minlength=tng)/tgsz
tll0 = -np.sum(twpg*np.log(tgsz)); rho_te = 1-tll/tll0

V = test_f['X']@beta; Vm = np.full(tng,-np.inf); np.maximum.at(Vm,tgid,V)
eV = np.exp(V-Vm[tgid]); se = np.bincount(tgid,weights=eV,minlength=tng); pp = eV/se[tgid]
top1 = sum(np.argmax(pp[tgid==g])==np.argmax(test_f['y'][tgid==g]) for g in range(tng))/tng

# SE, t-stat
H = np.zeros((len(beta),len(beta)))
for i in range(len(beta)):
    def grad_i(b, _i=i): return mnl_grad(b, train_f)[_i]
    H[i,:] = approx_fprime(beta, grad_i, 1e-5)
H = (H+H.T)/2
try: ses = np.sqrt(np.abs(np.diag(np.linalg.inv(H))))
except: ses = np.sqrt(np.abs(np.diag(np.linalg.pinv(H))))
ts = beta/np.where(ses>0,ses,1)

print(f'\n{"Feature":<22} {"β":>12} {"t-stat":>10}')
print('-'*46)
for label, b, t in zip(BOTH_LABELS, beta, ts):
    sig = '***' if abs(t)>2.576 else ''
    print(f'{label:<22} {b:>12.6f} {t:>10.2f} {sig}')
print('-'*46)
print(f'Train ρ²: {rho_tr:.4f}')
print(f'Test ρ²:  {rho_te:.4f}')
print(f'Top-1:    {top1*100:.1f}%')

print('\n기존 비교:')
print(f'  3. H3 OD + OTP walk:     Test ρ²=0.5249, Top-1=67.2%')
print(f'  4. H3 OD + EB walk:      Test ρ²=0.4732, Top-1=58.7%')
print(f'  6. H3 OD + OTP+EB walk:  Test ρ²={rho_te:.4f}, Top-1={top1*100:.1f}%')
