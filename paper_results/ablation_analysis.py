#!/usr/bin/env python3
"""
COMPREHENSIVE MULTI-SCALE ABLATION ANALYSIS (ALL 7 ABLATIONS)
Uses OOD eval paired instance data. Generates:
  - ablation_analysis/ablation_results.json  (clean numeric data)
  - ablation_analysis/ablation_tables.tex    (LaTeX tables)
Run: PYTHONPATH=. python paper_results/ablation_analysis.py
"""
import json, os, sys, random, glob
from pathlib import Path
from statistics import mean, stdev
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUTPUT = Path('paper_results/ablation_analysis'); OUTPUT.mkdir(parents=True, exist_ok=True)

MODEL_DIR = {'COAST':'vectra','B0':'b0','B1':'b1','B3':'b3','B5':'b5',
    'EdgeOff':'edgeoff','no_ownership':'no_ownership','no_lookahead':'no_lookahead'}
REGIMES = ['id_n50m3','ood_burst_dynamic','ood_n100m5','ood_n50m6','ood_sparse_spatial','ood_tight_tw']
REGIME_LABEL = {'id_n50m3':'ID (n50,m3)','ood_burst_dynamic':'Burst','ood_n100m5':'Scale n100',
    'ood_n50m6':'Fleet','ood_sparse_spatial':'Sparse','ood_tight_tw':'Tight TW'}

def load_all():
    results = defaultdict(lambda: defaultdict(dict))
    for regime in REGIMES:
        for name, dir_name in MODEL_DIR.items():
            fpath = Path('output/ood_eval') / dir_name / f'test_dvrptw_{regime}_500.infer.json'
            if not fpath.exists(): continue
            with open(fpath) as f: data = json.load(f)
            costs = data.get('costs',[])
            if not costs: continue
            results[regime][name] = {
                'mean': mean(costs), 'std': stdev(costs) if len(costs)>1 else 0,
                'n': len(costs), 'tw': data.get('total_tw_violations'),
                'skipped': data.get('total_skipped_customers'),
            }
    return results

def bootstrap_ci(a_costs, b_costs, n=2000, seed=42):
    rng = random.Random(seed)
    diffs = [b_i - a_i for a_i, b_i in zip(a_costs, b_costs)]
    N = len(diffs)
    means = [mean([diffs[rng.randint(0,N-1)] for _ in range(N)]) for _ in range(n)]
    means.sort()
    return mean(diffs), means[int(0.025*(len(means)-1))], means[int(0.975*(len(means)-1))]

print('='*100)
print('MULTI-SCALE ABLATION ANALYSIS (7 ablations + COAST)')
print('='*100)

results = load_all()
print(f'\nLoaded {sum(len(m) for m in results.values())} model×regime entries')

# ─── TABLE A1: MULTI-SCALE PERFORMANCE ────────────────────────
print('\n' + '═'*100)
print('TABLE A1: MULTI-SCALE PERFORMANCE')
print('═'*100)
header = f'{"Model":<16}' + ''.join(f' {REGIME_LABEL.get(r,r):>12}' for r in REGIMES)
print(header); print('─'*100)
model_ranks = defaultdict(list)
for name in MODEL_DIR:
    row = f'{name:<16}'
    for regime in REGIMES:
        d = results.get(regime,{}).get(name)
        if d: row += f' {d["mean"]:>8.2f}±{d["std"]:<4.2f}'; model_ranks[name].append(1)
        else: row += f' {"N/A":>12}'
    print(row)

# ─── TABLE A2: REMOVE-ONE ─────────────────────────────────────
print('\n' + '═'*100)
print('TABLE A2: REMOVE-ONE COMPONENT CONTRIBUTION')
print('═'*100)
print(f'Positive = removing component INCREASES cost → component benefits COAST\n')
header = f'{"Component":<20}' + ''.join(f' {REGIME_LABEL.get(r,r):>12}' for r in REGIMES)
print(header); print('─'*100)
comps = [('Ownership(H1)','COAST','no_ownership'),('Lookahead(H2)','COAST','no_lookahead'),
         ('Edge Feat.(H3)','COAST','EdgeOff'),('MLP Fusion(H4)','COAST','B5'),('All Features','COAST','B0')]
for label, full, base in comps:
    row = f'{label:<20}'
    for regime in REGIMES:
        df = results.get(regime,{}).get(full); db = results.get(regime,{}).get(base)
        if df and db: pct = (df['mean']-db['mean'])/db['mean']*100; row += f' {pct:>+11.1f}%'
        else: row += f' {"N/A":>12}'
    print(row)

# ─── TABLE A3: ISOLATED CONTRIBUTION ─────────────────────────
print('\n' + '═'*100)
print('TABLE A3: ISOLATED CONTRIBUTION (Remove-One from COAST)')
print('═'*100)
print(f'{"Regime":<18} {"-Ownership":>12} {"-Lookahead":>12} {"-Edge":>12} {"-MLP":>12} {"All":>10}')
print('─'*70)
for regime in REGIMES:
    r = results.get(regime,{}); c = r.get('COAST')
    if not c: continue
    no_own = r.get('no_ownership',{}).get('mean',c['mean'])
    no_look = r.get('no_lookahead',{}).get('mean',c['mean'])
    edge_off = r.get('EdgeOff',{}).get('mean',c['mean'])
    b5 = r.get('B5',{}).get('mean',c['mean'])
    b0 = r.get('B0',{}).get('mean',c['mean'])
    print(f'{REGIME_LABEL.get(regime,regime):<18} {no_own-c["mean"]:>+11.2f} {no_look-c["mean"]:>+11.2f} {edge_off-c["mean"]:>+11.2f} {b5-c["mean"]:>+11.2f} {b0-c["mean"]:>+9.2f}')

# ─── EXPORT JSON ──────────────────────────────────────────────
clean = {}
for regime, models in results.items():
    clean[REGIME_LABEL.get(regime,regime)] = {}
    for model, d in models.items():
        clean[REGIME_LABEL.get(regime,regime)][model] = {
            'mean_cost': round(d['mean'],4), 'std_cost': round(d['std'],4),
            'n': d['n'], 'tw_violations': d['tw'], 'skipped': d['skipped']}
with open(OUTPUT/'ablation_results.json','w') as f: json.dump(clean,f,indent=2)

# ─── EXPORT LATEX ─────────────────────────────────────────────
tex = r"""\begin{table}[t]
\centering
\caption{Multi-Scale Ablation: all 7 variants evaluated across 6 test regimes.}
\label{tab:ablation_multi_scale}
\small\begin{tabular}{l""" + 'r'*len(REGIMES) + r"""}
\toprule
Model & """ + ' & '.join(REGIME_LABEL.get(r,r) for r in REGIMES) + r""" \\ \midrule"""
for name in MODEL_DIR:
    tex += f'\n{name}'
    for regime in REGIMES:
        d = results.get(regime,{}).get(name)
        tex += f' & ${d["mean"]:.2f}\\pm{d["std"]:.2f}$' if d else ' & ---'
    tex += r' \\'
tex += r"""
\bottomrule\end{tabular}\end{table}

\begin{table}[t]
\centering
\caption{Remove-One: cost increase when removing each component (positive = component helps).}
\label{tab:remove_one}
\small\begin{tabular}{lrrrrr}
\toprule
Regime & $-$Ownership & $-$Lookahead & $-$Edge & $-$MLP & All \\ \midrule"""
for regime in REGIMES:
    r = results.get(regime,{}); c = r.get('COAST')
    if not c: continue
    def pct(base):
        d = r.get(base,{}).get('mean')
        return f'{(d-c["mean"])/c["mean"]*100:+.2f}\\%' if d else '---'
    tex += f'\n{REGIME_LABEL.get(regime,regime)} & ${pct("no_ownership")}$ & ${pct("no_lookahead")}$ & ${pct("EdgeOff")}$ & ${pct("B5")}$ & ${pct("B0")}$ \\\\'
tex += r"""
\bottomrule\end{tabular}\end{table}"""
with open(OUTPUT/'ablation_tables.tex','w') as f: f.write(tex)

print(f'\n✓ ablation_results.json → {OUTPUT/"ablation_results.json"}')
print(f'✓ ablation_tables.tex   → {OUTPUT/"ablation_tables.tex"}')
