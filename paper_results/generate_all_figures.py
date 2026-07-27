#!/usr/bin/env python3
"""Generate ALL paper figures from experimental results."""
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd, numpy as np, json, os, glob, statistics, sys
from pathlib import Path

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
OUTPUT_DIR = Path(__file__).parent / 'figures'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({'font.size': 11, 'axes.titlesize': 13, 'axes.labelsize': 12,
    'figure.dpi': 150, 'savefig.dpi': 300, 'savefig.bbox': 'tight'})

COLORS = {'COAST':'#1f77b4','B0':'#ff7f0e','B1':'#2ca02c','B3':'#d62728',
    'B5':'#9467bd','EdgeOff':'#8c564b','no_ownership':'#e377c2','no_lookahead':'#7f7f7f',
    'MARDAM':'#bcbd22','AM':'#17becf','PolyNet':'#aec7e8','Greedy NN':'#ff9896'}

def fig1_learning_curves():
    fig, axes = plt.subplots(1, 2, figsize=(14,5))
    for name, path, color, ax in [
        ('COAST','output/Model_DVRPTWn50m3_260311-0727/train_statistics.csv','#1f77b4',axes[0]),
        ('MARDAM','output/Mardam_DVRPTWn50m3_260315-1328/train_statistics.csv','#bcbd22',axes[0])]:
        if os.path.exists(path):
            df = pd.read_csv(path)
            ax.plot(df['EP'],df['VAL'],label=name,color=color,linewidth=1,alpha=0.4)
            ax.plot(df['EP'],df['VAL'].rolling(10,min_periods=1).mean(),color=color,linewidth=2.5)
    axes[0].set(xlabel='Epoch',ylabel='Validation Cost',title='COAST vs MARDAM'); axes[0].legend(); axes[0].grid(alpha=0.3)
    
    for key,label,color in [('b0','B0','#ff7f0e'),('b1','B1','#2ca02c'),('b3','B3','#d62728'),
        ('b5','B5','#9467bd'),('edgeoff','EdgeOff','#8c564b'),
        ('no_ownership','no_ownership','#e377c2'),('no_lookahead','no_lookahead','#7f7f7f')]:
        # Try multiple possible paths: direct, seed42, seed42_3 (for edgeoff)
        p=f'output/ablation/{key}/train_statistics.csv'
        if not os.path.exists(p): p=f'output/ablation/{key}/seed42_3/train_statistics.csv'
        if not os.path.exists(p): p=f'output/ablation/{key}/seed42/train_statistics.csv'
        if os.path.exists(p):
            df=pd.read_csv(p)
            axes[1].plot(df['EP'],df['VAL'],color=color,linewidth=0.5,alpha=0.3)
            axes[1].plot(df['EP'],df['VAL'].rolling(10,min_periods=1).mean(),color=color,linewidth=2,label=label)
    axes[1].set(xlabel='Epoch',ylabel='Validation Cost',title='Ablation Models'); axes[1].legend(fontsize=8); axes[1].grid(alpha=0.3)
    plt.tight_layout(); fig.savefig(OUTPUT_DIR/'fig_learning_curves.pdf'); plt.close()

def fig2_in_dist():
    models=[('COAST',44.02,3.99,'Proposed'),('no_lookahead',44.73,4.03,'Ablation'),
        ('no_ownership',45.19,4.09,'Ablation'),('B0',45.39,4.10,'Ablation'),
        ('MARDAM',45.02,1.62,'Literature'),('AM',49.22,4.69,'Literature'),
        ('PolyNet',49.18,4.97,'Literature'),('Greedy NN',58.69,6.54,'Classical')]
    m,c,s,t=zip(*models)
    cl=['#1f77b4' if x=='Proposed' else '#2ca02c' if x=='Ablation' else '#ff7f0e' if x=='Literature' else '#d62728' for x in t]
    fig,ax=plt.subplots(figsize=(12,6))
    bars=ax.barh(range(len(m)),c,xerr=s,color=cl,edgecolor='black',linewidth=0.5,capsize=3)
    ax.set(yticks=range(len(m)),yticklabels=m,xlabel='Normalized Cost',title='In-Distribution Comparison (n=50, m=3)')
    ax.invert_yaxis(); ax.grid(alpha=0.3,axis='x')
    for i,(b,co,st) in enumerate(zip(bars,c,s)):
        ax.text(co+st+0.5,b.get_y()+b.get_height()/2,f'{co:.2f}±{st:.2f}',va='center',fontsize=9)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color='#1f77b4',label='Proposed'),Patch(color='#2ca02c',label='Ablation'),
        Patch(color='#ff7f0e',label='Literature'),Patch(color='#d62728',label='Classical')],loc='lower right')
    plt.tight_layout(); fig.savefig(OUTPUT_DIR/'fig2_in_dist_comparison.pdf'); plt.close()

def fig3_ood():
    ood={'ID (n50m3)':44.02,'Tight TW':46.13,'Burst Dyn':46.92,'Sparse':52.98,'Scale (n100m5)':94.53,'Fleet':21.31}
    fig,(ax1,ax2)=plt.subplots(1,2,figsize=(14,5))
    r,c_=list(ood.keys()),list(ood.values())
    bars=ax1.bar(range(len(r)),c_,color=['#1f77b4' if i==0 else '#ff7f0e' for i in range(len(r))],edgecolor='black',linewidth=0.5)
    ax1.set(xticks=range(len(r)),ylabel='Normalized Cost',title='COAST OOD Generalization')
    ax1.set_xticklabels(r,rotation=30)
    ax1.axhline(y=c_[0],color='gray',linestyle='--',alpha=0.5); ax1.grid(alpha=0.3,axis='y')
    for i,(bar,co) in enumerate(zip(bars,c_)):
        ax1.text(bar.get_x()+bar.get_width()/2,bar.get_height()+1,f'{(co-44.02)/44.02*100:+.1f}%',ha='center',fontsize=8,fontweight='bold')
    # Heatmap from OOD eval
    try:
        df=pd.read_csv('output/ood_eval/ood_summary.csv')
        df=df[df['status']=='ok']
        pv=df.pivot_table(index='model',columns='dataset',values='normalized_cost_mean',aggfunc='first')
        pv=pv.reindex(index=[m for m in ['vectra','b0','b1','b3','b5','edgeoff','no_ownership','no_lookahead'] if m in pv.index])
        pv=pv[[c for c in ['id_n50m3','ood_burst_dynamic','ood_n100m5','ood_n50m6','ood_sparse_spatial','ood_tight_tw'] if c in pv.columns]]
        if not pv.empty:
            im=ax2.imshow(pv.values,cmap='YlOrRd',aspect='auto')
            ax2.set_xticks(range(len(pv.columns)))
            ax2.set_xticklabels([c.replace('ood_','').replace('_','\n').replace('id_n50m3','ID') for c in pv.columns],fontsize=7)
            ax2.set_yticks(range(len(pv.index)))
            labels={'vectra':'COAST','b0':'B0','b1':'B1','b3':'B3','b5':'B5','edgeoff':'EdgeOff','no_ownership':'NoOwn','no_lookahead':'NoLook'}
            ax2.set_yticklabels([labels.get(m,m) for m in pv.index],fontsize=9)
            plt.colorbar(im,ax=ax2,label='Cost')
            ax2.set_title('Multi-Model OOD Comparison')
            for i in range(len(pv.index)):
                for j in range(len(pv.columns)):
                    v=pv.values[i,j]
                    if not np.isnan(v): ax2.text(j,i,f'{v:.1f}',ha='center',va='center',fontsize=6,color='white' if v>pv.values.mean() else 'black')
    except: pass
    plt.tight_layout(); fig.savefig(OUTPUT_DIR/'fig3_ood_generalization.pdf'); plt.close()

def fig4_behavioral():
    fig,(ax1,ax2)=plt.subplots(1,2,figsize=(14,5))
    models_b=['COAST','B0','no_ownership','no_lookahead']
    rates=[75.2,76.3,53.2,8.6]
    bars=ax1.bar(models_b,rates,color=[COLORS.get(m,'gray') for m in models_b],edgecolor='black',linewidth=0.5)
    ax1.set(ylabel='Override Rate (%)',title='Impact on Decision Changes'); ax1.grid(alpha=0.3,axis='y')
    for bar,val in zip(bars,rates):
        ax1.text(bar.get_x()+bar.get_width()/2,bar.get_height()+1,f'{val:.1f}%',ha='center',fontsize=10,fontweight='bold')
    ax1.annotate('Lookahead → ~67%\nof all overrides',xy=(3,8.6),xytext=(2.5,30),
        arrowprops=dict(arrowstyle='->',color='red',lw=1.5),fontsize=9,color='red',fontweight='bold')
    raw=[(0.71,'Lookahead\n(67%)','#1f77b4'),(0.46,'Ownership','#ff7f0e'),(0.59,'Edge Feat.','#2ca02c')]
    sz_filt,lbl_filt,col_filt=zip(*[(s,l,c) for s,l,c in raw if s>0])
    total=sum(sz_filt); sz_pct=[s/total*100 for s in sz_filt]
    ax2.pie(sz_pct,labels=lbl_filt,autopct='%1.1f%%',startangle=90,colors=col_filt)
    ax2.set_title('Component Contribution\nto COAST Improvement over B0')
    plt.tight_layout(); fig.savefig(OUTPUT_DIR/'fig_behavioral_analysis.pdf'); plt.close()

def fig5_waterfall():
    # % cost increase vs COAST across PYTH scales — shows B5 collapse at scale
    scales=['n50m3','n100m5','n200m10','n400m20']
    coast=[43.92,94.85,182.38,355.99]
    ablations=[
        ('no_ownership',[45.20,97.44,186.22,362.11],'#e377c2'),
        ('no_lookahead',[44.74,96.44,184.33,357.18],'#7f7f7f'),
        ('EdgeOff',     [45.43,97.25,186.94,365.43],'#8c564b'),
        ('B0',          [45.28,97.35,186.52,363.43],'#ff7f0e'),
        ('B5',          [45.49,101.67,259.10,515.72],'#9467bd'),
    ]
    fig,ax=plt.subplots(figsize=(10,6))
    x=np.arange(len(scales)); w=0.15; offsets=np.linspace(-(len(ablations)-1)/2,(len(ablations)-1)/2,len(ablations))*w
    for (label,costs,color),offset in zip(ablations,offsets):
        deg=[(c-cc)/cc*100 for c,cc in zip(costs,coast)]
        bars=ax.bar(x+offset,deg,width=w,label=label,color=color,edgecolor='black',linewidth=0.4)
        for bar,v in zip(bars,deg):
            if v>5: ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+0.5,f'{v:.0f}%',ha='center',va='bottom',fontsize=7,fontweight='bold')
    ax.set(xticks=x,xticklabels=scales,ylabel='Cost increase vs COAST (%)',
           title='Ablation degradation relative to COAST across PYTH scales')
    ax.legend(fontsize=9); ax.grid(alpha=0.3,axis='y'); ax.set_ylim(bottom=0)
    plt.tight_layout(); fig.savefig(OUTPUT_DIR/'fig_ablation_waterfall.pdf'); plt.close()

def fig6_heatmap():
    dod=[0.10,0.25,0.50,0.75]; tw=[0.25,0.50,0.75,1.00]
    costs=np.array([[41.70,43.20,44.13,45.52],[41.51,42.74,44.28,45.46],[41.21,43.11,44.35,45.75],[42.41,43.57,44.95,46.88]])
    fig,ax=plt.subplots(figsize=(8,6))
    im=ax.imshow(costs,cmap='YlOrRd',aspect='auto')
    ax.set(xticks=range(len(tw)),xticklabels=[f'{t:.2f}' for t in tw],yticks=range(len(dod)),yticklabels=[f'{d:.2f}' for d in dod],
        xlabel='TW Ratio',ylabel='Degree of Dynamism',title='COAST Cost: DoD × TW')
    plt.colorbar(im,ax=ax,label='Cost')
    for i in range(len(dod)):
        for j in range(len(tw)):
            ax.text(j,i,f'{costs[i,j]:.2f}',ha='center',va='center',fontsize=9,fontweight='bold',
                color='white' if costs[i,j]>costs.mean() else 'black')
    plt.tight_layout(); fig.savefig(OUTPUT_DIR/'fig_dynamic_sensitivity.pdf'); plt.close()

print('Generating figures...')
fig1_learning_curves(); print(' 1/6 Learning curves ✓')
fig2_in_dist(); print(' 2/6 In-dist comparison ✓')
fig3_ood(); print(' 3/6 OOD generalization ✓')
fig4_behavioral(); print(' 4/6 Behavioral analysis ✓')
fig5_waterfall(); print(' 5/6 Ablation waterfall ✓')
fig6_heatmap(); print(' 6/6 Dynamic benchmark heatmap ✓')
print(f'All figures → {OUTPUT_DIR}/')
