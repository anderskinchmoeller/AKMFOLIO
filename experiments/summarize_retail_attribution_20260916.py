"""Summarize the three requested controlled comparisons."""
from pathlib import Path
import json
import hashlib
import sys
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from akm_hrp.backtest.engine import _compute_metrics
from akm_hrp.diagnostics.significance import newey_west_mean_test
OUT = ROOT/'runs/retail_attribution_20260916'
OLD = ROOT/'runs/full_scale_retail_alpha_ml_mpc_fixed_20260916_110248'

def read(path):
    return pd.read_csv(path,index_col=0,parse_dates=True).loc['2012-01-06':]

def metrics(r, cash=0.):
    m = _compute_metrics(r, pd.Series(0., index=r.index), risk_free_rate=cash)
    return {k:m[k] for k in ['cagr','mean_return','vol','sharpe','max_drawdown','total_return','n_obs']}

def main():
    ew = read(OLD/'partial/equal_weight_series.csv')
    original = read(OLD/'partial/retail_alpha_ml_mpc_series.csv')
    full_path = OUT/'full_model_series.csv'
    full = read(full_path) if full_path.exists() else original
    aligned = pd.concat([ew.portfolio_return.rename('ew'),full.portfolio_return.rename('model')],axis=1).dropna()
    assert len(aligned) == 765
    ratio = aligned.model.std()/aligned.ew.std()
    risk_rows=[]
    risk_series={}
    for cash in [0.,.03]:
        # Constant weekly cash return; sensitivity assumption, not observed T-bills.
        weekly_cash=cash/52
        matched=ratio*aligned.ew+(1-ratio)*weekly_cash
        for name,r in [('EW',aligned.ew),('Model',aligned.model),('EW at model volatility',matched)]:
            risk_rows.append({'cash_rate_assumption':cash,'portfolio':name,'ew_allocation':ratio if name.startswith('EW at') else np.nan,**metrics(r,cash)})
        risk_series[f'matched_ew_cash_{cash}']=matched
    pd.DataFrame(risk_rows).to_csv(OUT/'equal_risk_summary.csv',index=False)
    pd.DataFrame(risk_series).to_csv(OUT/'equal_risk_series.csv',index_label='date')
    if not (OUT/'no_ml_signal_series.csv').exists():
        print(pd.DataFrame(risk_rows).to_string(index=False))
        return
    selected=read(OUT/'selected_equal_weight_series.csv')
    common=read(OUT/'full_model_common_cost_series.csv')
    no_ml=read(OUT/'no_ml_signal_series.csv')
    series={'Broad EW, common costs':ew.portfolio_return,
            'Selected stocks EW, common costs':selected.portfolio_return,
            'Full model, common costs':common.portfolio_return,
            'Full model, native costs':full.portfolio_return,
            'No ML signal, native costs':no_ml.portfolio_return}
    frame=pd.concat(series,axis=1).dropna()
    assert len(frame)==765
    frame.to_csv(OUT/'comparison_weekly_returns.csv',index_label='date')
    summary=pd.DataFrame([{'portfolio':name,**metrics(r)} for name,r in frame.items()])
    summary.to_csv(OUT/'comparison_summary.csv',index=False)
    target_vol = min(frame.std())
    matched_all = frame.mul(target_vol/frame.std(),axis=1)
    pd.DataFrame([{'portfolio':name,'risky_allocation':float(target_vol/frame[name].std()),**metrics(r)} for name,r in matched_all.items()]).to_csv(OUT/'all_matched_risk_summary.csv',index=False)
    tests=[]
    for name,a,b in [('Selection','Selected stocks EW, common costs','Broad EW, common costs'),('Weighting','Full model, common costs','Selected stocks EW, common costs'),('ML contribution','Full model, native costs','No ML signal, native costs')]:
        active=frame[a]-frame[b]
        test={'comparison':name,**newey_west_mean_test(active,lags=13)}
        # Paired 13-week circular block bootstrap; retain serial dependence.
        rng=np.random.default_rng(20260916)
        n=len(active); draws=[]; values=active.to_numpy()
        for _ in range(5000):
            starts=rng.integers(0,n,size=int(np.ceil(n/13)))
            indices=((starts[:,None]+np.arange(13))%n).ravel()[:n]
            draws.append(values[indices].mean()*52)
        test['bootstrap_low_95'],test['bootstrap_high_95']=np.quantile(draws,[.025,.975])
        tests.append(test)
    pd.DataFrame(tests).to_csv(OUT/'paired_tests.csv',index=False)
    hashes=json.loads((OUT/'source_hashes.json').read_text())
    source_unchanged=all(hashlib.sha256((ROOT/f).read_bytes()).hexdigest()==h for f,h in hashes.items())
    checks=json.loads((OUT/'validation.json').read_text())
    checks['source_files_unchanged_during_run']=source_unchanged
    checks['scored_observations']=len(frame)
    checks['no_ml_signal_max_weight']=float(pd.read_csv(OUT/'no_ml_signal_diagnostics.csv')['signal_weight__ml_interaction_ensemble'].abs().max())
    (OUT/'validation.json').write_text(json.dumps(checks,indent=2))
    annual=(1+frame).groupby(frame.index.year).prod()-1
    annual.to_csv(OUT/'annual_returns.csv',index_label='year')
    print('EQUAL RISK\n'+pd.DataFrame(risk_rows).to_string(index=False))
    print('COMPARISON\n'+summary.to_string(index=False))
    print('PAIRED TESTS\n'+pd.DataFrame(tests).to_string(index=False))
    print('ANNUAL\n'+annual.round(4).to_string())
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,3,figsize=(15,4.5))
    panels=[('Equal risk · zero cash interest',{'EW + cash':risk_series['matched_ew_cash_0.0'],'Full model':aligned.model}),
            ('Selection and weighting · common costs',{'Broad EW':frame.iloc[:,0],'Selected EW':frame.iloc[:,1],'Full model':frame.iloc[:,2]}),
            ('ML signal contribution · native costs',{'Full model':frame.iloc[:,3],'ML signal disabled':frame.iloc[:,4]})]
    for ax,(title,items) in zip(axes,panels,strict=True):
        for (label,r),color in zip(items.items(),['#536878','#245CBA','#C66D2C'],strict=False):
            ax.plot(r.index,(1+r).cumprod()*100,label=label,color=color,lw=1.6)
        ax.set_title(title,fontsize=11,pad=12)
        ax.set_ylabel('Growth of 100')
        ax.grid(axis='y',alpha=.18)
        ax.legend(frameon=False,fontsize=9,loc='upper left')
    fig.suptitle('Retail model attribution | January 2012–August 2026',fontsize=15,x=.055,ha='left')
    fig.tight_layout(rect=(0,0,1,.93))
    fig.savefig(OUT/'comparison.png',dpi=180,bbox_inches='tight')
    plt.close(fig)

if __name__=='__main__':
    main()
