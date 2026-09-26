"""Descriptive PCA diagnostics shared by the five topic notebooks.

All outputs are exploratory, not held-out model evaluation.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from IPython.display import display


def fit_pca(reference, evaluation=None, standardize=True):
    reference = reference.replace([np.inf, -np.inf], np.nan)
    evaluation = reference.copy() if evaluation is None else evaluation.replace([np.inf, -np.inf], np.nan)
    medians = reference.median()
    columns = [c for c in reference if pd.notna(medians[c]) and reference[c].nunique() > 1]
    if len(columns) < 2:
        raise ValueError('PCA에 사용할 비상수 변수가 2개 이상 필요합니다.')
    ref = reference[columns].fillna(medians[columns])
    ev = evaluation[columns].fillna(medians[columns])
    scaler = StandardScaler(with_std=standardize)
    zref = scaler.fit_transform(ref)
    z = scaler.transform(ev)
    full = PCA(svd_solver='full').fit(zref)
    ratio = full.explained_variance_ratio_
    rank = int(np.sum(full.explained_variance_ > full.explained_variance_[0] * 1e-12))
    if rank < 2:
        raise ValueError('유효 순위가 2 미만입니다.')
    k95 = int(np.searchsorted(np.cumsum(ratio), .95) + 1)
    # Leave at least one positive-variance direction out for meaningful residuals.
    k = min(k95, rank - 1)
    scores = full.transform(z)
    rec = scores[:, :k] @ full.components_[:k] + full.mean_
    residual = z - rec
    spe = np.sum(residual ** 2, axis=1)
    t2 = np.sum(scores[:, :k] ** 2 / full.explained_variance_[:k], axis=1)
    refscores = full.transform(zref)
    ref_spe = np.sum((zref - (refscores[:, :k] @ full.components_[:k] + full.mean_)) ** 2, axis=1)
    ref_t2 = np.sum(refscores[:, :k] ** 2 / full.explained_variance_[:k], axis=1)
    return dict(model=full, scaler=scaler, columns=columns, omitted=[c for c in reference if c not in columns],
                reference_rows=len(ref), evaluation_rows=len(ev), rank=rank, k95=k95, k=k,
                ratio=ratio, scores=scores, residual=residual, spe=spe, t2=t2,
                ref_spe=ref_spe, ref_t2=ref_t2, medians=medians,
                evaluation_missing=int(evaluation[columns].isna().sum().sum()),
                standardize=standardize, reconstruction=rec)


def summary(result, name, reference_note):
    return {'analysis': name, 'reference': reference_note,
            'reference_rows':result['reference_rows'], 'evaluation_rows':result['evaluation_rows'],
            'features':len(result['columns']), 'rank':result['rank'],
            'PC1_PC2_variance_pct':float(result['ratio'][:2].sum()*100),
            'components_for_95pct':result['k95'], 'residual_components':result['k'],
            'residual_model_variance_pct':float(result['ratio'][:result['k']].sum()*100)}


def diagnostics(result, groups, title, images=False):
    groups = pd.Series(np.asarray(groups).astype(str))
    assert len(groups) == len(result['scores'])
    print(title)
    omitted = result['omitted'] if len(result['omitted']) <= 12 else f"{len(result['omitted'])}개 상수/전부 결측 변수 (목록: pca_result['omitted'])"
    print('제외 변수:', omitted, '/ 평가 데이터 중앙값 대체 셀:', result['evaluation_missing'])
    print(f"95% 설명 성분 수: {result['k95']} / 재구성·T² 계산 성분 수: {result['k']} / 유효 순위: {result['rank']}")
    variance = pd.DataFrame({'PC':np.arange(1,len(result['ratio'])+1),
                            'variance_pct':100*result['ratio'], 'cumulative_pct':100*np.cumsum(result['ratio'])})
    display(variance)
    fig,axes=plt.subplots(1,2,figsize=(13,4))
    axes[0].bar(variance.PC,variance.variance_pct,color='#2563eb')
    axes[0].set(title=title+' · 설명 분산',xlabel='주성분',ylabel='%')
    axes[1].plot(variance.PC,variance.cumulative_pct,marker='o')
    axes[1].axhline(95,color='gray',ls='--',label='95%')
    axes[1].set(xlabel='누적 성분 수',ylabel='누적 설명 분산 (%)',ylim=(0,102));axes[1].legend()
    plt.tight_layout();plt.show()

    fig,axes=plt.subplots(1,2,figsize=(13,5))
    colors=plt.get_cmap('tab10')
    for i,group in enumerate(sorted(groups.unique())):
        indices=np.flatnonzero(groups.to_numpy()==group)
        # Only the scatter is sampled; all statistics and errors use every row.
        selected=np.random.default_rng(42).choice(indices,min(1800,len(indices)),replace=False)
        label=f'{group} (n={len(indices):,})'
        axes[0].scatter(result['scores'][selected,0],result['scores'][selected,1],s=13,alpha=.5,color=colors(i%10),label=label)
        axes[1].scatter(result['t2'][selected],result['spe'][selected],s=13,alpha=.5,color=colors(i%10),label=label)
    axes[0].set(xlabel=f"PC1 ({result['ratio'][0]:.1%})",ylabel=f"PC2 ({result['ratio'][1]:.1%})",title='주성분 점수 · 그룹별 최대 1,800행 표시')
    axes[1].set(xlabel='T²',ylabel='SPE (잔차 제곱합)',title='정상/참조 구조 내 거리와 구조 밖 잔차')
    from matplotlib.ticker import FuncFormatter
    for axis,values in [('x',result['t2']),('y',result['spe'])]:
        pos=values[values>0]; threshold=float(np.median(pos)*.01) if len(pos) else .001
        getattr(axes[1],'set_'+axis+'scale')('symlog',linthresh=max(threshold,1e-8))
        getattr(axes[1],axis+'axis').set_major_formatter(FuncFormatter(lambda x,_:f'{x:.2g}'))
    axes[0].legend(fontsize=8);axes[1].legend(fontsize=8)
    plt.tight_layout();plt.show()

    metrics=pd.DataFrame({'group':groups,'T2':result['t2'],'SPE':result['spe']})
    display(metrics.groupby('group').agg(rows=('SPE','size'),T2_median=('T2','median'),SPE_median=('SPE','median'),SPE_p95=('SPE',lambda x:x.quantile(.95))))
    if not images:
        # Directions are signed unit eigenvectors, not causal importance or correlations.
        coefficients=pd.DataFrame(result['model'].components_[:2].T,index=result['columns'],columns=['PC1','PC2'])
        display(coefficients.rename_axis('주성분 계수 (단위 고유벡터)'))
        fig,axes=plt.subplots(1,2,figsize=(14,max(4,min(10,len(coefficients)*.35))))
        for ax,pc in zip(axes,['PC1','PC2']):
            top=coefficients[pc].abs().nlargest(12).index
            coefficients.loc[top,pc].sort_values().plot.barh(ax=ax,color='#0d9488')
            ax.set_title(pc+' · 절댓값 상위 12개 계수');ax.set_xlabel('주성분 계수 (부호는 임의)')
        plt.tight_layout();plt.show()
        contributions=pd.Series(np.mean(result['residual']**2,axis=0),index=result['columns']).sort_values(ascending=False)
        fig,ax=plt.subplots(figsize=(10,4));contributions.head(12).sort_values().plot.barh(ax=ax,color='#e87924')
        ax.set_title('재구성 오차 기여 · 변수별 평균 잔차 제곱');ax.set_xlabel('평균 잔차 제곱 (표준화 공간)')
        plt.tight_layout();plt.show()
    return metrics
