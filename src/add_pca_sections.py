"""Idempotently append PCA sections without replacing existing EDA cells."""
from pathlib import Path
from textwrap import dedent
import nbformat as nbf

ROOT=Path(__file__).resolve().parents[1]
TAG='pca-eda-v1'
COMMON='''
import sys, json
sys.path.insert(0,str(ROOT / 'src'))
from pca_eda import fit_pca, diagnostics, summary
PCA_OUTPUT=ROOT / 'outputs' / 'pca'
PCA_OUTPUT.mkdir(parents=True,exist_ok=True)
pca_summaries=[]
'''
INTRO='''
## 추가 EDA · PCA로 데이터 구조 비교

**계획:** 참조 데이터와 입력 변수를 명시하고 → 설명 분산과 PC1·PC2 → 주성분 계수 → T²·SPE 및 변수별 잔차를 확인합니다.

이 분석은 **탐색용**입니다. 학습·검증·테스트 성능을 산출하지 않습니다. 참조 데이터에서 중앙값 대체, 상수 열 제거, 표준화와 PCA를 학습하고 비교 대상에 적용합니다. ID·라벨·날짜 코드는 입력에서 제외합니다. 설명 분산은 참조 집합 기준이며, 산점도만 그룹별 최대 1,800행으로 제한하고 통계는 전체 행으로 계산합니다.

95% 설명에 필요한 성분 수와 별개로, SPE 계산은 **유효 순위보다 최소 1개 적은 성분**을 유지합니다. 따라서 유지 분산이 95% 미만일 수 있습니다. 모든 성분을 유지해 잔차가 0이 되는 것을 피하기 위한 EDA 설정이며 최적 성분 수는 아닙니다. T²는 유지한 축에서의 거리, SPE는 버린 축의 잔차 제곱합입니다. 두 지표는 확률도, 이상 정답도 아닙니다.

주성분 계수는 단위 고유벡터의 가중치이며 인과적 중요도가 아닙니다. 주성분 부호는 임의입니다. 주제 간 T²·SPE 절댓값이나 설명 분산으로 모델 성능 순위를 매기지 않습니다.
'''

def append(topic, sections):
    path=ROOT/f'notebooks/{topic:02d}_topic_{topic:02d}_eda.ipynb'
    nb=nbf.read(path,4)
    previous={(c.cell_type,c.source):c for c in nb.cells if TAG in c.metadata.get('tags',[])}
    nb.cells=[c for c in nb.cells if TAG not in c.metadata.get('tags',[])]
    for kind,source in [('md',INTRO),('code',COMMON),*sections,('code',f"(PCA_OUTPUT / 'topic{topic:02d}.json').write_text(json.dumps(pca_summaries,ensure_ascii=False,indent=2),encoding='utf-8')\ndisplay(pd.DataFrame(pca_summaries))")]:
        cell=nbf.v4.new_markdown_cell(dedent(source).strip()) if kind=='md' else nbf.v4.new_code_cell(dedent(source).strip())
        cell.metadata['tags']=[TAG]
        nb.cells.append(previous.get((cell.cell_type,cell.source),cell))
    nbf.write(nb,path)

append(1,[('md','''
### 사출성형기 적용 기준
공정값이 같은 라벨 보유 행은 PCA 참조 집합에서 한 번만 사용하여 복제 횟수의 영향을 줄입니다. 원래 행과 라벨은 삭제하지 않고 모두 투영합니다. CN7/RG3와 라벨 보유 여부를 색으로 비교하고, 라벨별 비교는 그룹 안에서 따로 표시합니다. 0/1의 의미는 여전히 확인이 필요합니다.
'''),('code','''
pca_frames=[]
for name,frame in clean.items():
    part=frame[features].copy();part['dataset']=name
    part['label']=frame['PassOrFail'] if 'PassOrFail' in frame else np.nan
    pca_frames.append(part)
pca_all=pd.concat(pca_frames,ignore_index=True)
reference=pca_all.loc[pca_all.label.notna(),features].drop_duplicates()
pca_result=fit_pca(reference,pca_all[features])
diagnostics(pca_result,pca_all.dataset,'사출성형기 · 라벨 보유 고유 공정값 기준')
pca_summaries.append(summary(pca_result,'01 사출성형기','라벨 보유 공정값 중복 제거; 전체 파일 투영'))
fig,axes=plt.subplots(1,2,figsize=(13,5))
for ax,group in zip(axes,['cn7','rg3']):
    for label,color in [(0,'#2563eb'),(1,'#e87924')]:
        mask=pca_all.dataset.eq('labeled_'+group)&pca_all.label.eq(label)
        ax.scatter(pca_result['scores'][mask,0],pca_result['scores'][mask,1],s=18 if label==0 else 55,alpha=.45 if label==0 else .9,c=color,label=f'라벨 {label} (n={mask.sum()})',marker='o' if label==0 else 'x')
    ax.set(title=group.upper()+' · 라벨별 투영',xlabel='PC1',ylabel='PC2');ax.legend()
plt.tight_layout();plt.show()
conflicts=[]
for name,frame in clean.items():
    if 'PassOrFail' not in frame:continue
    g=frame.groupby(features,dropna=False).PassOrFail.agg(['min','max','sum'])
    conflicts.append({'dataset':name,'상반 라벨 조합':int(g['min'].ne(g['max']).sum()),'충돌 조합 내 라벨1 행':int(g.loc[g['min'].ne(g['max']),'sum'].sum())})
display(pd.DataFrame(conflicts))
'''),('md','''
**해석:** CN7/RG3가 갈라지더라도 품질 라벨이 구분된다는 뜻은 아닙니다. 동일 공정값에 상반된 라벨이 있으면 PCA로도 구분할 수 없습니다. 미보유 데이터가 참조 분포에서 벗어난 정도는 스케일링·공정조건 차이일 수 있습니다. 추후 성능 검증에서는 중복 그룹을 같은 분할에 두고 PCA를 학습 데이터 안에서 다시 학습해야 합니다.
''')])

# Comparison notebook is authored only when empty or previously generated here.
comparison=ROOT/'notebooks/06_topic_comparison.ipynb'
if comparison.stat().st_size == 0 or nbf.read(comparison,4).metadata.get('pca_comparison_generated'):
    cells=[nbf.v4.new_markdown_cell('''# 06 · PCA 결과 비교와 주제 선정

1~5번 노트북의 추가 PCA에서 저장한 요약을 비교합니다. 먼저 해당 노트북을 실행하세요.

**설명 분산 비율은 모델 성능 점수가 아닙니다.** 변수 수, 스케일링 방식, 참조 집합이 다르므로 아래 값은 각 데이터의 압축 정도만 나타냅니다. T²·SPE의 절댓값은 주제 간 비교하지 않습니다. 모든 PCA 결과는 탐색용이며 홀드아웃 F1-score를 측정하지 않았습니다.
'''), nbf.v4.new_code_cell('''from pathlib import Path
import json
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager
from IPython.display import display
ROOT=next(p for p in [Path.cwd(),*Path.cwd().parents] if (p/'data').is_dir())
fonts={f.name for f in font_manager.fontManager.ttflist}
plt.rcParams['font.family']=next((f for f in ['Malgun Gothic','AppleGothic','NanumGothic'] if f in fonts),'DejaVu Sans')
plt.rcParams['axes.unicode_minus']=False
paths=[ROOT/'outputs'/'pca'/f'topic{i:02d}.json' for i in range(1,6)]
missing=[str(p) for p in paths if not p.exists()]
assert not missing,f'1~5번 노트북을 먼저 실행하세요: {missing}'
results=pd.DataFrame([row for p in paths for row in json.loads(p.read_text(encoding='utf-8'))])
display(results)
fig,axes=plt.subplots(1,2,figsize=(14,5))
results.plot.barh(x='analysis',y='PC1_PC2_variance_pct',ax=axes[0],legend=False,color='#2563eb')
axes[0].set(xlabel='PC1+PC2 설명 분산 (%)',ylabel='',xlim=(0,100),title='2차원에 보존되는 참조 집합 변동')
results.plot.barh(x='analysis',y=['components_for_95pct','residual_components'],ax=axes[1])
axes[1].set(xlabel='성분 수',ylabel='',title='95% 설명 성분과 잔차 분석 성분');axes[1].legend(['95% 설명','잔차 분석용'],fontsize=8)
plt.tight_layout();plt.show()
'''),nbf.v4.new_markdown_cell('''## 주제별 판단 기준

| 주제 | PCA로 확인할 구조 | 주제 선정 시 더 중요한 제약 |
|---|---|---|
| 사출성형기 | CN7/RG3, 라벨 유무와 품질 라벨의 투영 비교 | 동일 공정값의 상반 라벨, 소수 라벨 부족. PCA로 라벨 충돌을 해결할 수 없음 |
| 용접기 | 작업 날짜별 공정 변동과 큰 잔차 기록 | 개별 기록의 불량 라벨 없음. scaled CSV와 원본 대응 불명확 |
| 예지보전 | 정상 기준에서 원시 센서와 RMS의 상태별 차이 | 수집 날짜와 상태가 겹침. 겹치는 RMS 창은 독립 표본이 아님 |
| X-ray | 밝기·형상·원본 표시의 픽셀 변동 | 15장, 정상 참조 미확보, 축소 시 작은 결함 소실 가능 |
| 자원 최적화 | 생산 여부·계절·시간별 운영 구조 | 제조 이상 라벨 없음. 파생 변수와 예측 목표의 관계에 주의 |

**다음 결정:** PCA 산점도의 분리만으로 주제를 바꾸지 않습니다. 이상 정답의 신뢰도, 독립적인 검증 단위, 제공된 평가표와의 적합성이 우선입니다. 현재 3번은 상태 라벨이 있어 모델 비교를 진행할 후보이지만, 날짜/운전조건 효과를 분리할 수 있는지 검증해야 합니다. 1번은 라벨 충돌의 원인을 확인해야 하고, 2·4·5번은 각각 라벨·이미지·목표 정의의 보완이 필요합니다.

### 현재 데이터에서 관찰한 결과
- 사출성형기의 PC1+PC2는 약 36.9%만 설명하므로 2차원 겹침만으로 전체 변수의 분류 가능성을 결론 내릴 수 없습니다.
- 용접기는 95% 설명에 4개 성분 모두 필요합니다. 손실이 적은 차원 축소의 이점이 작으며, 3개 성분의 잔차는 보조 탐색 지표입니다.
- 예지보전 원시값은 PC1+PC2가 정상 참조 분산의 약 78.9%, RMS는 약 98.1%를 설명합니다. RMS에서 일부 이상 창은 정상 영역과 겹치고 일부는 크게 벗어납니다. 98.1%는 이상 검출 정확도가 아닙니다.
- X-ray의 큰 잔차는 물체 윤곽과 자세 차이에도 나타나며, 결함 위치를 의미하지 않습니다.
- 자원 최적화는 생산 여부에 따른 운영 구조 차이가 보이지만 이상 정답이나 비용 절감 근거로 바로 사용할 수 없습니다.
''')]
    nb=nbf.v4.new_notebook(cells=cells,metadata={'pca_comparison_generated':True,'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'}})
    nbf.write(nb,comparison)

append(2,[('md','''
### 용접기 적용 기준
Excel 원본의 가압력·전류·전압·통전시간 4개만 사용합니다. scaled CSV는 원본과 대응이 확인되지 않아 함께 학습하지 않습니다. 작업 날짜를 색으로 구분하며, 날짜별 불량 집계를 개별 생산 기록의 라벨로 사용하지 않습니다.
'''),('code','''
pca_result=fit_pca(raw[process])
date_groups=pd.to_datetime(raw['working time']).dt.strftime('%m-%d')
diagnostics(pca_result,date_groups,'용접기 · 원본 공정 변수 4개')
pca_summaries.append(summary(pca_result,'02 용접기','원본 전체 행; 작업 날짜별 투영'))
top=np.argsort(pca_result['spe'])[-10:][::-1]
display(raw.iloc[top][['idx','working time',*process]].assign(SPE=pca_result['spe'][top],T2=pca_result['t2'][top]))
'''),('md','''
**해석:** 날짜별로 다른 위치에 모이거나 재구성 오차가 높은 기록은 공정조건 변화 후보입니다. 불량 정답이 아니므로 이를 이용해 F1-score를 계산할 수 없습니다. 생산순번은 상위 오차 기록을 찾는 용도로만 표시합니다.
''')])

append(3,[('md','''
### 예지보전 적용 기준
정상 파일의 센서값만 참조로 학습하고 정상·이상을 모두 투영합니다. 이어서 앞선 EDA에서 계산한 10표본 RMS도 같은 방식으로 분석합니다. 정상 전체를 참조로 사용하므로 정상 점수는 학습 내 점수입니다. **두 분석 모두 테스트 성능이 아니며**, RMS 창끼리는 겹칠 수 있습니다.
'''),('code','''
pca_result=fit_pca(all_data.loc[all_data.Equipment_state.eq(0),sensors],all_data[sensors])
diagnostics(pca_result,all_data.Equipment_state.map({0:'상태 0 · 정상 파일',1:'상태 1 · 이상 파일'}),'예지보전 · 원시 센서 / 정상 참조')
pca_summaries.append(summary(pca_result,'03 예지보전 원시','정상 파일 전체 행 학습; 정상·이상 투영'))
'''),('code','''
pca_rms=fit_pca(rms.loc[rms.source.eq('press_data_normal'),sensors],rms[sensors])
diagnostics(pca_rms,rms.source,'예지보전 · 10표본 RMS / 정상 참조')
pca_summaries.append(summary(pca_rms,'03 예지보전 RMS','정상 RMS 창 학습; 정상·이상 RMS 투영'))
'''),('md','''
**해석:** 원시값과 RMS의 분리 양상을 비교하되, 겹치는 RMS 창이 표본 수를 늘려 보이게 한다는 점에 유의합니다. 정상·이상 날짜가 달라 센서 차이를 고장 효과로만 설명할 수 없습니다. 실제 이상 탐지 실험은 먼저 수집 구간을 분할하고, 정상 학습 구간만으로 표준화·PCA를 적합한 뒤 검증 구간에서 성분 수와 임계값을 정해야 합니다.
''')])

append(4,[('md','''
### X-ray 적용 기준
픽셀 중복을 제거한 15장만 사용합니다. 각 이미지를 종횡비를 유지해 64×64 캔버스에 배치하고, 여백은 해당 이미지의 밝기 중앙값으로 채웁니다. 픽셀값은 0~1로 변환 후 **평균 중심화만** 하며 픽셀별 표준편차로 나누지 않습니다. 15장으로 4,096개 픽셀 변수를 다루므로 유효 순위는 최대 14입니다.

이 분석은 이미지 외관 비교입니다. 작은 결함은 축소 과정에서 흐려질 수 있어 결함 검출용 특징이나 성능으로 해석하지 않습니다. 원본의 사각 표시·배경·형상이 주성분에 반영될 수 있습니다. 참조와 재구성 대상이 같아 오차가 낙관적으로 작습니다.
'''),('code','''
from PIL import ImageOps
image_vectors=[]
for _,row in unique_images.iterrows():
    with Image.open(row.path) as im:
        gray=im.convert('L');small=ImageOps.contain(gray,(64,64),method=Image.Resampling.LANCZOS)
        canvas=Image.new('L',(64,64),int(np.median(np.asarray(gray))))
        canvas.paste(small,((64-small.width)//2,(64-small.height)//2))
        image_vectors.append(np.asarray(canvas,dtype=float).reshape(-1)/255)
pixels=pd.DataFrame(image_vectors,columns=[f'pixel_{i}' for i in range(4096)])
pca_result=fit_pca(pixels,standardize=False)
diagnostics(pca_result,unique_images.stem.str[:3],'X-ray · 고유 이미지 64×64 / 중심화',images=True)
pca_summaries.append(summary(pca_result,'04 X-ray 픽셀','고유 이미지 전체; 중심화만 적용; 학습 내 오차'))
fig,axes=plt.subplots(1,3,figsize=(12,4))
axes[0].imshow(np.mean(image_vectors,axis=0).reshape(64,64),cmap='gray',vmin=0,vmax=1);axes[0].set_title('평균 이미지')
for i,ax in enumerate(axes[1:]):
    weights=np.zeros(4096)
    positions=[pixels.columns.get_loc(c) for c in pca_result['columns']]
    weights[positions]=pca_result['model'].components_[i]
    limit=np.max(np.abs(weights));ax.imshow(weights.reshape(64,64),cmap='RdBu_r',vmin=-limit,vmax=limit);ax.set_title(f'PC{i+1} 픽셀 계수')
for ax in axes:ax.axis('off')
plt.tight_layout();plt.show()
restored=np.array(image_vectors).copy()
positions=[pixels.columns.get_loc(c) for c in pca_result['columns']]
restored[:,positions]=pca_result['scaler'].inverse_transform(pca_result['reconstruction'])
indices=np.argsort(pca_result['spe'])[-3:][::-1]
fig,axes=plt.subplots(3,3,figsize=(10,9))
for row,idx in enumerate(indices):
    original=np.asarray(image_vectors[idx]).reshape(64,64);recon=restored[idx].reshape(64,64)
    axes[row,0].imshow(original,cmap='gray',vmin=0,vmax=1);axes[row,0].set_title(unique_images.iloc[idx].stem,fontsize=8)
    axes[row,1].imshow(recon,cmap='gray',vmin=0,vmax=1);axes[row,1].set_title(f"{pca_result['k']}개 성분 재구성")
    axes[row,2].imshow(np.abs(original-recon),cmap='magma',vmin=0,vmax=max(np.abs(np.asarray(image_vectors)-restored).max(),1e-9));axes[row,2].set_title('절대 잔차 · 공통 색 범위')
for ax in axes.flat:ax.axis('off')
plt.tight_layout();plt.show()
display(pd.DataFrame({'image':unique_images.stem.to_numpy(),'PC1':pca_result['scores'][:,0],'PC2':pca_result['scores'][:,1], 'SPE':pca_result['spe']}))
'''),('md','''
**해석:** 잔차가 높은 영역이 곧 결함 위치라는 뜻은 아닙니다. 형태·회전·밝기·표시 차이로도 잔차가 생깁니다. 정상 이미지가 별도로 확보되지 않아 정상 참조 기반 이상 검출로 평가하지 않습니다. 15장 전체를 사용한 설명 분산으로 새 이미지의 재구성 능력도 보장할 수 없습니다.
''')])

append(5,[('md','''
### 자원 최적화 적용 기준
유효 시각 6,120행에서 평균·생산량·기상·요금·인원·인건비를 사용합니다. 평균과 수학적으로 중복되는 15/30/45/60분, 날짜/시간 및 파생 날짜 코드는 제외합니다. 전체 운영 상태를 기술하는 EDA이므로 평균을 입력에 포함하지만, 향후 평균을 예측하는 모델의 사전 입력에 이를 포함해서는 안 됩니다.
'''),('code','''
pca_columns=['평균','생산량','기온','풍속','습도','강수량','전기요금(계절)','공장인원','인건비']
pca_result=fit_pca(valid_df[pca_columns])
diagnostics(pca_result,valid_df.production_state,'자원 최적화 · 운영 변수 9개')
pca_summaries.append(summary(pca_result,'05 자원 최적화','유효 시각 전체 행; 생산 여부별 비교'))
fig,axes=plt.subplots(1,2,figsize=(13,5))
for ax,color,title in [(axes[0],valid_df.timestamp.dt.month,'월별 운영 구조'),(axes[1],valid_df['시간'],'시간대별 운영 구조')]:
    pts=ax.scatter(pca_result['scores'][:,0],pca_result['scores'][:,1],c=color,s=8,alpha=.45,cmap='viridis')
    ax.set(title=title,xlabel='PC1',ylabel='PC2');fig.colorbar(pts,ax=ax)
plt.tight_layout();plt.show()
top=np.argsort(pca_result['spe'])[-10:][::-1]
display(valid_df.iloc[top][['timestamp',*pca_columns]].assign(SPE=pca_result['spe'][top]))
'''),('md','''
**해석:** 생산 유무 또는 계절에 따른 분리는 운영 패턴일 수 있으며 이상·낭비·비효율의 정답이 아닙니다. 생산량과 인원 등 강하게 연결된 변수는 같은 운영 정보를 중복 반영할 수 있습니다. 전처리와 PCA는 향후 시간 순서 학습 구간에서 다시 적합해야 합니다.
''')])
