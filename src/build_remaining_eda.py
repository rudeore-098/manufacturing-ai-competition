"""Build independent topic 2-5 notebooks; no source data is modified."""
from pathlib import Path
from textwrap import dedent
import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
COMMON = '''
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager
from IPython.display import display
SEED = 42
ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / 'data').is_dir())
pd.set_option('display.max_columns', 30)
fonts = {f.name for f in font_manager.fontManager.ttflist}
plt.rcParams['font.family'] = [next((f for f in ['Malgun Gothic','AppleGothic','NanumGothic'] if f in fonts), 'DejaVu Sans'), 'DejaVu Sans']
plt.rcParams.update({'axes.unicode_minus':False, 'figure.dpi':110, 'axes.spines.top':False, 'axes.spines.right':False})

def quality(df, exclude=()):
    x = df.drop(columns=list(exclude), errors='ignore')
    out = pd.DataFrame({'dtype':x.dtypes.astype(str), 'missing':x.isna().sum(),
                        'missing_pct':x.isna().mean()*100, 'unique':x.nunique()})
    print(f'행 {len(df):,} / 열 {len(df.columns)} / 제외 열을 뺀 중복 {x.duplicated().sum():,}')
    display(out)
    return out

def corr_plot(df, cols, title):
    corr = df[cols].corr(method='spearman')
    fig, ax = plt.subplots(figsize=(max(7,len(cols)*.65), max(5,len(cols)*.6)))
    im = ax.imshow(corr, cmap='RdBu_r', vmin=-1, vmax=1)
    ax.set_xticks(range(len(cols)),cols,rotation=70,ha='right')
    ax.set_yticks(range(len(cols)),cols); ax.set_title(title)
    fig.colorbar(im, ax=ax); plt.tight_layout(); plt.show()

def hist_grid(df, cols, title):
    fig, axes = plt.subplots(int(np.ceil(len(cols)/3)),3,figsize=(15,3.4*int(np.ceil(len(cols)/3))),squeeze=False)
    for ax,c in zip(axes.flat,cols):
        ax.hist(df[c].replace([np.inf,-np.inf],np.nan).dropna(),bins=40,color='#2563eb',alpha=.8)
        ax.set_title(c,fontsize=10); ax.set_ylabel('행 수')
    for ax in list(axes.flat)[len(cols):]: ax.set_visible(False)
    fig.suptitle(title); plt.tight_layout(); plt.show()
'''

class Notebook:
    def __init__(self, topic, title, plan):
        self.topic = topic
        self.cells = []
        self.md(f'# {topic:02d} · {title}\n\n## EDA 계획\n{plan}\n\n원본 데이터는 수정하지 않습니다. 위에서부터 실행하세요. 프로젝트 `.venv` 커널과 `notebooks/requirements.txt`의 패키지를 사용합니다. 수치와 도표는 전체 데이터를 기준으로 하며 표본·표시 범위를 제한하면 해당 셀에 명시합니다. 탐색 결과는 모델 성능이나 인과 효과를 의미하지 않습니다.')
        self.code(COMMON + f"\nDATA = next((ROOT / 'data').glob('{topic}.*'))")
    def md(self, s): self.cells.append(nbf.v4.new_markdown_cell(dedent(s).strip()))
    def code(self, s): self.cells.append(nbf.v4.new_code_cell(dedent(s).strip()))
    def save(self):
        nb = nbf.v4.new_notebook(cells=self.cells, metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'}})
        nbf.write(nb,ROOT/f'notebooks/{self.topic:02d}_topic_{self.topic:02d}_eda.ipynb')

n=Notebook(2,'용접기 데이터 탐색','''
1. Excel 세 시트와 scaled CSV의 역할·스키마·결측·중복을 확인합니다.
2. 원본 공정값의 분포, 문서에 기재된 수집 범위 이탈, 스케일링 결과를 비교합니다.
3. 설비·품목·날짜별 공정 변동과 변수 간 관계를 살펴봅니다.
4. 불량 결과의 집계 단위와 결합 가능성을 확인하고 집계 수준에서만 시각화합니다.
5. 품질 예측/이상 탐지 주제의 가능성과 검증에 필요한 추가 정보를 정리합니다.
''')
n.md('## 1. 원본과 스케일링 파일\n`data set` 시트의 설명·단위를 기준으로 해석합니다. `result.idx`를 개별 용접 기록 ID라고 가정하지 않습니다.')
n.code('''
book = next(DATA.rglob('*.xlsx'))
sheets = pd.read_excel(book, sheet_name=None)
raw, result, dictionary = sheets['Raw data'], sheets['result'], sheets['data set']
scaled = pd.read_csv(next(DATA.rglob('scaled_data.csv')), encoding='cp949')
display(pd.DataFrame([{'sheet':k,'rows':len(v),'columns':len(v.columns)} for k,v in sheets.items()]))
display(dictionary); display(raw.head()); display(result.head())
qraw=quality(raw, ['idx']); qscaled=quality(scaled, ['Unnamed: 0']); quality(result,['idx','Unnamed: 6'])
process=['weld force(bar)','weld current(kA)','weld Voltage(v)','weld time(ms)']
fig, ax=plt.subplots(figsize=(10,4))
missing_compare=pd.DataFrame({'원본':raw[process].isna().sum().to_numpy(),
                              'scaled CSV':scaled[['용접 가압력','전류','전압','통전시간']].isna().sum().to_numpy()},
                             index=['용접 가압력','전류','전압','통전시간'])
missing_compare.plot.bar(ax=ax,title='공정 변수별 결측 수 · 원본과 scaled CSV',rot=0)
ax.set_ylabel('결측 행 수');plt.tight_layout();plt.show()
''')
n.md('## 2. 원본 분포와 수집 범위\n범위 이탈은 설명 시트의 수집 범위를 벗어난 값입니다. 품질 불량의 정답으로 간주하거나 자동 제거하지 않습니다. 전체 범위와 중앙 98%를 함께 표시해 극단값에 가려진 분포를 확인합니다.')
n.code('''
ranges={'Thickness 1(mm)':(.3,2.3),'Thickness 2(mm)':(.3,2.3),
        'weld force(bar)':(1,12),'weld current(kA)':(12,18),'weld Voltage(v)':(1.5,3.5),'weld time(ms)':(30,120)}
display(raw[list(ranges)].describe().T)
range_audit=pd.DataFrame([{'feature':c,'low':lo,'high':hi,'outside':int((raw[c].notna() & ~raw[c].between(lo,hi)).sum()),'missing':int(raw[c].isna().sum())} for c,(lo,hi) in ranges.items()])
display(range_audit)
fig,axes=plt.subplots(4,2,figsize=(13,13))
for row,c in enumerate(process):
    v=raw[c].dropna(); lo,hi=v.quantile([.01,.99])
    axes[row,0].hist(v,bins=50,color='#2563eb');axes[row,0].set_title(c+' · 전체')
    axes[row,1].hist(v[v.between(lo,hi)],bins=40,color='#0d9488');axes[row,1].set_title(c+' · 1~99% 범위')
    for ax in axes[row]: ax.set_ylabel('행 수')
plt.tight_layout();plt.show()
''')
n.md('## 3. 스케일링 파일의 대응 여부\n원본의 min-max 변환과 CSV를 행별·정렬값별로 대조합니다. 일치하지 않으면 파일 간 순서/전처리 차이를 확인해야 하며, 원본 메타데이터나 불량 집계를 scaled CSV에 행 순서로 붙이지 않습니다.')
n.code('''
mapping=dict(zip(process,['용접 가압력','전류','전압','통전시간']))
audit=[]
fig,axes=plt.subplots(2,2,figsize=(12,7))
for ax,(a,b) in zip(axes.flat,mapping.items()):
    mm=(raw[a]-raw[a].min())/(raw[a].max()-raw[a].min())
    same_len=len(mm)==len(scaled)
    match=np.isclose(mm.to_numpy(),scaled[b].to_numpy(),equal_nan=True) if same_len else np.array([False])
    left,right=np.sort(mm.dropna()),np.sort(scaled[b].dropna())
    audit.append({'feature':b,'row_match_pct':match.mean()*100 if same_len else np.nan,
                  'sorted_values_match':len(left)==len(right) and np.allclose(left,right),
                  'raw_missing':raw[a].isna().sum(),'scaled_missing':scaled[b].isna().sum()})
    for values,label in [(mm,'원본 min-max'),(scaled[b],'scaled CSV')]:
        v=np.sort(values.dropna()); ax.plot(v,np.arange(1,len(v)+1)/len(v),label=label)
    ax.set_xscale('symlog',linthresh=.001)
    ax.set_xticks([0,.001,.01,.1,1],['0','0.001','0.01','0.1','1'])
    ax.set_xlim(0,1.05)
    ax.set(title=b,xlabel='스케일값 (symlog, 선형 구간 0~0.001)',ylabel='누적 비율');ax.legend(fontsize=8)
display(pd.DataFrame(audit));plt.tight_layout();plt.show()
''')
n.md('## 4. 날짜별 공정 변동과 변수 관계\n작업시간이 날짜 단위이므로 초 단위 시계열로 해석하지 않습니다. idx는 생산순번이며 모델 입력에서 제외할 후보입니다.')
n.code('''
keys=['Machine_Name','Item No','working time']
display(raw.groupby(keys).size().rename('records').to_frame())
daily=raw.groupby('working time')[process].median()
fig,axes=plt.subplots(2,2,figsize=(13,7))
for ax,c in zip(axes.flat,process):
    ax.plot(daily.index,daily[c],marker='o');ax.set_title(c+' · 일별 중앙값');ax.tick_params(axis='x',rotation=35)
plt.tight_layout();plt.show()
corr_plot(raw,process,'원본 공정 변수 Spearman 상관관계')
''')
n.md('## 5. 불량 집계의 결합 가능성\n`result`는 날짜·설비·품목별로 불량 유형이 반복되는 구조입니다. `defect`를 불량 건수로 해석한 **가설적 집계**만 보여 줍니다. 개별 행 라벨이나 불량률로 변환하지 않습니다. 결과가 없는 날짜는 0건으로 채우지 않습니다.')
n.code('''
display(result)
coverage=raw.groupby(keys).size().rename('process_records').to_frame().join(
    result.groupby(keys).agg(result_rows=('defect','size'), reported_defect_sum=('defect','sum')),how='outer')
display(coverage)
daily_defect=result.pivot_table(index='working time',columns='defect type',values='defect',aggfunc='sum')
fig,ax=plt.subplots(figsize=(11,4));daily_defect.plot.bar(stacked=True,ax=ax,rot=35)
ax.set_xticklabels(daily_defect.index.strftime('%Y-%m-%d'),rotation=35,ha='right')
ax.set_title('날짜별 defect 합계 · 유형 코드별 (건수 의미 확인 필요)');ax.set_ylabel('defect 합계')
plt.tight_layout();plt.show()
print('원본 관측 날짜:',raw['working time'].nunique(),'결과 관측 날짜:',result['working time'].nunique())
print('결과 미제공 그룹:',int(coverage.result_rows.isna().sum()))
''')
n.md('''
## 6. 주제 선정 해석
- 원본 11,939행은 결측과 문서상 수집 범위 이탈이 없습니다. 반면 scaled CSV에는 열별 15~23개 결측이 있고, 원본 min-max 변환과 값 분포도 일치하지 않습니다. 같은 행 수라는 이유만으로 동일 기록이라고 결합하면 안 됩니다.
- 설비와 품목이 각각 하나이고 소재 두께도 상수입니다. 관측은 9일에 한정되며 불량 결과가 없는 날짜가 1일 있습니다.
- 공정 조건 이상 탐지는 검토할 수 있지만, 범위 이탈을 실제 불량으로 취급하면 안 됩니다.
- 개별 용접 품질 분류에는 각 생산순번에 연결된 정답이 추가로 필요합니다. 날짜별 불량 값을 모든 생산 행에 복제하면 잘못된 라벨이 됩니다.
- 일별 불량 집계 예측은 관측 일수가 적어 검증 근거가 약합니다. 기록이 없는 날짜와 0건을 구별해야 합니다.
- 극단값 때문에 min-max 스케일링이 정상 영역을 압축할 수 있습니다. 전처리는 학습 구간 안에서 추정합니다.
- 날짜 단위 분할을 검토하고 설비·품목이 추가로 확보되는지 확인합니다.
'''); n.save()

n=Notebook(3,'소성가공 예지보전 데이터 탐색','''
1. 정상·이상 파일의 규모, 센서, 상태값, 결측·중복을 확인합니다.
2. 수집 기간과 측정 간격을 조사하고 불연속 지점에서 연속 구간을 나눕니다.
3. 상태별 센서 분포·상관관계 및 대표 연속 구간의 파형을 비교합니다.
4. 구간을 넘지 않는 1초 RMS 특징으로 상태 차이를 살펴봅니다.
5. 이상 탐지와 고장 예측의 차이 및 날짜/구간 단위 검증 한계를 정리합니다.
''')
n.md('## 1. 파일·상태·품질 확인\n파일명 normal/outlier와 상태 코드의 대응을 직접 확인합니다. 센서 단위는 추가 설명이 없어 원래 열 이름과 저장값을 사용합니다.')
n.code('''
frames={p.stem:pd.read_csv(p) for p in sorted(DATA.rglob('*.csv'))}
sensors=['AI0_Vibration','AI1_Vibration','AI2_Current']
for name,df in frames.items():
    print(name);quality(df,['Unnamed: 0'])
    df['time']=pd.to_datetime(df['TimeStamp'],errors='coerce')
    df['source']=name
all_data=pd.concat(frames.values(),ignore_index=True)
counts=pd.crosstab(all_data.source,all_data.Equipment_state)
display(counts)
fig,ax=plt.subplots(figsize=(9,4));counts.plot.bar(ax=ax,rot=0,title='파일별 Equipment_state 개수');ax.set_ylabel('행 수');plt.tight_layout();plt.show()
display(all_data.groupby('source')[sensors].agg(['mean','std','min','median','max']))
''')
n.md('## 2. 측정 간격과 연속 구간\n원본 순서에서 0.1초 간격인 행만 같은 구간으로 묶습니다(허용 오차 0.002초). 중복·역순 시각, 결측 시각, 공백은 구간 경계입니다. 파일 전체를 일정 주파수의 한 파형으로 연결하지 않습니다.')
n.code('''
time_audit=[]; segmented=[]
fig,axes=plt.subplots(1,2,figsize=(12,4))
for ax,(name,df) in zip(axes,frames.items()):
    dt=df.time.diff().dt.total_seconds()
    regular=np.isclose(dt,.1,atol=.002,rtol=0)
    df['segment']=(~regular).cumsum()
    segmented.append(df)
    time_audit.append({'source':name,'start':df.time.min(),'end':df.time.max(),'bad_time':df.time.isna().sum(),
                       'duplicate_time':df.time.duplicated().sum(),'reverse_steps':dt.lt(0).sum(),
                       'gaps_over_0.102s':dt.gt(.102).sum(),'segments':df.segment.nunique(),'median_step_s':dt.median()})
    ax.hist(dt.dropna(),bins=50,color='#2563eb');ax.set_yscale('log');ax.set_title(name);ax.set(xlabel='행 사이 간격 (초)',ylabel='간격 수 (log)')
display(pd.DataFrame(time_audit));plt.tight_layout();plt.show()
all_data=pd.concat(segmented,ignore_index=True)
lengths=all_data.groupby(['source','segment']).size().rename('rows')
display(lengths.groupby(level=0).describe())
''')
n.md('## 3. 상태별 분포와 파형\n대표 파형은 파일별 가장 긴 연속 구간의 처음 100개 행(최대 약 10초)입니다. 임의로 떨어진 구간을 선으로 잇지 않습니다.')
n.code('''
fig,axes=plt.subplots(1,3,figsize=(15,4))
for ax,c in zip(axes,sensors):
    bins=np.histogram_bin_edges(all_data[c].dropna(),bins=60)
    for name,df in frames.items():ax.hist(df[c].dropna(),bins=bins,density=True,histtype='step',label=name,linewidth=1.5)
    ax.set_title(c);ax.set_ylabel('밀도');ax.legend(fontsize=7)
plt.tight_layout();plt.show()
fig,axes=plt.subplots(3,2,figsize=(14,9),sharey='row')
for j,(name,df) in enumerate(frames.items()):
    segment=df.groupby('segment').size().idxmax(); sample=df[df.segment.eq(segment)].head(100)
    t=(sample.time-sample.time.iloc[0]).dt.total_seconds()
    for i,c in enumerate(sensors):
        axes[i,j].plot(t,sample[c],marker='.',lw=.8);axes[i,j].set_title(name+' · '+c);axes[i,j].set_xlabel('구간 시작 후 경과 초')
plt.tight_layout();plt.show()
for name,df in frames.items():corr_plot(df,sensors,name+' · Spearman 상관관계')
''')
n.md('## 4. 연속 구간 내부의 1초 RMS\n10개 표본이 연속인 창만 사용합니다. RMS는 진폭 크기의 요약이며 이상 정답을 새로 생성하지 않습니다. 겹치는 창은 서로 독립인 표본이 아니므로 무작위 분할하면 안 됩니다.')
n.code('''
rms_parts=[]
for (source,segment),g in all_data.groupby(['source','segment'],sort=False):
    rms=g[sensors].pow(2).rolling(10,min_periods=10).mean().pow(.5)
    rms['source']=source; rms['segment']=segment
    rms_parts.append(rms.dropna(subset=sensors))
rms=pd.concat(rms_parts,ignore_index=True)
display(rms.groupby('source')[sensors].agg(['count','median','mean','std']))
fig,axes=plt.subplots(1,3,figsize=(15,4))
for ax,c in zip(axes,sensors):
    groups=list(rms.groupby('source'))
    ax.boxplot([g[c] for _,g in groups],tick_labels=[name.replace('_data','') for name,_ in groups],showfliers=False)
    ax.set_title(c+' · 10표본 RMS');ax.tick_params(axis='x',rotation=15)
plt.tight_layout();plt.show()
print('파일별 시간대:');display(all_data.groupby('source').time.agg(['min','max']))
''')
n.md('''
## 5. 주제 선정 해석
- 정상 20,000행과 이상 600행이 있으며, 표본 수보다 독립적인 수집 세션 수가 검증의 핵심입니다.
- 0.1초 연속성 기준으로 정상은 600구간, 이상은 21구간으로 나뉩니다. 구간은 독립 세션을 보장하지 않습니다. AI0 진동의 표준편차는 정상 약 0.071, 이상 약 0.448로 차이가 있지만 수집 조건 영향을 구분해야 합니다.
- 정상은 7월 12일, 이상은 7월 17일에 수집되었습니다. 상태 차이와 날짜/운전조건 차이가 섞일 수 있어 분포 차이를 고장 효과로 단정할 수 없습니다.
- 현재 데이터는 이상 상태 탐지의 출발점입니다. 고장 발생 시각, 고장 전후 연속 기록, 정비 이력이 없으므로 잔여수명이나 사전 고장 예측을 직접 검증하기 어렵습니다.
- 새 정상/이상 세션을 확보하고 세션 단위로 분할해야 합니다. 겹치는 창은 같은 분할에 두고, 정상 학습 데이터만으로 임계값을 추정합니다.
- 측정 공백을 제거한 것처럼 이어붙여 FFT를 계산하면 시간축이 왜곡되므로, 본 EDA에서는 연속 구간 파형과 RMS를 사용합니다.
'''); n.save()

n=Notebook(4,'X-ray 검사장비 데이터 탐색','''
1. 전체 파일을 조사하고 이미지 디코딩으로 실제 이미지/비이미지 파일을 구분합니다.
2. 도구 예제 XML을 분리하고 X-ray 이름 패턴, 이미지·주석 매칭, 중복을 점검합니다.
3. 해상도·밝기·대비, 결함 개수와 박스 크기·위치를 시각화합니다.
4. 실제 이미지 위에 박스를 표시하고 VOC/YOLO 주석의 대응을 검증합니다.
5. 기존 train/test 목록의 파일 존재·중복을 확인하고 학습 가능 규모를 판단합니다.
''')
n.md('## 1. 파일 실체와 분석 범위\n확장자만으로 이미지라고 판단하지 않습니다. X-ray 원본 후보는 `001_` 또는 `002_`로 시작하는 시각 형식 파일명입니다. 이 규칙으로 도구 예제·결과 이미지를 제외하되, 목록과 제외 수를 표시합니다. 포함된 학습 코드나 노트북은 실행하지 않습니다.')
n.code('''
import re, hashlib
import xml.etree.ElementTree as ET
from PIL import Image, UnidentifiedImageError
from matplotlib.patches import Rectangle
files=sorted(p for p in DATA.rglob('*') if p.is_file())
display(pd.Series([p.suffix or '(없음)' for p in files]).value_counts().rename('files').to_frame())
pattern=re.compile(r'^(001|002)_\\d{8}_\\d{6}\\(\\d+\\)$')
image_rows=[]; unreadable=[]
for p in files:
    if p.suffix.lower() not in ['.jpg','.jpeg','.png','.bmp']:continue
    try:
        with Image.open(p) as im:
            im.load(); arr=np.asarray(im.convert('L'))
            image_rows.append({'path':p,'stem':p.stem,'width':im.width,'height':im.height,'format':im.format,
                               'mean':arr.mean(),'std':arr.std(),'candidate':bool(pattern.match(p.stem)),
                               'pixel_hash':hashlib.sha256(str(im.size).encode()+im.convert('RGB').tobytes()).hexdigest()})
    except (UnidentifiedImageError,OSError) as e:
        unreadable.append({'path':str(p.relative_to(DATA)),'reason':type(e).__name__})
images=pd.DataFrame(image_rows);candidates=images[images.candidate].copy()
unique_images=candidates.drop_duplicates('pixel_hash').copy()
display(pd.DataFrame(unreadable))
print('이미지 확장자:',len(images)+len(unreadable),'디코딩 성공:',len(images),'X-ray 후보 파일:',len(candidates),'픽셀 중복 제거:',len(unique_images))
display(candidates[['stem','format','width','height','pixel_hash']])
display(images.loc[~images.candidate,['path','format']])
''')
n.md('## 2. XML 매칭과 박스 유효성\n전체 XML과 X-ray 후보 XML을 별도로 집계합니다. 이미지가 없는 주석은 박스 분포의 주 분석에서 제외합니다. 좌표는 저장된 VOC 값을 그대로 사용하며 박스 폭은 xmax-xmin으로 계산합니다(1픽셀 원점 규약 확인 필요).')
n.code('''
annotations=[];boxes=[];xml_errors=[]
for p in sorted(DATA.rglob('*.xml')):
    try:
        root=ET.parse(p).getroot();w=float(root.findtext('size/width'));h=float(root.findtext('size/height'))
        objects=root.findall('object')
        annotations.append({'stem':p.stem,'candidate':bool(pattern.match(p.stem)),'xml':p,'width':w,'height':h,'objects':len(objects)})
        for obj in objects:
            coords=[float(obj.findtext('bndbox/'+k)) for k in ['xmin','ymin','xmax','ymax']]
            x1,y1,x2,y2=coords
            boxes.append({'stem':p.stem,'class':obj.findtext('name'),'x1':x1,'y1':y1,'x2':x2,'y2':y2,
                          'bw':x2-x1,'bh':y2-y1,'cx':(x1+x2)/(2*w),'cy':(y1+y2)/(2*h),'area_fraction':(x2-x1)*(y2-y1)/(w*h),
                          'valid':0<=x1<x2<=w and 0<=y1<y2<=h})
    except (ET.ParseError,TypeError,ValueError,ZeroDivisionError) as e:xml_errors.append({'file':str(p),'error':str(e)})
ann=pd.DataFrame(annotations);box=pd.DataFrame(boxes)
xann=ann[ann.candidate].copy();xann['image_present']=xann.stem.isin(candidates.stem)
matched=xann[xann.image_present].merge(unique_images,on='stem',suffixes=('_xml','_image'),validate='one_to_one')
matched['size_match']=(matched.width_xml==matched.width_image)&(matched.height_xml==matched.height_image)
matched_boxes=box[box.stem.isin(matched.stem)].copy()
display(pd.DataFrame({'metric':['전체 XML','X-ray 후보 XML','이미지 매칭 XML','이미지 미확보 XML','후보 이미지 중 XML 없음','XML 파싱 오류'],
                      'count':[len(ann),len(xann),len(matched),int((~xann.image_present).sum()),int((~unique_images.stem.isin(xann.stem)).sum()),len(xml_errors)]}))
display(box.groupby('class').size().rename('전체 XML 박스 수').to_frame())
display(xann.loc[~xann.image_present,['stem','objects']])
display(matched[['stem','objects','size_match']]);display(matched_boxes.groupby('class').size().rename('매칭 박스 수').to_frame())
print('매칭 박스 좌표 오류:',int((~matched_boxes.valid).sum()))
''')
n.md('## 3. 이미지 품질과 결함 크기\n통계는 픽셀 중복을 제거한 실제 X-ray 이미지와 해당 XML만 사용합니다. 작은 박스의 비율은 검출 해상도 선택에 도움이 되지만 성능을 보장하지 않습니다.')
n.code('''
fig,axes=plt.subplots(2,2,figsize=(12,8))
axes[0,0].scatter(unique_images.width,unique_images.height,s=80);axes[0,0].set(xlabel='폭 (px)',ylabel='높이 (px)',title='원본 해상도')
axes[0,1].scatter(unique_images['mean'],unique_images['std']);axes[0,1].set(xlabel='평균 밝기 (0~255)',ylabel='밝기 표준편차',title='밝기와 대비')
axes[1,0].hist(matched.objects,bins=np.arange(0,matched.objects.max()+2)-.5);axes[1,0].set(xlabel='이미지별 박스 수',ylabel='이미지 수')
axes[1,1].scatter(matched_boxes.bw,matched_boxes.bh);axes[1,1].set(xlabel='박스 폭 (px)',ylabel='박스 높이 (px)',title='결함 박스 크기')
plt.tight_layout();plt.show()
fig,ax=plt.subplots(figsize=(6,5));sc=ax.scatter(matched_boxes.cx,matched_boxes.cy,c=matched_boxes.area_fraction,cmap='viridis',s=55)
ax.set(xlim=(0,1),ylim=(1,0),xlabel='중심 x / 폭',ylabel='중심 y / 높이',title='결함 위치 (좌상단 원점)');fig.colorbar(sc,ax=ax,label='이미지 대비 박스 면적');plt.tight_layout();plt.show()
display(matched_boxes[['bw','bh','area_fraction']].describe())
''')
n.md('## 4. 이미지 위 주석 확인\n15장의 고유 X-ray 이미지를 모두 표시합니다. 작은 결함은 추가 확대 뷰로 확인합니다.')
n.code('''
fig,axes=plt.subplots(int(np.ceil(len(unique_images)/3)),3,figsize=(15,3.7*int(np.ceil(len(unique_images)/3))),squeeze=False)
for ax,(_,row) in zip(axes.flat,unique_images.iterrows()):
    with Image.open(row.path) as im:ax.imshow(im.convert('L'),cmap='gray',vmin=0,vmax=255)
    for _,b in matched_boxes[matched_boxes.stem.eq(row.stem)].iterrows():
        ax.add_patch(Rectangle((b.x1,b.y1),b.bw,b.bh,fill=False,edgecolor='#ff3b30',linewidth=1.4))
    ax.set_title(row.stem,fontsize=8);ax.axis('off')
for ax in list(axes.flat)[len(unique_images):]:ax.set_visible(False)
plt.tight_layout();plt.show()
fig,axes=plt.subplots(2,3,figsize=(12,8))
for ax,(_,b) in zip(axes.flat,matched_boxes.head(6).iterrows()):
    row=unique_images[unique_images.stem.eq(b.stem)].iloc[0]
    with Image.open(row.path) as im:ax.imshow(im.convert('L'),cmap='gray',vmin=0,vmax=255)
    ax.add_patch(Rectangle((b.x1,b.y1),b.bw,b.bh,fill=False,edgecolor='red'))
    ax.set_xlim(max(0,b.x1-20),min(row.width,b.x2+20));ax.set_ylim(min(row.height,b.y2+20),max(0,b.y1-20))
    ax.set_title(f'{b.stem[:3]} · {b.bw:.0f} x {b.bh:.0f} px');ax.axis('off')
plt.tight_layout();plt.show()
''')
n.md('## 5. YOLO 주석과 기존 분할 목록\nYOLO는 class, 중심 x/y, 폭/높이의 정규화 좌표입니다. VOC 변환값과 비교하고 기존 목록의 파일 존재 여부를 별도로 검사합니다. 원본 BMP가 있어도 목록의 JPG 경로가 없으면 기존 스크립트에서는 읽을 수 없습니다.')
n.code('''
yolo_dir=next(DATA.rglob('YOLO_darknet')); audits=[]
for _,row in matched.iterrows():
    p=yolo_dir/(row.stem+'.txt')
    vals=np.array([[float(v) for v in line.split()] for line in p.read_text().splitlines() if line.strip()]) if p.exists() else np.empty((0,5))
    b=matched_boxes[matched_boxes.stem.eq(row.stem)]
    expected=np.column_stack([np.zeros(len(b)),b.cx,b.cy,b.bw/row.width_xml,b.bh/row.height_xml])
    ordered=lambda a:a[np.lexsort(a.T[::-1])] if len(a) else a
    equal=vals.shape==expected.shape and np.allclose(ordered(vals),ordered(expected),atol=1e-6)
    audits.append({'stem':row.stem,'yolo_present':p.exists(),'boxes':len(vals),'voc_yolo_match':equal})
display(pd.DataFrame(audits))
split_rows=[]
for name in ['train','test']:
    path=next(DATA.rglob(name+'.txt'))
    for line in path.read_text().splitlines():
        if not line.strip():continue
        rel=Path(line.strip()); stem=rel.stem
        hashes=candidates.loc[candidates.stem.eq(stem),'pixel_hash']
        split_rows.append({'split':name,'entry':line,'stem':stem,'listed_path_exists':(path.parent/rel).exists(),
                           'source_image_available':len(hashes)>0,'pixel_hash':hashes.iloc[0] if len(hashes) else None})
splits=pd.DataFrame(split_rows);display(splits)
display(splits.groupby('split').agg(entries=('entry','size'),listed_files=('listed_path_exists','sum'),available_sources=('source_image_available','sum')))
hash_sets={k:set(g.pixel_hash.dropna()) for k,g in splits.groupby('split')}
print('train/test 픽셀 중복:',len(hash_sets.get('train',set())&hash_sets.get('test',set())))
print('train/test 이름 중복:',len(set(splits.loc[splits.split.eq('train'),'stem'])&set(splits.loc[splits.split.eq('test'),'stem'])))
''')
n.md('''
## 6. 주제 선정 해석
- 이미지 확장자를 가진 코드·설명 파일과 도구 예제 XML이 섞여 있으므로 전체 파일 수를 학습 표본 수로 사용하면 안 됩니다.
- 현재 고유 X-ray 원본은 15장입니다. 수십만 픽셀이나 여러 박스가 있어도 독립 이미지 15장이라는 한계가 사라지지 않습니다.
- XML 833개 중 X-ray 이름 패턴은 35개이며, 15개만 원본과 연결되고 20개는 이미지가 없습니다. 연결된 결함 박스는 23개이고 폭 3~8px, 높이 6~8px입니다.
- 기존 train/test 목록 15개 경로 중 4개만 그대로 존재합니다. BMP 원본은 15장 모두 있으므로 후속 학습 시 경로/포맷 정리가 필요합니다.
- 결함 박스가 매우 작아 원본 해상도 보존과 확대/타일링 실험이 필요합니다. 타일을 나눈다면 원본 이미지 단위로 학습·검증을 분리합니다.
- 확대 뷰의 일부 원본에는 빨간 EDA 박스 외에 두꺼운 밝은색/어두운색 사각 표시가 보입니다. 검사 단계에서 추가한 표시인지 확인하고, 라벨 단서가 이미지에 포함된 경우 표시 없는 원본을 확보해야 합니다.
- 이미지가 없는 X-ray 주석의 원본, 추가 검사 이미지, 정상 무결함 이미지 및 촬영 로트 정보를 확보하는 것이 우선입니다.
- 동봉된 결과 그림과 학습 로그는 재현한 모델 성능이 아닙니다. 본 EDA는 데이터 가용성·주석 상태만 검증합니다.
''');n.save()

n=Notebook(5,'자원 최적화 데이터 탐색','''
1. 데이터 기간·열·결측·중복·날짜/시간 유효성을 점검합니다.
2. 15/30/45/60분과 평균 열의 수학적 관계, 날짜 파생 열의 일관성을 확인합니다.
3. 유효 시간 기록으로 날짜·시간대·요일별 패턴과 생산량 관계를 살펴봅니다.
4. 무생산 시간대의 자원 사용, 생산량 대비 지표, 기상·요금·인원 변수 관계를 확인합니다.
5. 예측 시점에 이용 가능한 변수와 최적화에 필요한 단위·제약조건을 구분합니다.
''')
n.md('## 1. 품질과 시간축 점검\n`시간`은 0~23의 정수로 가정하여 검사합니다. 잘못된 시간값을 다음 날짜로 넘기거나 행 순서로 복원하지 않습니다. 파일명에 augumented가 있으나 증강 방식은 제공되지 않았습니다.')
n.code('''
path=next(DATA.rglob('*.csv'));raw=pd.read_csv(path,encoding='utf-8-sig')
display(raw.head());q=quality(raw)
day=pd.to_datetime(raw['날짜'].astype(str),format='%Y%m%d',errors='coerce')
valid_hour=raw['시간'].between(0,23)&raw['시간'].mod(1).eq(0)
valid=day.notna()&valid_hour
df=raw.copy();df['date']=day;df['timestamp']=pd.NaT
df.loc[valid,'timestamp']=day[valid]+pd.to_timedelta(raw.loc[valid,'시간'],unit='h')
issues=pd.DataFrame({'issue':['잘못된 날짜','잘못된 시간','전체 중복 행','날짜/시간 키 중복(첫 행 제외)','유효 시각 중복(첫 행 제외)'],
                     'rows':[int(day.isna().sum()),int((~valid_hour).sum()),int(raw.duplicated().sum()),int(raw.duplicated(['날짜','시간']).sum()),int(df.loc[valid,'timestamp'].duplicated().sum())]})
display(issues);display(raw.loc[~valid,['날짜','시간']].groupby('날짜').agg(['count','min','max']))
valid_df=df.loc[valid].copy().sort_values('timestamp')
print('전체 기간:',day.min(),day.max(),'유효 시간 행:',len(valid_df))
coverage=valid_df.groupby('date')['시간'].nunique().reindex(pd.date_range(day.min(),day.max(),freq='D'),fill_value=0)
fig,axes=plt.subplots(1,2,figsize=(13,4))
q.missing[q.missing.gt(0)].plot.bar(ax=axes[0],color='#e87924',title='열별 결측 수')
axes[1].plot(coverage.index,coverage);axes[1].set(title='날짜별 유효 시간대 개수',ylabel='고유 시간대 수',ylim=(0,25));axes[1].tick_params(axis='x',rotation=30)
plt.tight_layout();plt.show()
''')
n.md('## 2. 파생 열과 분포\n평균이 4개 구간의 평균과 일치하는지 검사합니다. 예측 목표가 평균일 때 같은 시간의 구간값을 입력하면 목표값이 사실상 노출될 수 있습니다. 소수 인원값은 실제 인원 수인지 별도 지표인지 확인해야 합니다.')
n.code('''
quarters=['15분','30분','45분','60분']
mean4=raw[quarters].mean(axis=1)
checks=pd.Series({'평균과 4구간 평균의 정확 일치 행':int(np.isclose(raw['평균'],mean4).sum()),
                  '평균과 4구간 평균의 차이 1 이내 행':int((raw['평균']-mean4).abs().le(1).sum()),
                  '날짜와 m 불일치':int((day.notna()&raw.m.ne(day.dt.month)).sum()),
                  '날짜와 d 불일치':int((day.notna()&raw.d.ne(day.dt.day)).sum()),
                  'day와 ISO 요일 불일치':int((day.notna()&raw.day.ne(day.dt.dayofweek+1)).sum()),
                  '공장인원 소수값 행':int((raw['공장인원'].notna()&raw['공장인원'].mod(1).ne(0)).sum())})
display(checks.to_frame('rows'))
hist_grid(raw,['평균','생산량','기온','풍속','습도','강수량','전기요금(계절)','공장인원','인건비'],'주요 변수 분포 (전체 원본 행)')
fig,ax=plt.subplots(figsize=(7,3));ax.hist(raw['평균']-mean4,bins=30);ax.set(title='저장된 평균 - 4구간 산술평균',xlabel='차이',ylabel='행 수');plt.tight_layout();plt.show()
''')
n.md('## 3. 시간대·요일·날짜 패턴\n이 절부터 시간 분석에는 유효 시각만 사용합니다. 일별 평균은 관측 시간의 평균이며 누락 시간은 0으로 채우지 않습니다. 평균 열의 단위가 불명확하여 에너지 총량으로 환산하지 않습니다.')
n.code('''
valid_df['weekday']=valid_df.timestamp.dt.dayofweek
hourly=valid_df.groupby('시간')[['평균','생산량']].median()
daily=valid_df.groupby('date')[['평균','생산량']].mean()
fig,axes=plt.subplots(2,2,figsize=(14,8))
for i,c in enumerate(['평균','생산량']):
    axes[0,i].plot(daily.index,daily[c]);axes[0,i].set_title(c+' · 관측 시간의 일별 평균');axes[0,i].tick_params(axis='x',rotation=25)
    axes[1,i].plot(hourly.index,hourly[c],marker='o');axes[1,i].set_title(c+' · 시간대별 중앙값');axes[1,i].set_xlabel('시각')
plt.tight_layout();plt.show()
pivot=valid_df.pivot_table(index='weekday',columns='시간',values='평균',aggfunc='mean').reindex(index=range(7),columns=range(24))
fig,ax=plt.subplots(figsize=(12,4));im=ax.imshow(pivot,aspect='auto',cmap='YlOrRd')
ax.set_yticks(range(7),['월','화','수','목','금','토','일']);ax.set_xticks(range(24));ax.set_title('요일 × 시간대 · 평균 열의 관측 평균')
fig.colorbar(im,ax=ax,label='원자료 값');plt.tight_layout();plt.show()
display(valid_df.groupby('weekday').size().rename('시간 행 수').to_frame())
''')
n.md('## 4. 생산량과 자원 사용 관계\n생산량이 0인 시간과 양수인 시간을 분리합니다. 평균/생산량은 원자료 기준 비율로, 단위가 확인되지 않아 kWh/개 또는 비용으로 부르지 않습니다. 원인·낭비 여부는 운영 조건을 확인해야 합니다.')
n.code('''
valid_df['production_state']=np.where(valid_df['생산량'].eq(0),'생산량 0','생산량 양수')
display(valid_df.groupby('production_state').agg(rows=('평균','size'),mean_resource=('평균','mean'),median_resource=('평균','median')))
fig,axes=plt.subplots(1,3,figsize=(16,4))
hb=axes[0].hexbin(valid_df['생산량'],valid_df['평균'],gridsize=35,mincnt=1,bins='log');fig.colorbar(hb,ax=axes[0],label='행 수 (log)')
axes[0].set(xlabel='생산량',ylabel='평균',title='생산량과 평균의 관계')
groups=list(valid_df.groupby('production_state'))
axes[1].boxplot([g['평균'] for _,g in groups],tick_labels=[k for k,_ in groups],showfliers=False);axes[1].set_title('생산 상태별 평균 · 극단점 숨김')
positive=valid_df[valid_df['생산량'].gt(0)].copy();positive['ratio']=positive['평균']/positive['생산량']
axes[2].scatter(positive['생산량'],positive.ratio,s=7,alpha=.3);axes[2].set(xlabel='생산량 (>0)',ylabel='평균 / 생산량',yscale='log',title='생산량 대비 자원 지표 (log)')
axes[2].set_yticks([.01,.1,1,10,100],['0.01','0.1','1','10','100'])
plt.tight_layout();plt.show()
display(positive.ratio.describe(percentiles=[.5,.9,.95,.99]).to_frame('평균 / 생산량'))
corr_plot(valid_df,['평균','생산량','기온','풍속','습도','강수량','전기요금(계절)','공장인원','인건비'],'유효 시간 행 · Spearman 상관관계')
''')
n.md('''
## 5. 예측·최적화 주제 설계
- 전체 6,168행 중 48행은 시간 범위를 벗어나며, 2021-07-13과 2021-07-15 각 24행입니다. 유효한 6,120행 중 생산량 0은 2,609행입니다.
- 저장된 평균은 모든 행에서 4개 구간의 산술평균과 차이가 1 이내입니다. 이 관계는 목표 누출 가능성을 뒷받침합니다.
- 현재 데이터로 시간대별 자원 수요와 생산량의 관계를 탐색할 수 있습니다. 미래 수요 예측은 시간 순서 분할과 과거 값만으로 만든 지연 특징이 필요합니다.
- 같은 시간의 15/30/45/60분 값은 평균을 직접 설명하는 값이므로 평균 예측의 사전 입력에서 제외합니다. 실제 생산량·기상·인원도 예측 시점에 알려진 계획값인지 사후 실측인지 구분합니다.
- `시간` 오류 행은 원본과 대조하여 복구해야 합니다. 현재 노트북은 해당 행을 시간 분석에서만 제외하고 값 분포에는 포함합니다.
- 비용 최적화에는 전력/에너지 단위, 요금의 적용 방식, 인건비 단위, 생산능력·수요·납기·인원 제약이 필요합니다. 단위 불명의 값을 곱해 실제 비용이나 절감액을 만들지 않습니다.
- 파일명에 증강을 암시하는 표현이 있으므로 원본/증강 구분과 생성 규칙을 확보하고 같은 원본에서 파생된 행은 분할을 공유해야 합니다.
''');n.save()
