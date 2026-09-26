"""Measure topic-selection evidence and append it to the comparison notebook."""
from pathlib import Path
import json
import hashlib
import xml.etree.ElementTree as ET
import pandas as pd
import nbformat as nbf
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]
data=ROOT/'data'
rows=[]
details={}
parts=[]
for path in sorted(data.glob('1*/**/moldset_labeled_*.csv')):
    frame=pd.read_csv(path)
    frame['group']=path.stem.rsplit('_',1)[1]
    parts.append(frame)
df=pd.concat(parts,ignore_index=True)
features=[c for c in df if c not in ['PassOrFail','group'] and not c.startswith('Unnamed:')]
groups=df.groupby(['group',*features],dropna=False).PassOrFail.agg(['min','max','sum','size'])
conflict=groups['min'].ne(groups['max'])
positive=int(df.PassOrFail.eq(1).sum())
details['topic1']={'labeled_rows':len(df),'label1_rows':positive,'label1_pct':positive/len(df)*100,
                  'unique_feature_groups':len(groups),'duplicate_rows':len(df)-len(groups),
                  'duplicate_pct':(len(df)-len(groups))/len(df)*100,'conflicting_groups':int(conflict.sum()),
                  'label1_in_conflict':int(groups.loc[conflict,'sum'].sum()),
                  'label1_in_conflict_pct':groups.loc[conflict,'sum'].sum()/positive*100}
total=sum(len(pd.read_csv(p)) for p in data.glob('1*/**/*.csv'))
rows.append({'주제':'1 사출성형기','원자료 규모':f'{total:,}행','개별 이상 라벨':f'{len(df):,}행; 코드 1 {positive}행 ({positive/len(df):.2%})',
             '품질·라벨 제약':f'라벨 보유 중 중복 {len(df)-len(groups):,}행; 상반 라벨 {int(conflict.sum())}조합',
             '검증 단위':'CN7/RG3 2그룹; 수집 시각 정보 없음'})
book=next(data.glob('2*/**/*.xlsx'))
raw=pd.read_excel(book,sheet_name='Raw data');result=pd.read_excel(book,sheet_name='result')
scaled=pd.read_csv(next(data.glob('2*/**/*.csv')),encoding='cp949').iloc[:,1:]
details['topic2']={'raw_rows':len(raw),'raw_missing_cells':int(raw.isna().sum().sum()),'raw_dates':raw['working time'].nunique(),
                  'result_rows':len(result),'result_dates':result['working time'].nunique(),'scaled_missing_cells':int(scaled.isna().sum().sum())}
rows.append({'주제':'2 용접기','원자료 규모':f'{len(raw):,}행','개별 이상 라벨':'확인 불가; 불량 집계 23행',
             '품질·라벨 제약':f'원본 결측 0; scaled CSV 결측 {scaled.isna().sum().sum()}셀',
             '검증 단위':f'원본 {raw["working time"].nunique()}일 / 결과 {result["working time"].nunique()}일; 설비·품목 각 1개'})
press=pd.concat([pd.read_csv(p).assign(source=p.stem) for p in data.glob('3*/**/*.csv')],ignore_index=True)
counts=press.Equipment_state.value_counts()
details['topic3']={'rows':len(press),'state1_rows':int(counts[1]),'state1_pct':counts[1]/len(press)*100,
                  'missing_cells':int(press.isna().sum().sum()),'dates_per_state':{str(k):g.TimeStamp.str[:10].nunique() for k,g in press.groupby('Equipment_state')}}
rows.append({'주제':'3 예지보전','원자료 규모':f'{len(press):,}행','개별 이상 라벨':f'{len(press):,}행; 이상 {counts[1]}행 ({counts[1]/len(press):.2%})',
             '품질·라벨 제약':'결측 0; 동일 시각·센서 중복 1행','검증 단위':'정상 1일 / 이상 1일; 서로 다른 날짜'})
root=next(data.glob('4.*'));images={}
for path in root.rglob('*.bmp'):
    with Image.open(path) as im:
        key=hashlib.sha256(str(im.size).encode()+im.convert('RGB').tobytes()).hexdigest()
        images[key]=path.stem
box_count=0;annotated=0
for stem in images.values():
    xml=next(root.rglob(stem+'.xml'))
    n=len(ET.parse(xml).getroot().findall('object'))
    box_count+=n;annotated+=int(n>0)
details['topic4']={'unique_images':len(images),'matched_boxes':box_count,'images_with_boxes':annotated,'images_without_boxes':len(images)-annotated}
rows.append({'주제':'4 X-ray','원자료 규모':f'고유 이미지 {len(images)}장','개별 이상 라벨':f'결함 박스 {box_count}개 / {annotated}장',
             '품질·라벨 제약':f'무박스 이미지 {len(images)-annotated}장; X-ray 주석 20개 원본 미확보','검증 단위':'이미지 15장; 독립 촬영 로트 미확인'})
resource=pd.read_csv(next(data.glob('5*/**/*.csv')))
bad=int((~resource['시간'].between(0,23)).sum())
details['topic5']={'rows':len(resource),'valid_time_rows':len(resource)-bad,'bad_time_rows':bad,'bad_time_pct':bad/len(resource)*100,
                  'missing_cells':int(resource.isna().sum().sum()),'dates':resource['날짜'].nunique()}
rows.append({'주제':'5 자원 최적화','원자료 규모':f'{len(resource):,}행','개별 이상 라벨':'없음; 생산량·자원 사용 수치 존재',
             '품질·라벨 제약':f'시간 오류 {bad}행 ({bad/len(resource):.2%}); 결측 {resource.isna().sum().sum()}셀',
             '검증 단위':f'{resource["날짜"].nunique()}일; 유효 시간 {len(resource)-bad:,}행'})
out=ROOT/'outputs/pca/topic_evidence.json'
out.write_text(json.dumps({'comparison':rows,'details':details},ensure_ascii=False,indent=2,default=int),encoding='utf-8')
path=ROOT/'notebooks/06_topic_comparison.ipynb';nb=nbf.read(path,4)
tag='objective-evidence-v1'
nb.cells=[c for c in nb.cells if tag not in c.metadata.get('tags',[])]
cells=[nbf.v4.new_markdown_cell('''## 객관적 데이터 근거 종합

원본 파일에서 다시 계산한 수치입니다. 임의 배점 합계는 만들지 않습니다. 행·이미지·박스는 단위가 달라 표본 수를 직접 순위화할 수 없습니다. 1번의 코드 1은 불량 의미를 확인해야 하며, 2번 집계값과 5번 생산 여부는 개별 이상 정답으로 취급하지 않습니다.
'''),nbf.v4.new_code_cell('''evidence=json.loads((ROOT/'outputs/pca/topic_evidence.json').read_text(encoding='utf-8'))
display(pd.DataFrame(evidence['comparison']))
display(pd.DataFrame(evidence['details']).T)
d=evidence['details']
print(f"사출성형기 라벨 보유 중 중복 비율: {d['topic1']['duplicate_pct']:.2f}%")
print(f"사출성형기 코드 1 중 반대 라벨의 동일 입력이 있는 비율: {d['topic1']['label1_in_conflict_pct']:.2f}%")
print(f"예지보전 이상 행 / 사출성형기 코드 1 행: {d['topic3']['state1_rows']/d['topic1']['label1_rows']:.2f}배 (독립 표본 수 비교 아님)")
'''),nbf.v4.new_markdown_cell('''### 평가표에 대한 객관적 결론

- 모델 개발 40점과 영향요인·오류분석 15점은 예측값과 정답을 비교할 수 있어야 합니다. 현재 개별 이진 상태/품질 코드가 있는 표형 데이터는 1번과 3번입니다. 4번에는 위치 주석이 있지만 고유 이미지가 15장이고 무박스 이미지가 없습니다.
- 1번은 코드 1이 42행이며, 이 중 36행(85.71%)은 동일 입력의 코드 0 기록이 있습니다. 3번은 이상 600행이 있으나 정상·이상 각각 단 하루의 수집 데이터입니다. 두 주제 모두 행 수를 독립적인 사건 수로 간주하면 안 됩니다.
- 따라서 **현재 데이터로 평가 요구사항을 구현하기 위한 우선 후보는 3번**입니다. 이는 모델 성능 1위라는 결론이 아닙니다. 수집 날짜가 상태와 완전히 겹치는 제약을 해결하지 않으면 다른 날짜로의 일반화는 입증되지 않습니다.
- PCA 설명 분산은 압축 정도입니다. 예지보전 RMS 98.13%를 F1-score 또는 정확도로 해석하지 않습니다. 실제 홀드아웃 F1·재현율·PR-AUC는 아직 측정하지 않았으므로 총점이나 성능 순위를 수치로 제시하지 않습니다.
''')]
for c in cells:c.metadata['tags']=[tag]
nb.cells.extend(cells);nbf.write(nb,path)
print(json.dumps(details,ensure_ascii=False,indent=2,default=int))
