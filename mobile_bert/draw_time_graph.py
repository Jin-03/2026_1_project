import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. 데이터 로드 (정제된 데이터)
# 데이터에 'timestamp_created' 컬럼이 포함되어 있어야 합니다.
df = pd.read_csv("silksong_reviews_cleaned.csv")

# 2. Unix 타임스탬프를 날짜(Datetime) 객체로 변환 (초 단위 기준)
# df['date'] = pd.to_datetime(df['timestamp_created'], unit='s')



# 2. ✨ 변경된 부분: '%Y-%m-%d %H:%M:%S' 형식의 문자열을 날짜 객체로 변환
# format 인자에 저장된 형태를 그대로 지정해 주면 됩니다.
df['date'] = pd.to_datetime(df['timestamp_updated'], format='%Y-%m-%d %H:%M:%S')

# 변환이 잘 되었는지 상위 5개 확인
print(df[['timestamp_updated', 'date']].head())
print(df['timestamp_updated'].dtype)
print(df['date'].dtype) # <M8[ns] (datetime64) 라고 뜨면 성공!


# 3. 날짜에서 '연도-월'만 추출 (예: 2025-09, 2025-10)
df['month'] = df['date'].dt.to_period('M').astype(str)



# 4. 그래프 범례 가독성을 위해 True/False를 문자열로 변경
df['Review Type'] = df['voted_up'].map({
    True: 'Recommended (Positive)',
    False: 'Not Recommended (Negative)',
    1: 'Recommended (Positive)',
    0: 'Not Recommended (Negative)'
})

# 5. 시간 순서대로 월 정렬
df = df.sort_values('month')

# 6. 막대 그래프 그리기 (Grouped Bar Chart)
plt.figure(figsize=(13, 6))

# hue 변수에 'Review Type'을 지정하면 긍/부정이 서로 다른 색상의 막대로 표현됩니다.
sns.countplot(
    data=df,
    x='month',
    hue='Review Type',
    palette=['skyblue', 'salmon'],  # 긍정은 하늘색, 부정은 연홍색
    edgecolor='black',
    alpha=0.8
)

# 그래프 디테일 설정
plt.title("Monthly Review Trend: Recommended vs Not Recommended", fontsize=14, fontweight='bold', pad=15)
plt.xlabel("Month (Year-MM)", fontsize=11, labelpad=10)
plt.ylabel("Number of Reviews", fontsize=11, labelpad=10)

# X축 레이블(월)이 겹치지 않도록 45도 회전
plt.xticks(rotation=45)

# 가독성을 위한 그리드 배치 (Y축 방향만)
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.legend(title="Sentiment", loc='upper right')

plt.tight_layout()
plt.show()