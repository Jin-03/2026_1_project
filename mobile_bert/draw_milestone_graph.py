import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. 데이터 로드 및 날짜 변환
df = pd.read_csv("silksong_reviews_cleaned.csv")
df['date'] = pd.to_datetime(df['timestamp_updated'], format='%Y-%m-%d %H:%M:%S')


# 2. ✨ 핵심: 패치노트 기준 시점(Milestones) 정의
# 출시일, 패치 1.1, 패치 1.2, 그리고 데이터 수집 종료일까지의 타임라인을 설정합니다.
milestones = [
    pd.Timestamp('2025-09-04'),  # 게임 출시일 (Launch)
    pd.Timestamp('2025-09-12'),  # Patch Version 1.0.28497 Now Live
    pd.Timestamp('2025-09-22'),  # Patch Version 1.0.28650 Now Live
    pd.Timestamp('2025-10-14'),  # Patch Version 1.0.28891 Now Live
    pd.Timestamp('2025-11-07'),  # Patch Version 1.0.29242 Now Live
    pd.Timestamp('2026-03-16'),  # Patch Version 1.0.30000 Now Live
    pd.Timestamp('2026-05-27')   # now
]

# 각 구간에 부여할 이름 (에라 라벨)
era_labels = [
    '2025. 9. 4. - 9. 11. ',
    '2025. 9. 12. - 9. 21. ',
    '2025. 9. 22. - 10. 13. ',
    '2025. 10. 14. - 11. 6. ',
    '2025. 11. 7. - 2026. 3. 16. ',
    '2026. 3. 16. - 5. 27. '
]

# 3. pd.cut을 활용해 각 리뷰가 어떤 패치 버전 구간에 속하는지 자동 분류
# 순성문자 날짜 데이터를 기준으로 구간을 나누어 줍니다.
df['Era'] = pd.cut(df['date'], bins=milestones, labels=era_labels, include_lowest=True)

# 결측치(구간 외 날짜) 제거
df = df.dropna(subset=['Era'])

# 4. 리뷰 타입 문자열 매핑 (범례용)
df['Review Type'] = df['voted_up'].map({
    True: 'Recommended (Positive)',
    False: 'Not Recommended (Negative)',
    1: 'Recommended (Positive)',
    0: 'Not Recommended (Negative)'
})

# 5. 에라별 긍정/부정 막대 그래프 그리기
plt.figure(figsize=(12, 6))

sns.countplot(
    data=df,
    x='Era',
    hue='Review Type',
    palette=['skyblue', 'salmon'],
    edgecolor='black',
    alpha=0.8
)

# 그래프 서식 세팅
plt.title("Review Distribution by Patch Eras (Milestones)", fontsize=14, fontweight='bold', pad=15)
plt.xlabel("Game Version / Era", fontsize=11, labelpad=10)
plt.ylabel("Number of Reviews", fontsize=11, labelpad=10)
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.legend(title="Sentiment")

plt.tight_layout()
plt.show()