import pandas as pd



# 1. 이전 단계에서 에라(Era) 분리가 완료된 데이터 로드
df = pd.read_csv("silksong_reviews_cleaned.csv")
df['date'] = pd.to_datetime(df['timestamp_updated'], format='%Y-%m-%d %H:%M:%S')

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

# 결측치 방지를 위해 Era가 없는 행은 제거
df = df.dropna(subset=['Era'])


def get_stratified_sample(data, target_size=1000):
    """
    긍정/부정 별로 에라(Era)의 비율을 유지하며 target_size만큼 샘플링하는 함수
    """
    sampled_list = []

    # 에라별 비율(Percentage) 계산
    # 예: Launch 에라 80%, Patch 1.1 에라 15%, Patch 1.2 에라 5%
    era_proportions = data['Era'].value_counts(normalize=True)

    print("💡 이 그룹의 패치 에라별 실제 데이터 분포 비율:")
    for era, prop in era_proportions.items():
        print(f"   - {era}: {prop * 100:.2f}%")

    # 각 에라별로 비율에 맞게 개수 배분하여 추출
    for era, prop in era_proportions.items():
        # 추출해야 할 개수 계산 (반올림)
        sample_n = int(round(prop * target_size))

        # 해당 에라의 데이터만 필터링
        era_data = data[data['Era'] == era]

        # 만약 데이터가 계산된 개수보다 적을 경우를 대비한 안전장치
        actual_sample_n = min(sample_n, len(era_data))

        # 랜덤 샘플링 (random_state를 고정하여 언제 돌려도 같은 결과가 나오게 함)
        era_sample = era_data.sample(n=actual_sample_n, random_state=42)
        sampled_list.append(era_sample)

    # 분할 추출한 데이터 하나로 합치기
    return pd.concat(sampled_list)


print("=== 🔵 [1단계] 긍정 리뷰(Recommended) 층화 추출 시작 ===")
pos_df = df[df['voted_up'] == True]
final_pos_sampled = get_stratified_sample(pos_df, target_size=1000)
print(f"➔ 긍정 리뷰 최종 추출 개수: {len(final_pos_sampled)}개\n")

print("=== 🔴 [2단계] 부정 리뷰(Not Recommended) 층화 추출 시작 ===")
neg_df = df[df['voted_up'] == False]
final_neg_sampled = get_stratified_sample(neg_df, target_size=1000)

# ✨ [추가] 만약 반올림 오차로 1000건이 넘었다면 무작위로 1000건만 슬라이싱
if len(final_neg_sampled) > 1000:
    final_neg_sampled = final_neg_sampled.sample(n=1000, random_state=42)

print(f"➔ 부정 리뷰 최종 추출 개수: {len(final_neg_sampled)}개\n")



# 2. 긍정 1000건 + 부정 1000건 합치기
bert_train_dataset = pd.concat([final_pos_sampled, final_neg_sampled]).reset_index(drop=True)

# 3. 데이터가 골고루 섞이도록 전체 셔플(Shuffle) 한 번 해주기
bert_train_dataset = bert_train_dataset.sample(frac=1, random_state=42).reset_index(drop=True)

# 4. 최종 모델 학습용 데이터셋 저장
bert_train_dataset.to_csv("silksong_bert_train.csv", index=False, encoding='utf-8-sig')

print("=" * 50)
print(f"🎉 MobileBERT 학습용 데이터셋 구축 완료!")
print(f"💾 'silksong_bert_train.csv'에 총 {len(bert_train_dataset)}개의 균형 잡힌 데이터가 저장되었습니다.")