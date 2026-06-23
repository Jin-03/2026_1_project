import pandas as pd

def main():
    data = pd.read_csv("silksong_reviews_playtime.csv")
    tier_labels = ['1. Light (<= 2h)', '2. Regular (2h - 30h)', '3. Heavy (> 30h)']

    # 최종 층화 추출 함수
    def final_stratified_sampling(data, target_size=1000):
        sampled = []
        # 긍정/부정 각각 수행
        for status in [True, False]:
            subset = data[data['voted_up'] == status]
            # 플레이타임 티어별 비율 유지
            for tier in tier_labels:
                tier_subset = subset[subset['playtime_tier'] == tier]
                n = int(round((len(subset[subset['playtime_tier'] == tier]) / len(subset)) * target_size))

                # 가중치를 반영한 무작위 추출
                if len(tier_subset) > 0:
                    weights = tier_subset['sample_weight'] / tier_subset['sample_weight'].sum()
                    sampled_item = tier_subset.sample(n=min(n, len(tier_subset)), weights=weights, random_state=42)
                    sampled.append(sampled_item)
        return pd.concat(sampled)


    # 실행
    final_df = final_stratified_sampling(data, target_size=1000)

    # 최종 확인 및 저장
    print(f"최종 학습 데이터셋 구성 완료: {len(final_df)}건")
    final_df.to_csv("silksong_final_train.csv", index=False, encoding='utf-8')

if __name__ == '__main__':
    main()