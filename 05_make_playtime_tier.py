import pandas as pd

def main():
    # 1. 데이터 로드 및 전처리
    df = pd.read_csv("silksong_reviews_cleaned.csv")

    # 2. 플레이타임 티어 생성 (학습용 기준)
    bins = [-1, 2, 30, float('inf')]
    tier_labels = ['1. Light (<= 2h)', '2. Regular (2h - 30h)', '3. Heavy (> 30h)']
    df['playtime_tier'] = pd.cut(df['playtime_at_review'], bins=bins, labels=tier_labels)

    # 3. 행동 유형 생성
    df['extra_time'] = df['playtime_forever'] - df['playtime_at_review']

    # 4. 유저 행동 유형(User Behavior Type) 분류 함수
    def segment_behavior(row):
        is_positive = row['voted_up']
        extra_time = row['extra_time']

        if is_positive:
            if extra_time < 2:
                return 'Satisfied'  # 만족하고 깔끔하게 졸업한 유저
            else:
                return 'Loyal Fan'  # 너무 좋아서 계속 붙잡고 있는 유저
        else:
            if extra_time < 2:
                return 'Rage Quit'  # 욕하고 미련 없이 완전히 떠난 유저
            else:
                return 'Love-Hate'  # 욕하면서도 계속 붙잡고 있는 애증의 유저

    df['behavior_type'] = df.apply(segment_behavior, axis=1)

    # 5. 분석 가치 기반 가중치 할당
    # Love-Hate 부정 리뷰는 2배의 가중치를 주어 샘플링 시 더 잘 뽑히게 함
    def get_weight(row):
        if not row['voted_up'] and row['behavior_type'] == 'Love-Hate': return 3.0
        if not row['voted_up']: return 1.5
        return 1.0

    df['sample_weight'] = df.apply(get_weight, axis=1)

    # 결과 확인
    print("\n실크송 유저들의 플레이타임 분포:")
    print(df['playtime_tier'].value_counts())
    print("\n 비율:")
    print(df['playtime_tier'].value_counts(normalize=True) * 100)


    print("\n실크송 유저들의 행동 유형 분포:")
    print(df['behavior_type'].value_counts())
    print("\n 비율:")
    print(df['behavior_type'].value_counts(normalize=True) * 100)

    df.to_csv("silksong_reviews_playtime.csv", index=False, encoding='utf-8')


if __name__ == '__main__':
    main()