import pandas as pd

# 1. 학습용 균형 데이터셋 로드
df = pd.read_csv("silksong_bert_train.csv")

# 리뷰 텍스트를 모두 소문자로 변경 (대소문자 구분 없이 단어를 찾기 위함)
df['review_lower'] = df['review'].astype(str).str.lower()

# 2. 강력한 부정/긍정 단어 정의
negative_words = ['disappointed', 'terrible', 'boring', 'worst', 'trash', 'waste', 'refund']
positive_words = ['amazing', 'perfect', 'masterpiece', 'love', 'best', 'incredible', 'beautiful']

# 3. 모순 데이터 필터링
# 조건 A: 유저는 '추천(True)'했는데, 텍스트에는 심각한 부정 단어가 포함된 경우
suspicious_pos = df[
    (df['voted_up'] == True) &
    (df['review_lower'].apply(lambda x: any(word in x for word in negative_words)))
]

# 조건 B: 유저는 '비추천(False)'했는데, 텍스트에는 극찬하는 단어가 포함된 경우
suspicious_neg = df[
    (df['voted_up'] == False) &
    (df['review_lower'].apply(lambda x: any(word in x for word in positive_words)))
]

print(f"🚨 검토가 필요한 '긍정 라벨 노이즈' 의심 건수: {len(suspicious_pos)}건")
print(f"🚨 검토가 필요한 '부정 라벨 노이즈' 의심 건수: {len(suspicious_neg)}건")
print("=" * 60)

# 4. 의심 데이터 눈으로 확인하기
print("👀 [확인] '긍정(추천)' 라벨이지만 부정 단어가 섞인 리뷰 샘플 3개:")
for i, row in suspicious_pos.head(3).iterrows():
    print(f"- [{row['Era']}] {row['review'][:120]}...")
print("-" * 60)

print("👀 [확인] '부정(비추천)' 라벨이지만 긍정 단어가 섞인 리뷰 샘플 3개:")
for i, row in suspicious_neg.head(3).iterrows():
    print(f"- [{row['Era']}] {row['review'][:120]}...")