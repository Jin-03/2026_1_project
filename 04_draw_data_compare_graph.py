import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def main():
    # 1. 데이터 로드 및 글자 수 계산
    df_raw = pd.read_csv("silksong_reviews.csv")
    df_cleaned = pd.read_csv("silksong_reviews_cleaned.csv")

    # 2. 1행 2열의 그래프 레이아웃 설정
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))

    # ------------------------------------------------------------------
    # [왼쪽 그래프: 0~100글자 구간 줌인 및 겹쳐 그리기 (Overlay)]
    # ------------------------------------------------------------------
    # 원본 데이터는 투명도(alpha)를 주어 붉은색으로, 정제 데이터는 파란색으로 겹쳐 그립니다.
    sns.histplot(data=df_raw, x='text_len', color='salmon', label='Raw Data',
                 binrange=(0, 100), bins=10, alpha=0.5, ax=axes[0])
    sns.histplot(data=df_cleaned, x='text_len', color='skyblue', label='Cleaned Data',
                 binrange=(0, 100), bins=10, alpha=0.7, ax=axes[0])

    axes[0].set_title("Micro View: Short Review Filtering (0-100 Chars)", fontsize=13, fontweight='bold')
    axes[0].set_xlabel("Character Length")
    axes[0].set_ylabel("Count")
    axes[0].legend()  # 범례 표시


    sns.histplot(data=df_raw, x='text_len', color='salmon', label='Raw Data',
                 binrange=(0, 1000), bins=20, alpha=0.5, ax=axes[1])
    sns.histplot(data=df_cleaned, x='text_len', color='skyblue', label='Cleaned Data',
                 binrange=(0, 1000), bins=20, alpha=0.7, ax=axes[1])

    axes[1].set_title("0-1000 Chars", fontsize=13, fontweight='bold')
    axes[1].set_xlabel("Character Length")
    axes[1].set_ylabel("Count")
    axes[1].legend()  # 범례 표시


    # ------------------------------------------------------------------
    # [오른쪽 그래프: 전체 데이터 개수 변화 비교 (Bar Chart)]
    # ------------------------------------------------------------------
    raw_count = len(df_raw)
    cleaned_count = len(df_cleaned)

    bars = axes[2].bar(['Before (Raw)', 'After (Cleaned)'], [raw_count, cleaned_count],
                       color=['salmon', 'skyblue'], width=0.4)

    axes[2].set_title("Macro View: Total Review Count Reduction", fontsize=13, fontweight='bold')
    axes[2].set_ylabel("Number of Reviews")

    # 막대 그래프 위에 정확한 숫자 표시하기
    for bar in bars:
        yval = bar.get_height()
        axes[2].text(bar.get_x() + bar.get_width() / 2.0, yval + (raw_count * 0.01),
                     f'{yval:,}', ha='center', va='bottom', fontweight='bold', fontsize=11)

    # 상단 여백을 조금 더 줘서 글자가 잘리지 않게 조절
    axes[2].set_ylim(0, raw_count * 1.1)

    plt.tight_layout()
    plt.savefig("graph/data_compare_graph.png")
    plt.show()


if __name__ == '__main__':
    main()