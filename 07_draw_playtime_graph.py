import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def draw_playtime_graph(data_path, graph_path):
    # 1. 샘플링된 학습용 데이터셋 로드
    df = pd.read_csv(data_path)

    # 2. X축 정렬 순서 강제 고정 (Light -> Regular -> Heavy)
    tier_order = ['1. Light (<= 2h)', '2. Regular (2h - 30h)', '3. Heavy (> 30h)']
    df['playtime_tier'] = pd.Categorical(df['playtime_tier'], categories=tier_order, ordered=True)
    behavior_order = ['Satisfied', 'Loyal Fan', 'Rage Quit', 'Love-Hate']

    # 3. 그래프 그리기
    plt.figure(figsize=(12, 6))

    ax = sns.countplot(
        data=df,
        x='playtime_tier',
        hue='behavior_type',
        hue_order=behavior_order,
        order=tier_order,
        palette=['skyblue', 'blue', 'salmon', 'red'],  # 긍정: 하늘색, 부정: 연홍색
        edgecolor='black',
        alpha=0.8,

    )

    # 4. 막대 상단에 정확한 데이터 개수(Count) 표시하기
    for p in ax.patches:
        height = p.get_height()
        if height > 0:  # 데이터가 있는 막대만 표시
            ax.annotate(f'{int(height):,}',
                        (p.get_x() + p.get_width() / 2., height),
                        ha='center', va='center',
                        xytext=(0, 8),  # 막대 상단에서의 마진
                        textcoords='offset points',
                        fontweight='bold', fontsize=10, color='black')

    # 5. 그래프 세부 서식 세팅 (폰트 에러 방지 영문 표기)
    plt.title("Review Sentiment Distribution by Playtime Tiers", fontsize=14, fontweight='bold', pad=15)
    plt.xlabel("User Playtime Tier", fontsize=11, labelpad=10)
    plt.ylabel("Number of Reviews", fontsize=11, labelpad=10)

    # 숫자가 위에 적히므로 Y축 상단 여백을 조금 더 넓혀줍니다.
    plt.ylim(0, df['playtime_tier'].value_counts().max() * 0.7)

    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.legend(title="Sentiment", loc='upper right')

    plt.tight_layout()
    plt.savefig(graph_path)
    plt.show()


def main():
    data_all = "silksong_reviews_playtime.csv"
    data_all_graph = "graph/silksong_reviews_playtime_graph.png"
    data_sampled = "silksong_final_train.csv"
    data_sampled_graph = "graph/silksong_final_train.png"

    draw_playtime_graph(data_all, data_all_graph)
    draw_playtime_graph(data_sampled, data_sampled_graph)




if __name__ == '__main__':
    main()