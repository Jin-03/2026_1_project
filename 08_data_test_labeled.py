import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def count_label(data_path, output_path):
    df = pd.read_csv(data_path, encoding = "utf-8")

    label = list(df["label"])
    print(label)

    label_count = []
    for i in range(max(label)+1):
        label_count.append({
            "index": i,
            "value": label.count(i)
        })

    print(label_count)
    pd.DataFrame(label_count).to_csv(output_path, index=False)


def draw_label_graph(data_path, graph_path):
    df = pd.read_csv(data_path, encoding = "utf-8")

    fig, ax = plt.subplots(figsize=(5, 4))
    sns.barplot(x = df['index'], y = df['value'], order=[1,2,0,3], legend=False, palette=['skyblue', 'violet', 'salmon', 'gray'])
    for p in ax.patches:
        ax.text(p.get_x() + (p.get_width() / 2),  # 가로 위치
                p.get_y() + p.get_height(),  # 세로 위치
                f"{p.get_height()}",  # 값 + 표시방법 소수 둘째자리까지
                ha='center')  # 좌우정렬 중간으로

    plt.savefig(graph_path)
    plt.show()


def main():
    data_labeled_path = "silksong_final_train_labeled.csv"
    label_count_path = "label_count.csv"
    graph_path = "graph/label_count.png"
    count_label(data_labeled_path, label_count_path)
    draw_label_graph(label_count_path, graph_path)


if __name__ == "__main__":
    main()