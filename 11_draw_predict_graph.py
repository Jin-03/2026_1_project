import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns



def count_predict(data_path, output_path):
    df = pd.read_csv(data_path, encoding = "utf-8")

    predict = list(df["prediction"])
    print(predict)

    predict_count = []
    for i in range(max(predict)+1):
        predict_count.append({
            "index": i,
            "value": predict.count(i)
        })

    print(predict_count)
    pd.DataFrame(predict_count).to_csv(output_path, index=False)


def draw_predict_graph(data_path, graph_path):
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
    data_predict_path = "silksong_reviews_predict.csv"
    predict_count_path = "predict_count.csv"
    graph_path = "graph/predict_graph.png"
    count_predict(data_predict_path, predict_count_path)
    draw_predict_graph(predict_count_path, graph_path)



if __name__ == "__main__":
    main()