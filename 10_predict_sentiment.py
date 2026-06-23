import torch
from transformers import MobileBertTokenizer, MobileBertForSequenceClassification
import pandas as pd
import os

# 1. 모델과 토크나이저 불러오기
model_path = 'mobilebert_silksong.pt'  # 저장된 폴더 경로
tokenizer = MobileBertTokenizer.from_pretrained('mobilebert-uncased')
model = MobileBertForSequenceClassification.from_pretrained(model_path)

# 2. 추론 모드 설정 (모델을 평가 모드로 변경)
model.eval()


def predict_sentiment(review_text):
    # 3. 텍스트 토큰화
    inputs = tokenizer(review_text, return_tensors="pt", truncation=True, max_length=512, padding="max_length")

    # 4. 모델 예측
    with torch.no_grad():
        outputs = model(**inputs)

    # 5. 결과 해석
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    prediction = torch.argmax(probs, dim=-1).item()

    confidence = probs[0][prediction].item() * 100
    #
    # print(f"리뷰: {review_text}")
    # print(f"예측 결과: {prediction} ({confidence:.2f}% 확신)")
    # if prediction == 0:
    #     sentiment = 'negative'
    # elif prediction == 1:
    #     sentiment = 'positive'
    # elif prediction == 2:
    #     sentiment = 'neutral'
    # elif prediction == 3:
    #     sentiment = 'unrelated'

    return prediction, confidence

#
# # 6. 실제 사용 예시
# sample_review = "The boss design is too punishing, I really struggled to pass."
# sentiment, conf = predict_sentiment(sample_review)
#
# print(f"리뷰: {sample_review}")
# print(f"예측 결과: {sentiment} ({conf:.2f}% 확신)")

def main():
    review = pd.read_csv('silksong_reviews_playtime.csv')

    prediction = []
    confidence = []
    # len(review)
    for i in range(len(review)):
        review_text = review['review'].iloc[i]
        pred, conf = predict_sentiment(review_text)
        prediction.append(pred)
        confidence.append(conf)

        df = pd.DataFrame({'prediction': [pred], 'confidence': [conf]})
        csv_file = 'silksong_predict.csv'
        file_exists = os.path.exists(csv_file)

        df.to_csv(
            csv_file,
            mode="a",
            index=False,
            header=not file_exists,
            encoding="utf-8",
        )
        print(len(prediction), '번째 저장')


    review['prediction'] = prediction
    review['confidence'] = confidence
    review.to_csv('silksong_reviews_predict.csv', index=False)


if __name__ == "__main__":
    main()