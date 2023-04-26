import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline


train = pd.read_csv('lenta-ru-news.csv.gz',
                    header=None,
                    names=['url', 'title', 'text', 'topic', 'tags'],).dropna()

topic = train['topic']
news = train['text']

# Создаем модель классификации текста
text_classifier = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('clf', MultinomialNB())
])

# Обучаем модель на обучающих данных
text_classifier.fit([x for x in news], [x for x in topic])

# Тестируем модель на новых данных
test_data = ['Удобнее и быстрее всего подать заявку в Сбербанк Онлайн в разделе «Кредиты».'
             ' Также подать заявку можно с помощью бесплатного сервиса Кредитный потенциал,'
             ' который дополнительно рассчитает максимальный размер платежеспособности и определит финальный размер процентной ставки.'
             ' Или, вы можете прийти в офис банка, где есть консультант по кредитам. Чтобы найти такой офис,'
             ' на странице «Отделения и банкоматы» нажмите кнопку «Выбрать услуги» и в столбце «Кредиты» отметьте нужный пункт.']
predictions = text_classifier.predict(test_data)

# Выводим результаты
for text, predicted_topic in zip(test_data, predictions):
    print(f'Тема текста  - {predicted_topic}')
