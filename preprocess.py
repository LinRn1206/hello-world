import pandas as pd
import re
import jieba
import jieba.analyse
from snownlp import SnowNLP
from gensim import corpora, models
from PIL.Image import item
from config import FIELD_UNITS, STOPWORDS

def preprocess_data(df):
    numeric_cols = list(FIELD_UNITS.keys())
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')
    df = df.dropna(subset=numeric_cols).query('follower_count >= 0')
    if 'create_time' in df.columns:
        df['create_time'] = pd.to_datetime(df['create_time'])
        df['hour'] = df['create_time'].dt.hour
        df['date'] = df['create_time'].dt.date
    if 'ip_location' in df.columns:
        df['province'] = df['ip_location'].str.replace('IP属地：', '', regex=False)
    for field in ['caption', 'item_title', 'username']:
        if field in df.columns:
            df[field] = df[field].astype(str).str.replace('\n', ' ', regex=False)
    return df

def preprocess_time_fields(df, timezone='Asia/Shanghai'):
    df['create_time'] = pd.to_datetime(df['create_time'], errors='coerce', utc=True)
    df['create_time'] = df['create_time'].dt.tz_convert(timezone)
    df['hour'] = df['create_time'].dt.hour
    df['date'] = df['create_time'].dt.date
    df['month'] = df['create_time'].dt.month
    now = pd.Timestamp.now(tz=timezone)
    df = df[(df['create_time'] < now) & (df['create_time'] > pd.Timestamp('2015-01-01', tz=timezone))]
    return df

def clean_text(text):
    return re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9#]', '', str(text))

def tokenize(text, stopwords=STOPWORDS):
    words = jieba.lcut(clean_text(text))
    return [w for w in words if w not in stopwords and w.strip()]

def enrich_text_features(df):
    df['item_title_tokens'] = df['item_title'].apply(lambda x: tokenize(x))
    df['caption_tokens'] = df['caption'].apply(lambda x: tokenize(x))
    df['item_title_keywords'] = df['item_title'].apply(lambda x: jieba.analyse.extract_tags(x, topK=5))
    df['caption_keywords'] = df['caption'].apply(lambda x: jieba.analyse.extract_tags(x, topK=5))
    df['item_title_sentiment'] = df['item_title'].apply(lambda x: SnowNLP(str(x)).sentiments)
    df['caption_sentiment'] = df['caption'].apply(lambda x: SnowNLP(str(x)).sentiments)
    texts = df['caption_tokens'].tolist()
    dictionary = corpora.Dictionary(texts)
    corpus = [dictionary.doc2bow(text) for text in texts]
    lda = models.LdaModel(corpus, num_topics=5, id2word=dictionary, passes=10)
    df['caption_topic'] = [max(lda[doc], key=lambda x: x[1])[0] if lda[doc] else -1 for doc in corpus]
    return df