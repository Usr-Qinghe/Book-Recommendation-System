import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import streamlit as st
import pickle
import torch

# 加载数据集
file_path = 'bookstoscrape.csv'
data = pd.read_csv(file_path)

# 数据预处理
def process_price(price_str):
    if isinstance(price_str, str):
        return float(price_str.replace(',', ''))
    elif isinstance(price_str, (int, float)):
        return float(price_str)
    else:
        raise ValueError(f"Unsupported type: {type(price_str)}")

data['Price'] = data['Price'].apply(process_price)

# 对星级评分进行编码
star_rating_mapping = {'One': 1, 'Two': 2, 'Three': 3, 'Four': 4, 'Five': 5}
data['Star Rating'] = data['Star Rating'].map(star_rating_mapping)

# 文本预处理和特征提取
data['Title'] = data['Title'].str.lower().str.replace(r'[^\w\s]', '')
tfidf_vectorizer = TfidfVectorizer(max_features=500, stop_words='english')
tfidf_matrix = tfidf_vectorizer.fit_transform(data['Title'])

# 导入模型
model_path = 'model.pkl'
with open(model_path, 'rb') as file:
    linear_regression = pickle.load(file)

# 设计用户界面
st.title('Book Recommendation System')
user_input = st.text_input('Enter the title of a book you like:')
if user_input:
    # 用户输入书籍标题
    user_input_vector = tfidf_vectorizer.transform([user_input])
    # 计算用户输入与所有书籍的余弦相似度
    cosine_similarities = torch.cosine_similarity(torch.tensor(user_input_vector.toarray()), torch.tensor(tfidf_matrix.toarray()))
    
    # 确保 cosine_similarities 是一个1维张量
    if cosine_similarities.ndim == 0:
        cosine_similarities = cosine_similarities.view(1)
    
    # 获取相似度最高的书籍索引
    similar_indices = cosine_similarities.argsort(descending=True)[:5]
    
    # 为用户生成推荐
    recommendations = data.iloc[similar_indices]['Title']
    st.write(f'Recommendations for "{user_input}":')
    for i, book in enumerate(recommendations, 1):
        st.write(f'{i}. {book}')