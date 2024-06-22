import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import numpy as np
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

# 随机选择五本书进行展示
random_indices = np.random.choice(data.index, 5, replace=False)
random_books = data.loc[random_indices, 'Title']

# 用户选择喜欢的书籍
user_choice = st.selectbox('Choose a book you like:', random_books)

if user_choice:
    # 用户选择的书籍的索引
    user_choice_index = data[data['Title'] == user_choice].index[0]
    user_choice_vector = tfidf_matrix[user_choice_index]
    
    # 计算用户选择与所有书籍的余弦相似度
    cosine_similarities = torch.cosine_similarity(torch.tensor(user_choice_vector.toarray()), torch.tensor(tfidf_matrix.toarray()))
    
    # 获取相似度最高的书籍索引（除了用户选择的书籍）
    similar_indices = cosine_similarities.argsort(descending=True)[1:6]
    
    # 为用户生成推荐
    recommendations = data.iloc[similar_indices]['Title']
    st.write(f'Recommendations based on "{user_choice}":')
    for i, book in enumerate(recommendations, 1):
        st.write(f'{i}. {book}')
