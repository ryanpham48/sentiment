#----------------------------------------------------------------------------------------------------
#### XÂY DỰNG GUI : Sentiment Analysis Application
#### ĐỒ ÁN TN : Data Science 
#----------------------------------------------------------------------------------------------------
# import libraries
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib
import streamlit as st
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import seaborn as sns
import io

#----------------------------------------------------------------------------------------------------
# Part 1: Load Data and Models
#----------------------------------------------------------------------------------------------------
@st.cache_data
def load_data():
    """Load review dataset."""
    return pd.read_csv('valid_reviews.csv')

@st.cache_resource
def load_model_and_vectorizer():
    """Load pretrained model and CountVectorizer."""
    model = joblib.load('random_forest_model.pkl')  # Load trained model
    count_vectorizer = joblib.load('tfidf_vectorizer.pkl')  # Load CountVectorizer
    return model, count_vectorizer

# Load data and models
valid_reviews = load_data()
model, count_vectorizer = load_model_and_vectorizer()

# Load files
emoji_file = 'emojicon.txt'
teencode_file = 'teencode.txt'
english_file = 'english-vnmese.txt'
wrong_word_file = 'wrong-word.txt'
stopwords_file = 'vietnamese-stopwords.txt'

# Đọc dữ liệu từ các file
with open(emoji_file, 'r', encoding="utf8") as file:
    emoji_dict = {line.split('\t')[0]: line.split('\t')[1].strip() for line in file if '\t' in line}

with open(teencode_file, 'r', encoding="utf8") as file:
    teencode_dict = {line.split('\t')[0]: line.split('\t')[1].strip() for line in file if '\t' in line}

with open(english_file, 'r', encoding="utf8") as file:
    english_dict = {line.split('\t')[0]: line.split('\t')[1].strip() for line in file if '\t' in line}

with open(wrong_word_file, 'r', encoding="utf8") as file:
    wrong_words = set(file.read().splitlines())

with open(stopwords_file, 'r', encoding="utf8") as file:
    stopwords = set(file.read().splitlines())
#----------------------------------------------------------------------------------------------------
# Part 2: Text Preprocessing and Sentiment Analysis
#----------------------------------------------------------------------------------------------------
import re
import regex 
from underthesea import word_tokenize, pos_tag, sent_tokenize
# Chuẩn hóa unicode tiếng việt
def loaddicchar():
    uniChars = "àáảãạâầấẩẫậăằắẳẵặèéẻẽẹêềếểễệđìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵÀÁẢÃẠÂẦẤẨẪẬĂẰẮẲẴẶÈÉẺẼẸÊỀẾỂỄỆĐÌÍỈĨỊÒÓỎÕỌÔỒỐỔỖỘƠỜỚỞỠỢÙÚỦŨỤƯỪỨỬỮỰỲÝỶỸỴÂĂĐÔƠƯ"
    unsignChars = "aaaaaaaaaaaaaaaaaeeeeeeeeeeediiiiiooooooooooooooooouuuuuuuuuuuyyyyyAAAAAAAAAAAAAAAAAEEEEEEEEEEEDIIIOOOOOOOOOOOOOOOOOOOUUUUUUUUUUUYYYYYAADOOU"

    dic = {}
    char1252 = 'à|á|ả|ã|ạ|ầ|ấ|ẩ|ẫ|ậ|ằ|ắ|ẳ|ẵ|ặ|è|é|ẻ|ẽ|ẹ|ề|ế|ể|ễ|ệ|ì|í|ỉ|ĩ|ị|ò|ó|ỏ|õ|ọ|ồ|ố|ổ|ỗ|ộ|ờ|ớ|ở|ỡ|ợ|ù|ú|ủ|ũ|ụ|ừ|ứ|ử|ữ|ự|ỳ|ý|ỷ|ỹ|ỵ|À|Á|Ả|Ã|Ạ|Ầ|Ấ|Ẩ|Ẫ|Ậ|Ằ|Ắ|Ẳ|Ẵ|Ặ|È|É|Ẻ|Ẽ|Ẹ|Ề|Ế|Ể|Ễ|Ệ|Ì|Í|Ỉ|Ĩ|Ị|Ò|Ó|Ỏ|Õ|Ọ|Ồ|Ố|Ổ|Ỗ|Ộ|Ờ|Ớ|Ở|Ỡ|Ợ|Ù|Ú|Ủ|Ũ|Ụ|Ừ|Ứ|Ử|Ữ|Ự|Ỳ|Ý|Ỷ|Ỹ|Ỵ'.split(
        '|')
    charutf8 = "à|á|ả|ã|ạ|ầ|ấ|ẩ|ẫ|ậ|ằ|ắ|ẳ|ẵ|ặ|è|é|ẻ|ẽ|ẹ|ề|ế|ể|ễ|ệ|ì|í|ỉ|ĩ|ị|ò|ó|ỏ|õ|ọ|ồ|ố|ổ|ỗ|ộ|ờ|ớ|ở|ỡ|ợ|ù|ú|ủ|ũ|ụ|ừ|ứ|ử|ữ|ự|ỳ|ý|ỷ|ỹ|ỵ|À|Á|Ả|Ã|Ạ|Ầ|Ấ|Ẩ|Ẫ|Ậ|Ằ|Ắ|Ẳ|Ẵ|Ặ|È|É|Ẻ|Ẽ|Ẹ|Ề|Ế|Ể|Ễ|Ệ|Ì|Í|Ỉ|Ĩ|Ị|Ò|Ó|Ỏ|Õ|Ọ|Ồ|Ố|Ổ|Ỗ|Ộ|Ờ|Ớ|Ở|Ỡ|Ợ|Ù|Ú|Ủ|Ũ|Ụ|Ừ|Ứ|Ử|Ữ|Ự|Ỳ|Ý|Ỷ|Ỹ|Ỵ".split(
        '|')
    for i in range(len(char1252)):
        dic[char1252[i]] = charutf8[i]
    return dic
# Đưa toàn bộ dữ liệu qua hàm này để chuẩn hóa lại
def covert_unicode(text):
    dicchar = loaddicchar()
    return regex.sub(
        r'à|á|ả|ã|ạ|ầ|ấ|ẩ|ẫ|ậ|ằ|ắ|ẳ|ẵ|ặ|è|é|ẻ|ẽ|ẹ|ề|ế|ể|ễ|ệ|ì|í|ỉ|ĩ|ị|ò|ó|ỏ|õ|ọ|ồ|ố|ổ|ỗ|ộ|ờ|ớ|ở|ỡ|ợ|ù|ú|ủ|ũ|ụ|ừ|ứ|ử|ữ|ự|ỳ|ý|ỷ|ỹ|ỵ|À|Á|Ả|Ã|Ạ|Ầ|Ấ|Ẩ|Ẫ|Ậ|Ằ|Ắ|Ẳ|Ẵ|Ặ|È|É|Ẻ|Ẽ|Ẹ|Ề|Ế|Ể|Ễ|Ệ|Ì|Í|Ỉ|Ĩ|Ị|Ò|Ó|Ỏ|Õ|Ọ|Ồ|Ố|Ổ|Ỗ|Ộ|Ờ|Ớ|Ở|Ỡ|Ợ|Ù|Ú|Ủ|Ũ|Ụ|Ừ|Ứ|Ử|Ữ|Ự|Ỳ|Ý|Ỷ|Ỹ|Ỵ',
        lambda x: dicchar[x.group()], text)
# Hàm chuẩn hóa các từ có ký tự lặp
def normalize_repeated_characters(text):
    return regex.sub(r'(.)\1+', r'\1', text)
# Hàm xử lý emoji với dấu câu
def remove_emoji_punctuation(text):
    text = re.sub(r'([^\w\s])\.', r'\1', text)  # Xử lý emoji + dấu câu, loại bỏ dấu câu sau emoji nếu cần
    return text
    
# Hàm xử lý emoji, teen code, và từ sai
def process_text(
    text, 
    emoji_dict, 
    teencode_dict, 
    wrong_words):
    """
    Process text by handling emojis, teen code, and removing unwanted characters.
    Preserve specific words in positive, neutral, and negative word lists.
    """
    document = text.lower()
    document = document.replace("’", '')  # Remove unwanted characters
    document = regex.sub(r'\.+', ".", document)  # Normalize dots
    new_sentence = ''
    
    for sentence in sent_tokenize(document):
        # Step 1: Handle emoji
        sentence = ' '.join(emoji_dict[word] if word in emoji_dict else word for word in sentence.split())
        
        # Step 2: Handle teencode
        sentence = ' '.join(teencode_dict[word] if word in teencode_dict else word for word in sentence.split())
        
        # Step 3: Extract only valid words
        pattern = r'(?i)\b[a-záàảãạăắằẳẵặâấầẩẫậéèẻẽẹêếềểễệóòỏõọôốồổỗộơớờởỡợíìỉĩịúùủũụưứừửữựýỳỷỹỵđ]+\b'
        sentence = ' '.join(regex.findall(pattern, sentence))
        
        # Step 4: Remove wrong words
        sentence = ' '.join('' if word in wrong_words else word for word in sentence.split())
        
        new_sentence += sentence + '. '  # Recombine sentences
    
        # Step 5: Remove excess whitespace
        document = regex.sub(r'\s+', ' ', new_sentence).strip()  
    return document

# Hàm xử lý từ đặc biệt
def process_special_word(text):
    # có thể có nhiều từ đặc biệt cần ráp lại với nhau
    new_text = ''
    text_lst = text.split()
    i= 0
    # không, chẳng, chả...
    if 'không'or 'chẳng' or 'chả' or 'kém' in text_lst:
        while i <= len(text_lst) - 1:
            word = text_lst[i]
            #print(word)
            #print(i)
            if  word == 'không' or word == 'chẳng' or word == 'chả' or word == 'kém' :
                next_idx = i+1
                if next_idx <= len(text_lst) -1:
                    word = word +'_'+ text_lst[next_idx]
                i= next_idx + 1
            else:
                i = i+1
            new_text = new_text + word + ' '
    else:
        new_text = text
    return new_text.strip()

# Hàm xử lý POS tagging và lọc từ loại
def process_postag_thesea(text):
    new_document = ''
    for sentence in sent_tokenize(text):
        sentence = sentence.replace('.','')
        ###### POS tag
        lst_word_type = ['N','Np','A','AB','V','R'] # Giới hạn từ loại cần thiết
        # lst_word_type = ['A','AB','V','VB','VY','R']
        sentence = ' '.join( word[0] if word[1].upper() in lst_word_type else '' for word in pos_tag(process_special_word(word_tokenize(sentence, format="text"))))
        new_document = new_document + sentence + ' '
    ###### DEL excess blank space
    new_document = regex.sub(r'\s+', ' ', new_document).strip()
    return new_document
    
# Hàm loại bỏ từ dừng
def remove_stopwords(text):
    words = text.split()
    words = [word for word in words if word not in stopwords]
    return " ".join(words)

# Hàm tổng hợp tiền xử lý đầy đủ bao gồm POS Tagging
def clean_text(text, emoji_dict, teencode_dict, wrong_words):
    text = normalize_repeated_characters(text)  # Chuẩn hóa ký tự lặp
    text = covert_unicode(text)
    text = remove_emoji_punctuation(text)
    text = process_text(text, emoji_dict, teencode_dict, wrong_words)  # Làm sạch
    text = process_special_word(text)  # Xử lý từ đặc biệt
    text = process_postag_thesea(text)  # POS tagging và lọc từ loại
    text = remove_stopwords(text)  # Loại bỏ từ dừng

    return text

def predict_sentiment(text, model, vectorizer):
    """
    Dự đoán cảm xúc của văn bản đầu vào.
    """
    try:
        # Xử lý văn bản
        processed_text = clean_text(text, emoji_dict, teencode_dict, wrong_words)

        # Vector hóa văn bản
        vectorized_text = vectorizer.transform([processed_text])

        # Dự đoán
        prediction = model.predict(vectorized_text)
        print(f"Kết quả thô từ mô hình: {prediction}")  # Log giá trị trả về để kiểm tra

        # Ánh xạ nhãn cảm xúc
        sentiment_mapping = {
            0: "Tiêu cực",
            1: "Trung lập",
            2: "Tích cực",
            'negative': "Tiêu cực",
            'neutral': "Trung lập",
            'positive': "Tích cực"
        }

        # Lấy nhãn cảm xúc
        return sentiment_mapping.get(prediction[0], f"Giá trị cảm xúc không mong đợi: {prediction[0]}")

    except Exception as e:
        print(f"Lỗi trong quá trình dự đoán cảm xúc: {e}")
        return "Lỗi trong dự đoán"
#----------------------------------------------------------------------------------------------------
# Part 3: Build App
#----------------------------------------------------------------------------------------------------
# GUI
st.title("Data Science Project - Sentiment Analysis for Customer Reviews")
st.write("## Sử dụng phản hồi từ khách hàng để cải thiện chất lượng sản phẩm và dịch vụ.")

menu = st.sidebar.selectbox('Menu', ['Overview', 'Model Evaluation', 'Dự đoán từ văn bản', 'Product ID Prediction'])
st.sidebar.write("""#### Thành viên thực hiện:
                 Phạm Anh Vũ & Lý Quốc Hồng Phúc""")
st.sidebar.image('phucly.png')
st.sidebar.image('vupham.jpg')
st.sidebar.write("""#### Giảng viên hướng dẫn: Khuất Thùy Phương""")
st.sidebar.image('khuat_thuy_phuong.jpg')
st.sidebar.write("""#### Thời gian thực hiện: 05/12/2024""")
#----------------------------------------------------------------------------------------------------
if menu == 'Overview':
    st.subheader('Tổng Quan Dự Án')
    st.image('Sentiment-Analysis.png')
    st.write("""
    **Objective**: Dự án xoay quanh Hasaki.vn, một cửa hàng mỹ phẩm chuyên cung cấp các sản phẩm làm đẹp đa dạng. Mục tiêu chính của dự án là sử dụng phản hồi từ khách hàng để cải thiện chất lượng sản phẩm và dịch vụ. Cụ thể:
    1. Phân tích các đánh giá của khách hàng để xác định những điểm mạnh (phản hồi tích cực) và những điểm cần cải thiện (phản hồi tiêu cực).

    2. Dựa vào kết quả phân tích để hiểu rõ hơn về sở thích của khách hàng, khắc phục các vấn đề tồn tại và đưa ra quyết định dựa trên dữ liệu để nâng cao chất lượng dịch vụ.

    """)
    st.image('Hasaki.logo.wide.jpg')
    st.write("""
    - **Key Features**: Lựa chọn thuật toán phù hợp trong “Machine Learning with Python” cho bài toán Sentiment classification (Positive, Neutral, Negative). Sử dụng Random Forest.
    - **Analysis Tools**: Data cleaning, visualization, wordclouds, and machine learning models.
    - **Thư viên sử dụng**:
- numpy, pandas, matplotlib, seaborn
- underthesea (để đọc dữ liệu tiếng việt)
- worldcloud ( để thành 2 bảng, 1 là positive với các từ như yêu thích, tốt... tiếp là negative như kém, tệ...)
- scikit-learn
- pyspark để dùng machine learning,
    """)
    

#----------------------------------------------------------------------------------------------------
elif menu == 'Model Evaluation':
    st.subheader('Model Evaluation')
    st.write('### Valid Reviews Dataset sau khi đã tiền xử lý và làm sạch')
    st.table(valid_reviews.head())

    # Display sentiment distribution
    st.write('### Phân phối đánh giá trước khi xử lý')
    st.image('distribution.png')

    # WordClouds
    st.write('### WordClouds')
    st.image('positive.png')
    st.image('neutral.png')
    st.image('negative.png')

    
    st.write('#### Cân Bằng Dữ Liệu Trước và Sau SMOTE')
    st.image('smote.png')
    st.write("""
    - **Độ chính xác**: ~98%
    - **Điểm mạnh**: Cao nhất về độ chính xác và hiệu suất tốt trên mọi nhãn. Tách biệt tốt giữa các lớp cảm xúc.
    - **Cân nhắc**:Thời gian xử lý dài hơn, yêu cầu tài nguyên tính toán cao hơn.
    - #### Kết luận: Random Forest là mô hình phù hợp nhất, khuyến khích sử dụng cho dữ liệu này.""")
    st.image('evaluation.png')
    st.write('### Ma Trận Nhầm Lẫn')
    st.image('matrix.png')

#----------------------------------------------------------------------------------------------------
elif menu == 'Dự đoán từ văn bản':
    st.subheader('Dự đoán từ văn bản')
    st.write('Nhập văn bản đánh giá hoặc tải lên tệp CSV để hệ thống dự đoán cảm xúc.')

    choice = st.radio("Lựa chọn phương thức nhập liệu:", ["Nhập văn bản", "Tải lên tệp CSV"])

    # Option 1: Text Input
    if choice == "Nhập văn bản":
        input_text = st.text_area("Nhập văn bản đánh giá:")

        if st.button("Dự đoán cảm xúc"):
            if input_text.strip():  # Kiểm tra xem văn bản có trống không
                try:
                    # Tiền xử lý văn bản
                    processed_text = clean_text(
                        input_text, emoji_dict, teencode_dict, wrong_words)

                    # Hiển thị văn bản sau khi tiền xử lý
                    st.write("**Văn bản sau khi tiền xử lý:**")
                    st.code(processed_text)

                    # Chuyển văn bản đã xử lý thành vector
                    vectorized_text = count_vectorizer.transform([processed_text])

                    # Dự đoán cảm xúc
                    prediction = model.predict(vectorized_text)

                    # Ánh xạ kết quả dự đoán thành nhãn cảm xúc
                    sentiment_mapping = {
            0: "Tiêu cực",
            1: "Trung lập",
            2: "Tích cực",
            'negative': "Tiêu cực",
            'neutral': "Trung lập",
            'positive': "Tích cực"
        }
                    sentiment = sentiment_mapping.get(prediction[0], "Không xác định")

                    # Hiển thị kết quả dự đoán
                    st.write(f"**Kết quả dự đoán:** {sentiment}")

                except Exception as e:
                    # Hiển thị lỗi nếu xảy ra trong quá trình dự đoán
                    st.error(f"Lỗi trong quá trình dự đoán: {e}")
            else:
                # Thông báo nếu không có văn bản nhập vào
                st.warning("Vui lòng nhập văn bản để dự đoán cảm xúc!")

    # Option 2: CSV File Upload
    elif choice == "Tải lên tệp CSV":
        uploaded_file = st.file_uploader("Chọn tệp CSV", type=["csv"])

        if uploaded_file is not None:
            try:
                # Đọc tệp CSV
                data = pd.read_csv(uploaded_file)

                # Kiểm tra cột review_text có tồn tại
                if "noi_dung_binh_luan" not in data.columns:
                    st.error("Tệp CSV phải chứa cột 'noi_dung_binh_luan'.")
                else:
                    # Tiền xử lý văn bản
                    data["processed_text"] = data["noi_dung_binh_luan"].apply(
                        lambda x: clean_text(x, emoji_dict, teencode_dict, wrong_words)
                        if isinstance(x, str)
                        else None
                    )

                    # Hiển thị văn bản đã qua tiền xử lý
                    st.write("**Văn bản sau khi tiền xử lý:**")
                    st.dataframe(data[["noi_dung_binh_luan", "processed_text"]].head(10))

                    # Chuyển văn bản đã xử lý thành vector
                    vectorized_texts = count_vectorizer.transform(data["processed_text"].dropna())

                    # Dự đoán cảm xúc
                    predictions = model.predict(vectorized_texts)

                    # Ánh xạ kết quả dự đoán thành nhãn cảm xúc
                    sentiment_mapping = {
            0: "Tiêu cực",
            1: "Trung lập",
            2: "Tích cực",
            'negative': "Tiêu cực",
            'neutral': "Trung lập",
            'positive': "Tích cực"
        }
                    data["sentiment"] = [
                        sentiment_mapping.get(pred, "Không xác định") for pred in predictions
                    ]

                    # Hiển thị dữ liệu với kết quả dự đoán
                    st.write("**Kết quả dự đoán:**")
                    st.dataframe(data[["noi_dung_binh_luan", "sentiment"]].head(10))

                    # Tải xuống kết quả
                    output = io.BytesIO()
                    data.to_csv(output, index=False)
                    output.seek(0)
                    st.download_button(
                        label="Tải xuống kết quả dự đoán",
                        data=output,
                        file_name="predictions.csv",
                        mime="text/csv",
                    )
            except Exception as e:
                st.error(f"Lỗi trong quá trình xử lý tệp CSV: {e}")

#----------------------------------------------------------------------------------------------------
elif menu == 'Product ID Prediction':
    st.subheader('Dự đoán cảm xúc theo mã sản phẩm')
    st.write("Phân tích cảm xúc của khách hàng cho một hoặc nhiều mã sản phẩm.")

    choice = st.radio("Chọn phương thức nhập liệu:", ["Nhập mã sản phẩm", "Tải lên tập CSV"])

    if choice == "Nhập mã sản phẩm":
        product_id = st.text_input("Nhập mã sản phẩm:")
        if st.button("Phân tích sản phẩm"):
            if product_id.strip():
                try:
                    # Chuyển đổi product_id thành số nguyên
                    product_id = int(product_id.strip())
                    product_reviews = valid_reviews[valid_reviews['ma_san_pham'] == product_id]
                    
                    if not product_reviews.empty:
                        st.write(f"Phân tích cho mã sản phẩm: {product_id}")
                        st.dataframe(product_reviews.head())
                        
                        # Phân tích đánh giá
                        sentiment_counts = product_reviews['sentiment_rate'].value_counts()
                        
                        # Phân loại đánh giá
                        positive_reviews = " ".join(product_reviews[product_reviews['sentiment_rate'] == 'positive']['noi_dung_binh_luan_clean'].dropna())
                        neutral_reviews = " ".join(product_reviews[product_reviews['sentiment_rate'] == 'neutral']['noi_dung_binh_luan_clean'].dropna())
                        negative_reviews = " ".join(product_reviews[product_reviews['sentiment_rate'] == 'negative']['noi_dung_binh_luan_clean'].dropna())

                        # Vẽ WordCloud
                        st.write('### WordCloud cho sản phẩm:')
                        if positive_reviews:
                            wc_pos = WordCloud(background_color='white', max_words=100).generate(positive_reviews)
                            st.image(wc_pos.to_array(), caption="Positive Reviews WordCloud")
                        if neutral_reviews:
                            wc_neu = WordCloud(background_color='white', max_words=100).generate(neutral_reviews)
                            st.image(wc_neu.to_array(), caption="Neutral Reviews WordCloud")
                        if negative_reviews:
                            wc_neg = WordCloud(background_color='white', max_words=100).generate(negative_reviews)
                            st.image(wc_neg.to_array(), caption="Negative Reviews WordCloud")
                        
                        # Liệt kê top từ
                        def get_top_keywords(text, top_n=10):
                            words = text.split()
                            word_counts = pd.Series(words).value_counts()
                            return word_counts.head(top_n).to_dict()

                        positive_keywords = get_top_keywords(positive_reviews)
                        neutral_keywords = get_top_keywords(neutral_reviews)
                        negative_keywords = get_top_keywords(negative_reviews)

                        st.write("### Từ khóa phổ biến nhất:")
                        st.write("Top từ tích cực:", positive_keywords)
                        st.write("Top từ trung lập:", neutral_keywords)
                        st.write("Top từ tiêu cực:", negative_keywords)

                        # Hiển thị thông tin thống kê
                        st.write("### Thống kê cảm xúc:")
                        st.write({
                            "Mã sản phẩm": product_id,
                            "Tổng đánh giá tích cực": sentiment_counts.get('positive', 0),
                            "Tổng đánh giá trung lập": sentiment_counts.get('neutral', 0),
                            "Tổng đánh giá tiêu cực": sentiment_counts.get('negative', 0)
                        })
                    else:
                        st.warning(f"Không tìm thấy đánh giá cho mã sản phẩm: {product_id}")
                except ValueError:
                    st.error("Mã sản phẩm phải là một số nguyên hợp lệ!")
            else:
                st.warning("Vui lòng nhập mã sản phẩm hợp lệ!")

    elif choice == "Tải lên tập CSV":
    st.subheader("Tải lên tệp CSV")

    # File uploader
    uploaded_file = st.file_uploader("Chọn tệp CSV có cột 'product_id'", type=["csv"])

    if uploaded_file is None:
        st.info("Vui lòng tải lên tệp CSV để bắt đầu phân tích.")
    else:
        try:
            # Đọc file CSV
            data = pd.read_csv(uploaded_file)

            # Kiểm tra cột 'product_id' có tồn tại
            if "product_id" not in data.columns:
                st.error("Tệp CSV phải chứa cột 'product_id'.")
            else:
                # Xử lý giá trị trong cột 'product_id'
                data["product_id"] = pd.to_numeric(data["product_id"], errors="coerce")  # Chuyển đổi các giá trị hợp lệ
                data = data.dropna(subset=["product_id"])  # Loại bỏ các hàng có giá trị NaN
                data["product_id"] = data["product_id"].astype(int)  # Chuyển đổi thành số nguyên

                # Lấy danh sách product_id duy nhất
                product_ids = data["product_id"].unique()
                st.write(f"Tìm thấy {len(product_ids)} mã sản phẩm:")
                st.write(product_ids)

                # Kết quả phân tích sẽ được lưu ở đây
                analysis_results = []

                # Phân tích từng product ID
                for product_id in product_ids:
                    st.subheader(f"Phân tích cho mã sản phẩm: {product_id}")
                    product_reviews = valid_reviews[valid_reviews['ma_san_pham'] == product_id]
                    if not product_reviews.empty:
                        st.dataframe(product_reviews.head())

                        # Tạo WordCloud cho từng loại cảm xúc
                        sentiments = product_reviews['sentiment_rate'].unique()
                        sentiment_text = {
                            sentiment: " ".join(
                                product_reviews[product_reviews['sentiment_rate'] == sentiment]['noi_dung_binh_luan_clean']
                            )
                            for sentiment in sentiments
                        }
                        for sentiment, text in sentiment_text.items():
                            wordcloud = WordCloud(background_color='white', max_words=100).generate(text)
                            st.image(wordcloud.to_array(), caption=f"WordCloud - {sentiment}")

                        # Lấy từ khóa chính
                        top_keywords = {
                            sentiment: pd.Series(text.split()).value_counts().head(10).to_dict()
                            for sentiment, text in sentiment_text.items()
                        }

                        # Lưu kết quả phân tích
                        analysis_results.append({
                            "Product ID": product_id,
                            "Total Reviews": len(product_reviews),
                            "Sentiments": sentiments.tolist(),
                            "Top Keywords": top_keywords
                        })
                    else:
                        st.warning(f"Không tìm thấy đánh giá cho mã sản phẩm: {product_id}")

                # Tải xuống kết quả phân tích
                if analysis_results:
                    st.success("Phân tích hoàn tất. Bạn có thể tải xuống kết quả.")
                    results_df = pd.DataFrame(analysis_results)
                    output = io.BytesIO()
                    results_df.to_csv(output, index=False)
                    output.seek(0)

                    st.download_button(
                        label="Tải xuống kết quả phân tích",
                        data=output,
                        file_name="analysis_results.csv",
                        mime="text/csv",
                    )
        except Exception as e:
            st.error(f"Lỗi khi xử lý tệp: {e}")
