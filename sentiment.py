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
    wrong_words, 
    positive_words, 
    neutral_words, 
    negative_words
):
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
        
        # Step 5: Preserve sentiment-related words
        sentence = ' '.join(
            word if word in positive_words or word in neutral_words or word in negative_words else word
            for word in sentence.split()
        )
        
        new_sentence += sentence + '. '  # Recombine sentences
    
    # Step 6: Remove excess whitespace
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
        sentence = sentence.replace('.', '')
        ###### POS tag
        lst_word_type = ['N','Np','A','AB','V','R'] # Giới hạn từ loại cần thiết
        sentence = ' '.join(
            word[0] if word[1].upper() in lst_word_type else '' 
            for word in pos_tag(process_special_word(word_tokenize(sentence, format="text")))
        )
        new_document = new_document + sentence + ' '
    ###### Loại bỏ khoảng trắng thừa
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
    st.subheader('Product ID Prediction')
    st.write("Analyze customer sentiment for one or more product IDs.")

    choice = st.radio("Choose Input Method:", ["Nhập mã sản phẩm", "Tải lên tập CSV"])

    if choice == "Nhập mã sản phẩm":
        product_id = st.text_input("Enter Product ID:")
    if st.button("Analyze Product"):
        if product_id.strip():
            try:
                # Chuyển đổi product_id thành số nguyên
                product_id = int(product_id.strip())
                product_reviews = valid_reviews[valid_reviews['ma_san_pham'] == product_id]
                if not product_reviews.empty:
                    st.write(f"Analysis for Product ID: {product_id}")
                    st.dataframe(product_reviews.head())
                else:
                    st.warning(f"No reviews found for Product ID: {product_id}")
            except ValueError:
                st.error("Product ID must be a valid integer!")
        else:
            st.warning("Please enter a valid Product ID.")

    elif choice == "Tải lên tập CSV":
        st.subheader("Upload a CSV File")

    # File uploader
    uploaded_file = st.file_uploader("Choose a CSV file with 'product_id' column", type=["csv"])

    if uploaded_file is None:
        # Nhắc nhở nhẹ nhàng
        st.info("Please upload a CSV file to start the analysis.")
    else:
        # Chỉ xử lý khi đã có tệp được tải lên
        try:
            # Đọc file CSV
            data = pd.read_csv(uploaded_file)

            # Kiểm tra cột 'product_id' có tồn tại
            if "product_id" not in data.columns:
                st.error("The CSV file must contain a 'product_id' column.")
            else:
                # Xử lý giá trị trong cột 'product_id'
                data["product_id"] = pd.to_numeric(data["product_id"], errors="coerce")  # Chuyển đổi các giá trị hợp lệ
                data = data.dropna(subset=["product_id"])  # Loại bỏ các hàng có giá trị NaN
                data["product_id"] = data["product_id"].astype(int)  # Chuyển đổi thành số nguyên
                
                # Lấy danh sách product_id duy nhất
                product_ids = data["product_id"].unique()
                st.write(f"Found {len(product_ids)} Product IDs:")
                st.write(product_ids)

                # Phân tích từng product ID
                for product_id in product_ids:
                    st.subheader(f"Analysis for Product ID: {product_id}")
                    product_reviews = valid_reviews[valid_reviews['ma_san_pham'] == product_id]
                    if not product_reviews.empty:
                        st.dataframe(product_reviews.head())
                    else:
                        st.warning(f"No reviews found for Product ID: {product_id}")
        except Exception as e:
            st.error(f"Error processing the file: {e}")
