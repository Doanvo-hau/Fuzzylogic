import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf

# Load mô hình đã được huấn luyện
model = tf.keras.models.load_model('D:\Tien\Tien.h5')  # Đường dẫn tới file mô hình đã được huấn luyện

def predict(image):
    # Tiến hành dự đoán
    img = image.resize((100, 100))  # Resize ảnh thành kích thước đầu vào của mô hình
    img_array = np.array(img) / 255.0  # Chuẩn hóa giá trị pixel về khoảng 0-1
    img_array = np.expand_dims(img_array, axis=0)  # Mở rộng kích thước ma trận thành (1, 224, 224, 3)

    # Dự đoán tiền
    prediction = model.predict(img_array)
    result = np.argmax(prediction)
    # Danh sách các lớp tiền
    class_names = ['','1K', '2K', '5K', '10K', '20K', '50K', '100K', '200K', '500K'] 
    return class_names[result]

# Giao diện web app
st.title('NHẬN DẠNG TIỀN VIỆT NAM')

# Upload file ảnh
uploaded_file = st.file_uploader('PLEASE UPLOAD IMAGE', type=['png', 'jpg', 'jpeg'])

if uploaded_file is not None:
    # Đọc và hiển thị ảnh
    image = Image.open(uploaded_file)
    st.image(image, caption='IMAGE', use_column_width=True)

    # Dự đoán tiền khi nhấn nút
    if st.button('PREDICT'):
        result = predict(image)
        st.success(f'VN_BANKNOTES IS: {result}')
