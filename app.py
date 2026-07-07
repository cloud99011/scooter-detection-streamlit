import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np

# Настраиваем страницу браузера
st.set_page_config(page_title="Детектор самокатов у метро", page_icon="🛴")

st.title("🛴 Учет количества самокатов около метро")
st.write("Загрузите фотографию, и искусственный интеллект посчитает количество самокатов в кадре.")

# Путь к твоей обученной модели
MODEL_PATH = "C:/Users/Gay/Desktop/detector/runs/detect/scooter_project/scooter_model/weights/best.pt"

# Кэшируем модель в памяти, чтобы сайт не зависал при каждой загрузке фото
@st.cache_resource
def load_model():
    return YOLO(MODEL_PATH)

try:
    model = load_model()
    
    # Компонент для загрузки файла
    uploaded_file = st.file_uploader("Выберите изображение...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Открываем изображение через Pillow
        image = Image.open(uploaded_file)
        
        # Разделяем интерфейс на две колонки для красоты
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Исходное фото")
            st.image(image, use_container_width=True)
            
        with col2:
            st.subheader("Результат распознавания")
            
            # Запускаем предсказание модели
            # Переводим картинку в формат numpy для YOLO
            results = model.predict(source=np.array(image), conf=0.25)
            
            # Считаем количество объектов
            scooter_count = len(results[0].boxes)
            
            # Рендерим картинку с нарисованными рамками
            # plot() возвращает массив BGR, переводим его в RGB для корректных цветов
            res_plotted = results[0].plot()[:, :, ::-1]
            
            st.image(res_plotted, use_container_width=True)
            
        # Красивая плашка с итоговым числом
        st.success(f" Найдено самокатов: {scooter_count}")

except Exception as e:
    st.error(f"Ошибка загрузки модели. Проверьте, завершилось ли обучение и правильный ли путь указан. Технический текст: {e}")