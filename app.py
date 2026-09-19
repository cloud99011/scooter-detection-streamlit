import sqlite3
from datetime import datetime
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st
from ultralytics import YOLO

# Настройки страницы Streamlit
st.set_page_config(
    page_title="Учет самокатов у метро", page_icon="🛴", layout="wide"
)

# Пути к БД и модели
DB_PATH = "scooter_history.db"
MODEL_PATH = "C:/Users/Gay/Desktop/detector/runs/detect/scooter_project/scooter_model/weights/best.pt"


# 1. Инициализация и создание таблицы в SQLite
def init_db():
  conn = sqlite3.connect(DB_PATH)
  cursor = conn.cursor()
  cursor.execute("""
        CREATE TABLE IF NOT EXISTS detections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            filename TEXT,
            scooter_count INTEGER,
            status TEXT
        )
    """)
  conn.commit()
  conn.close()


# 2. Функция сохранения записи в SQLite
def save_detection(filename, count, status):
  conn = sqlite3.connect(DB_PATH)
  cursor = conn.cursor()
  timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
  cursor.execute(
      """
        INSERT INTO detections (timestamp, filename, scooter_count, status)
        VALUES (?, ?, ?, ?)
    """,
      (timestamp, filename, count, status),
  )
  conn.commit()
  conn.close()


# 3. Функция чтение всей истории из SQLite
def get_history():
  conn = sqlite3.connect(DB_PATH)
  df = pd.read_sql_query(
      """
        SELECT 
            id AS 'ID', 
            timestamp AS 'Дата и время', 
            filename AS 'Файл', 
            scooter_count AS 'Количество', 
            status AS 'Статус' 
        FROM detections 
        ORDER BY id DESC
    """,
      conn,
  )
  conn.close()
  return df


# 4. Загрузка модели с кэшированием
@st.cache_resource
def load_model():
  return YOLO(MODEL_PATH)


# Запуск создания БД при старте приложения
init_db()

# Интерфейс Streamlit
st.title("🛴 Учет количества самокатов около метро")

tab1, tab2 = st.tabs(["Детекция объектов", "История и экспорт отчета"])

# Вкладка 1: Обработка фото
with tab1:
  st.write(
      "Загрузите фотографию для автоматического подсчета количества самокатов"
      " в кадре."
  )

  uploaded_file = st.file_uploader(
      "Выберите изображение...", type=["jpg", "jpeg", "png"]
  )

  if uploaded_file is not None:
    try:
      model = load_model()
      image = Image.open(uploaded_file)

      col1, col2 = st.columns(2)

      with col1:
        st.subheader("Исходное фото")
        st.image(image, use_container_width=True)

      with col2:
        st.subheader("Результат распознавания")
        # Распознавание объектов
        results = model.predict(source=np.array(image), conf=0.25)
        scooter_count = len(results[0].boxes)

        # Отрисовка рамок объектов
        res_plotted = results[0].plot()[:, :, ::-1]
        st.image(res_plotted, use_container_width=True)

      # Статус нагрузки
      status = "Превышение нормы" if scooter_count > 10 else "Норма"

      if status == "Норма":
        st.success(
            f"Найдено самокатов: {scooter_count} шт. | Статус: {status}"
        )
      else:
        st.warning(
            f"Найдено самокатов: {scooter_count} шт. | Статус: {status}"
        )

      # Автоматическое сохранение в SQLite
      save_detection(uploaded_file.name, scooter_count, status)
      st.info("Результат автоматически сохранен в историю (SQLite).")

    except Exception as e:
      st.error(f"Ошибка при обработке изображения или загрузке модели: {e}")

# Вкладка 2: Просмотр истории и скачивание CSV
with tab2:
  st.subheader("Журнал учета и экспорт данных")

  df_history = get_history()

  if not df_history.empty:
    st.dataframe(df_history, use_container_width=True)

    # Генерация CSV-файла для экспорта
    csv_data = df_history.to_csv(index=False).encode("utf-8-sig")

    st.download_button(
        label="📥 Скачать отчет (CSV)",
        data=csv_data,
        file_name=f"report_scooters_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv",
    )
  else:
    st.info(
        "История распознаваний пуста. Загрузите изображение во вкладке «Детекция"
        " объектов»."
    )