import os
import shutil
import zipfile
import pickle
import glob
import argparse
import numpy as np
import pandas as pd
import torch
import faiss
from PIL import Image
from tqdm import tqdm
from ultralytics import YOLO
from torchvision import transforms, models
import matplotlib.pyplot as plt
import clip  # Библиотека CLIP от OpenAI

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional

# ======================================
# Изначальные настройки и инициализация
# ======================================

ZIP_FOLDER = 'zip_files'          # Директория с zip файлами
DATA_DIR = 'zip_files'            # Директория с папками кепок
INDEX_FILE = 'faiss_index.bin'    # Файл для сохранения FAISS индекса
METADATA_FILE = 'metadata.pkl'    # Файл для сохранения метаданных
EXCEL_FILE = 'Кепки.xlsx'         # Excel файл с названиями групп
DEVICE = 'cpu'                    # Использовать CPU

# Инициализация модели YOLO для детекции кепок
# Замените на путь к вашей обученной модели YOLO
yolo_model = YOLO('weights/best.pt')
yolo_model.to(DEVICE)

# Определяем метод ресэмплинга для Pillow
try:
    from PIL import Resampling
    RESAMPLE_METHOD = Resampling.LANCZOS
except ImportError:
    RESAMPLE_METHOD = Image.LANCZOS

# Инициализация модели CLIP
clip_model, preprocess_clip = clip.load("ViT-L/14", device=DEVICE)  # Загружаем более тяжелую модель
clip_model.eval()


# ======================================
# Обёрнутые в функции вспомогательные методы
# ======================================

def organize_zip_files(excel_file, zip_folder):
    """
    Организует zip файлы по группам на основе Excel файла.
    """
    df = pd.read_excel(excel_file)
    zip_files = [f for f in os.listdir(zip_folder) if f.endswith('.zip')]
    df_titles = df['Название по группе '].tolist()

    for title in df_titles:
        group_folder = os.path.join(zip_folder, title)
        os.makedirs(group_folder, exist_ok=True)

        matched = False
        for zip_file in zip_files:
            if os.path.splitext(zip_file)[0] == title:
                matched = True
                zip_file_path = os.path.join(zip_folder, zip_file)
                shutil.move(zip_file_path, group_folder)
                with zipfile.ZipFile(os.path.join(group_folder, zip_file), 'r') as zip_ref:
                    zip_ref.extractall(group_folder)
                print(f"Файл {zip_file} перемещен и разархивирован в {group_folder}.")
                break

        if not matched:
            print(f"Эта группа отсутствует в zip_files: {title}")

    missing_titles = [title for title in df_titles if title not in [os.path.splitext(f)[0] for f in zip_files]]
    if missing_titles:
        print("Эти названия не найдены в zip_files:", missing_titles)
    else:
        print("Все названия из датафрейма найдены в zip_files и обработаны.")


def resize_image_if_needed(image):
    """
    Масштабирует изображение до минимального размера 640x640, если необходимо.
    """
    desired_size = 640
    if image.width < desired_size or image.height < desired_size:
        new_width = max(desired_size, image.width)
        new_height = max(desired_size, image.height)
        image = image.resize((new_width, new_height), RESAMPLE_METHOD)
        print(f"Изображение масштабировано до размера: {image.size}")
    else:
        print(f"Размер изображения подходит: {image.size}")
    return image


def extract_features_clip(image):
    """
    Извлекает вектор признаков из изображения с помощью модели CLIP.
    """
    input_tensor = preprocess_clip(image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        features = clip_model.encode_image(input_tensor)
    features = features.cpu().numpy().flatten()
    features = features / np.linalg.norm(features)  # Нормализация
    return features.astype('float32')


def detect_and_extract_features(image_path):
    """
    Детектирует кепку на изображении и извлекает признаки с помощью CLIP.
    Возвращает список векторов признаков или None, если кепка не обнаружена.
    """
    try:
        print(f"\nОбрабатываем изображение: {image_path}")
        image = Image.open(image_path).convert("RGB")
        image = resize_image_if_needed(image)  # Масштабируем при необходимости

        # Детекция с помощью YOLO
        results = yolo_model.predict(image, imgsz=640)
        print(f"Количество детектированных объектов: {len(results[0].boxes)}")

        if results and len(results) > 0:
            result = results[0]
            if result.boxes is not None and len(result.boxes) > 0:
                features_list = []
                for box in result.boxes.xyxy.cpu().numpy():
                    x1, y1, x2, y2 = map(int, box)
                    print(f"Координаты обнаруженной кепки: {(x1, y1, x2, y2)}")
                    cropped_image = image.crop((x1, y1, x2, y2))
                    print("Кепка обрезана из изображения.")

                    # Извлечение признаков с помощью CLIP
                    features = extract_features_clip(cropped_image)
                    print("Признаки извлечены с помощью CLIP.")
                    features_list.append(features)

                return features_list
        else:
            print("Не обнаружено ни одного объекта на изображении.")
    except Exception as e:
        print(f"Общая ошибка при обработке изображения {image_path}: {e}")

    return None


def build_feature_database(data_dir, metadata_file):
    """
    Проходит по всем изображениям в DATA_DIR, детектирует кепки, извлекает признаки и сохраняет их.
    """
    feature_list = []
    metadata = []

    # Поддерживаемые форматы изображений
    supported_extensions = ['.png', '.webp', '.jpg', '.jpeg', '.bmp', '.tiff', '.gif']

    image_paths = []
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if os.path.splitext(file)[1].lower() in supported_extensions:
                image_paths.append(os.path.join(root, file))

    print(f"Найдено {len(image_paths)} изображений для обработки.")

    if not image_paths:
        print("Нет изображений для обработки в указанной директории.")
        return None, None

    print("Начинаем обработку изображений и извлечение признаков...")
    for image_path in tqdm(image_paths):
        cap_name = os.path.basename(os.path.dirname(image_path))  # Название папки как название кепки
        features = detect_and_extract_features(image_path)
        if features is not None:
            for feature in features:
                feature_list.append(feature)
                metadata.append({
                    'cap_name': cap_name,
                    'image_path': image_path
                })
            print(f"Признаки успешно извлечены для: {image_path}")
        else:
            print(f"Кепка не обнаружена или не удалось извлечь признаки на изображении {image_path}")

    if feature_list:
        # Объединяем все векторы признаков в одну матрицу
        features_matrix = np.vstack(feature_list)
        print(f"Извлечено {features_matrix.shape[0]} векторов признаков.")

        # Сохраняем метаданные
        try:
            with open(metadata_file, 'wb') as f:
                pickle.dump(metadata, f)
            print(f"Метаданные сохранены в {metadata_file}.")
        except Exception as meta_e:
            print(f"Ошибка при сохранении метаданных: {meta_e}")

        return features_matrix, metadata
    else:
        print("Не найдено признаков для индексации.")
        return None, None


def create_faiss_index(features_matrix, index_file):
    """
    Создает FAISS индекс для векторов признаков.
    """
    try:
        dimension = features_matrix.shape[1]  # Размерность векторов
        index = faiss.IndexFlatIP(dimension)   # Inner Product -> косинусное сходство при нормировке
        faiss.normalize_L2(features_matrix)    # Нормализуем векторы
        index.add(features_matrix)
        faiss.write_index(index, index_file)
        print(f"FAISS индекс создан и сохранён в {index_file}")
        return index
    except Exception as e:
        print(f"Ошибка при создании FAISS индекса: {e}")
        return None


def load_faiss_index(index_file, metadata_file):
    """
    Загружает FAISS индекс и метаданные.
    """
    if not os.path.exists(index_file):
        print(f"FAISS индекс не найден: {index_file}")
        return None, None
    if not os.path.exists(metadata_file):
        print(f"Файл метаданных не найден: {metadata_file}")
        return None, None

    try:
        index = faiss.read_index(index_file)
        with open(metadata_file, 'rb') as f:
            metadata = pickle.load(f)
        print("FAISS индекс и метаданные успешно загружены.")
        return index, metadata
    except Exception as e:
        print(f"Ошибка при загрузке FAISS индекса или метаданных: {e}")
        return None, None


def search_similar_cap(uploaded_image_path, index_file=INDEX_FILE, metadata_file=METADATA_FILE, top_k=1):
    """
    Принимает путь к загруженному изображению, обрабатывает его и ищет похожую кепку в базе данных.
    Возвращает список из top_k самых похожих кепок.
    """
    index, metadata = load_faiss_index(index_file, metadata_file)
    if index is None or metadata is None:
        msg = "Не удалось загрузить FAISS индекс или метаданные."
        print(msg)
        return msg

    features = detect_and_extract_features(uploaded_image_path)
    if features is None:
        return "На загруженном изображении не обнаружена кепка или не удалось извлечь признаки."

    try:
        results = []
        for feature in features:
            query_features = feature / np.linalg.norm(feature)
            query_features = query_features.reshape(1, -1).astype('float32')
            print("Вектор запроса нормализован.")

            # Поиск в FAISS
            D, I = index.search(query_features, top_k)
            print(f"Поиск завершен. Найдено {len(I[0])} похожих кепок.")

            # Составляем результаты
            for idx, score in zip(I[0], D[0]):
                if idx < len(metadata):
                    cap_info = metadata[idx]
                    results.append({
                        'cap_name': cap_info['cap_name'],
                        'image_path': cap_info['image_path'],
                        'similarity_score': float(score)
                    })
                    print(f"Похожая кепка: {cap_info['cap_name']} с сходством {score:.4f}")
                else:
                    print(f"Неверный индекс метаданных: {idx}")
        return results
    except Exception as e:
        print(f"Ошибка при поиске похожей кепки: {e}")
        return f"Ошибка при поиске похожей кепки: {e}"


def display_results(uploaded_image_path, results):
    """
    Отображает загруженное изображение и найденные похожие изображения с их названиями и значениями сходства.
    Для FastAPI обычно не используется, но может пригодиться локально для отладки.
    """
    if not isinstance(results, list):
        print("Нет результатов для отображения.")
        return

    top_k = len(results)
    total_images = top_k + 1

    max_display = 5
    if total_images > max_display + 1:
        print(f"Отображается максимум {max_display} похожих изображений.")
        top_k = max_display
        total_images = top_k + 1
        results = results[:max_display]

    plt.figure(figsize=(5 * total_images, 5))

    try:
        uploaded_image = Image.open(uploaded_image_path).convert("RGB")
        plt.subplot(1, total_images, 1)
        plt.imshow(uploaded_image)
        plt.title("Загруженная кепка")
        plt.axis('off')
    except Exception as e:
        print(f"Ошибка при открытии загруженного изображения: {e}")
        return

    for i, res in enumerate(results):
        try:
            similar_image = Image.open(res['image_path']).convert("RGB")
            plt.subplot(1, total_images, i + 2)
            title = f"Название: {res['cap_name']}\nСходство: {res['similarity_score']:.4f}"
            plt.imshow(similar_image)
            plt.title(title)
            plt.axis('off')
        except Exception as e:
            print(f"Ошибка при открытии изображения {res['image_path']}: {e}")

    plt.tight_layout()
    plt.show()


# ======================================
# Обёртка в FastAPI
# ======================================

app = FastAPI(title="Caps FAISS Search API")


class SearchRequest(BaseModel):
    """
    Модель входных данных для поиска.
    """
    image_path: str
    top_k: int = 1


@app.post("/build_db")
def build_database_endpoint():
    """
    Эндпоинт для создания базы данных:
    1) Организует zip-файлы
    2) Извлекает признаки
    3) Создает FAISS индекс
    """
    try:
        print("\nОрганизация zip файлов...")
        organize_zip_files(EXCEL_FILE, ZIP_FOLDER)

        print("\nЗапуск создания базы данных признаков...")
        features_matrix, metadata = build_feature_database(DATA_DIR, METADATA_FILE)

        if features_matrix is not None:
            index = create_faiss_index(features_matrix, INDEX_FILE)
            if index is not None:
                return {"status": "ok", "message": "База данных и индекс успешно созданы."}
            else:
                return {"status": "error", "message": "Не удалось создать FAISS индекс."}
        else:
            return {"status": "error", "message": "Не удалось создать базу данных признаков."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/search")
def search_endpoint(req: SearchRequest):
    """
    Эндпоинт для поиска похожих кепок по изображению и количеству top_k.
    """
    try:
        results = search_similar_cap(req.image_path, top_k=req.top_k)
        if isinstance(results, str):
            # Если вернулась строка — это сообщение об ошибке или предупреждение
            return {"status": "error", "message": results}
        else:
            return {"status": "ok", "results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
