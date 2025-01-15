import os
import shutil
import zipfile
import pickle
import numpy as np
import pandas as pd
import torch
import faiss
from PIL import Image
from tqdm import tqdm
from ultralytics import YOLO
import clip


class CapsRecognizer:
    def __init__(self, device='cpu', yolo_weights='static/weights/best.pt', clip_model_name="ViT-L/14"):
        self.device = device
        self.zip_folder = 'static/zip_files'
        self.data_dir = 'static/zip_files'
        self.index_file = 'static/faiss_index.bin'
        self.metadata_file = 'static/metadata.pkl'
        self.excel_file = 'static/Кепки.xlsx'

        # Инициализация YOLO
        self.yolo_model = YOLO(yolo_weights)
        self.yolo_model.to(self.device)

        # Инициализация CLIP
        self.clip_model, self.preprocess_clip = clip.load(clip_model_name, device=self.device)
        self.clip_model.eval()

        # Определяем метод ресэмплинга для Pillow
        try:
            from PIL import Resampling
            self.resample_method = Resampling.LANCZOS
        except ImportError:
            self.resample_method = Image.LANCZOS

    def organize_zip_files(self):
        """
        Организует zip файлы по группам на основе Excel файла.
        """
        df = pd.read_excel(self.excel_file)
        zip_files = [f for f in os.listdir(self.zip_folder) if f.endswith('.zip')]
        df_titles = df['Название по группе '].tolist()

        for title in df_titles:
            group_folder = os.path.join(self.zip_folder, title)
            os.makedirs(group_folder, exist_ok=True)

            matched = False
            for zip_file in zip_files:
                if os.path.splitext(zip_file)[0] == title:
                    matched = True
                    zip_file_path = os.path.join(self.zip_folder, zip_file)
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

    def resize_image_if_needed(self, image):
        """
        Масштабирует изображение до минимального размера 640x640, если необходимо.
        """
        desired_size = 640
        if image.width < desired_size or image.height < desired_size:
            new_width = max(desired_size, image.width)
            new_height = max(desired_size, image.height)
            image = image.resize((new_width, new_height), self.resample_method)
            print(f"Изображение масштабировано до размера: {image.size}")
        else:
            print(f"Размер изображения подходит: {image.size}")
        return image

    def extract_features_clip(self, image):
        """
        Извлекает вектор признаков из изображения с помощью модели CLIP.
        """
        input_tensor = self.preprocess_clip(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            features = self.clip_model.encode_image(input_tensor)
        features = features.cpu().numpy().flatten()
        features = features / np.linalg.norm(features)  # Нормализация
        return features.astype('float32')

    def detect_and_extract_features(self, image_path):
        """
        Детектирует кепку на изображении и извлекает признаки с помощью CLIP.
        """
        try:
            print(f"\nОбрабатываем изображение: {image_path}")
            image = Image.open(image_path).convert("RGB")
            image = self.resize_image_if_needed(image)

            # Детекция с помощью YOLO
            results = self.yolo_model.predict(image, imgsz=640)
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
                        features = self.extract_features_clip(cropped_image)
                        print("Признаки извлечены с помощью CLIP.")
                        features_list.append(features)

                    return features_list
            else:
                print("Не обнаружено ни одного объекта на изображении.")
        except Exception as e:
            print(f"Ошибка при обработке изображения {image_path}: {e}")

        return None

    def build_feature_database(self):
        """
        Создаёт базу признаков и метаданных.
        """
        feature_list = []
        metadata = []
        supported_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.webp', '.tiff']

        image_paths = [
            os.path.join(root, file)
            for root, _, files in os.walk(self.data_dir)
            for file in files
            if os.path.splitext(file)[1].lower() in supported_extensions
        ]

        print(f"Найдено {len(image_paths)} изображений для обработки.")

        for image_path in tqdm(image_paths):
            cap_name = os.path.basename(os.path.dirname(image_path))
            features = self.detect_and_extract_features(image_path)
            if features:
                for feature in features:
                    feature_list.append(feature)
                    metadata.append({
                        'cap_name': cap_name,
                        'image_path': image_path
                    })

        if feature_list:
            features_matrix = np.vstack(feature_list)
            with open(self.metadata_file, 'wb') as f:
                pickle.dump(metadata, f)
            return features_matrix, metadata
        else:
            return None, None

    def create_faiss_index(self, features_matrix):
        """
        Создаёт FAISS индекс.
        """
        dimension = features_matrix.shape[1]
        index = faiss.IndexFlatIP(dimension)
        faiss.normalize_L2(features_matrix)
        index.add(features_matrix)
        faiss.write_index(index, self.index_file)
        return index

    def load_faiss_index(self):
        """
        Загружает FAISS индекс и метаданные.
        """
        if not os.path.exists(self.index_file) or not os.path.exists(self.metadata_file):
            return None, None
        index = faiss.read_index(self.index_file)
        with open(self.metadata_file, 'rb') as f:
            metadata = pickle.load(f)
        return index, metadata

    def search_similar_cap(self, uploaded_image_path, top_k=1):
        """
        Поиск похожей кепки.
        """
        index, metadata = self.load_faiss_index()
        if not index or not metadata:
            return "Индекс или метаданные отсутствуют."

        features = self.detect_and_extract_features(uploaded_image_path)
        if not features:
            return "Не удалось извлечь признаки."

        results = []
        for feature in features:
            query_features = feature / np.linalg.norm(feature)
            query_features = query_features.reshape(1, -1).astype('float32')
            D, I = index.search(query_features, top_k)
            for idx, score in zip(I[0], D[0]):
                if idx < len(metadata):
                    cap_info = metadata[idx]
                    results.append({
                        'cap_name': cap_info['cap_name'],
                        'image_path': cap_info['image_path'],
                        'similarity_score': float(score)
                    })
        return results
