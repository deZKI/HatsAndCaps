import logging
import mimetypes
import os
import shutil
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from starlette.responses import FileResponse

from .ai import CapsRecognizer

logger = logging.getLogger(__name__)

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# Импортируем CapsRecognizer из вашего модуля

app = FastAPI(title="Caps FAISS Search API")

# Инициализация CapsRecognizer
caps_recognizer = CapsRecognizer(
    device='cpu',
    yolo_weights='static/weights/best.pt',
    clip_model_name="ViT-L/14"
)


def save_upload_file(upload_file: UploadFile, destination: Path) -> Path:
    """
    Сохраняет загруженный файл на сервере.
    """
    try:
        with destination.open("wb") as buffer:
            shutil.copyfileobj(upload_file.file, buffer)
        return destination
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка при сохранении файла: {e}")


@app.post("/build_db")
def build_database_endpoint():
    """
    Эндпоинт для создания базы данных:
    1) Организует zip-файлы
    2) Извлекает признаки
    3) Создает FAISS индекс
    """
    try:
        # Организация zip-файлов
        print("\nОрганизация zip файлов...")
        caps_recognizer.organize_zip_files()

        # Создание базы данных признаков
        print("\nЗапуск создания базы данных признаков...")
        features_matrix, metadata = caps_recognizer.build_feature_database()

        if features_matrix is not None:
            # Создание FAISS индекса
            index = caps_recognizer.create_faiss_index(features_matrix)
            if index is not None:
                return {"status": "ok", "message": "База данных и индекс успешно созданы."}
            else:
                return {"status": "error", "message": "Не удалось создать FAISS индекс."}
        else:
            return {"status": "error", "message": "Не удалось создать базу данных признаков."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/search_image")
def search_endpoint(image: UploadFile = File(...), top_k: int = 1):
    """
    Эндпоинт для поиска похожих кепок по изображению и количеству top_k.
    """
    try:
        # Проверка и сохранение загруженного файла
        upload_dir = Path("static/uploads")
        upload_dir.mkdir(parents=True, exist_ok=True)
        uploaded_image_path = save_upload_file(image, upload_dir / image.filename)

        # Поиск похожих кепок
        results = caps_recognizer.search_similar_cap(str(uploaded_image_path), top_k=top_k)

        if isinstance(results, str):
            # Если вернулась строка — это сообщение об ошибке или предупреждение
            return {"status": "error", "message": results}
        else:
            return {"status": "ok", "results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/images/{image_path:path}")
def get_image(image_path: str):
    """
    Эндпоинт для получения изображения по имени.
    """
    # Базовая директория для хранения изображений
    base_dir = Path("static").resolve()

    # Путь к запрашиваемому файлу
    file_path = (base_dir / image_path).resolve()

    # Проверка, что путь находится внутри base_dir
    if not file_path.is_relative_to(base_dir):
        logger.warning(f"Попытка доступа за пределы директории: {file_path}")
        raise HTTPException(status_code=400, detail="Недопустимый путь")

    # Проверка существования файла
    if not file_path.exists():
        logger.error(f"Запрошенное изображение не найдено: {file_path}")
        raise HTTPException(status_code=404, detail="Изображение не найдено")

    # Определение MIME-типа файла
    mime_type, _ = mimetypes.guess_type(file_path)
    if not mime_type:
        mime_type = "application/octet-stream"

    # Логируем успешный запрос
    logger.info(f"Отправка файла: {file_path}")

    # Возвращаем файл как ответ
    return FileResponse(
        path=file_path,
        media_type=mime_type,
        filename=file_path.name
    )