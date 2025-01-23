import json
import logging
from pathlib import Path

import aiofiles
import aiohttp
from aiogram import Bot, Dispatcher, F
from aiogram.fsm.storage.memory import MemoryStorage
from aiogram.types import FSInputFile, Message
from PIL import Image

from utils import TempFileManager

logger = logging.getLogger(__name__)


class TelegramBot:
    def __init__(self, api_token, analysis_service, rabbitmq_handler):
        self.bot = Bot(token=api_token)
        self.dp = Dispatcher(storage=MemoryStorage())
        self.analysis_service = analysis_service
        self.rabbitmq_handler = rabbitmq_handler

        # Регистрация хендлеров
        self._register_handlers()

    def _register_handlers(self):
        self.dp.message.register(self.send_welcome, F.text == "/start")
        self.dp.message.register(self.handle_image, F.content_type == "photo")

    async def send_welcome(self, message: Message):
        """
        Приветственное сообщение при вводе команды /start.
        """
        logger.info(f"Пользователь {message.from_user.id} начал работу с ботом.")
        await message.reply(
            "Привет! 👋\n"
            "Я бот для анализа изображений. Просто отправьте мне фото, и я постараюсь найти похожие объекты. 📷"
        )

    async def handle_image(self, message: Message):
        """
        Обработка изображений, отправленных пользователем.
        """
        logger.info(f"Получено фото от пользователя {message.from_user.id}")
        user_data = {
            "telegram_id": message.from_user.id,
            "username": message.from_user.username,
            "message": "Фото",
        }

        try:
            # Скачиваем изображение
            photo = message.photo[-1]
            file_info = await self.bot.get_file(photo.file_id)
            file_path = file_info.file_path

            # Работа с временным файлом
            with TempFileManager(suffix=".jpg") as temp_file:
                await self.bot.download_file(file_path, temp_file)

                # Анализ изображения
                analysis_result = await self.analysis_service.analyze_image(temp_file)
                user_data["analysis_result"] = analysis_result

                # Отправка результатов пользователю
                await self.process_analysis_result(message, analysis_result)

            # Отправка данных в RabbitMQ
            await self.rabbitmq_handler.send_to_queue(json.dumps(user_data))

        except Exception as e:
            logger.error(f"Ошибка обработки изображения: {e}")
            await message.reply("Произошла ошибка при обработке изображения. 😞")

    async def process_analysis_result(self, message: Message, analysis_result: dict):
        """
        Обрабатывает результат анализа и отправляет данные пользователю.
        """
        if analysis_result.get("status") != "ok":
            await message.reply(f"Ошибка анализа: {analysis_result.get('message', 'Неизвестная ошибка')}")
            return

        results = analysis_result.get("results", [])
        if not results:
            await message.reply("Извините, похожие объекты не найдены.")
            return

        for res in results:
            await self.send_result(message, res)

        await message.reply("Вот похожие объекты, которые я нашел для вас! 😊")

    from io import BytesIO

    from aiogram.types import FSInputFile
    from PIL import Image

    async def send_result(self, message: Message, result: dict):
        """
        Скачивает изображение из другого сервиса, сохраняет его локально, а затем отправляет пользователю.
        """
        cap_name = result.get("cap_name", "Без имени")
        similarity_score = result.get("similarity_score", 0.0)
        image_path = result.get("image_path")

        if not image_path:
            logger.warning(f"У объекта {cap_name} отсутствует путь к изображению.")
            return

        # Формируем полный URL для загрузки изображения
        image_url = f"{self.analysis_service.service_url}/images/{image_path.lstrip('/')}"
        image_url = image_url.replace("\\", "/")  # Исправляем обратные слэши

        # Путь для временного сохранения загруженного изображения
        temp_dir = Path("temp_images")
        temp_dir.mkdir(parents=True, exist_ok=True)
        local_image_path = temp_dir / Path(image_path).name

        try:
            # Скачиваем изображение
            async with aiohttp.ClientSession() as session:
                async with session.get(image_url) as response:
                    if response.status == 200:
                        async with aiofiles.open(local_image_path, "wb") as f:
                            await f.write(await response.read())
                    else:
                        logger.warning(f"Не удалось загрузить изображение: {image_url}, статус: {response.status}")
                        return

            # Конвертируем изображение, если оно в формате WEBP
            if local_image_path.suffix.lower() == ".webp":
                converted_path = temp_dir / f"{local_image_path.stem}.jpg"
                with Image.open(local_image_path) as img:
                    img.convert("RGB").save(converted_path, "JPEG")
                local_image_path = converted_path

            # Отправляем изображение пользователю
            photo_file = FSInputFile(local_image_path)
            await self.bot.send_photo(
                chat_id=message.chat.id,
                photo=photo_file,
                caption=f"Название: {cap_name}\nСхожесть: {similarity_score:.2f}"
            )

        except Exception as e:
            logger.error(f"Ошибка при обработке изображения {image_url}: {e}")
            await message.reply(f"Не удалось обработать изображение: {cap_name}")
        finally:
            # Удаляем временные файлы
            try:
                if local_image_path.exists():
                    local_image_path.unlink()
            except Exception as cleanup_error:
                logger.warning(f"Не удалось удалить временный файл {local_image_path}: {cleanup_error}")

    async def run(self):
        """
        Запуск бота.
        """
        logger.info("Бот запущен и готов к обработке сообщений.")
        await self.dp.start_polling(self.bot)
