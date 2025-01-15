import logging
import os
import json
import tempfile
from aiogram import Bot, Dispatcher, F
from aiogram.types import Message
from aiogram.fsm.storage.memory import MemoryStorage
from app.utils.temp_file import TempFileManager

logger = logging.getLogger(__name__)

class TelegramBot:
    def __init__(self, api_token, analysis_service, rabbitmq_handler):
        self.bot = Bot(token=api_token)
        self.dp = Dispatcher(storage=MemoryStorage())
        self.analysis_service = analysis_service
        self.rabbitmq_handler = rabbitmq_handler

        # Регистрация хендлеров
        self.register_handlers()

    def register_handlers(self):
        self.dp.message.register(self.send_welcome, F.text == "/start")
        self.dp.message.register(self.handle_image, F.content_type == "photo")

    async def send_welcome(self, message: Message):
        logger.info(f"Пользователь {message.from_user.id} начал работу с ботом.")
        await message.reply(
            "Привет! 👋\n"
            "Я бот для анализа изображений. Просто отправьте мне фото, и я постараюсь найти похожие объекты. 📷"
        )

    async def handle_image(self, message: Message):
        logger.info(f"Получено фото от пользователя {message.from_user.id}")
        user_data = {
            "telegram_id": message.from_user.id,
            "username": message.from_user.username,
            "message": "Фото",
        }

        try:
            photo = message.photo[-1]
            file_info = await self.bot.get_file(photo.file_id)
            file_path = file_info.file_path

            # Работа с временным файлом
            with TempFileManager(suffix=".jpg") as temp_file:
                await self.bot.download_file(file_path, temp_file)

                # Анализ изображения
                analysis_result = await self.analysis_service.analyze_image(temp_file)

            # Обработка результата анализа
            user_data["analysis_result"] = analysis_result

            if analysis_result.get("status") == "ok":
                results = analysis_result.get("results", [])
                if not results:
                    await message.reply("Извините, похожие объекты не найдены.")
                    return

                answer = "\n".join(
                    [f" - {r.get('cap_name', 'Без имени')}, схожесть: {r.get('similarity_score', 0):.2f}" for r in results]
                )
                await message.reply(f"Найдены похожие объекты:\n{answer}")
            else:
                await message.reply(f"Ошибка: {analysis_result.get('message', 'Неизвестная ошибка')}")

            # Отправка данных в RabbitMQ
            await self.rabbitmq_handler.send_to_queue(json.dumps(user_data))

        except Exception as e:
            logger.error(f"Ошибка обработки изображения: {e}")
            await message.reply("Произошла ошибка при обработке изображения.")

    async def run(self):
        logger.info("Бот запущен и готов к обработке сообщений.")
        await self.dp.start_polling(self.bot)
