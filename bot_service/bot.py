import logging
import os
import aiohttp
import asyncio
import json
import aio_pika
import tempfile
from aiogram import Bot, Dispatcher, F
from aiogram.types import Message
from aiogram.fsm.storage.memory import MemoryStorage
from dotenv import load_dotenv

# Загрузка переменных окружения
load_dotenv()

# Инициализация логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Переменные окружения
API_TOKEN = os.getenv("TELEGRAM_API_TOKEN")
ANALYSIS_SERVICE_URL = os.getenv("ANALYSIS_SERVICE_URL")
RABBITMQ_URL = os.getenv("RABBITMQ_URL")

# Проверка обязательных переменных окружения
if not all([API_TOKEN, ANALYSIS_SERVICE_URL, RABBITMQ_URL]):
    raise ValueError("Необходимо указать TELEGRAM_API_TOKEN, ANALYSIS_SERVICE_URL и RABBITMQ_URL в .env файле")

# Инициализация бота и диспетчера
bot = Bot(token=API_TOKEN)
dp = Dispatcher(storage=MemoryStorage())


async def send_to_rabbitmq(data: str):
    """
    Отправка данных в RabbitMQ.
    """
    try:
        connection = await aio_pika.connect_robust(RABBITMQ_URL)
        async with connection:
            channel = await connection.channel()
            queue = await channel.declare_queue("database_queue", durable=True)
            await channel.default_exchange.publish(
                aio_pika.Message(body=data.encode()),
                routing_key=queue.name,
            )
        logger.info("Данные успешно отправлены в RabbitMQ")
    except Exception as e:
        logger.error(f"Ошибка при отправке в RabbitMQ: {e}")


async def analyze_image(file_path: str):
    """
    Отправка изображения на анализ сервису и получение результата.
    """
    try:
        async with aiohttp.ClientSession() as session:
            with open(file_path, "rb") as f:
                form_data = aiohttp.FormData()
                form_data.add_field("file", f, filename=os.path.basename(file_path), content_type="image/jpeg")
                form_data.add_field("top_k", "2")  # Пример параметра для анализа

                async with session.post(f"{ANALYSIS_SERVICE_URL}/search_image", data=form_data) as resp:
                    if resp.status != 200:
                        logger.error(f"Ошибка анализа изображения, код: {resp.status}")
                        return {"status": "error", "message": "Ошибка анализа изображения"}
                    return await resp.json()
    except aiohttp.ClientError as e:
        logger.error(f"Ошибка соединения с сервисом анализа: {e}")
        return {"status": "error", "message": "Сервис анализа временно недоступен"}


@dp.message(F.content_type == "photo")
async def handle_image(message: Message):
    """
    Обработка входящих изображений:
    - Скачиваем фото.
    - Отправляем запрос на сервис image analysis.
    - Сохраняем результат в RabbitMQ.
    - Отправляем ответ пользователю.
    """
    logger.info(f"Получено фото от пользователя {message.from_user.id}")

    user_data = {
        "telegram_id": message.from_user.id,
        "username": message.from_user.username,
        "message": "Фото",
    }

    # Работа с временным файлом
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as temp_file:
        file_path = temp_file.name
        await message.photo[-1].download(destination=file_path)

    # Анализ изображения
    analysis_result = await analyze_image(file_path)

    # Удаление временного файла
    os.remove(file_path)

    # Обработка результата анализа
    user_data["analysis_result"] = analysis_result

    if analysis_result.get("status") == "ok":
        results = analysis_result.get("results", [])
        if not results:
            await message.reply("Извините, похожие объекты не найдены.")
            return

        # Формирование ответа
        answer = "\n".join(
            [f" - {r.get('cap_name', 'Без имени')}, схожесть: {r.get('similarity_score', 0):.2f}" for r in results]
        )
        await message.reply(f"Найдены похожие кепки:\n{answer}")
    else:
        await message.reply(f"Ошибка: {analysis_result.get('message', 'Неизвестная ошибка')}")

    # Отправка данных в RabbitMQ
    await send_to_rabbitmq(json.dumps(user_data))


@dp.message(F.text == "/start")
async def send_welcome(message: Message):
    """
    Приветственное сообщение при вводе команды /start.
    """
    logger.info(f"Пользователь {message.from_user.id} начал работу с ботом.")
    await message.reply(
        "Привет! 👋\n"
        "Я бот для анализа изображений. Просто отправьте мне фото, и я постараюсь найти похожие объекты. 📷\n\n"
        "Если возникнут вопросы, напишите /help."
    )


async def main():
    """
    Основная функция запуска бота.
    """
    logger.info("Бот запущен и готов к обработке сообщений.")
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())
