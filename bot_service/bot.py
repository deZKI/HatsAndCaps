import logging
import os
import aiohttp
import asyncio
import json
import aio_pika
from aiogram import Bot, Dispatcher, F
from aiogram.types import Message
from aiogram.fsm.storage.memory import MemoryStorage
from aiogram.types import ContentType
from dotenv import load_dotenv

load_dotenv()

API_TOKEN = os.getenv("TELEGRAM_API_TOKEN")
ANALYSIS_SERVICE_URL = os.getenv("ANALYSIS_SERVICE_URL")
RABBITMQ_URL = os.getenv("RABBITMQ_URL")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

bot = Bot(token=API_TOKEN)
dp = Dispatcher(storage=MemoryStorage())


# Функция для отправки данных в RabbitMQ
async def send_to_rabbitmq(data):
    connection = await aio_pika.connect_robust(RABBITMQ_URL)
    async with connection:
        channel = await connection.channel()
        queue = await channel.declare_queue("database_queue", durable=True)
        await channel.default_exchange.publish(
            aio_pika.Message(body=data.encode()),
            routing_key=queue.name,
        )


@dp.message(content_types=[ContentType.PHOTO])
async def handle_image(message: Message):
    """
    Обработка входящих изображений:
      - Скачиваем фото
      - Отправляем запрос на сервис image analysis
      - Сохраняем результат в RabbitMQ
      - Отправляем ответ пользователю
    """
    user_data = {
        "telegram_id": message.from_user.id,
        "username": message.from_user.username,
        "message": "Фото"
    }

    # 1. Скачиваем файл на диск (или в оперативную память)
    photo = message.photo[-1]  # Берём самое большое по размеру
    file_path = f"{photo.file_id}.jpg"
    await photo.download(destination=file_path)

    # 2. Отправляем запрос на сервис, который обрабатывает изображение (CLIP+YOLO+FAISS)
    #    Для примера используем параметр top_k=2
    async with aiohttp.ClientSession() as session:
        try:
            # Чтобы отправить файл, используем параметр "files"
            with open(file_path, "rb") as f:
                form_data = aiohttp.FormData()
                form_data.add_field("file", f, filename=file_path, content_type="image/jpeg")
                form_data.add_field("top_k", "2")  # если нужно несколько похожих

                async with session.post(
                        f"{ANALYSIS_SERVICE_URL}/search_image",
                        data=form_data
                ) as resp:
                    if resp.status != 200:
                        await message.reply("Ошибка при анализе изображения.")
                        return
                    analysis_result = await resp.json()
        except aiohttp.ClientError:
            await message.reply("Сервис анализа изображений временно недоступен.")
            return
        finally:
            # Удаляем временный файл, чтобы не засорять дисковое пространство
            if os.path.exists(file_path):
                os.remove(file_path)

    # 3. Добавляем результат анализа к данным пользователя
    user_data["analysis_result"] = analysis_result

    # 4. Отправляем данные в RabbitMQ для сохранения
    await send_to_rabbitmq(json.dumps(user_data))

    # 5. Готовим ответ пользователю
    if analysis_result.get("status") == "ok":
        results = analysis_result.get("results", [])
        if not results:
            # Пустой список — значит модель ничего не нашла
            await message.reply("Извините, похожие объекты не найдены.")
            return

        # Собираем описание найденных результатов
        answer_parts = []
        for r in results:
            cap_name = r.get("cap_name")
            score = r.get("similarity_score")
            answer_parts.append(f" - {cap_name}, схожесть: {score:.2f}")
        answer_text = "Найдены похожие кепки:\n" + "\n".join(answer_parts)
    else:
        answer_text = f"Ошибка: {analysis_result}"

    await message.reply(answer_text)


async def main():
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())
