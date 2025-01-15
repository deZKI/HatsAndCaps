import os
import django
import asyncio
import aio_pika
import json
import html
from dotenv import load_dotenv
from asgiref.sync import sync_to_async

load_dotenv()

RABBITMQ_URL = os.getenv("RABBITMQ_URL")

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'config.settings')
django.setup()

from bot_data.models import User, MessageHistory

async def save_message_to_db(telegram_id, username, text, analysis_result):
    # Используем sync_to_async для работы с базой данных
    user, _ = await sync_to_async(User.objects.get_or_create)(
        telegram_id=telegram_id, defaults={"username": username}
    )

    decoded_result = {k: html.unescape(v) if isinstance(v, str) else v
                      for k, v in analysis_result.items()}
    await MessageHistory.objects.acreate(
        user=user,
        message=text,
        analysis_result=decoded_result
    )

async def main():
    connection = await aio_pika.connect_robust(RABBITMQ_URL)
    queue_name = "database_queue"

    async with connection:
        channel = await connection.channel()
        queue = await channel.declare_queue(queue_name, durable=True)

        async with queue.iterator() as queue_iter:
            async for message in queue_iter:
                async with message.process():
                    data = json.loads(message.body.decode())
                    telegram_id = data["telegram_id"]
                    username = data["username"]
                    text = data["message"]
                    analysis_result = data["analysis_result"]

                    # Сохраняем данные в базе данных через асинхронный вызов
                    await save_message_to_db(telegram_id, username, text, analysis_result)

if __name__ == "__main__":
    asyncio.run(main())
