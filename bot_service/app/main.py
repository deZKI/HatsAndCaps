import asyncio
import os

from dotenv import load_dotenv

from handlers import TelegramBot
from services.analysis import ImageAnalysisService
from services.rabbitmq import RabbitMQHandler

load_dotenv()


async def main():
    # Проверка обязательных переменных окружения
    api_token = os.getenv("TELEGRAM_API_TOKEN")
    analysis_service_url = os.getenv("ANALYSIS_SERVICE_URL")
    rabbitmq_url = os.getenv("RABBITMQ_URL")

    if not all([api_token, analysis_service_url, rabbitmq_url]):
        raise ValueError("Необходимо указать TELEGRAM_API_TOKEN, ANALYSIS_SERVICE_URL и RABBITMQ_URL в .env файле")

    # Инициализация сервисов
    analysis_service = ImageAnalysisService(analysis_service_url)
    rabbitmq_handler = RabbitMQHandler(rabbitmq_url, queue_name="database_queue")
    telegram_bot = TelegramBot(api_token, analysis_service, rabbitmq_handler)

    await telegram_bot.run()


if __name__ == "__main__":
    asyncio.run(main())
