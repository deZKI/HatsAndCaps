import logging

import aio_pika

logger = logging.getLogger(__name__)


class RabbitMQHandler:
    def __init__(self, rabbitmq_url: str, queue_name: str):
        self.rabbitmq_url = rabbitmq_url
        self.queue_name = queue_name

    async def send_to_queue(self, data: str):
        try:
            connection = await aio_pika.connect_robust(self.rabbitmq_url)
            async with connection:
                channel = await connection.channel()
                queue = await channel.declare_queue(self.queue_name, durable=True)
                await channel.default_exchange.publish(
                    aio_pika.Message(body=data.encode()),
                    routing_key=queue.name,
                )
            logger.info("Данные успешно отправлены в RabbitMQ")
        except Exception as e:
            logger.error(f"Ошибка при отправке в RabbitMQ: {e}")
