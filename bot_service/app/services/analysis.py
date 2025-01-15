import aiohttp
import logging

logger = logging.getLogger(__name__)


class ImageAnalysisService:
    def __init__(self, service_url: str):
        self.service_url = service_url

    async def analyze_image(self, file_path: str, top_k: int = 2):
        try:
            async with aiohttp.ClientSession() as session:
                with open(file_path, "rb") as f:
                    form_data = aiohttp.FormData()
                    form_data.add_field("image", f, filename=file_path, content_type="image/jpeg")
                    form_data.add_field("top_k", str(top_k))

                    async with session.post(f"{self.service_url}/search_image", data=form_data) as resp:
                        if resp.status != 200:
                            logger.error(f"Ошибка анализа изображения, код: {resp.status}")
                            return {"status": "error", "message": "Ошибка анализа изображения"}
                        return await resp.json()
        except aiohttp.ClientError as e:
            logger.error(f"Ошибка соединения с сервисом анализа: {e}")
            return {"status": "error", "message": "Сервис анализа временно недоступен"}
