from io import BytesIO

from PIL import Image


def convert_webp_to_jpg(image_bytes):
    """
    Конвертирует изображение из формата WEBP в JPG.
    """
    image = Image.open(BytesIO(image_bytes))
    converted_image = BytesIO()
    image.convert("RGB").save(converted_image, format="JPEG")
    converted_image.seek(0)
    return converted_image
