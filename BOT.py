import asyncio
from aiogram import Bot, Dispatcher, types

# Замените на ваш токен
TOKEN = "7954192436:AAGmeydvSJhrxONiq4yc275SunsfhtHErG0"

async def send_message(user_id, text):
    bot = Bot(token=TOKEN)
    dp = Dispatcher()
    """Асинхронно отправляет сообщение пользователю."""
    try:
        await bot.send_message(user_id, text)
        print(f"Сообщение отправлено пользователю {user_id}")
    except Exception as e:
        print(f"Ошибка при отправке сообщения пользователю {user_id}: {e}")


async def main():
    #  Пример использования:
    await send_message(user_id=123456789, text="<b>Привет!</b> Это тестовое сообщение.") # Замените 123456789 на ID пользователя


if __name__ == "__main__":
    asyncio.run(main())
