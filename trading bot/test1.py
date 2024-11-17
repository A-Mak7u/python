import yfinance as yf
import talib
import numpy as np
from aiogram import Bot
import asyncio
from datetime import datetime

API_TOKEN = '8173543278:AAFxSllt2Baepmbu_1SUU9HTqb260Gy9NEs'
CHAT_ID = '696572396'

tickers = ['SBER.ME', 'GAZP.ME', 'LKOH.ME']  # Тикеры для анализа волатильности
bot = Bot(token=API_TOKEN)

# Функция для получения данных о котировках
def get_stock_data(ticker, period='2d', interval='5m'):
    stock = yf.Ticker(ticker)
    data = stock.history(period=period, interval=interval)
    return data

# Функция для расчета исторической волатильности
def calculate_historical_volatility(data):
    returns = data['Close'].pct_change().dropna()
    volatility = np.std(returns) * np.sqrt(252)
    return volatility

# Функция для эмуляции торговли
def scalping_strategy(ticker):
    data = get_stock_data(ticker)
    close_prices = data['Close'].values
    sma_short = talib.SMA(close_prices, timeperiod=5)
    rsi = talib.RSI(close_prices, timeperiod=14)

    if close_prices[-1] > sma_short[-1] and rsi[-1] < 30:
        buy_price = close_prices[-1]
        target_price = buy_price * 1.005  # Цель: 0.5% роста
        return {
            "ticker": ticker,
            "buy_price": buy_price,
            "target_price": target_price
        }
    return None
# Функция для отправки сообщения в Telegram
async def send_telegram_message(text):
    print(f"Sending message: {text}")
    await bot.send_message(CHAT_ID, text)

# Функция анализа и эмуляции торговли в режиме реального времени
async def analyze_and_notify():
    while True:
        now = datetime.now()
        # Проверка рабочего времени биржи
        if now.weekday() >= 5 or now.hour < 10 or now.hour > 18:
            print("Биржа не работает. Ожидаем следующего рабочего дня.")
            await asyncio.sleep(3600)
            continue

        for ticker in tickers:
            result = scalping_strategy(ticker)
            if result:
                message = f"Эмуляция сделки:\nТикер: {result['ticker']}\nЦена покупки: {result['buy_price']:.2f}\nЦелевая цена продажи: {result['target_price']:.2f}"
                await send_telegram_message(message)
            else:
                await send_telegram_message(f"Для {ticker} сигналов на покупку нет.")

        await asyncio.sleep(300)  # Пауза на 5 минут перед следующей проверкой

# Основная функция для запуска анализа
async def main():
    print("Starting bot for real-time trading emulation...")
    await analyze_and_notify()

if __name__ == "__main__":
    asyncio.run(main())
