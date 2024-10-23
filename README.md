import asyncio
import aiohttp
import json
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from datetime import datetime, timedelta
from binance.client import Client

# Konfigurera loggning
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class OpenInterestBot:
    def __init__(self, api_key, api_secret, symbol='XAUUSDT', initial_capital=1000, risk_per_trade=0.02, take_profit=0.05, stop_loss=0.02):
        self.client = Client(api_key, api_secret)
        self.symbol = symbol
        self.history = pd.DataFrame(columns=['timestamp', 'open_interest', 'volume', 'price'])
        self.capital = initial_capital
        self.position = 0
        self.buy_price = 0
        self.risk_per_trade = risk_per_trade
        self.take_profit = take_profit
        self.stop_loss = stop_loss
        self.trend_signals = []

    async def fetch_market_data(self):
        """Hämtar marknadsdata (pris, volym, öppet intresse) från Binance."""
        try:
            klines = self.client.get_historical_klines(self.symbol, Client.KLINE_INTERVAL_1MINUTE, "1 hour ago UTC")
            for kline in klines:
                self.history = self.history.append({
                    'timestamp': datetime.fromtimestamp(kline[0] / 1000),
                    'open_interest': None,  # Placeholder, ska fyllas i med verkliga data
                    'volume': float(kline[5]),
                    'price': float(kline[4])  # Stänger priset
                }, ignore_index=True)
            logging.info("Marknadsdata har hämtats.")
        except Exception as e:
            logging.error(f"Fel vid hämtning av marknadsdata: {e}")

    async def fetch_open_interest(self):
        """Simulerad funktion för att hämta öppet intresse från en extern API."""
        # Här skulle du använda aiohttp för att hämta öppet intresse från en API
        open_interest = np.random.random() * 1000  # Simulerad data
        volume = self.history['volume'].iloc[-1] if not self.history.empty else 0
        return open_interest, volume

    async def record_open_interest(self):
        open_interest, volume = await self.fetch_open_interest()
        if open_interest is not None and volume is not None:
            timestamp = pd.Timestamp.now()
            self.history = self.history.append({'timestamp': timestamp, 'open_interest': open_interest, 'volume': volume}, ignore_index=True)
            logging.info(f"Recorded open interest for {self.symbol}: {open_interest} at {timestamp}")

    def analyze_trend(self):
        if len(self.history) < 10:  # Krav på data
            logging.warning("Inte tillräckligt med data för att analysera trend.")
            return "hold"

        # Beräkna tekniska indikatorer
        self.history['SMA_5'] = self.history['price'].rolling(window=5).mean()
        self.history['SMA_10'] = self.history['price'].rolling(window=10).mean()
        self.history['RSI'] = self.calculate_rsi(self.history['price'])

        latest_price = self.history['price'].iloc[-1]
        latest_rsi = self.history['RSI'].iloc[-1]

        # Trendanalys
        if self.history['SMA_5'].iloc[-1] > self.history['SMA_10'].iloc[-1] and latest_rsi < 30:
            return "strong_buy"
        elif self.history['SMA_5'].iloc[-1] < self.history['SMA_10'].iloc[-1] and latest_rsi > 70:
            return "strong_sell"
        else:
            return "hold"

    def calculate_rsi(self, prices, period=14):
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def calculate_position_size(self):
        risk_amount = self.capital * self.risk_per_trade
        if self.buy_price > 0:
            position_size = risk_amount / (self.stop_loss * self.buy_price)
            return position_size
        return 0

    async def place_order(self, action):
        if action == "strong_buy" and self.position == 0:
            position_size = self.calculate_position_size()
            if position_size > 0:
                self.position = position_size
                self.buy_price = self.history['price'].iloc[-1]
                self.capital -= self.buy_price * position_size
                await self.execute_trade('buy', position_size)
                logging.info(f"Köpt {position_size:.4f} enheter av {self.symbol} till pris {self.buy_price:.2f}. Återstående kapital: {self.capital:.2f}.")
        elif action == "strong_sell" and self.position > 0:
            sell_price = self.history['price'].iloc[-1]
            await self.execute_trade('sell', self.position)
            self.capital += sell_price * self.position
            logging.info(f"Sålt {self.position:.4f} enheter av {self.symbol} till pris {sell_price:.2f}. Totalt kapital: {self.capital:.2f}.")
            self.position = 0

    async def execute_trade(self, action, amount):
        try:
            # Här skulle du implementera API-anropet för att utföra handeln
            logging.info(f"Utför {action} handel för {amount} enheter av {self.symbol}.")
        except Exception as e:
            logging.error(f"Fel vid utförande av handel: {e}")

    def check_stop_loss_take_profit(self):
        if self.position > 0:
            current_price = self.history['price'].iloc[-1]
            if current_price >= self.buy_price * (1 + self.take_profit):
                logging.info(f"Take profit utlösts för {self.symbol} till pris {current_price:.2f}.")
                self.position = 0
            elif current_price <= self.buy_price * (1 - self.stop_loss):
                logging.info(f"Stop loss utlösts för {self.symbol} till pris {current_price:.2f}.")
                self.position = 0

    async def run_bot(self):
        while True:
            await self.fetch_market_data()
            await self.record_open_interest()
            action = self.analyze_trend()
            await self.place_order(action)
            self.check_stop_loss_take_profit()
            await asyncio.sleep(60)  # Vänta i 60 sekunder innan nästa iteration

async def main():
    with open('config.json') as config_file:
        config = json.load(config_file)

    api_key = config["api_key"]
    api_secret = config["api_secret"]
    initial_capital = config["initial_capital"]
    risk_per_trade = config["risk_per_trade"]
    take_profit = config["take_profit"]
    stop_loss = config["stop_loss"]

    bot = OpenInterestBot(api_key, api_secret, initial_capital=initial_capital, risk_per_trade=risk_per_trade, take_profit=take_profit, stop_loss=stop_loss)
    await bot.run_bot()

if __name__ == "__main__":
    asyncio.run(main())
