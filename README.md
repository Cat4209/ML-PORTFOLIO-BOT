# ML Portfolio Bot — BTC + ETH

> Алгоритмична търговска система базирана на машинно обучение за криптовалутни пазари.  
> Архитектура V45 | +19.8%/год | MaxDD −15.8% | Sharpe 2.24

---

##  Какво прави ботът

Ботът анализира пазарните условия в реално време и генерира търговски сигнали за BTC и ETH едновременно. Всеки актив има собствен независим ML модел, а рискът се управлява на ниво портфейл — не на ниво отделна позиция.

Сигналите пристигат директно в Telegram с точна точка на вход, Take Profit, Stop Loss и препоръчан размер на позицията.

---

##  Архитектура

### ML модели
- **XGBoost + LightGBM** — ансамбъл от два мултикласови класификатора
- **3 класа:** LONG / SHORT / Flat (без позиция)
- **Walk-forward преобучение** на всеки 90 дни — моделите не се застояват

### Входни характеристики
| Група | Характеристики |
|---|---|
| Технически | RSI, ATR, Bollinger Bands, EMA 21/55/200, моментум 3/7/14/30д |
| Макро | SPX momentum, DXY momentum, VIX режим |
| Sentiment | Fear & Greed Index (7-дневна MA) |
| Funding | Funding Rate + 7-дневна MA + промяна |
| Cross-asset | ETH/BTC корелация, дивергенция, ratio |

### Управление на риска
- **Sniper Entry** — лимитен ордер на −0.4% под цената (LONG) / +0.4% над (SHORT), вместо пазарен
- **Soft Macro Scaling** — размерът на позицията се намалява при неблагоприятни макро условия (FNG, SPX, DXY, Funding)
- **Half-Kelly** — оптимален размер на позицията по формулата на Кели, намален наполовина
- **Portfolio Kelly Penalty** — при едновременни сигнали на BTC и ETH (корелация 0.88), всяка позиция се намалява автоматично
- **Breakeven Trailing** — стопът се придвижва до вход при достигане на 65% от пътя до TP
- **Portfolio Hard Stop** — при просадка −15% на целия портфейл, ботът спира за 30 дни

---

##  Резултати (Out-of-Sample тест)

| Метрика | BTC | ETH | Портфейл |
|---|---|---|---|
| Доходност/год | — | — | **+19.8%** |
| Max Drawdown | — | — | **−15.8%** |
| Sharpe Ratio | — | — | **2.24** |
| Сделки/год | ~20 | ~20 | ~40 |

> Тестът е Walk-Forward Out-of-Sample върху последните ~2.2 години данни.  
> Резултатите не са гаранция за бъдеща доходност.

---

##  Структура на проекта

```
ml-portfolio-bot/
│
├── model_v45.py          # Обучение на модели + Walk-Forward тест + Optuna оптимизация
├── live_bot_v45.py       # Генериране на сигнали + форматиране на Telegram съобщения
├── bot_listener_v45.py   # Постоянно работещ Telegram слушател
│
├── models_v45/           # Запазени модели (създава се автоматично)
│   ├── v45_btc_usd.joblib
│   └── v45_eth_usd.joblib
│
├── onchain_cache/        # Кеш за Fear & Greed данни
│   └── fear_greed.csv
│
├── funding_rate.csv      # Funding Rate история за BTC
├── bot_state_v45.json    # Текущи отворени позиции
├── portfolio_state_v45.json  # История на сделките + Hard Stop статус
├── trades_journal_v45.csv    # Пълен журнал на сделките
│
├── requirements.txt
└── README.md
```

---

## 🚀 Инсталация и стартиране

### 1. Клониране и инсталация
```bash
git clone https://github.com/<your-username>/ml-portfolio-bot.git
cd ml-portfolio-bot
pip install -r requirements.txt
```

### 2. Конфигурация
В `live_bot_v45.py` попълни своите Telegram данни:
```python
TG_TOKEN   = "YOUR_TELEGRAM_BOT_TOKEN_HERE"
TG_CHAT_ID = "YOUR_TELEGRAM_CHAT_ID_HERE"
```

> **Как да получиш токен:** [@BotFather](https://t.me/BotFather) → `/newbot`  
> **Как да получиш Chat ID:** [@userinfobot](https://t.me/userinfobot)

### 3. Подготви данните
```bash
# Постави fear_greed.csv в onchain_cache/
# Постави funding_rate.csv в корена на проекта
```

### 4. Обучи моделите (еднократно, ~30–60 мин)
```bash
python model_v45.py
```
Това ще създаде `models_v45/v45_btc_usd.joblib` и `models_v45/v45_eth_usd.joblib`.

### 5. Стартирай бота

**Еднократен сигнал:**
```bash
python live_bot_v45.py
```

**Постоянен Telegram слушател:**
```bash
python bot_listener_v45.py
```

---

## 📱 Telegram команди

| Команда | Описание |
|---|---|
| `/signal` | Текущи сигнали за BTC и ETH с вход, TP, SL и размер на позиция |
| `/status` | Отворени позиции с текущ P&L |
| `/portfolio` | История на последните 20 сделки + Win Rate + общ P&L |
| `/market` | Макро преглед — Fear & Greed, SPX, DXY, Funding Rate |
| `/help` | Списък с команди |

---

## 🔄 Workflow

```
Всеки ден (или при /signal):
  1. Зареждане на данни  →  yfinance (OHLCV) + макро (SPX, DXY, VIX, FNG)
  2. ML предсказване     →  XGBoost + LightGBM → вероятности LONG / SHORT / Flat
  3. Macro Scaling       →  намаляване на сигнала при лоши макро условия
  4. Kelly sizing        →  изчисляване на размера на позицията
  5. Portfolio penalty   →  намаляване при едновременни сигнали (корелация)
  6. Hard Stop проверка  →  ако портфейлът е в пауза → без сигнали
  7. Telegram            →  изпращане на форматирано съобщение
```

---

## ⚠️ Важни бележки

- Ботът **не изпраща ордери автоматично** — сигналите се изпълняват ръчно
- Използвай **лимитен ордер**, не пазарен (Sniper Entry логиката разчита на това)
- Сигналът е валиден **до края на деня** — ако лимитът не е ударен, ордерът се отменя
- При активен **Hard Stop** ботът не генерира сигнали до края на паузата

---

## 📦 Зависимости

```
numpy, pandas, scikit-learn
xgboost, lightgbm, optuna
yfinance, requests, matplotlib
joblib, tqdm
```

Пълен списък: `requirements.txt`

---

## 📜 Лиценз

MIT License — свободно използване с упоменаване на автора.
