"""
BOT LISTENER V45 — постоянно работещ слушател на Telegram.
Импортира цялата логика от live_bot_v45.py.

Стартиране:
    python bot_listener_v45.py

Команди в Telegram:
    /signal   — сигнали BTC+ETH
    /status   — отворени позиции
    /portfolio — последни сделки
    /market   — макро преглед на пазара
    /help     — списък с команди
"""

import collections
if not hasattr(collections, 'MutableMapping'):
    import collections.abc
    collections.MutableMapping = collections.abc.MutableMapping

import sys
import time
import json
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))
try:
    from live_bot_v45 import (
        send_telegram, get_updates,
        run_portfolio_signal, format_portfolio_signal, format_status,
        check_hard_stop, process_signals,
        load_json, save_json,
        TG_CHAT_ID, ASSETS,
        STATE_FILE, PORTFOLIO_FILE,
    )
except ImportError as e:
    print(f"❌ Грешка при импорт от live_bot_v45.py: {e}")
    sys.exit(1)

OFFSET_FILE = "listener_offset_v45.txt"


# ==============================================================
# ОФСЕТ — за да не се обработват стари съобщения при рестарт
# ==============================================================
def load_offset():
    try:
        if Path(OFFSET_FILE).exists():
            return int(Path(OFFSET_FILE).read_text().strip())
    except:
        pass
    return None


def save_offset(offset: int):
    try:
        Path(OFFSET_FILE).write_text(str(offset))
    except:
        pass


# ==============================================================
# ОБРАБОТКА НА КОМАНДИ
# ==============================================================
def process_command(text: str) -> str | None:
    """Обработва командата и връща отговор или None."""
    text = text.lower().strip()

    # ── /signal — главна команда ──────────────────────
    if text in ("/signal", "/сигнал", "/s"):
        send_telegram("⏳ _Зареждам данни и изчислявам сигнали..._")
        try:
            results    = run_portfolio_signal()
            hard_stop, portfolio_dd, days = check_hard_stop()
            msg = format_portfolio_signal(
                results, hard_stop, portfolio_dd, days)
            # Запазваме състоянието ако има сигнал
            process_signals(results)
            return msg
        except Exception as e:
            return f"❌ Грешка при изчисление на сигнала:\n`{e}`"

    # ── /status — отворени позиции ───────────────────
    elif text in ("/status", "/статус", "/st"):
        send_telegram("⏳ _Зареждам данни..._")
        try:
            results = run_portfolio_signal()
            state   = load_json(STATE_FILE)
            _, dd, _ = check_hard_stop()
            return format_status(results, state, dd)
        except Exception as e:
            return f"❌ Грешка:\n`{e}`"

    # ── /portfolio — история на сделките ─────────────
    elif text in ("/portfolio", "/портфейл", "/p"):
        port   = load_json(PORTFOLIO_FILE)
        trades = port.get("completed_trades", [])
        if not trades:
            return (
                "╔══ 📋 *ИСТОРИЯ НА СДЕЛКИТЕ* ══╗\n\n"
                "  _Все още няма завършени сделки_\n\n"
                "╚══════════════════════════╝"
            )
        recent = trades[-20:]
        pnls   = [t["pnl_pct"] for t in recent]
        wins   = sum(1 for p in pnls if p > 0)
        total  = sum(pnls)
        avg    = total / len(pnls)
        best   = max(pnls)
        worst  = min(pnls)

        lines = [
            "╔══ 📋 *ИСТОРИЯ НА СДЕЛКИТЕ V45* ══╗",
            f"",
            f"  Последни сделки:  `{len(recent)}`",
            f"  Win Rate:         `{wins/len(recent):.0%}`",
            f"  Общ P&L:          `{total:+.1f}%`",
            f"  Средна сделка:    `{avg:+.1f}%`",
            f"  Най-добра:        `{best:+.1f}%`",
            f"  Най-лоша:         `{worst:+.1f}%`",
            f"",
            "  *Последни 10:*",
        ]
        for t in reversed(recent[-10:]):
            em  = "✅" if t["pnl_pct"] > 0 else "❌"
            tkr = t.get("ticker", "?").replace("-USD", "")
            lines.append(
                f"  {em} {tkr} {t['side'][:1]}  "
                f"`{t['pnl_pct']:+.1f}%`  `{t['date']}`"
            )
        lines.append("╚══════════════════════════╝")
        return "\n".join(lines)

    # ── /market — макро преглед ───────────────────────
    elif text in ("/market", "/пазар", "/m"):
        send_telegram("⏳ _Зареждам макро данни..._")
        try:
            results = run_portfolio_signal()
            # Вземаме макро от първия актив (еднакво за всички)
            r = next(
                (v for v in results.values() if not v.get("error")),
                None
            )
            if r is None:
                return "❌ Няма данни"

            from live_bot_v45 import _fng_label, _macro_status
            fng     = r.get("fng", 50)
            spx_mom = r.get("spx_mom", 0)
            dxy_mom = r.get("dxy_mom", 0)
            fund    = r.get("fund", 0)

            lines = [
                "╔══ 🌍 *МАКРО ПРЕГЛЕД* ══╗",
                f"",
                f"  🧠 Fear & Greed: `{fng:.0f}/100`",
                f"  _{_fng_label(fng)}_",
                f"",
                f"  📈 SPX 7д: `{spx_mom*100:+.1f}%`",
                f"  💵 DXY 7д: `{dxy_mom*100:+.1f}%`",
                f"  🔄 Funding: `{fund:.4f}`",
                f"",
                f"  🎯 Режим: *{_macro_status(spx_mom, dxy_mom)}*",
            ]

            for ticker, rv in results.items():
                if rv.get("error"): continue
                em    = rv["emoji"]
                lbl   = rv["label"]
                ls    = rv.get("long_scale",  0)
                ss    = rv.get("short_scale", 0)
                lines += [
                    f"",
                    f"  {em} *{lbl}* скейлинг:",
                    f"  LONG scale:  `{ls:.2f}` {'✅' if ls > 0.6 else '⚠️' if ls > 0.3 else '❌'}",
                    f"  SHORT scale: `{ss:.2f}` {'✅' if ss > 0.6 else '⚠️' if ss > 0.3 else '❌'}",
                ]

            lines.append("╚══════════════════════╝")
            return "\n".join(lines)
        except Exception as e:
            return f"❌ Грешка:\n`{e}`"

    # ── /help ─────────────────────────────────────────────
    elif text in ("/help", "/start", "/помощ"):
        return (
            "╔══ 🤖 *V45 PORTFOLIO BOT* ══╗\n\n"
            "  *Портфейл:* BTC + ETH\n"
            "  *Стратегия:* +19.8%/год, MaxDD -15.8%\n\n"
            "  📡 *Команди:*\n"
            "  `/signal` — сигнали LONG/SHORT\n"
            "  `/status` — отворени позиции\n"
            "  `/portfolio` — история на сделките\n"
            "  `/market` — макро преглед на пазара\n"
            "  `/help` — това меню\n\n"
            "  ⚙️ *Архитектура:*\n"
            "  • XGBoost + LightGBM (многокласови)\n"
            "  • LONG + SHORT + Sniper Entry\n"
            "  • Soft Macro Scaling (FNG+SPX+DXY)\n"
            "  • Portfolio Kelly Penalty\n"
            "  • Hard Stop -15% → пауза 30 дни\n\n"
            "╚══════════════════════════╝"
        )

    return None


# ==============================================================
# ГЛАВЕН ЦИКЪЛ
# ==============================================================
def run_listener():
    print("=" * 55)
    print("  BOT LISTENER V45 — МУЛТИ-АКТИВ ПОРТФЕЙЛ")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("  Активи: BTC-USD + ETH-USD")
    print("=" * 55)

    offset = load_offset()

    send_telegram(
        "🚀 *V45 Portfolio Bot стартира!*\n\n"
        "Активи: ₿ BTC + Ξ ETH\n"
        "Стратегия: V45 | +19.8%/год | MaxDD -15.8%\n\n"
        "_Команди: /signal /status /portfolio /market_"
    )

    while True:
        try:
            updates = get_updates(offset)

            for upd in updates:
                offset = upd["update_id"] + 1
                save_offset(offset)

                msg  = upd.get("message", {})
                text = msg.get("text", "").strip()
                chat = str(msg.get("chat", {}).get("id", ""))

                if chat != str(TG_CHAT_ID):
                    continue
                if not text.startswith("/"):
                    continue

                ts = datetime.now().strftime("%H:%M:%S")
                print(f"  [{ts}] Команда: {text}")

                response = process_command(text)
                if response:
                    send_telegram(response)
                    print(f"  [{ts}] Отговорът е изпратен ({len(response)} символа)")

            time.sleep(2)

        except KeyboardInterrupt:
            print("\n  Слушателят е спрян.")
            send_telegram("🔴 *V45 Bot е спрян*")
            break
        except Exception as e:
            print(f"  ❌ Грешка: {e}")
            time.sleep(10)


if __name__ == "__main__":
    run_listener()
