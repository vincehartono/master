# fx_bot.py  (Python 3.9 compatible) — TP decay fix
from ib_insync import *
import pandas as pd
import numpy as np
import os, time
from datetime import datetime, timezone
from typing import Optional, Dict

# =========================
# CONFIG
# =========================
TWS_HOST = '127.0.0.1'
TWS_PORT = 7497            # Paper TWS: 7497, Live TWS: 7496
CLIENT_ID = 1

FX_PAIRS = ['AUDCHF', 'AUDJPY', 'EURAUD', 'USDJPY']
TRADE_SIZE = 10000

BAR_SIZE = '5 mins'
DURATION = '5 D'
WHAT_TO_SHOW = 'MIDPOINT'

SHORT_MA = 5
LONG_MA = 20
RSI_PERIOD = 14
RSI_BUY_MAX = 60
RSI_SELL_MIN = 40

SL_PIPS = 20
TP_INITIAL_PCT = 0.003            # 0.3%
TP_DECAY_PCT_PER_MIN = 0.001      # 0.1%/min
TP_MIN_PCT = 0.0                  # don’t go below breakeven
LOOP_SLEEP_SEC = 60

LOG_DIR = 'logs'
LOG_FILE = os.path.join(LOG_DIR, 'fx_trades_log.csv')

# =========================
# HELPERS
# =========================
def ensure_log_file():
    os.makedirs(LOG_DIR, exist_ok=True)
    if not os.path.exists(LOG_FILE):
        with open(LOG_FILE, 'w', encoding='utf-8') as f:
            f.write('timestamp,pair,action,signal,price,sl,tp,orderId,status,sma_short,sma_long,rsi,notes\n')

def log_row(**kw):
    ensure_log_file()
    def _fmt(x): return '' if x is None else str(x)
    line = (
        f"{_fmt(kw.get('timestamp'))},{_fmt(kw.get('pair'))},{_fmt(kw.get('action'))},{_fmt(kw.get('signal'))},"
        f"{_fmt(kw.get('price'))},{_fmt(kw.get('sl'))},{_fmt(kw.get('tp'))},{_fmt(kw.get('orderId'))},{_fmt(kw.get('status'))},"
        f"{_fmt(kw.get('sma_short'))},{_fmt(kw.get('sma_long'))},{_fmt(kw.get('rsi'))},{_fmt(kw.get('notes'))}\n"
    )
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(line)

def pip_size_for(pair: str) -> float:
    return 0.01 if 'JPY' in pair.upper() else 0.0001

def round_to_pip(price: float, pair: str) -> float:
    pip = pip_size_for(pair)
    return round(price / pip) * pip

def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = (delta.clip(lower=0)).ewm(alpha=1/period, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1/period, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi

def get_history(ib: IB, contract: Contract) -> Optional[pd.DataFrame]:
    bars = ib.reqHistoricalData(
        contract,
        endDateTime='',
        durationStr=DURATION,
        barSizeSetting=BAR_SIZE,
        whatToShow=WHAT_TO_SHOW,
        useRTH=False,
        formatDate=1
    )
    df = util.df(bars)
    if df is None or df.empty:
        return None
    df['SMA_S'] = df['close'].rolling(SHORT_MA).mean()
    df['SMA_L'] = df['close'].rolling(LONG_MA).mean()
    df['RSI'] = compute_rsi(df['close'], RSI_PERIOD)
    return df

def signal_from_crossover(df: pd.DataFrame) -> str:
    if df is None or len(df) < max(LONG_MA, RSI_PERIOD) + 2:
        return 'HOLD'
    s_now, l_now = df['SMA_S'].iloc[-2], df['SMA_L'].iloc[-2]
    s_prev, l_prev = df['SMA_S'].iloc[-3], df['SMA_L'].iloc[-3]
    rsi_now = df['RSI'].iloc[-2]
    if np.isnan([s_now, l_now, s_prev, l_prev, rsi_now]).any():
        return 'HOLD'
    if s_now > l_now and s_prev <= l_prev and rsi_now < RSI_BUY_MAX:
        return 'BUY'
    if s_now < l_now and s_prev >= l_prev and rsi_now > RSI_SELL_MIN:
        return 'SELL'
    return 'HOLD'

def latest_metrics(df: pd.DataFrame) -> dict:
    row = df.iloc[-2]
    return {'time': row['date'], 'close': float(row['close']),
            'sma_s': float(row['SMA_S']), 'sma_l': float(row['SMA_L']),
            'rsi': float(row['RSI'])}

def describe(df: pd.DataFrame, pair: str):
    r = df.iloc[-2]
    print(f"{pair}  {r['date']}  close={r['close']:.5f}  SMA{SHORT_MA}={r['SMA_S']:.5f}  SMA{LONG_MA}={r['SMA_L']:.5f}  RSI={r['RSI']:.2f}")

def get_positions_by_base(ib: IB) -> Dict[str, float]:
    pos = {}
    for p in ib.positions():
        if p.contract.secType == 'CASH':
            pos[p.contract.symbol] = pos.get(p.contract.symbol, 0.0) + float(p.position)
    return pos

# === Track live trades with child Trades (not just Orders) ===
class LiveTrade:
    def __init__(self, parent_trade: Trade, tp_trade: Trade, sl_trade: Trade,
                 direction: str, entry_price: float, pair: str):
        self.parent_trade = parent_trade
        self.tp_trade = tp_trade
        self.sl_trade = sl_trade
        self.direction = direction
        self.entry_price = entry_price
        self.pair = pair
        self.entry_time = datetime.utcnow().replace(tzinfo=timezone.utc)
        self.initial_tp_pct = TP_INITIAL_PCT

active_trades: Dict[str, LiveTrade] = {}  # key: pair

def place_bracket_percent_tp(ib: IB, contract: Contract, pair: str, action: str, qty: int, entry_price_ref: float) -> LiveTrade:
    pip = pip_size_for(pair)
    sl_dist = SL_PIPS * pip
    tp_pct = TP_INITIAL_PCT

    if action.upper() == 'BUY':
        sl_price = round_to_pip(entry_price_ref - sl_dist, pair)
        tp_price = round_to_pip(entry_price_ref * (1.0 + tp_pct), pair)
        parent_action, child_action = 'BUY', 'SELL'
    else:
        sl_price = round_to_pip(entry_price_ref + sl_dist, pair)
        tp_price = round_to_pip(entry_price_ref * (1.0 - tp_pct), pair)
        parent_action, child_action = 'SELL', 'BUY'

    parent = MarketOrder(parent_action, qty); parent.transmit = False
    t_parent = ib.placeOrder(contract, parent)
    ib.sleep(0.5)
    parentId = t_parent.order.orderId

    tp = LimitOrder(child_action, qty, tp_price); tp.parentId = parentId; tp.transmit = False
    sl = StopOrder(child_action, qty, sl_price);  sl.parentId = parentId; sl.transmit = True

    t_tp = ib.placeOrder(contract, tp)
    t_sl = ib.placeOrder(contract, sl)

    return LiveTrade(t_parent, t_tp, t_sl, action.upper(), entry_price_ref, pair)

def child_active(trade: Trade) -> bool:
    st = trade.orderStatus.status
    return st not in ('Filled', 'Cancelled', 'ApiCancelled', 'Inactive')

def update_dynamic_tp(ib: IB, lt: LiveTrade, contract: Contract):
    """
    Reduce TP by 0.1% per minute from entry, not below TP_MIN_PCT.
    Modify the existing TP order using the same orderId.
    """
    now = datetime.utcnow().replace(tzinfo=timezone.utc)
    minutes = int((now - lt.entry_time).total_seconds() // 60)
    decay = minutes * TP_DECAY_PCT_PER_MIN
    effective_pct = max(lt.initial_tp_pct - decay, TP_MIN_PCT)

    if lt.direction == 'BUY':
        new_tp = round_to_pip(lt.entry_price * (1.0 + effective_pct), lt.pair)
        # Only tighten (lower) TP for BUY
        cur = lt.tp_trade.order.lmtPrice
        if cur is None or new_tp < cur:
            lt.tp_trade.order.lmtPrice = new_tp
            ib.placeOrder(contract, lt.tp_trade.order)  # modify existing orderId
            print(f"🔧 Adjusted TP for {lt.pair}: {cur} -> {new_tp}")
            return new_tp
    else:
        new_tp = round_to_pip(lt.entry_price * (1.0 - effective_pct), lt.pair)
        cur = lt.tp_trade.order.lmtPrice
        if cur is None or new_tp > cur:
            lt.tp_trade.order.lmtPrice = new_tp
            ib.placeOrder(contract, lt.tp_trade.order)
            print(f"🔧 Adjusted TP for {lt.pair}: {cur} -> {new_tp}")
            return new_tp
    return None

# =========================
# MAIN
# =========================
def main():
    ensure_log_file()

    ib = IB()
    print(f"Connecting to TWS {TWS_HOST}:{TWS_PORT} ...")
    ib.connect(TWS_HOST, TWS_PORT, clientId=CLIENT_ID)
    print("Connected:", ib.isConnected())

    contracts = {pair: Forex(pair) for pair in FX_PAIRS}
    ib.qualifyContracts(*contracts.values())

    try:
        while True:
            base_pos = get_positions_by_base(ib)

            # === Maintain active trades (adjust TP & prune when truly done)
            to_delete = []
            for pair, lt in active_trades.items():
                base = pair[:3]
                # Adjust TP while position is still open and TP order is active
                if abs(base_pos.get(base, 0.0)) != 0.0 and child_active(lt.tp_trade):
                    new_tp = update_dynamic_tp(ib, lt, contracts[pair])
                    if new_tp is not None:
                        log_row(timestamp=datetime.utcnow().isoformat(), pair=pair, action='TP_ADJUST',
                                signal='', price='', sl='',
                                tp=new_tp, orderId=lt.parent_trade.order.orderId, status='',
                                sma_short='', sma_long='', rsi='', notes='TP decay applied')
                # Remove only when position is flat AND both children are inactive
                if abs(base_pos.get(base, 0.0)) == 0.0 and not child_active(lt.tp_trade) and not child_active(lt.sl_trade):
                    to_delete.append(pair)
            for k in to_delete:
                del active_trades[k]

            # === Scan for new entries
            for pair in FX_PAIRS:
                print(f"\n🔄 {time.strftime('%H:%M:%S')} Checking {pair}...")
                contract = contracts[pair]
                base = pair[:3]

                holding = (abs(base_pos.get(base, 0.0)) != 0.0) or (pair in active_trades)
                if holding:
                    print(f"⛔ Holding {pair} (position or active bracket). Skipping entry.")
                    log_row(timestamp=datetime.utcnow().isoformat(), pair=pair, action='HOLD', signal='HOLD',
                            price='', sl='', tp='', orderId='', status='', sma_short='', sma_long='', rsi='',
                            notes='Holding position or active bracket')
                    continue

                try:
                    df = get_history(ib, contract)
                    if df is None:
                        print("❌ No data returned.")
                        continue

                    describe(df, pair)
                    sig = signal_from_crossover(df)
                    m = latest_metrics(df)
                    print(f"📊 Signal: {sig}")

                    if sig in ('BUY', 'SELL'):
                        entry_ref = m['close']
                        print(f"✅ {sig} {pair} — {TRADE_SIZE} units @ ~{entry_ref:.5f}")
                        try:
                            lt = place_bracket_percent_tp(ib, contract, pair, sig, TRADE_SIZE, entry_ref)
                            active_trades[pair] = lt
                            ib.sleep(1.0)
                            status = lt.parent_trade.orderStatus.status
                            order_id = lt.parent_trade.order.orderId
                            log_row(timestamp=datetime.utcnow().isoformat(), pair=pair, action=sig, signal=sig,
                                    price=round(m['close'], 5),
                                    sl=lt.sl_trade.order.auxPrice if hasattr(lt.sl_trade.order, 'auxPrice') else '',
                                    tp=lt.tp_trade.order.lmtPrice,
                                    orderId=order_id, status=status,
                                    sma_short=round(m['sma_s'], 5), sma_long=round(m['sma_l'], 5), rsi=round(m['rsi'], 2),
                                    notes='Bracket submitted (percent TP with decay)')
                        except Exception as e:
                            print(f"⚠️ Order placement error on {pair}: {e}")
                            log_row(timestamp=datetime.utcnow().isoformat(), pair=pair, action='ERROR', signal=sig,
                                    price=round(m['close'], 5), sl='', tp='', orderId='', status='',
                                    sma_short=round(m['sma_s'], 5), sma_long=round(m['sma_l'], 5), rsi=round(m['rsi'], 2),
                                    notes=f"Order error: {e}")
                    else:
                        log_row(timestamp=datetime.utcnow().isoformat(), pair=pair, action='HOLD', signal=sig,
                                price=round(m['close'], 5), sl='', tp='', orderId='', status='',
                                sma_short=round(m['sma_s'], 5), sma_long=round(m['sma_l'], 5), rsi=round(m['rsi'], 2),
                                notes='No entry criteria')
                        print("⏸ No trade executed.")

                    time.sleep(0.4)

                except Exception as e:
                    print(f"⚠️ Error processing {pair}: {e}")
                    log_row(timestamp=datetime.utcnow().isoformat(), pair=pair, action='ERROR', signal='',
                            price='', sl='', tp='', orderId='', status='', sma_short='', sma_long='', rsi='',
                            notes=f"Loop error: {e}")

            time.sleep(LOOP_SLEEP_SEC)

    except KeyboardInterrupt:
        print("\n🛑 Stopping loop...")

    finally:
        ib.disconnect()
        print("Disconnected.")

if __name__ == '__main__':
    main()
