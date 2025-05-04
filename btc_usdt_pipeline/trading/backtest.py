# trading/backtest.py
"""
Contains the backtesting engine logic.
Refactored from scripts/backtest.py to use centralized config and helpers.
"""
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple
import math

# Use absolute imports from the package
from btc_usdt_pipeline import config
from btc_usdt_pipeline.utils.helpers import setup_logger
# Note: generate_signals is called *before* the backtest function usually.
# The backtest function receives the signals as input.

logger = setup_logger('backtest.log')

def run_backtest(df: pd.DataFrame,
                 signals: np.ndarray,
                 initial_equity: float = config.INITIAL_EQUITY,
                 atr_col: str = config.BACKTEST_ATR_COLUMN,
                 sl_multiplier: float = config.ATR_STOP_LOSS_MULTIPLIER,
                 tp_multiplier: float = config.ATR_TAKE_PROFIT_MULTIPLIER,
                 commission_rate: float = config.COMMISSION_RATE,
                 slippage_points: float = config.SLIPPAGE_POINTS,
                 risk_fraction: float = config.RISK_FRACTION
                 ) -> Tuple[List[float], List[Dict[str, Any]]]:
    """
    Runs an event-driven backtest with position sizing, commissions, and slippage.

    Args:
        df (pd.DataFrame): DataFrame with OHLC data and the ATR column, indexed by time.
        signals (np.ndarray): Array of trading signals ("Long", "Short", "Flat") corresponding to df rows.
        initial_equity (float): Starting capital for the backtest.
        atr_col (str): Name of the column containing ATR values for SL/TP calculation.
        sl_multiplier (float): Multiplier for ATR to set stop loss distance.
        tp_multiplier (float): Multiplier for ATR to set take profit distance.
        commission_rate (float): Commission fee per trade (e.g., 0.001 for 0.1%).
        slippage_points (float): Price difference due to slippage on entry/exit.
        risk_fraction (float): Fraction of equity to risk per trade.

    Returns:
        Tuple[List[float], List[Dict[str, Any]]]:
            - equity_curve: List of equity values over time.
            - trade_log: List of dictionaries detailing each executed trade.
    """
    if len(df) != len(signals):
        logger.error(f"DataFrame length ({len(df)}) and signals length ({len(signals)}) mismatch. Cannot run backtest.")
        return [initial_equity], []

    if atr_col not in df.columns:
        logger.error(f"ATR column '{atr_col}' not found in DataFrame. Cannot run backtest.")
        return [initial_equity], []

    logger.info(f"Starting backtest with Initial Equity: ${initial_equity:,.2f}, Risk/Trade: {risk_fraction:.2%}, SL: {sl_multiplier}*ATR, TP: {tp_multiplier}*ATR, Comm: {commission_rate:.4%}, Slippage: {slippage_points} pts")

    equity = float(initial_equity)
    equity_curve = [equity]
    position = 0 # 0: Flat, 1: Long, -1: Short
    position_size = 0.0 # Size of the current position in base asset (e.g., BTC)
    entry_price = 0.0
    trade_log: List[Dict[str, Any]] = []
    stop_loss = None
    take_profit = None
    current_trade_index = -1 # Index in trade_log for the currently open trade

    # Iterate through each bar/row in the DataFrame
    for i, (index, row) in enumerate(df.iterrows()):
        signal = signals[i]
        close = row['close']
        high = row['high']
        low = row['low']
        atr = row[atr_col]

        # Ensure ATR is valid for SL/TP calculation
        if pd.isna(atr) or atr <= 0:
            logger.warning(f"Invalid ATR ({atr}) at index {index}. Skipping trade checks for this bar.")
            equity_curve.append(float(equity)) # Equity remains unchanged
            continue

        # --- Check for SL/TP Hits for Open Positions ---
        exit_price = None
        pnl = 0.0
        trade_closed = False

        if position == 1: # Currently Long
            # Check Stop Loss first (most conservative)
            if low <= stop_loss:
                exit_price = stop_loss - slippage_points # Apply slippage on exit
                trade_closed = True
                log_reason = "SL"
            # Check Take Profit
            elif high >= take_profit:
                exit_price = take_profit - slippage_points # Apply slippage on exit
                trade_closed = True
                log_reason = "TP"

            if trade_closed:
                pnl = (exit_price - entry_price) * position_size
                commission = abs(pnl) * commission_rate # Commission on exit value
                net_pnl = pnl - commission
                equity += net_pnl
                logger.info(f"Long {log_reason} hit at ~{exit_price:.2f} (Index: {index}). Entry: {entry_price:.2f}, Size: {position_size:.4f}, PnL: {net_pnl:.2f}, Equity: {equity:.2f}")
                trade_log[current_trade_index].update({
                    'Exit': exit_price,
                    'Exit_idx': index,
                    'PnL': net_pnl,
                    'Commission': commission,
                    'Exit Reason': log_reason
                })
                position = 0
                position_size = 0.0
                stop_loss, take_profit = None, None

        elif position == -1: # Currently Short
            # Check Stop Loss first
            if high >= stop_loss:
                exit_price = stop_loss + slippage_points # Apply slippage on exit
                trade_closed = True
                log_reason = "SL"
            # Check Take Profit
            elif low <= take_profit:
                exit_price = take_profit + slippage_points # Apply slippage on exit
                trade_closed = True
                log_reason = "TP"

            if trade_closed:
                pnl = (entry_price - exit_price) * position_size # PnL for short
                commission = abs(pnl) * commission_rate # Commission on exit value
                net_pnl = pnl - commission
                equity += net_pnl
                logger.info(f"Short {log_reason} hit at ~{exit_price:.2f} (Index: {index}). Entry: {entry_price:.2f}, Size: {position_size:.4f}, PnL: {net_pnl:.2f}, Equity: {equity:.2f}")
                trade_log[current_trade_index].update({
                    'Exit': exit_price,
                    'Exit_idx': index,
                    'PnL': net_pnl,
                    'Commission': commission,
                    'Exit Reason': log_reason
                })
                position = 0
                position_size = 0.0
                stop_loss, take_profit = None, None

        # --- Check for New Entry Signals (only if flat) ---
        if position == 0 and equity > 0: # Can only enter if flat and solvent
            enter_trade = False
            sl_distance_pts = sl_multiplier * atr
            tp_distance_pts = tp_multiplier * atr

            if signal == "Long" and sl_distance_pts > 0:
                enter_trade = True
                entry_price = close + slippage_points # Apply slippage on entry
                stop_loss = entry_price - sl_distance_pts
                take_profit = entry_price + tp_distance_pts
                risk_per_unit = entry_price - stop_loss
                trade_type = "Long"

            elif signal == "Short" and sl_distance_pts > 0:
                enter_trade = True
                entry_price = close - slippage_points # Apply slippage on entry
                stop_loss = entry_price + sl_distance_pts
                take_profit = entry_price - tp_distance_pts
                risk_per_unit = stop_loss - entry_price
                trade_type = "Short"

            if enter_trade and risk_per_unit > 0:
                # Calculate position size based on risk fraction
                equity_to_risk = equity * risk_fraction
                position_size = equity_to_risk / risk_per_unit
                # Ensure position size is not greater than available equity allows (simple check)
                position_size = min(position_size, equity / entry_price)

                if position_size > 0:
                    position = 1 if trade_type == "Long" else -1
                    entry_cost = entry_price * position_size
                    commission = entry_cost * commission_rate # Commission on entry
                    equity -= commission # Deduct entry commission immediately

                    trade_info = {
                        'Type': trade_type,
                        'Entry': entry_price,
                        'Stop': stop_loss,
                        'Target': take_profit,
                        'Entry_idx': index,
                        'Size': position_size,
                        'Entry Cost': entry_cost,
                        'Commission': commission, # Store entry commission
                        'Exit': None,
                        'Exit_idx': None,
                        'PnL': None,
                        'Exit Reason': None
                    }
                    trade_log.append(trade_info)
                    current_trade_index = len(trade_log) - 1
                    logger.info(f"{trade_type} Entry at ~{entry_price:.2f} (Index: {index}). Size: {position_size:.4f}, SL: {stop_loss:.2f}, TP: {take_profit:.2f}, Entry Comm: {commission:.2f}")
                else:
                    logger.warning(f"Calculated position size is zero or negative at index {index}. Cannot enter trade.")
                    position = 0 # Ensure stays flat

        # Append current equity to the curve
        # Ensure equity doesn't go below zero (or handle bankruptcy)
        equity = max(equity, 0)
        equity_curve.append(float(equity))

    # If position is still open at the end, mark it based on the last close price
    if position != 0 and current_trade_index >= 0 and equity > 0:
        last_close = df['close'].iloc[-1]
        exit_price = last_close # Assume exit at last close, apply slippage
        exit_price += slippage_points if position == -1 else -slippage_points

        if position == 1:
            pnl = (exit_price - entry_price) * position_size
        else: # position == -1
            pnl = (entry_price - exit_price) * position_size

        commission = abs(pnl) * commission_rate # Commission on exit value
        net_pnl = pnl - commission
        equity += net_pnl # Update final equity
        equity_curve[-1] = float(max(equity, 0)) # Ensure last point is float

        trade_log[current_trade_index].update({
            'Exit': exit_price,
            'Exit_idx': df.index[-1],
            'PnL': net_pnl,
            'Commission': trade_log[current_trade_index].get('Commission', 0) + commission, # Sum entry and exit commissions
            'Exit Reason': 'EndOfData'
        })
        logger.info(f"Closing open {trade_log[current_trade_index]['Type']} position at end of data (~{exit_price:.2f}). PnL: {net_pnl:.2f}, Final Equity: {equity:.2f}")

    logger.info(f"Backtest finished. Final Equity: {equity_curve[-1]:.2f}. Total Trades Logged: {len(trade_log)}")
    return [float(e) for e in equity_curve], trade_log
