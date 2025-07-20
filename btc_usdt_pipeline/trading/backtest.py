# trading/backtest.py
"""
Contains the backtesting engine logic.
Refactored from scripts/backtest.py to use centralized config and helpers.
"""
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple, Optional
import math

# Use absolute imports from the package
from btc_usdt_pipeline.utils.config_manager import config_manager
from btc_usdt_pipeline.utils.logging_config import setup_logging
from btc_usdt_pipeline.exceptions import ParameterValidationError, DataAlignmentError
from btc_usdt_pipeline.types import TradeLogType, MetricsDict
# Note: generate_signals is called *before* the backtest function usually.
# The backtest function receives the signals as input.

logger = setup_logging(log_filename='backtest.log')

def validate_inputs(df: pd.DataFrame, signals: Any, required_cols: Optional[List[str]] = None, logger: Optional[Any] = None) -> pd.DataFrame:
    """
    Validates DataFrame and signals for backtesting:
    - Checks required columns
    - No NaNs in critical columns
    - Signal array and DataFrame match in length
    - Index is properly ordered
    Raises ParameterValidationError or DataAlignmentError on failure.
    """
    logger = logger or setup_logging('backtest.log')
    required_cols = required_cols or ['open', 'high', 'low', 'close']
    for col in required_cols:
        if col not in df.columns:
            logger.error(f"Required column '{col}' missing from DataFrame.")
            raise ParameterValidationError(f"Required column '{col}' missing from DataFrame.")
        if df[col].isnull().any():
            logger.error(f"NaN values found in required column '{col}'.")
            raise ParameterValidationError(f"NaN values found in required column '{col}'.")
        if col in ['close', 'high', 'low'] and (df[col] <= 0).any():
            logger.error(f"Non-positive values found in '{col}'.")
            raise ParameterValidationError(f"Non-positive values found in '{col}'.")

    if len(df) != len(signals):
        logger.error(f"DataFrame length ({len(df)}) and signals length ({len(signals)}) mismatch.")
        raise DataAlignmentError(f"DataFrame length ({len(df)}) and signals length ({len(signals)}) mismatch.")
    if isinstance(df.index, (pd.DatetimeIndex, pd.RangeIndex, pd.Index)):
        if not df.index.is_monotonic_increasing:
            logger.warning("DataFrame index is not monotonic increasing. Sorting.")
            df = df.sort_index()
    return df

def run_backtest(
    df: pd.DataFrame,
    signals: np.ndarray,
    initial_equity: float = None,
    atr_col: str = None,
    sl_multiplier: float = None,
    tp_multiplier: float = None,
    commission_rate: float = None,
    slippage_points: float = None,
    risk_fraction: float = None
) -> Tuple[List[float], TradeLogType]:
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
        Tuple[List[float], TradeLogType]:
            - equity_curve: List of equity values over time.
            - trade_log: List of dictionaries detailing each executed trade.
    """
    # --- Input validation and alignment ---
    from btc_usdt_pipeline.utils.data_processing import align_and_validate_data
    required_cols = ['open', 'high', 'low', 'close']
    if atr_col is not None:
        required_cols.append(atr_col)
    df = validate_inputs(df, signals, required_cols=required_cols, logger=logger)
    df, signals = align_and_validate_data(df, signals, arr_name="signals", logger=logger)

    # --- Parameter validation ---
    initial_equity = initial_equity if initial_equity is not None else config_manager.get('backtest.initial_equity')
    atr_col = atr_col if atr_col is not None else config_manager.get('backtest.backtest_atr_column')
    sl_multiplier = sl_multiplier if sl_multiplier is not None else config_manager.get('backtest.atr_stop_loss_multiplier')
    tp_multiplier = tp_multiplier if tp_multiplier is not None else config_manager.get('backtest.atr_take_profit_multiplier')
    commission_rate = commission_rate if commission_rate is not None else config_manager.get('backtest.commission_rate')
    slippage_points = slippage_points if slippage_points is not None else config_manager.get('backtest.slippage_points')
    risk_fraction = risk_fraction if risk_fraction is not None else config_manager.get('backtest.risk_fraction')

    if sl_multiplier <= 0 or tp_multiplier <= 0:
        logger.error("sl_multiplier and tp_multiplier must be positive.")
        raise ParameterValidationError("sl_multiplier and tp_multiplier must be positive.")
    if not (0 < risk_fraction <= 1):
        logger.error("risk_fraction must be between 0 and 1.")
        raise ParameterValidationError("risk_fraction must be between 0 and 1.")
    if commission_rate < 0:
        logger.error("commission_rate must be non-negative.")
        raise ParameterValidationError("commission_rate must be non-negative.")
    if len(df) != len(signals):
        logger.error(f"DataFrame length ({len(df)}) and signals length ({len(signals)}) mismatch. Cannot run backtest.")
        raise DataAlignmentError(f"DataFrame length ({len(df)}) and signals length ({len(signals)}) mismatch.")

    if atr_col not in df.columns:
        logger.error(f"ATR column '{atr_col}' not found in DataFrame. Cannot run backtest.")
        return [initial_equity], []

    logger.info(f"Starting backtest with Initial Equity: ${initial_equity:,.2f}, Risk/Trade: {risk_fraction:.2%}, SL: {sl_multiplier}*ATR, TP: {tp_multiplier}*ATR, Comm: {commission_rate:.4%}, Slippage: {slippage_points} pts")

    # --- Slippage sanity check: warn if slippage is excessive ---
    avg_price = df['close'].mean() if not df.empty else 0
    if avg_price > 0 and slippage_points / avg_price > 0.1:
        logger.warning(f"Slippage ({slippage_points}) is more than 10% of average price ({avg_price:.2f}). Results may be unrealistic.")

    equity = float(initial_equity)
    equity_curve = [equity]
    position = 0 # 0: Flat, 1: Long, -1: Short
    position_size = 0.0 # Size of the current position in base asset (e.g., BTC)
    entry_price = 0.0
    trade_log: TradeLogType = []
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
        log_reason = None  # Initialize before use

        # --- Slippage Model ---
        # For LONG positions:
        #   ENTRY: Add slippage to entry price (pay more when buying)
        #   EXIT (TP/SL): Subtract slippage from exit price (receive less when selling)
        # For SHORT positions:
        #   ENTRY: Subtract slippage from entry price (receive less when selling)
        #   EXIT (TP/SL): Add slippage to exit price (pay more when buying back)


        if position == 1: # Currently Long
            # Check Take Profit first
            if take_profit is not None and high is not None and high >= take_profit:
                exit_price = take_profit - slippage_points if take_profit is not None and slippage_points is not None else None # LONG exit: receive less
                trade_closed = True
                log_reason = "TP"
            # Check Stop Loss
            elif stop_loss is not None and low is not None and low <= stop_loss:
                exit_price = stop_loss - slippage_points if stop_loss is not None and slippage_points is not None else None # LONG exit: receive less
                trade_closed = True
                log_reason = "SL"

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
            # Check Take Profit first
            if take_profit is not None and low is not None and low <= take_profit:
                exit_price = take_profit + slippage_points if take_profit is not None and slippage_points is not None else None # SHORT exit: pay more
                trade_closed = True
                log_reason = "TP"
            # Check Stop Loss
            elif stop_loss is not None and high is not None and high >= stop_loss:
                exit_price = stop_loss + slippage_points if stop_loss is not None and slippage_points is not None else None # SHORT exit: pay more
                trade_closed = True
                log_reason = "SL"

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
                entry_price = close + slippage_points # LONG entry: pay more
                stop_loss = entry_price - sl_distance_pts
                take_profit = entry_price + tp_distance_pts
                risk_per_unit = entry_price - stop_loss
                trade_type = "Long"

            elif signal == "Short" and sl_distance_pts > 0:
                enter_trade = True
                entry_price = close - slippage_points # SHORT entry: receive less
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
        if last_close is not None:
            if position == 1:
                exit_price = last_close - slippage_points # LONG exit: receive less
                pnl = (exit_price - entry_price) * position_size
            else: # position == -1
                exit_price = last_close + slippage_points # SHORT exit: pay more
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

def estimate_total_slippage_cost(trade_log: TradeLogType) -> float:
    """
    Estimate the total slippage cost across all trades in the trade log.
    Args:
        trade_log (TradeLogType): List of trade dictionaries from run_backtest.
    Returns:
        float: Total slippage cost (absolute sum of entry and exit slippage).
    """
    total_slippage = 0.0
    for trade in trade_log:
        if trade['Type'] == 'Long':
            entry_slip = trade['Entry'] - trade['Entry_idx'] if trade['Entry'] is not None and trade['Entry_idx'] is not None else 0
            exit_slip = trade['Exit'] - trade['Target'] if trade['Exit'] is not None and trade['Target'] is not None else 0
        elif trade['Type'] == 'Short':
            entry_slip = trade['Entry_idx'] - trade['Entry'] if trade['Entry'] is not None and trade['Entry_idx'] is not None else 0
            exit_slip = trade['Exit'] - trade['Target'] if trade['Exit'] is not None and trade['Target'] is not None else 0
        else:
            entry_slip = 0
            exit_slip = 0
        total_slippage += abs(entry_slip) + abs(exit_slip)
    return total_slippage
