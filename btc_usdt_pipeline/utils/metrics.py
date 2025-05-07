import numpy as np
from typing import List, Dict, Any, Optional


def calculate_performance_metrics(
    equity_curve: List[float],
    trade_log: List[Dict[str, Any]],
    risk_free_rate: float = 0.0
) -> Dict[str, Any]:
    """
    Calculate comprehensive performance metrics for a trading strategy.

    Args:
        equity_curve (List[float]): List of equity values over time.
        trade_log (List[Dict[str, Any]]): List of trade dictionaries with PnL and other trade info.
        risk_free_rate (float, optional): Annual risk-free rate for Sharpe/Sortino. Default is 0.0.

    Returns:
        Dict[str, Any]: Dictionary containing:
            - sharpe: Sharpe ratio (annualized)
            - sortino: Sortino ratio (annualized)
            - calmar: Calmar ratio
            - cagr: Compound annual growth rate
            - max_drawdown: Maximum drawdown (as a negative float)
            - total_return: Total return over the period
            - trades: Number of trades
            - win_ratio: Proportion of winning trades
            - profit_factor: Ratio of gross profit to gross loss
    """
    equity = np.array(equity_curve)
    returns = np.diff(equity) / equity[:-1] if len(equity) > 1 else np.array([])
    mean_return = np.mean(returns) if len(returns) > 0 else 0
    std_return = np.std(returns) if len(returns) > 0 else 0
    downside_std = np.std(returns[returns < 0]) if np.any(returns < 0) else 0
    periods_per_year = 252  # Assume daily bars; adjust if needed
    cagr = (equity[-1] / equity[0]) ** (periods_per_year / len(equity)) - 1 if len(equity) > 1 else 0
    sharpe = ((mean_return - risk_free_rate / periods_per_year) / std_return * np.sqrt(periods_per_year)) if std_return > 0 else np.nan
    sortino = ((mean_return - risk_free_rate / periods_per_year) / downside_std * np.sqrt(periods_per_year)) if downside_std > 0 else np.nan
    running_max = np.maximum.accumulate(equity)
    drawdown = (equity - running_max) / running_max
    max_drawdown = drawdown.min() if len(drawdown) > 0 else 0
    calmar = cagr / abs(max_drawdown) if max_drawdown != 0 else np.nan
    total_return = (equity[-1] / equity[0] - 1) if len(equity) > 1 else 0
    # Trade stats
    wins = [t for t in trade_log if t.get('PnL') is not None and t['PnL'] > 0]
    losses = [t for t in trade_log if t.get('PnL') is not None and t['PnL'] <= 0]
    win_ratio = len(wins) / len(trade_log) if trade_log else 0
    profit_factor = sum(t['PnL'] for t in wins) / abs(sum(t['PnL'] for t in losses)) if losses else float('inf')
    return {
        'sharpe': sharpe,
        'sortino': sortino,
        'calmar': calmar,
        'cagr': cagr,
        'max_drawdown': max_drawdown,
        'total_return': total_return,
        'trades': len(trade_log),
        'win_ratio': win_ratio,
        'profit_factor': profit_factor
    }
