import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objs as go
import plotly.express as px
from typing import List, Dict, Any, Optional
from btc_usdt_pipeline.utils.metrics import calculate_performance_metrics
from btc_usdt_pipeline.visualization.backtest_viz import *

def visualize_equity_curve(equity_curve: List[float], title: str = "Equity Curve", index: Optional[pd.Index] = None, use_plotly: bool = False):
    """
    Plot the equity curve using Matplotlib or Plotly.
    Args:
        equity_curve: List of equity values.
        title: Chart title.
        index: Optional index (e.g., pd.DatetimeIndex) for x-axis.
        use_plotly: If True, use Plotly; else use Matplotlib.
    """
    if use_plotly:
        x = index if index is not None else list(range(len(equity_curve)))
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x, y=equity_curve, mode='lines', name='Equity'))
        fig.update_layout(title=title, xaxis_title='Time', yaxis_title='Equity')
        fig.show()
    else:
        plt.figure(figsize=(12, 6))
        x = index if index is not None else list(range(len(equity_curve)))
        plt.plot(x, equity_curve, label='Equity')
        plt.title(title)
        plt.xlabel('Time')
        plt.ylabel('Equity')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

def visualize_drawdown(equity_curve: List[float], title: str = "Drawdown Analysis", use_plotly: bool = False):
    """
    Plot drawdown curve from equity curve.
    Args:
        equity_curve: List of equity values.
        title: Chart title.
        use_plotly: If True, use Plotly; else use Matplotlib.
    """
    equity = np.array(equity_curve)
    running_max = np.maximum.accumulate(equity)
    drawdown = (equity - running_max) / running_max
    if use_plotly:
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=drawdown, mode='lines', name='Drawdown'))
        fig.update_layout(title=title, xaxis_title='Time', yaxis_title='Drawdown (%)')
        fig.show()
    else:
        plt.figure(figsize=(12, 4))
        plt.plot(drawdown, label='Drawdown')
        plt.title(title)
        plt.xlabel('Time')
        plt.ylabel('Drawdown (%)')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

def visualize_trades(df: pd.DataFrame, trade_log: List[Dict[str, Any]], title: str = "Trade Analysis", use_plotly: bool = False):
    """
    Plot price chart with trade entry/exit points.
    Args:
        df: DataFrame with price data (must include 'close').
        trade_log: List of trade dicts from backtest.
        title: Chart title.
        use_plotly: If True, use Plotly; else use Matplotlib.
    """
    if use_plotly:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df.index, y=df['close'], mode='lines', name='Close'))
        for trade in trade_log:
            entry_idx = trade['Entry_idx']
            exit_idx = trade['Exit_idx']
            entry_price = trade['Entry']
            exit_price = trade['Exit']
            color = 'green' if trade['PnL'] and trade['PnL'] > 0 else 'red'
            fig.add_trace(go.Scatter(x=[entry_idx], y=[entry_price], mode='markers', marker=dict(color=color, symbol='triangle-up'), name='Entry'))
            fig.add_trace(go.Scatter(x=[exit_idx], y=[exit_price], mode='markers', marker=dict(color=color, symbol='x'), name='Exit'))
        fig.update_layout(title=title, xaxis_title='Time', yaxis_title='Price')
        fig.show()
    else:
        plt.figure(figsize=(14, 6))
        plt.plot(df.index, df['close'], label='Close', color='blue')
        for trade in trade_log:
            entry_idx = trade['Entry_idx']
            exit_idx = trade['Exit_idx']
            entry_price = trade['Entry']
            exit_price = trade['Exit']
            color = 'green' if trade['PnL'] and trade['PnL'] > 0 else 'red'
            plt.scatter(entry_idx, entry_price, marker='^', color=color, label='Entry')
            plt.scatter(exit_idx, exit_price, marker='x', color=color, label='Exit')
        plt.title(title)
        plt.xlabel('Time')
        plt.ylabel('Price')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

def visualize_return_heatmap(df: pd.DataFrame, freq: str = 'M', use_plotly: bool = False):
    """
    Plot a heatmap of monthly or yearly returns.
    Args:
        df: DataFrame with 'close' price and datetime index.
        freq: 'M' for monthly, 'Y' for yearly.
        use_plotly: If True, use Plotly; else use Matplotlib.
    """
    returns = df['close'].pct_change().dropna()
    if freq == 'M':
        period = returns.index.to_period('M')
    elif freq == 'Y':
        period = returns.index.to_period('Y')
    else:
        raise ValueError("freq must be 'M' or 'Y'")
    returns_df = pd.DataFrame({'return': returns, 'period': period})
    heatmap = returns_df.pivot_table(index=returns_df.index.year, columns=returns_df.index.month if freq=='M' else returns_df.index.year, values='return', aggfunc='sum')
    if use_plotly:
        fig = px.imshow(heatmap, labels=dict(x="Month", y="Year", color="Return"), title="Return Heatmap")
        fig.show()
    else:
        plt.figure(figsize=(10, 6))
        plt.title("Return Heatmap")
        plt.xlabel("Month" if freq=='M' else "Year")
        plt.ylabel("Year")
        plt.imshow(heatmap, aspect='auto', cmap='RdYlGn', interpolation='nearest')
        plt.colorbar(label='Return')
        plt.tight_layout()
        plt.show()

def export_html_report(equity_curve: List[float], trade_log: List[Dict[str, Any]], df: pd.DataFrame, filename: str = "backtest_report.html"):
    """
    Export a simple HTML report with equity curve, drawdown, and trade stats using Plotly.
    """
    import plotly.offline as pyo
    metrics = calculate_performance_metrics(equity_curve, trade_log)
    # Equity curve
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=df.index, y=equity_curve, mode='lines', name='Equity'))
    fig1.update_layout(title="Equity Curve")
    # Drawdown
    equity = np.array(equity_curve)
    running_max = np.maximum.accumulate(equity)
    drawdown = (equity - running_max) / running_max
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=df.index, y=drawdown, mode='lines', name='Drawdown'))
    fig2.update_layout(title="Drawdown")
    # Trade stats
    stats_html = f"""
    <h2>Performance Metrics</h2>
    <ul>
        <li>Total Trades: {metrics['trades']}</li>
        <li>Win Ratio: {metrics['win_ratio']:.2%}</li>
        <li>Profit Factor: {metrics['profit_factor']:.2f}</li>
        <li>Max Drawdown: {metrics['max_drawdown']:.2%}</li>
        <li>Total Return: {metrics['total_return']:.2%}</li>
        <li>Sharpe Ratio: {metrics['sharpe']:.2f}</li>
        <li>Sortino Ratio: {metrics['sortino']:.2f}</li>
        <li>Calmar Ratio: {metrics['calmar']:.2f}</li>
        <li>CAGR: {metrics['cagr']:.2%}</li>
    </ul>
    """
    with open(filename, "w") as f:
        f.write("<html><head><title>Backtest Report</title></head><body>")
        f.write("<h1>Backtest Report</h1>")
        f.write(stats_html)
        f.write(pyo.plot(fig1, include_plotlyjs='cdn', output_type='div'))
        f.write(pyo.plot(fig2, include_plotlyjs='cdn', output_type='div'))
        f.write("</body></html>")
