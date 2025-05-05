import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional
from datetime import datetime
import matplotlib.pyplot as plt
import json

# --- Performance Metrics ---
def calculate_performance_metrics(equity_curve: List[float], trade_log: List[Dict[str, Any]], risk_free_rate: float = 0.0) -> Dict[str, Any]:
    equity = np.array(equity_curve)
    returns = np.diff(equity) / equity[:-1]
    mean_return = np.mean(returns)
    std_return = np.std(returns)
    downside_std = np.std(returns[returns < 0]) if np.any(returns < 0) else 0
    cagr = (equity[-1] / equity[0]) ** (252 / len(equity)) - 1 if len(equity) > 1 else 0
    sharpe = (mean_return - risk_free_rate/252) / std_return * np.sqrt(252) if std_return > 0 else np.nan
    sortino = (mean_return - risk_free_rate/252) / downside_std * np.sqrt(252) if downside_std > 0 else np.nan
    max_drawdown = np.min((equity - np.maximum.accumulate(equity)) / np.maximum.accumulate(equity))
    calmar = cagr / abs(max_drawdown) if max_drawdown != 0 else np.nan
    return {
        'sharpe': sharpe,
        'sortino': sortino,
        'calmar': calmar,
        'cagr': cagr,
        'max_drawdown': max_drawdown,
        'total_return': (equity[-1] / equity[0] - 1) if len(equity) > 1 else 0,
        'trades': len(trade_log)
    }

def generate_returns_table(equity_curve: List[float], freq: List[str] = ['M', 'Y'], index: Optional[pd.Index] = None) -> Dict[str, pd.DataFrame]:
    if index is None:
        index = pd.date_range(datetime.today(), periods=len(equity_curve), freq='D')
    equity = pd.Series(equity_curve, index=index)
    returns = equity.pct_change().fillna(0)
    tables = {}
    if 'M' in freq:
        monthly = returns.resample('M').apply(lambda x: (x+1).prod()-1)
        tables['monthly'] = monthly
    if 'Y' in freq:
        yearly = returns.resample('Y').apply(lambda x: (x+1).prod()-1)
        tables['yearly'] = yearly
    return tables

def generate_drawdown_table(equity_curve: List[float], top_n: int = 5, index: Optional[pd.Index] = None) -> pd.DataFrame:
    if index is None:
        index = pd.date_range(datetime.today(), periods=len(equity_curve), freq='D')
    equity = pd.Series(equity_curve, index=index)
    running_max = equity.cummax()
    drawdown = (equity - running_max) / running_max
    dd_periods = drawdown[drawdown < 0]
    dd_table = dd_periods.sort_values().head(top_n)
    return dd_table

def compare_strategies(strategy_results: Dict[str, Dict[str, Any]], metrics: List[str] = ['sharpe', 'sortino', 'cagr']) -> pd.DataFrame:
    rows = []
    for name, res in strategy_results.items():
        row = {'strategy': name}
        for m in metrics:
            row[m] = res.get(m, np.nan)
        rows.append(row)
    return pd.DataFrame(rows)

# --- Report Generation ---
def to_html_report(results: Dict[str, Any], title: str, output_path: str):
    html = f"<html><head><title>{title}</title></head><body>"
    html += f"<h1>{title}</h1>"
    html += "<h2>Performance Metrics</h2><ul>"
    for k, v in results['metrics'].items():
        html += f"<li>{k}: {v:.4f}</li>"
    html += "</ul>"
    html += "<h2>Monthly Returns</h2>" + results['tables']['monthly'].to_frame('Return').to_html()
    html += "<h2>Yearly Returns</h2>" + results['tables']['yearly'].to_frame('Return').to_html()
    html += "<h2>Top Drawdowns</h2>" + results['drawdowns'].to_frame('Drawdown').to_html()
    html += "</body></html>"
    with open(output_path, 'w') as f:
        f.write(html)

def to_pdf_report(results: Dict[str, Any], title: str, output_path: str):
    # Simple HTML to PDF using pdfkit (requires wkhtmltopdf)
    import pdfkit
    html_path = output_path.replace('.pdf', '.html')
    to_html_report(results, title, html_path)
    pdfkit.from_file(html_path, output_path)

def to_json_report(results: Dict[str, Any], output_path: str):
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)

# --- Dashboard Templates ---
def dashboard_template(results: Dict[str, Any], title: str = "Performance Dashboard") -> str:
    # Returns HTML string for embedding or serving
    html = f"<html><head><title>{title}</title></head><body>"
    html += f"<h1>{title}</h1>"
    html += "<h2>Performance Metrics</h2><ul>"
    for k, v in results['metrics'].items():
        html += f"<li>{k}: {v:.4f}</li>"
    html += "</ul>"
    html += "<h2>Monthly Returns</h2>" + results['tables']['monthly'].to_frame('Return').to_html()
    html += "<h2>Yearly Returns</h2>" + results['tables']['yearly'].to_frame('Return').to_html()
    html += "<h2>Top Drawdowns</h2>" + results['drawdowns'].to_frame('Drawdown').to_html()
    html += "</body></html>"
    return html
