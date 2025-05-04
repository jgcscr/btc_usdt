"""
backtest.py
Entry point script to run a backtest using the btc_usdt_pipeline package.
Run as: python -m scripts.backtest
"""
# Removed sys.path manipulation
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt # For plotting results

from btc_usdt_pipeline import config
from btc_usdt_pipeline.trading.backtest import run_backtest # Import the core function
from btc_usdt_pipeline.utils.helpers import setup_logger, calculate_metrics, print_trade_summary, plot_equity_curve, save_json

# Use a specific logger for this script
logger = setup_logger('backtest_script.log')

def main():
    logger.info("--- Running Backtest Script ---")
    try:
        # Load enriched data
        logger.info(f"Loading enriched data from: {config.ENRICHED_DATA_PATH}")
        # Add note about memory usage
        logger.info("Note: For very large datasets on memory-constrained environments (like Colab), consider loading data in chunks or using memory-efficient dtypes.")
        df = pd.read_parquet(config.ENRICHED_DATA_PATH)
        # Cast numeric columns to memory-efficient types
        for col in df.select_dtypes(include=['float64', 'float']).columns:
            df[col] = df[col].astype('float32')
        for col in df.select_dtypes(include=['int64', 'int']).columns:
            df[col] = df[col].astype('int32')
        if 'open_time' in df.columns:
            df['open_time'] = pd.to_datetime(df['open_time'])
            df = df.set_index('open_time')
        logger.info(f"Loaded enriched data: {df.shape}")

        # Load generated signals
        signals_path = config.RESULTS_DIR / 'generated_signals.csv'
        logger.info(f"Loading signals from: {signals_path}")
        if not signals_path.exists():
            logger.error(f"Signals file not found at {signals_path}. Run signal generation first.")
            print(f"Error: Signals file not found at {signals_path}. Run signal generation first.")
            return
        signals_df = pd.read_csv(signals_path, index_col='open_time', parse_dates=True)
        logger.info(f"Loaded {len(signals_df)} signals.")

        # Align data and signals
        common_index = df.index.intersection(signals_df.index)
        if common_index.empty:
            logger.error("No common index found between enriched data and signals. Cannot align for backtest.")
            print("Error: Cannot align data and signals.")
            return

        df_aligned = df.loc[common_index]
        signals_aligned = signals_df.loc[common_index, 'signal'].values

        logger.info(f"Data and signals aligned. Using {len(df_aligned)} rows for backtesting.")

        if df_aligned.empty:
            logger.error("Aligned data subset for backtesting is empty.")
            print("Error: Aligned data subset for backtesting is empty.")
            return

        # Check if necessary ATR column exists
        if config.BACKTEST_ATR_COLUMN not in df_aligned.columns:
             logger.error(f"Required ATR column '{config.BACKTEST_ATR_COLUMN}' not found in data. Cannot run backtest.")
             print(f"Error: Required ATR column '{config.BACKTEST_ATR_COLUMN}' not found.")
             return

        # Run Backtest (using the enhanced backtester from the package)
        logger.info("Running backtest engine...")
        equity_curve, trade_log = run_backtest(
            df=df_aligned,
            signals=signals_aligned,
            initial_equity=config.INITIAL_EQUITY,
            atr_col=config.BACKTEST_ATR_COLUMN,
            sl_multiplier=config.ATR_STOP_LOSS_MULTIPLIER,
            tp_multiplier=config.ATR_TAKE_PROFIT_MULTIPLIER,
            commission_rate=config.COMMISSION_RATE,
            slippage_points=config.SLIPPAGE_POINTS,
            risk_fraction=config.RISK_FRACTION
        )

        # Calculate and Print Metrics
        if equity_curve and trade_log:
            metrics = calculate_metrics(equity_curve, trade_log, initial_equity=config.INITIAL_EQUITY)
            logger.info(f"Backtest Metrics: {metrics}")
            print("\n--- Backtest Results ---")
            print(f"Initial Equity: ${config.INITIAL_EQUITY:,.2f}")
            print(f"Final Equity:   ${metrics['Final Equity']:,.2f}")
            print(f"Net Profit:     ${metrics['Net Profit']:,.2f} ({metrics['Net Profit %']:.2f}%)")
            print(f"Total Trades:   {metrics['Total Trades']}")
            print(f"Winning Trades: {metrics['Winning Trades']} ({metrics['Win Rate %']:.2f}%)")
            print(f"Losing Trades:  {metrics['Losing Trades']}")
            print(f"Max Drawdown:   {metrics['Max Drawdown %']:.2f}%")
            print(f"Sharpe Ratio:   {metrics['Sharpe Ratio']:.2f}") # Assuming daily Sharpe for now
            print(f"Avg Trade PnL:  ${metrics['Avg Trade PnL']:,.2f}")
            print(f"Profit Factor:  {metrics['Profit Factor']:.2f}")
            print("------------------------")

            # Print trade summary
            print_trade_summary(trade_log)

            # Save results (Directory creation handled by plot_equity_curve and save_json in helpers)
            results_path = config.RESULTS_DIR / 'backtest_results.json'
            results_data = {
                'metrics': metrics,
                'trade_log': trade_log # Consider converting datetime index in log if needed
            }
            # Convert Timestamps in trade_log to strings for JSON serialization
            for trade in results_data['trade_log']:
                for key in ['Entry_idx', 'Exit_idx']:
                    if key in trade and isinstance(trade[key], pd.Timestamp):
                        trade[key] = trade[key].isoformat()

            # Use save_json helper which handles directory creation
            save_json(results_data, results_path)
            logger.info(f"Backtest results saved to {results_path}")
            print(f"Backtest results saved to {results_path}")

            # Plot equity curve (plot_equity_curve helper handles directory creation)
            equity_curve_path = config.RESULTS_DIR / 'equity_curve.png'
            plot_equity_curve(equity_curve, df_aligned.index, save_path=equity_curve_path) # Pass aligned index
            logger.info(f"Equity curve plot saved to {equity_curve_path}")
            print(f"Equity curve plot saved to {equity_curve_path}")

        else:
            logger.warning("Backtest completed but generated no equity curve or trade log.")
            print("Backtest completed but generated no equity curve or trade log.")

    except FileNotFoundError as e:
        logger.error(f"Required file not found: {e}. Cannot run backtest.")
        print(f"Error: Required file not found ({e}). Cannot run backtest.")
    except Exception as e:
        logger.error(f"An error occurred during backtest script: {e}", exc_info=True)
        print(f"An error occurred during backtest script: {e}")

    logger.info("--- Backtest Script Finished ---")

if __name__ == '__main__':
    main()
