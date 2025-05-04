"""
optimize_indicators.py
Entry point script to run indicator optimization using the btc_usdt_pipeline package.
Run as: python -m scripts.optimize_indicators
"""
# Removed sys.path manipulation

from btc_usdt_pipeline.optimize.optimize_indicators import main as optimize_indicators_main
from btc_usdt_pipeline.utils.helpers import setup_logger

logger = setup_logger('optimize_indicators_script.log')

if __name__ == '__main__':
    logger.info("--- Running Indicator Optimization Script ---")
    print("--- Running Indicator Optimization Script ---")
    try:
        optimize_indicators_main()
    except Exception as e:
        logger.error(f"Error during indicator optimization: {e}", exc_info=True)
        print(f"Error during indicator optimization: {e}")
    logger.info("--- Indicator Optimization Script Finished ---")
    print("--- Indicator Optimization Script Finished ---")
