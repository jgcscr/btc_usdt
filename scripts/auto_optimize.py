"""
auto_optimize.py
Entry point script to run auto optimization using the btc_usdt_pipeline package.
Run as: python -m scripts.auto_optimize
"""
# Removed sys.path manipulation

from btc_usdt_pipeline.optimize.auto_optimize import main as auto_optimize_main
from btc_usdt_pipeline.utils.helpers import setup_logger

logger = setup_logger('auto_optimize_script.log')

if __name__ == '__main__':
    logger.info("--- Running Auto Optimization Script ---")
    print("--- Running Auto Optimization Script ---")
    try:
        auto_optimize_main()
    except Exception as e:
        logger.error(f"Error during auto optimization: {e}", exc_info=True)
        print(f"Error during auto optimization: {e}")
    logger.info("--- Auto Optimization Script Finished ---")
    print("--- Auto Optimization Script Finished ---")
