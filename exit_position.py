import time
import schedule
from datetime import datetime
from dhanhq import dhanhq  # Import the dhanhq library

# --- Configuration ---
# !! IMPORTANT: Store your credentials securely. Avoid hardcoding directly.
# Consider environment variables or a config file.
CLIENT_ID = "1103956228"
ACCESS_TOKEN = "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzUxMiJ9.eyJpc3MiOiJkaGFuIiwicGFydG5lcklkIjoiIiwiZXhwIjoxNzQ1ODg3OTcwLCJ0b2tlbkNvbnN1bWVyVHlwZSI6IlNFTEYiLCJ3ZWJob29rVXJsIjoiIiwiZGhhbkNsaWVudElkIjoiMTEwMzk1NjIyOCJ9.jHHGztzKA2dD_2Fhgw4bUBLeR07QXXQc6F1efWu5yR5O2Q983SYASyFEcMgbT9bKVg45WNkDFfi3LfM_rs7eQg" # This is the long-term access token

# Time to exit positions (HH:MM in 24-hour format, IST)
EXIT_TIME = "15:18"

# --- DhanHQ Client Initialization ---
try:
    dhan = dhanhq(CLIENT_ID, ACCESS_TOKEN)
    print("DhanHQ client initialized successfully.")
    # Optional: Verify connection by fetching profile info or funds
    # profile_info = dhan.get_profile()
    # print("Profile Info:", profile_info)
except Exception as e:
    print(f"Error initializing DhanHQ client: {e}")
    # Exit if initialization fails
    exit()

# --- Function to Exit All Positions ---
def exit_all_positions():
    """
    Fetches all open positions and places market orders to close them.
    """
    print(f"--- Running Exit Positions Job at {datetime.now()} ---")
    try:
        # 1. Fetch Open Positions
        positions_response = dhan.get_positions()

        if positions_response.get('status') == 'failure':
            print(f"Error fetching positions: {positions_response.get('remarks', 'Unknown error')}")
            return

        positions_data = positions_response.get('data', [])

        if not positions_data:
            print("No open positions found.")
            print("--- Exit Job Finished ---")
            return

        print(f"Found {len(positions_data)} open positions. Attempting to close...")

        # 2. Iterate and Place Closing Orders
        for position in positions_data:
            try:
                security_id = str(position.get('securityId'))
                symbol = position.get('tradingSymbol', 'N/A')
                net_qty = int(position.get('netQty', 0))
                exchange = position.get('exchangeSegment')
                product_type = position.get('productType') # e.g., 'INTRADAY', 'CNC', 'MARGIN', 'MTF', 'BO', 'CO'

                # Skip if quantity is zero (already closed or no net position)
                if net_qty == 0:
                    print(f"Skipping {symbol} (Security ID: {security_id}) - Net quantity is zero.")
                    continue

                # Determine transaction type for closing order
                if net_qty > 0:  # Long position, need to SELL
                    transaction_type = dhan.SELL
                    closing_qty = net_qty
                    action = "Selling"
                elif net_qty < 0: # Short position, need to BUY
                    transaction_type = dhan.BUY
                    closing_qty = abs(net_qty) # Order quantity must be positive
                    action = "Buying to cover"
                else: # Should not happen if net_qty check above works, but for safety
                    continue

                print(f"{action} {closing_qty} of {symbol} (Security ID: {security_id}, Product: {product_type})")

                # Place Market Order to Close
                # Ensure product_type matches the one you want to close (e.g., dhan.INTRA)
                order_response = dhan.place_order(
                    security_id=security_id,
                    exchange_segment=exchange, # e.g., dhan.NSE_EQ, dhan.NSE_FNO, etc. Pass the string from position data.
                    transaction_type=transaction_type,
                    quantity=closing_qty,
                    order_type=dhan.MARKET, # Using Market order for immediate exit
                    product_type=product_type, # CRITICAL: Match position's product type
                    price=0 # Required for market order
                    # validity='DAY' # Default is DAY
                )

                # Log order placement result
                if order_response.get('status') == 'success' and order_response.get('data', {}).get('orderId'):
                    print(f"Successfully placed closing order for {symbol}. Order ID: {order_response['data']['orderId']}")
                else:
                    error_msg = order_response.get('remarks', {}).get('message', 'Unknown order placement error')
                    print(f"Error placing closing order for {symbol}: {error_msg}")
                    # Consider adding logic here to retry or handle partial failures

            except Exception as e:
                print(f"Error processing position {position.get('tradingSymbol', 'N/A')}: {e}")
                # Log this error and continue with the next position

        print("--- Finished processing all positions ---")

    except Exception as e:
        print(f"An unexpected error occurred in exit_all_positions: {e}")
    finally:
        print("--- Exit Job Finished ---")


# --- Scheduling ---
print(f"Scheduling job to run daily at {EXIT_TIME} IST.")
schedule.every().day.at(EXIT_TIME).do(exit_all_positions)

# --- Main Loop ---
if __name__ == "__main__":
    print("Scheduler started. Waiting for the scheduled time...")
    # Run once immediately if needed for testing (optional)
    # exit_all_positions()

    while True:
        schedule.run_pending()
        time.sleep(10) 
