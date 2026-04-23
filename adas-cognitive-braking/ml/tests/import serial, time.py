import serial
import threading
import time
import pandas as pd
import sys

# --- CONFIGURATION ---
PORT = "COM4"
BAUD = 115200
# IMPORTANT: Update this path to your CSV file location
CSV_PATH = r"C:\Users\madir\OneDrive\Desktop\GRAD PROJECT\Open ACC Dataset\openacc_ttc_3000_smoothed_input live.csv"
MAX_ROWS_TO_SEND = 100 # <--- Specify the number of values (rows) to store in flash
SLEEP_DELAY = 0.05       # Delay between sending rows (in seconds)
TERMINATION_CMD = "END_DATA_STREAM\n"

# --- Open serial port ---
try:
    # Set a small timeout for readline operations
    ser = serial.Serial(PORT, BAUD, timeout=1) 
except serial.SerialException as e:
    print(f"❌ Error opening serial port {PORT}: {e}")
    sys.exit(1) # Exit the program if the port cannot be opened

# Try to prevent ESP32 reset and wait for initialization
ser.setDTR(False)
ser.setRTS(False)
print("⏳ Waiting for ESP32 initialization...")
time.sleep(2) 
ser.reset_input_buffer()

# --- Thread 1: Reader (listens continuously) ---
def read_from_esp(ser):
    """Continuously listens for incoming data from the serial port."""
    while True:
        try:
            if ser.in_waiting > 0:
                # Read line and decode/strip whitespace
                line = ser.readline().decode(errors="ignore").strip()
                if line:
                    # Print the acknowledgment messages from the ESP32
                    print(f"💬 ESP32 Acknowledgment: {line}")
        except serial.SerialException:
            # Break the loop if the port is closed externally or disconnected
            break
        except Exception as e:
            # Catch other potential errors
            # print(f"⚠️ Reader error: {e}") 
            break

# --- Thread 2: Writer (streams dataset) ---
def send_dataset(ser, csv_path):
    """Reads CSV, limits the rows, and streams data to the serial port."""
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"❌ File not found: {csv_path}")
        return

    # Limit the DataFrame to MAX_ROWS_TO_SEND
    df_to_send = df.head(MAX_ROWS_TO_SEND)
    num_rows = len(df_to_send)

    print(f"\n📦 Starting transfer of {num_rows} rows (specified by MAX_ROWS_TO_SEND)...")
    
    # Iterate over the limited DataFrame
    for index, row in df_to_send.iterrows():
        # Assuming the first 3 columns contain the data points
        # Ensure the row ends with a newline character (\n)
        msg = f"{row.iloc[0]},{row.iloc[1]},{row.iloc[2]}\n"
        
        try:
            ser.write(msg.encode('utf-8'))
            
            # Print update periodically
            if (index % 10 == 0) or (index == num_rows - 1): 
                 print(f"📤 Sent ({index+1}/{num_rows}): {msg.strip()}")
            
            time.sleep(SLEEP_DELAY) # Delay to allow ESP32 time to process and write to flash

        except serial.SerialException:
            print("❌ Serial connection lost during write.")
            break
    
    # Send the final termination command required by the ESP32 sketch
    ser.write(TERMINATION_CMD.encode('utf-8'))
    print(f"\n✅ Finished sending dataset and sent termination command: {TERMINATION_CMD.strip()}")

# --- Launch threads ---

# 1. Start Reader Thread (daemon=True means it won't block program exit)
reader = threading.Thread(target=read_from_esp, args=(ser,), daemon=True)
reader.start()

# 2. Start Sender Thread
sender = threading.Thread(target=send_dataset, args=(ser, CSV_PATH))
sender.start()

# 3. Wait for the Sender to finish its job
print("\n--- Main thread waiting for data stream to complete ---")
sender.join()

# 4. Give the Reader a moment to process the final acknowledgment from ESP32
# The ESP32 will then enter the dump routine after this wait.
time.sleep(1) 

# --- Clean up ---
ser.close()
print("\n🔚 Port closed. NOW open the Arduino Serial Monitor immediately to view stored data.")