import serial
import time
import random
import numpy as np

# ----------------------------
#  SERIAL CONFIGURATION
# ----------------------------
SERIAL_PORT = "COM4"      # ← change this to your ESP32 port (e.g. COM5 or /dev/ttyUSB0)
BAUD_RATE = 115200
DELAY_SEC = 0.75          # delay between each send (seconds)
NUM_SAMPLES = 3000
STREAM_COUNT = 500

# ----------------------------
#  SYNTHETIC DATA GENERATION
# ----------------------------
def generate_synthetic_data(num_samples=3000):
    """
    Generates synthetic Time-to-Collision (TTC),
    Relative Speed, and Ego Speed data similar to real driving conditions.
    """
    ttc_values = np.clip(np.random.normal(3.5, 1.2, num_samples), 0.5, 8.0)
    rel_speed = np.clip(np.random.normal(-1.5, 1.0, num_samples), -5.0, 3.0)
    ego_speed = np.clip(np.random.normal(18.0, 5.0, num_samples), 0, 35.0)

    return list(zip(ttc_values, rel_speed, ego_speed))

# ----------------------------
#  SERIAL CONNECTION HANDLER
# ----------------------------
def connect_esp32():
    print("🔌 [STEP 1] Connecting to ESP32...")
    try:
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=2)
        time.sleep(2)
        print(f"✅ [OK] Connected to ESP32 on {SERIAL_PORT} at {BAUD_RATE} baud.")
        return ser
    except Exception as e:
        print(f"❌ [ERROR] Could not open serial port: {e}")
        exit()

# ----------------------------
#  STREAM SYNTHETIC DATA
# ----------------------------
def stream_data(ser, data, num_streams=500):
    print("\n🚀 [STEP 2] Starting data streaming...")
    print("----------------------------------------------------------")
    print(f"📊 Streaming {num_streams} samples every {DELAY_SEC}s.\n")

    try:
        for i, (ttc, rel, ego) in enumerate(data[:num_streams]):
            msg = f"{ttc:.2f},{rel:.2f},{ego:.2f}\n"
            ser.write(msg.encode())

            # Read any ESP32 serial response
            time.sleep(0.05)
            response = ""
            while ser.in_waiting:
                line = ser.readline().decode(errors="ignore").strip()
                if line:
                    response = line

            print(f"[{i+1:03d}] Sent → {msg.strip()}")
            if response:
                print(f"     ↳ ESP32: {response}")
            else:
                print("     ↳ ESP32: (no response yet)")

            time.sleep(DELAY_SEC)

        print("\n✅ [DONE] Data streaming completed successfully.")
    except KeyboardInterrupt:
        print("\n⛔ Interrupted by user.")
    except Exception as e:
        print(f"\n❌ [ERROR] Streaming failed: {e}")
    finally:
        ser.close()
        print("🔒 Serial connection closed.\n")

# ----------------------------
#  MAIN EXECUTION
# ----------------------------
if __name__ == "__main__":
    print("===============================================")
    print("   🚗 Cognitive Braking Assist – Live Streamer ")
    print("===============================================\n")

    print("[INFO] Generating synthetic dataset...")
    synthetic_data = generate_synthetic_data(NUM_SAMPLES)
    print(f"✅ Generated {NUM_SAMPLES} synthetic samples.\n")

    ser = connect_esp32()
    print("⏳ Waiting for ESP32 ready message...\n")

    # Optional handshake read
    start_time = time.time()
    while time.time() - start_time < 5:
        if ser.in_waiting:
            msg = ser.readline().decode(errors="ignore").strip()
            if msg:
                print(f"💬 ESP32 says: {msg}")
        time.sleep(0.2)

    print("\n📡 ESP32 communication verified.")
    stream_data(ser, synthetic_data, STREAM_COUNT)
