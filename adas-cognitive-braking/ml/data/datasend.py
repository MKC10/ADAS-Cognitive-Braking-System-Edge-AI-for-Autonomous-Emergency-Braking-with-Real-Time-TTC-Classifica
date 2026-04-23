import serial
import time
import numpy as np

# ==============================
# 🔧 CONFIGURATION
# ==============================
STREAM_SAMPLES = 500           # number of live samples
STREAM_DELAY = 0.75            # delay between samples (seconds)
SERIAL_PORT = "COM4"           # your ESP32 port
BAUD_RATE = 115200
START_KEYWORD = "ESP32 ready"  # optional ESP32 startup message
# ==============================

# === Generate synthetic data approximating your training dataset ===
np.random.seed(42)

# Typical realistic driving ranges (adjust as needed)
spacing_m = np.clip(np.random.normal(5.0, 2.0, STREAM_SAMPLES), 0.3, 15.0)   # 0.3–15 m
rel_speed_mps = np.clip(np.random.normal(-2.0, 1.5, STREAM_SAMPLES), -8.0, 4.0)  # -8 to 4 m/s
ego_speed_mps = np.clip(np.random.normal(10.0, 3.0, STREAM_SAMPLES), 0.5, 25.0)  # 0.5–25 m/s

# Derived ratios
spacing_over_speed = spacing_m / (np.abs(ego_speed_mps) + 1e-6)
rel_over_speed = rel_speed_mps / (np.abs(ego_speed_mps) + 1e-6)

print(f"✅ Generated {STREAM_SAMPLES} synthetic samples.")
print("Example sample:")
print(f"Spacing={spacing_m[0]:.2f}, RelSpeed={rel_speed_mps[0]:.2f}, EgoSpeed={ego_speed_mps[0]:.2f}, "
      f"Spacing/Ego={spacing_over_speed[0]:.2f}, Rel/Ego={rel_over_speed[0]:.2f}")

# === Open serial connection ===
try:
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
    time.sleep(2)
    print(f"\n🔌 Connected to {SERIAL_PORT} at {BAUD_RATE} baud.")
    print("⏳ Waiting for ESP32 to announce readiness...")

    # Optional: wait for ESP32 startup message
    ready = False
    for _ in range(50):  # ~10 seconds max
        line = ser.readline().decode(errors="ignore").strip()
        if line:
            print("ESP32:", line)
        if START_KEYWORD.lower() in line.lower():
            ready = True
            break
        time.sleep(0.2)

    if not ready:
        print("⚠️ ESP32 didn’t send readiness message. Proceeding anyway...")
    else:
        print("🚀 ESP32 is ready — starting live 5-parameter stream!\n")

    # === Stream all parameters ===
    for i in range(STREAM_SAMPLES):
        line = f"{spacing_m[i]:.2f},{rel_speed_mps[i]:.2f},{ego_speed_mps[i]:.2f},{spacing_over_speed[i]:.3f},{rel_over_speed[i]:.3f}\n"
        ser.write(line.encode('utf-8'))

        resp = ser.readline().decode(errors="ignore").strip()
        if resp:
            print(f"ESP32 → {resp}")
        else:
            print(f"Sent sample[{i}] = {line.strip()} (no response yet)")

        time.sleep(STREAM_DELAY)

    print("\n✅ Streaming finished successfully.")

except serial.SerialException as e:
    print(f"❌ Serial error: {e}")

finally:
    if 'ser' in locals() and ser.is_open:
        ser.close()
        print("🔒 Serial connection closed.")
