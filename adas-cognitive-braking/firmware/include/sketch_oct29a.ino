#include <ESP32Servo.h>
#include "model_weights.h"   // your trained MLP weights and biases

// ===================== Pin Configuration =====================
#define LED_SAFE     25   // Yellow  → SAFE
#define LED_CAUTION  26   // Amber   → CAUTION
#define LED_BRAKE    27   // Red     → BRAKE
#define SERVO_PIN    14   // PWM-capable pin for servo signal

// ===================== Servo Configuration =====================
const int SERVO_MIN_ANGLE   = 10;
const int SERVO_MAX_ANGLE   = 100;
const int SERVO_BRAKE_ANGLE = 100;   // position when braking

// Sweep speeds (deg per update)
const float SAFE_SPEED_DEG   = 3.0f;   // fast sweep in SAFE
const float CAUT_SPEED_DEG   = 1.2f;   // slower sweep in CAUTION

const unsigned long SERVO_UPDATE_PERIOD_MS = 25;    // time between updates
const unsigned long BRAKE_HOLD_MS          = 800;   // how long to hold brake

// ===================== Class Indexes =====================
const int IDX_SAFE    = 0;
const int IDX_CAUTION = 1;
const int IDX_BRAKE   = 2;

// ===================== Globals =====================
String inputLine = "";
float spacing, rel_speed, ego_speed, spacing_over_speed, rel_over_speed;

Servo brakeServo;

// Servo motion state
float sweepAngle         = SERVO_MIN_ANGLE;
int   sweepDir           = 1;  // +1 or -1
unsigned long lastServoUpdate = 0;
unsigned long lastBrakeTime   = 0;
int currentState = IDX_SAFE;

// ===================== Serial Input =====================
// Expected line: spacing, rel_speed, ego_speed, spacing_over_speed, rel_over_speed
bool readSerialInput() {
  while (Serial.available()) {
    char c = Serial.read();

    if (c == '\n') {
      inputLine.trim();
      if (inputLine.length() == 0) return false;

      int parsed = sscanf(
        inputLine.c_str(),
        "%f,%f,%f,%f,%f",
        &spacing, &rel_speed, &ego_speed,
        &spacing_over_speed, &rel_over_speed
      );
      inputLine = "";
      return (parsed == 5);
    } else {
      inputLine += c;
    }
  }
  return false;
}

// ===================== Activations =====================
float relu(float x) {
  return (x > 0.0f) ? x : 0.0f;
}

void softmax(float *input, int len) {
  float maxVal = input[0];
  for (int i = 1; i < len; i++) {
    if (input[i] > maxVal) maxVal = input[i];
  }

  float sum = 0.0f;
  for (int i = 0; i < len; i++) {
    input[i] = expf(input[i] - maxVal);
    sum += input[i];
  }
  if (sum <= 0.0f) sum = 1.0f;
  for (int i = 0; i < len; i++) {
    input[i] /= sum;
  }
}

// ===================== MLP Forward Pass =====================
void inferMLP(float *input, float *output) {
  // Layer 1: 5 → 32
  float h1[32];
  for (int j = 0; j < 32; j++) {
    float sum = Bias0[j];
    for (int i = 0; i < 5; i++) {
      sum += input[i] * W0[i][j];
    }
    h1[j] = relu(sum);
  }

  // Layer 2: 32 → 16
  float h2[16];
  for (int j = 0; j < 16; j++) {
    float sum = Bias1[j];
    for (int i = 0; i < 32; i++) {
      sum += h1[i] * W1[i][j];
    }
    h2[j] = relu(sum);
  }

  // Output: 16 → 3 (Safe / Caution / Brake)
  for (int j = 0; j < 3; j++) {
    float sum = Bias2[j];
    for (int i = 0; i < 16; i++) {
      sum += h2[i] * W2[i][j];
    }
    output[j] = sum;
  }

  softmax(output, 3);
}

// ===================== Setup =====================
void setup() {
  Serial.begin(115200);

  pinMode(LED_SAFE, OUTPUT);
  pinMode(LED_CAUTION, OUTPUT);
  pinMode(LED_BRAKE, OUTPUT);
  digitalWrite(LED_SAFE, LOW);
  digitalWrite(LED_CAUTION, LOW);
  digitalWrite(LED_BRAKE, LOW);

  brakeServo.attach(SERVO_PIN);
  sweepAngle = SERVO_MIN_ANGLE;
  brakeServo.write((int)sweepAngle);

  lastServoUpdate = millis();
  lastBrakeTime   = 0;

  delay(1000);
  Serial.println("🚗 ESP32 Cognitive Braking Assist – Continuous Servo Pattern");
}

// ===================== Main Loop =====================
void loop() {
  if (!readSerialInput()) return;

  // ----- Inference -----
  float features[5] = {
    spacing, rel_speed, ego_speed, spacing_over_speed, rel_over_speed
  };
  float output[3];
  inferMLP(features, output);

  // Argmax → state
  int predicted = IDX_SAFE;
  float maxVal  = output[IDX_SAFE];
  for (int i = 1; i < 3; i++) {
    if (output[i] > maxVal) {
      maxVal = output[i];
      predicted = i;
    }
  }
  currentState = predicted;

  String label;
  if (predicted == IDX_SAFE)      label = "✅ SAFE";
  else if (predicted == IDX_CAUTION) label = "⚠️  CAUTION";
  else                               label = "🚨 BRAKE";

  // ----- LEDs -----
  digitalWrite(LED_SAFE,    (predicted == IDX_SAFE)    ? HIGH : LOW);
  digitalWrite(LED_CAUTION, (predicted == IDX_CAUTION) ? HIGH : LOW);
  digitalWrite(LED_BRAKE,   (predicted == IDX_BRAKE)   ? HIGH : LOW);

  // ----- Servo motion control -----
  unsigned long now = millis();

  if (predicted == IDX_BRAKE) {
    // mark last time we saw BRAKE
    lastBrakeTime = now;
  }

  if (now - lastServoUpdate >= SERVO_UPDATE_PERIOD_MS) {
    lastServoUpdate = now;

    // If we are braking or just braked recently: hold at brake angle
    if ( (currentState == IDX_BRAKE) ||
         (now - lastBrakeTime < BRAKE_HOLD_MS) ) {

      brakeServo.write(SERVO_BRAKE_ANGLE);

    } else {
      // SAFE or CAUTION → continuous sweep
      float speedDeg = (currentState == IDX_SAFE)
                         ? SAFE_SPEED_DEG   // fast
                         : CAUT_SPEED_DEG;  // slower

      sweepAngle += sweepDir * speedDeg;

      // reflect at ends
      if (sweepAngle > SERVO_MAX_ANGLE) {
        sweepAngle = SERVO_MAX_ANGLE;
        sweepDir   = -1;
      } else if (sweepAngle < SERVO_MIN_ANGLE) {
        sweepAngle = SERVO_MIN_ANGLE;
        sweepDir   = 1;
      }

      brakeServo.write((int)sweepAngle);
    }
  }

  // ----- Debug -----
  Serial.printf(
    "spacing=%.2f rel=%.2f ego=%.2f s/e=%.3f r/e=%.3f → %s | "
    "Prob(S=%.2f, C=%.2f, B=%.2f) | servoAngle=%.1f\n",
    spacing, rel_speed, ego_speed, spacing_over_speed, rel_over_speed,
    label.c_str(),
    output[IDX_SAFE], output[IDX_CAUTION], output[IDX_BRAKE],
    sweepAngle
  );
}
