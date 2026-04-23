/**
 * @file config.h
 * @brief Configuration Header - Pin definitions and constants (based on sketch_oct29a.ino)
 */

#ifndef CONFIG_H
#define CONFIG_H

// ═══════════════════════════════════════════════════════════
//  GPIO PIN DEFINITIONS
// ═══════════════════════════════════════════════════════════

#define LED_SAFE     25   ///< Yellow LED - indicates SAFE state (TTC > 5s)
#define LED_CAUTION  26   ///< Amber LED - indicates CAUTION state (2s < TTC <= 5s)
#define LED_BRAKE    27   ///< Red LED - indicates BRAKE state (TTC <= 2s)
#define SERVO_PIN    14   ///< PWM-capable GPIO pin for servo signal (MG996R)

// ═══════════════════════════════════════════════════════════
//  UART / SERIAL CONFIGURATION
// ═══════════════════════════════════════════════════════════

#define UART_BAUD_RATE   115200   ///< Serial communication speed (baud)

// ═══════════════════════════════════════════════════════════
//  SERVO CONFIGURATION
// ═══════════════════════════════════════════════════════════

#define SERVO_MIN_ANGLE       10    ///< Minimum servo angle (degrees)
#define SERVO_MAX_ANGLE      100    ///< Maximum servo angle (degrees)
#define SERVO_BRAKE_ANGLE    100    ///< Servo position when applying full braking

// Servo motion control
#define SAFE_SPEED_DEG        3.0f  ///< Sweep speed in SAFE state (degrees per update)
#define CAUTION_SPEED_DEG     1.2f  ///< Sweep speed in CAUTION state (degrees per update)
#define SERVO_UPDATE_PERIOD_MS 25   ///< Time between servo position updates (milliseconds)
#define BRAKE_HOLD_MS         800   ///< Duration to hold brake position after BRAKE state (milliseconds)

// ═══════════════════════════════════════════════════════════
//  NEURAL NETWORK ARCHITECTURE
// ═══════════════════════════════════════════════════════════

#define INPUT_SIZE   5   ///< Input layer: 5 features
#define HIDDEN1_SIZE 32  ///< First hidden layer: 32 neurons
#define HIDDEN2_SIZE 16  ///< Second hidden layer: 16 neurons
#define OUTPUT_SIZE  3   ///< Output layer: 3 classes (SAFE, CAUTION, BRAKE)

// Class definitions (output layer indices)
#define CLASS_SAFE    0   ///< Output index 0: SAFE state (TTC > 5 seconds)
#define CLASS_CAUTION 1   ///< Output index 1: CAUTION state (2s < TTC <= 5s)
#define CLASS_BRAKE   2   ///< Output index 2: BRAKE state (TTC <= 2 seconds)

// ═══════════════════════════════════════════════════════════
//  FEATURE DEFINITIONS
// ═══════════════════════════════════════════════════════════

// Input feature indices (for feature vector [5])
#define FEATURE_SPACING_M           0   ///< Distance to leading vehicle (meters)
#define FEATURE_REL_SPEED_MPS       1   ///< Relative velocity (m/s)
#define FEATURE_EGO_SPEED_MPS       2   ///< Ego vehicle speed (m/s)
#define FEATURE_SPACING_OVER_SPEED  3   ///< Derived: spacing / ego_speed
#define FEATURE_REL_OVER_SPEED      4   ///< Derived: rel_speed / ego_speed

// ═══════════════════════════════════════════════════════════
//  DATA STREAM FORMAT
// ═══════════════════════════════════════════════════════════

/**
 * Expected CSV input from Python streamer (e.g., datasend.py):
 * 
 * Format:
 *   spacing_m,rel_speed_mps,ego_speed_mps,spacing_over_speed,rel_over_speed
 * 
 * Example (highway scenario):
 *   5.20,-2.10,18.50,0.281,-0.114
 * 
 * Where:
 *   - spacing_m: Distance to vehicle ahead (meters), range [0.3, 15.0]
 *   - rel_speed_mps: Rate of closure (m/s), negative = approaching, range [-8.0, 4.0]
 *   - ego_speed_mps: Vehicle speed (m/s), range [0.5, 25.0]
 *   - spacing_over_speed: Feature ratio, used for normalization
 *   - rel_over_speed: Feature ratio, used for normalization
 */

// ═══════════════════════════════════════════════════════════
//  MODEL METADATA (from training)
// ═══════════════════════════════════════════════════════════

#define MODEL_ACCURACY_TRAIN   0.965f  ///< Training accuracy: 96.5%
#define MODEL_ACCURACY_VAL     0.965f  ///< Validation accuracy: 96.5%
#define MODEL_ACCURACY_TEST    0.948f  ///< Test accuracy: 94.8%
#define MODEL_LATENCY_MS       16.8f   ///< Average inference latency (milliseconds)
#define MODEL_LATENCY_MAX_MS   20.0f   ///< Peak inference latency (milliseconds)

// Training data
#define DATASET_SIZE           3001    ///< Size of training dataset
#define DATASET_SOURCE         "OpenACC"  ///< Training data source

// ═══════════════════════════════════════════════════════════
//  TTC THRESHOLDS (for reference - not used in inference)
// ═══════════════════════════════════════════════════════════

#define TTC_SAFE_THRESHOLD    5.0f    ///< TTC > 5 seconds → SAFE
#define TTC_CAUTION_THRESHOLD 2.0f    ///< 2 < TTC <= 5 seconds → CAUTION
#define TTC_BRAKE_THRESHOLD   2.0f    ///< TTC <= 2 seconds → BRAKE

// ═══════════════════════════════════════════════════════════
//  DEBUG / LOGGING
// ═══════════════════════════════════════════════════════════

#define DEBUG_MODE     1   ///< Enable debug serial output (1=enabled, 0=disabled)
#define DEBUG_VERBOSE  0   ///< Extra detailed logging

// ═══════════════════════════════════════════════════════════
//  ASSERTIONS (Compile-time validation)
// ═══════════════════════════════════════════════════════════

#if (INPUT_SIZE != 5)
    #error "INPUT_SIZE must be 5 (features from datasend.py)"
#endif

#if (OUTPUT_SIZE != 3)
    #error "OUTPUT_SIZE must be 3 (SAFE, CAUTION, BRAKE)"
#endif

#endif // CONFIG_H
