/**
 * @file servo_control.c
 * @brief Servo Control Implementation
 * 
 * Extracted from sketch_oct29a.ino:
 * - Servo sweep pattern logic
 * - Brake hold timing
 * - State-based servo movement
 */

#include "include/servo_control.h"
#include "include/config.h"
#include <Arduino.h>
#include <ESP32Servo.h>

// ═══════════════════════════════════════════════════════════
//  INTERNAL STATE
// ═══════════════════════════════════════════════════════════

static Servo brake_servo;
static float sweep_angle = SERVO_MIN_ANGLE;
static int sweep_direction = 1;  // +1 or -1
static float sweep_speed = SAFE_SPEED_DEG;
static unsigned long last_servo_update_ms = 0;
static unsigned long last_brake_time_ms = 0;
static int is_initialized = 0;
static int is_sweeping = 0;

// ═══════════════════════════════════════════════════════════
//  INITIALIZATION
// ═══════════════════════════════════════════════════════════

void servo_control_init(void)
{
    if (is_initialized) {
        return;
    }
    
    // Attach servo to GPIO pin
    brake_servo.attach(SERVO_PIN);
    
    // Set initial position
    sweep_angle = SERVO_MIN_ANGLE;
    brake_servo.write((int)sweep_angle);
    
    // Initialize state
    sweep_direction = 1;
    sweep_speed = SAFE_SPEED_DEG;
    last_servo_update_ms = millis();
    last_brake_time_ms = 0;
    is_sweeping = 1;
    
    is_initialized = 1;
    
    #if DEBUG_MODE
    Serial.println("[SERVO] Initialized on GPIO " STR(SERVO_PIN));
    #endif
}

// ═══════════════════════════════════════════════════════════
//  SERVO CONTROL (Main Update Function)
// ═══════════════════════════════════════════════════════════

/**
 * @brief Main servo update - extracted from sketch_oct29a.ino lines 176-211
 * 
 * State machine:
 * 1. BRAKE state → hold at SERVO_BRAKE_ANGLE (100°)
 * 2. Recently braked (within BRAKE_HOLD_MS) → stay at brake angle
 * 3. SAFE/CAUTION → sweep pattern (different speeds)
 */
void servo_control_update(int current_state, unsigned long now)
{
    // Track when brake was last applied
    if (current_state == CLASS_BRAKE) {
        last_brake_time_ms = now;
    }
    
    // Update servo position every SERVO_UPDATE_PERIOD_MS
    if (now - last_servo_update_ms >= SERVO_UPDATE_PERIOD_MS) {
        last_servo_update_ms = now;
        
        // ─────────────────────────────────────────────────────
        // BRAKE or POST-BRAKE: Hold at brake angle
        // ─────────────────────────────────────────────────────
        if ((current_state == CLASS_BRAKE) ||
            (now - last_brake_time_ms < BRAKE_HOLD_MS)) {
            
            brake_servo.write(SERVO_BRAKE_ANGLE);
            sweep_angle = SERVO_BRAKE_ANGLE;
            is_sweeping = 0;
        }
        // ─────────────────────────────────────────────────────
        // SAFE or CAUTION: Continuous sweep pattern
        // ─────────────────────────────────────────────────────
        else {
            is_sweeping = 1;
            
            // Choose sweep speed based on state
            float speed = (current_state == CLASS_SAFE)
                         ? SAFE_SPEED_DEG      // 3.0°/update in SAFE
                         : CAUTION_SPEED_DEG;  // 1.2°/update in CAUTION
            
            // Update sweep angle
            sweep_angle += sweep_direction * speed;
            
            // Reflect at boundaries
            if (sweep_angle > SERVO_MAX_ANGLE) {
                sweep_angle = SERVO_MAX_ANGLE;
                sweep_direction = -1;
            } else if (sweep_angle < SERVO_MIN_ANGLE) {
                sweep_angle = SERVO_MIN_ANGLE;
                sweep_direction = 1;
            }
            
            // Write to servo
            brake_servo.write((int)sweep_angle);
        }
    }
}

// ═══════════════════════════════════════════════════════════
//  DIRECT SERVO CONTROL
// ═══════════════════════════════════════════════════════════

void servo_control_set_angle(int angle)
{
    // Clamp to valid range
    if (angle < SERVO_MIN_ANGLE) angle = SERVO_MIN_ANGLE;
    if (angle > SERVO_MAX_ANGLE) angle = SERVO_MAX_ANGLE;
    
    sweep_angle = (float)angle;
    brake_servo.write(angle);
    is_sweeping = 0;
}

int servo_control_get_angle(void)
{
    return (int)sweep_angle;
}

void servo_control_brake(void)
{
    servo_control_set_angle(SERVO_BRAKE_ANGLE);
}

void servo_control_safe(void)
{
    servo_control_set_angle(SERVO_MIN_ANGLE);
}

// ═══════════════════════════════════════════════════════════
//  SWEEP CONTROL
// ═══════════════════════════════════════════════════════════

void servo_control_start_sweep(float speed_deg)
{
    sweep_speed = speed_deg;
    is_sweeping = 1;
    sweep_direction = 1;
}

void servo_control_stop_sweep(void)
{
    is_sweeping = 0;
}

// ═══════════════════════════════════════════════════════════
//  DEBUG / DIAGNOSTICS
// ═══════════════════════════════════════════════════════════

void servo_control_print_status(void)
{
    Serial.println("\n╔════════════════════════════════════╗");
    Serial.println("║       Servo Control Status         ║");
    Serial.println("╚════════════════════════════════════╝");
    Serial.printf("Current Angle: %.1f°\n", sweep_angle);
    Serial.printf("Min/Max: %d° / %d°\n", SERVO_MIN_ANGLE, SERVO_MAX_ANGLE);
    Serial.printf("Brake Angle: %d°\n", SERVO_BRAKE_ANGLE);
    Serial.printf("Sweeping: %s\n", is_sweeping ? "YES" : "NO");
    Serial.printf("Sweep Speed: %.1f°/update\n", sweep_speed);
    Serial.printf("Update Period: %d ms\n", SERVO_UPDATE_PERIOD_MS);
    Serial.printf("Brake Hold: %d ms\n\n", BRAKE_HOLD_MS);
}

void servo_control_test_sweep(void)
{
    Serial.println("\n[SERVO TEST] Running full sweep...");
    Serial.println("Sweeping from 10° to 100°...\n");
    
    for (int angle = SERVO_MIN_ANGLE; angle <= SERVO_MAX_ANGLE; angle += 10) {
        servo_control_set_angle(angle);
        Serial.printf("  Angle: %d°\n", angle);
        delay(200);
    }
    
    Serial.println("\nSweeping from 100° back to 10°...\n");
    
    for (int angle = SERVO_MAX_ANGLE; angle >= SERVO_MIN_ANGLE; angle -= 10) {
        servo_control_set_angle(angle);
        Serial.printf("  Angle: %d°\n", angle);
        delay(200);
    }
    
    servo_control_set_angle(SERVO_MIN_ANGLE);
    Serial.println("[SERVO TEST] Complete. Returned to SAFE (10°).\n");
}

// ═══════════════════════════════════════════════════════════
//  HELPER MACRO (for debug output)
// ═══════════════════════════════════════════════════════════

#define STR(x) #x
