/**
 * @file servo_control.h
 * @brief Servo Control Interface
 * 
 * High-level servo control based on braking states (SAFE/CAUTION/BRAKE)
 * Supports sweep patterns and hold-on-brake functionality.
 * 
 * Based on sketch_oct29a.ino servo control logic
 */

#ifndef SERVO_CONTROL_H
#define SERVO_CONTROL_H

#include <stdint.h>

// ═══════════════════════════════════════════════════════════
//  SERVO STATE MANAGEMENT
// ═══════════════════════════════════════════════════════════

/**
 * @brief Initialize servo control module
 * 
 * Sets up GPIO pin and initial servo position.
 * Must be called during setup()
 */
void servo_control_init(void);

/**
 * @brief Update servo position based on current state
 * 
 * Implements state machine:
 * - SAFE: Continuous fast sweep (3.0°/update)
 * - CAUTION: Continuous slow sweep (1.2°/update)
 * - BRAKE: Hold at 100° for BRAKE_HOLD_MS, then resume sweep
 * 
 * @param current_state Current classification state (0=SAFE, 1=CAUTION, 2=BRAKE)
 * @param now Current time in milliseconds (from millis())
 */
void servo_control_update(int current_state, unsigned long now);

/**
 * @brief Manually set servo angle
 * 
 * @param angle Desired angle (10-100 degrees)
 */
void servo_control_set_angle(int angle);

/**
 * @brief Get current servo angle
 * 
 * @return Last commanded angle (10-100 degrees)
 */
int servo_control_get_angle(void);

/**
 * @brief Force servo to brake position
 * 
 * Immediately moves servo to SERVO_BRAKE_ANGLE (100°)
 */
void servo_control_brake(void);

/**
 * @brief Force servo to safe position
 * 
 * Immediately moves servo to SERVO_MIN_ANGLE (10°)
 */
void servo_control_safe(void);

// ═══════════════════════════════════════════════════════════
//  SWEEP CONTROL
// ═══════════════════════════════════════════════════════════

/**
 * @brief Start continuous sweep pattern
 * 
 * @param speed_deg Sweep speed in degrees per update
 *                  Typical values: 3.0f (fast), 1.2f (slow)
 */
void servo_control_start_sweep(float speed_deg);

/**
 * @brief Stop sweep and hold current position
 */
void servo_control_stop_sweep(void);

// ═══════════════════════════════════════════════════════════
//  DEBUG / DIAGNOSTICS
// ═══════════════════════════════════════════════════════════

/**
 * @brief Print servo state information
 */
void servo_control_print_status(void);

/**
 * @brief Test servo with full sweep (10° to 100°)
 * 
 * Useful for hardware validation during development.
 * Moves through range with 200ms delay between steps.
 */
void servo_control_test_sweep(void);

#endif // SERVO_CONTROL_H
