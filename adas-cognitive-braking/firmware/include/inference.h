/**
 * @file inference.h
 * @brief Neural Network Inference Interface
 * 
 * Provides interface for MLP inference (5→32→16→3)
 * Forward pass uses quantized weights from model_weights.h
 * 
 * Based on sketch_oct29a.ino inferMLP() function
 */

#ifndef INFERENCE_H
#define INFERENCE_H

#include <stdint.h>

// ═══════════════════════════════════════════════════════════
//  INFERENCE FUNCTIONS
// ═══════════════════════════════════════════════════════════

/**
 * @brief Initialize inference module and load weights
 * 
 * Must be called during setup()
 */
void inference_initialize(void);

/**
 * @brief Run MLP forward pass
 * 
 * Performs 3-layer neural network inference:
 * Input (5) → Hidden1 (32, ReLU) → Hidden2 (16, ReLU) → Output (3, Softmax)
 * 
 * @param input Input feature vector [5]:
 *              input[0] = spacing (meters)
 *              input[1] = rel_speed (m/s)
 *              input[2] = ego_speed (m/s)
 *              input[3] = spacing_over_speed
 *              input[4] = rel_over_speed
 * 
 * @param output Output probabilities [3]:
 *               output[0] = P(SAFE)
 *               output[1] = P(CAUTION)
 *               output[2] = P(BRAKE)
 *               Values normalized by softmax to sum to 1.0
 */
void inference_mlp(const float *input, float *output);

/**
 * @brief ReLU activation function
 * 
 * @param x Input value
 * @return max(0, x)
 */
float inference_relu(float x);

/**
 * @brief Softmax activation (numerically stable)
 * 
 * Converts raw outputs to probabilities that sum to 1.0
 * Uses max normalization for numerical stability
 * 
 * @param input Raw output values [3] (modified in-place)
 * @param len Array length (3)
 */
void inference_softmax(float *input, int len);

/**
 * @brief Get predicted class from output probabilities
 * 
 * @param output Softmax output [3]
 * @return Class with highest probability (0, 1, or 2)
 */
int inference_get_class(const float *output);

/**
 * @brief Get confidence of predicted class
 * 
 * @param output Softmax output [3]
 * @return Maximum probability value (0.0 to 1.0)
 */
float inference_get_confidence(const float *output);

// ═══════════════════════════════════════════════════════════
//  DEBUG FUNCTIONS
// ═══════════════════════════════════════════════════════════

/**
 * @brief Print model information
 */
void inference_print_info(void);

/**
 * @brief Test inference with known input
 */
void inference_test(void);

#endif // INFERENCE_H
