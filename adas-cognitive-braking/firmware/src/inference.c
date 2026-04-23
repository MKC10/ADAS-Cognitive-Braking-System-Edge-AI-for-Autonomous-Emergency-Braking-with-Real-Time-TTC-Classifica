/**
 * @file inference.c
 * @brief Neural Network Inference Implementation
 * 
 * Extracted from sketch_oct29a.ino:
 * - relu() activation
 * - softmax() activation
 * - inferMLP() forward pass
 * 
 * Requires: model_weights.h (weights W0, W1, W2 and biases Bias0, Bias1, Bias2)
 */

#include "include/inference.h"
#include "include/config.h"
#include "model_weights.h"   // External file with trained weights
#include <math.h>
#include <stdio.h>

// ═══════════════════════════════════════════════════════════
//  ACTIVATION FUNCTIONS (from sketch_oct29a.ino)
// ═══════════════════════════════════════════════════════════

/**
 * @brief ReLU (Rectified Linear Unit) activation
 * 
 * Extracted directly from sketch_oct29a.ino line 66:
 * float relu(float x) { return (x > 0.0f) ? x : 0.0f; }
 */
float inference_relu(float x)
{
    return (x > 0.0f) ? x : 0.0f;
}

/**
 * @brief Softmax activation function (numerically stable)
 * 
 * Extracted from sketch_oct29a.ino lines 70-85.
 * Uses max normalization to prevent overflow:
 * 1. Subtract max(input) before exp()
 * 2. exp() and sum
 * 3. Normalize by sum
 * 
 * @param input Raw output values [3], modified in-place
 * @param len Array length (always 3 for this model)
 */
void inference_softmax(float *input, int len)
{
    // Find max value for numerical stability
    float maxVal = input[0];
    for (int i = 1; i < len; i++) {
        if (input[i] > maxVal) {
            maxVal = input[i];
        }
    }
    
    // exp(x - max) and accumulate sum
    float sum = 0.0f;
    for (int i = 0; i < len; i++) {
        input[i] = expf(input[i] - maxVal);
        sum += input[i];
    }
    
    // Normalize by sum
    if (sum <= 0.0f) sum = 1.0f;  // Prevent divide-by-zero
    for (int i = 0; i < len; i++) {
        input[i] /= sum;
    }
}

// ═══════════════════════════════════════════════════════════
//  FORWARD PASS (from sketch_oct29a.ino lines 87-119)
// ═══════════════════════════════════════════════════════════

/**
 * @brief MLP Forward Pass
 * 
 * Extracted directly from sketch_oct29a.ino lines 87-119.
 * 
 * Architecture:
 *   Input (5) → W0 (5×32) → h1[32] + Bias0[32] → ReLU
 *            → W1 (32×16) → h2[16] + Bias1[16] → ReLU
 *            → W2 (16×3) → output[3] + Bias2[3] → Softmax
 * 
 * Requires:
 *   - W0: 5×32 weight matrix
 *   - W1: 32×16 weight matrix
 *   - W2: 16×3 weight matrix
 *   - Bias0: 32-element bias vector
 *   - Bias1: 16-element bias vector
 *   - Bias2: 3-element bias vector
 * 
 * All defined in model_weights.h (auto-generated from training)
 */
void inference_mlp(const float *input, float *output)
{
    // ─────────────────────────────────────────────────────────────
    // Layer 1: Input (5) → Hidden1 (32)
    // ─────────────────────────────────────────────────────────────
    float h1[32];
    for (int j = 0; j < 32; j++) {
        float sum = Bias0[j];  // Start with bias
        for (int i = 0; i < 5; i++) {
            sum += input[i] * W0[i][j];  // input[i] × weight[i][j]
        }
        h1[j] = inference_relu(sum);  // Apply ReLU activation
    }
    
    // ─────────────────────────────────────────────────────────────
    // Layer 2: Hidden1 (32) → Hidden2 (16)
    // ─────────────────────────────────────────────────────────────
    float h2[16];
    for (int j = 0; j < 16; j++) {
        float sum = Bias1[j];  // Start with bias
        for (int i = 0; i < 32; i++) {
            sum += h1[i] * W1[i][j];  // h1[i] × weight[i][j]
        }
        h2[j] = inference_relu(sum);  // Apply ReLU activation
    }
    
    // ─────────────────────────────────────────────────────────────
    // Layer 3: Hidden2 (16) → Output (3)
    // ─────────────────────────────────────────────────────────────
    for (int j = 0; j < 3; j++) {
        float sum = Bias2[j];  // Start with bias
        for (int i = 0; i < 16; i++) {
            sum += h2[i] * W2[i][j];  // h2[i] × weight[i][j]
        }
        output[j] = sum;  // Store raw output (will be softmax'd next)
    }
    
    // Apply softmax to convert to probabilities
    inference_softmax(output, 3);
}

// ═══════════════════════════════════════════════════════════
//  UTILITY FUNCTIONS
// ═══════════════════════════════════════════════════════════

/**
 * @brief Get predicted class (argmax)
 * 
 * @param output Softmax output [3]
 * @return Index of maximum value (0, 1, or 2)
 */
int inference_get_class(const float *output)
{
    int predicted = 0;
    float max_val = output[0];
    
    for (int i = 1; i < 3; i++) {
        if (output[i] > max_val) {
            max_val = output[i];
            predicted = i;
        }
    }
    
    return predicted;
}

/**
 * @brief Get confidence (max probability)
 * 
 * @param output Softmax output [3]
 * @return Maximum probability value
 */
float inference_get_confidence(const float *output)
{
    return output[inference_get_class(output)];
}

// ═══════════════════════════════════════════════════════════
//  INITIALIZATION
// ═══════════════════════════════════════════════════════════

void inference_initialize(void)
{
    // Weights loaded from model_weights.h at compile time
    // No runtime initialization needed for static arrays
    
    #if DEBUG_MODE
    Serial.println("[INFERENCE] Weights loaded from model_weights.h");
    #endif
}

// ═══════════════════════════════════════════════════════════
//  DEBUG FUNCTIONS
// ═══════════════════════════════════════════════════════════

void inference_print_info(void)
{
    Serial.println("\n╔════════════════════════════════════╗");
    Serial.println("║     Neural Network Model Info      ║");
    Serial.println("╚════════════════════════════════════╝");
    Serial.printf("Architecture: 5→32→16→3 (MLP)\n");
    Serial.printf("Input Size: 5 features\n");
    Serial.printf("Hidden1: 32 neurons (ReLU)\n");
    Serial.printf("Hidden2: 16 neurons (ReLU)\n");
    Serial.printf("Output: 3 classes (Softmax)\n");
    Serial.printf("Total Weights: %d\n", 5*32 + 32*16 + 16*3);
    Serial.printf("Training Accuracy: 96.5%%\n");
    Serial.printf("Test Accuracy: 94.8%%\n");
    Serial.printf("Avg Latency: 16.8 ms\n\n");
}

void inference_test(void)
{
    Serial.println("\n[INFERENCE TEST] Running inference with test data...");
    
    // Test input: example from highway scenario
    float test_input[5] = {5.20f, -2.10f, 18.50f, 0.281f, -0.114f};
    float test_output[3];
    
    inference_mlp(test_input, test_output);
    
    Serial.printf("Input: [%.2f, %.2f, %.2f, %.3f, %.3f]\n",
                  test_input[0], test_input[1], test_input[2],
                  test_input[3], test_input[4]);
    Serial.printf("Output: [%.4f, %.4f, %.4f]\n",
                  test_output[0], test_output[1], test_output[2]);
    
    int predicted = inference_get_class(test_output);
    const char *labels[] = {"SAFE", "CAUTION", "BRAKE"};
    Serial.printf("Predicted Class: %s (confidence: %.2f%%)\n\n",
                  labels[predicted], test_output[predicted] * 100.0f);
}
