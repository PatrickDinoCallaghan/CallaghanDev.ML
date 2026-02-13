using CallaghanDev.ML.Enums;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace CallaghanDev.ML.AccelerationManagers
{
    public interface IAccelerationManager
    {
        float[] CalculateDotProduct(float[,] matrix, float[] vector);
        (float[] activation, float[] derivative) ActivateLayer(float[] dot, float[] bias, ActivationType activationType);
        float[] CalculateOutputGradients(float[] cost, float[] derivative);
        float[] CalculateHiddenGradients(float[,] weights, float[] nextDeltas, float[] derivative);
        float[,] UpdateWeights(float[,] weights, float[] deltas, float[] prevActivations, float learningRate, float lambda);
        float[] UpdateBias(float[] bias, float[] deltas, float learningRate);



        /// <summary>
        /// Matrix-matrix multiplication: C = A * B
        /// Critical for transformer operations
        /// </summary>
        float[,] MatrixMultiply(float[,] A, float[,] B);

        /// <summary>
        /// Matrix-matrix multiplication with transpose: C = A * B^T
        /// Essential for attention: Q * K^T
        /// </summary>
        float[,] MatrixMultiplyTranspose(float[,] A, float[,] B);

        /// <summary>
        /// Element-wise matrix scaling
        /// </summary>
        float[,] MatrixScale(float[,] matrix, float scalar);

        /// <summary>
        /// Element-wise matrix addition
        /// </summary>
        float[,] MatrixAdd(float[,] A, float[,] B);

        /// <summary>
        /// Row-wise softmax with optional masking for attention
        /// </summary>
        float[,] Softmax(float[,] matrix, bool[,] mask = null);

        /// <summary>
        /// Layer normalization for transformer blocks
        /// </summary>
        float[,] LayerNorm(float[,] input, float[] gamma, float[] beta, float epsilon = 1e-5f);


        /// <summary>
        /// Batch matrix-vector dot products: computes weights · inputRows[i] for each row.
        /// Replaces row-by-row CalculateDotProduct loops in MatMulWithBias/ProjectToVocab.
        /// </summary>
        float[,] BatchDotProduct(float[,] weights, float[,] inputMatrix);

        /// <summary>
        /// Parallel per-head attention forward: computes all heads concurrently.
        /// Returns concatenated [seqLen, embeddingDim] output.
        /// </summary>
        float[,] MultiHeadAttentionForward(float[,] Q, float[,] K, float[,] V, int numHeads, float scale, bool[,] mask = null);

        /// <summary>
        /// Parallel per-head attention backward: computes dQ, dK, dV across all heads concurrently.
        /// </summary>
        (float[,] dQ, float[,] dK, float[,] dV) MultiHeadAttentionBackward(float[,] Q, float[,] K, float[,] V, float[,] dConcatenated, int numHeads, float scale, bool useDecoderMask = false);

        /// <summary>
        /// Backprop through linear projection: accumulates weight/bias grads and input gradient.
        /// Parallelizes the triple nested loop in BackpropLinearProjection.
        /// </summary>
        void BackpropLinearProjection(float[,] input, float[,] dOutput, float[,] weights, float[,] weightGrad, float[] biasGrad, float[,] dInput);

        /// <summary>
        /// Layer norm forward with cache outputs, parallelized across rows.
        /// </summary>
        (float[,] output, float[] means, float[] variances, float[,] normalized) LayerNormForward(float[,] input, float[] gamma, float[] beta, float epsilon = 1e-5f);

        /// <summary>
        /// Layer norm backward, parallelized across rows.
        /// </summary>
        (float[,] dInput, float[] dGamma, float[] dBeta) LayerNormBackward(float[,] dOut, float[,] normalized, float[] gamma, float[,] input, float[] mean, float[] variance, float epsilon = 1e-5f);

        /// <summary>
        /// Computes squared Frobenius norm of a matrix (for gradient clipping).
        /// </summary>
        float MatrixSquaredNorm(float[,] matrix);

        /// <summary>
        /// In-place scale all elements of a matrix.
        /// </summary>
        void MatrixScaleInPlace(float[,] matrix, float scale);

        /// <summary>
        /// In-place scale all elements of a vector.
        /// </summary>
        void VectorScaleInPlace(float[] vector, float scale);

        /// <summary>
        /// SGD weight update: weights -= lr * gradients, parallelized.
        /// </summary>
        void MatrixUpdate(float[,] weights, float[,] gradients, float learningRate);
        void VectorUpdate(float[] weights, float[] gradients, float learningRate);

        /// <summary>
        /// Zero out a matrix (can use SIMD/parallel clearing).
        /// </summary>
        void ZeroMatrix(float[,] matrix);
        void Dispose();
    }
}
