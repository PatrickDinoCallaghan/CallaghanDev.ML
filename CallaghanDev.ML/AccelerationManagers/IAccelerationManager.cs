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

        void Dispose();
    }
}
