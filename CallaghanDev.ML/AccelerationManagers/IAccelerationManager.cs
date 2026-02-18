using CallaghanDev.ML.Enums;
using CallaghanDev.ML.Transformers.TACAMT;
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

        float[,] MatrixAddBias(float[,] matrix, float[] bias);

        float[,] EmbedTokensWithPosition(float[,] tokenEmbedding, int[] tokenIds, float[,] positionalEncoding, int seqLen, int embeddingDim);

        float[,] AddBiasAndPositionalEncoding(float[,] projected, float[] bias, float[,] positionalEncoding, int seqLen, int embeddingDim);

        (float loss, float[,] dLogits) CrossEntropyLossAndGradient(float[,] logits, int[] targets, int effectiveLen);

        (float loss, float[,] dOutput) MSELossAndGradient(float[,] predictions, float[,] targets, int effectiveLen);

        float[,] BackpropOutputProjection(float[,] dLogits, float[,] input, float[,] weights, float[,] weightGrad, float[] biasGrad, int seqLen, int outputDim, int embeddingDim);

        /// <summary>
        /// Backprop through input projection (continuous inputs).
        /// Accumulates into weightGrad and biasGrad.
        /// </summary>
        void BackpropInputProjection(float[,] dX, float[,] continuousInput, float[,] weightGrad, float[] biasGrad, int seqLen, int embeddingDim, int inputFeatureDim);

        /// <summary>
        /// Accumulates embedding gradients: embeddingGrad[tokenIds[i], j] += dX[i, j]
        /// </summary>
        void AccumulateTokenEmbeddingGrad(float[,] embeddingGrad, float[,] dX, int[] tokenIds, int seqLen, int embeddingDim);

        /// <summary>
        /// Element-wise vector accumulation: target[j] += source[j]
        /// </summary>
        void AccumulateVectorGradients(float[] targetGrad, float[] sourceGrad);

        /// <summary>
        /// Squared L2 norm of a vector (for gradient clipping).
        /// </summary>
        float VectorSquaredNorm(float[] vector);

        /// <summary>
        /// Extract rows [startRow, endRow) from a matrix.
        /// </summary>
        float[,] SliceRows(float[,] matrix, int startRow, int endRow);

        /// <summary>
        /// Copy one row from a matrix into a new vector.
        /// </summary>
        float[] ExtractRow(float[,] matrix, int rowIndex, int cols);

        /// <summary>
        /// Copy a vector into one row of a matrix.
        /// </summary>
        void SetRow(float[,] matrix, int rowIndex, float[] values, int cols);

        /// <summary>
        /// Create a lower-triangular causal mask: mask[i,j] = (j &lt;= i).
        /// </summary>
        bool[,] CreateCausalMask(int seqLen);

        /// <summary>
        /// In-place element-wise: target[i,j] += source[i,j]
        /// </summary>
        void MatrixAccumulate(float[,] target, float[,] source);

        /// <summary>
        /// Zero out a vector.
        /// </summary>
        void ZeroVector(float[] vector);

        void SigmoidInPlace(float[,] matrix);

        float[,] ContentAwareCrossAttentionForward(float[,] Q, float[,] K, float[,] V, int numHeads, float scale, float[,,] decayBias, out float[][,] attentionWeights, out float[][,] scoresPreSoftmax);

        float[,] FFNForwardBatch(float[,] input, int seqLen, int outputDim, Func<float[], float[]> forwardPassFn);

        void ApplyContextTypeEmbedding(float[,] contextHidden,  float[,] typeEmbedding,int[] typeIndices);

        float[,] ComputeTimeDiffMatrix(int priceSeqLen, float[] keyArrivalTimes);

        float[,] MeanPoolRows(float[,] hidden, int[] storyOffsets, int[] storyCounts, int numStories, int embeddingDim);

        float[,] EmbedWithBiasAndPositional(float[,] projected, float[] bias, float[,] positionalEncoding, int seqLen, int embeddingDim);

        float[] ComputeMemoryAttentionScores(float[,] priceHidden, int lastPos, float[,] contextHidden, int totalCtx, float scale);

        float[,] ProjectOutputBatch(float[,] hidden, float[,] outputProjection, float[] outputBias, int seqLen, int outputDim);

        public (float[,,] decayBias, ContentAwareDecayCache cache) ContentAwareDecayForward(float[,] queryEmbeddings, float[,] keyEmbeddings, float[,] timeDiffs, float[] keyTimesFromRef, CallaghanDev.ML.Transformers.TACAMT.ContentAwareDecayNetwork network, bool isTraining = false, Random dropoutRng = null);

        public float[,] ContentAwareCrossAttentionWithCache(float[,] Q, float[,] K, float[,] V, float[,] timeDiffs, float[] keyTimesFromRef, float[,] queryEmbeddings, float[,] keyEmbeddings, TransformerBlock block, BlockCache bc, int PriceEmbeddingDim, int PriceNumHeads, bool isTraining = false, Random dropoutRng = null);


        void Dispose();
    }
}
