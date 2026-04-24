using CallaghanDev.ML.AccelerationManagers;
using System;

namespace CallaghanDev.ML.Transformers.MultiTypeTransformer
{
    public class MultiHeadAttention
    {
        private readonly int _embeddingDim;
        private readonly int _numHeads;
        private readonly int _headDim;
        private readonly IAccelerationManager _accel;

        public float[,] WQ { get; set; }
        public float[,] WK { get; set; }
        public float[,] WV { get; set; }
        public float[,] WO { get; set; }
        public float[] BiasQ { get; set; }
        public float[] BiasK { get; set; }
        public float[] BiasV { get; set; }
        public float[] BiasO { get; set; }

        public MultiHeadAttention(int embeddingDim, int numHeads, IAccelerationManager accel, Random random = null)
        {
            if (embeddingDim <= 0)
                throw new ArgumentOutOfRangeException(nameof(embeddingDim), "Embedding dimension must be positive.");
            if (numHeads <= 0)
                throw new ArgumentOutOfRangeException(nameof(numHeads), "Number of heads must be positive.");
            if (embeddingDim % numHeads != 0)
                throw new ArgumentException("Embedding dimension must be divisible by number of heads.", nameof(numHeads));

            int headDim = embeddingDim / numHeads;
            if ((headDim & 1) != 0)
                throw new ArgumentException("RoPE requires an even per-head dimension.", nameof(embeddingDim));

            _embeddingDim = embeddingDim;
            _numHeads = numHeads;
            _headDim = headDim;
            _accel = accel ?? throw new ArgumentNullException(nameof(accel));

            random ??= new Random();

            WQ = InitWeights(embeddingDim, embeddingDim, random);
            WK = InitWeights(embeddingDim, embeddingDim, random);
            WV = InitWeights(embeddingDim, embeddingDim, random);
            WO = InitWeights(embeddingDim, embeddingDim, random);

            BiasQ = new float[embeddingDim];
            BiasK = new float[embeddingDim];
            BiasV = new float[embeddingDim];
            BiasO = new float[embeddingDim];
        }

        private float[,] InitWeights(int rows, int cols, Random random)
        {
            if (rows <= 0) throw new ArgumentOutOfRangeException(nameof(rows));
            if (cols <= 0) throw new ArgumentOutOfRangeException(nameof(cols));

            var weights = new float[rows, cols];
            float std = MathF.Sqrt(2.0f / (rows + cols));

            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < cols; j++)
                {
                    float u1 = MathF.Max(1e-7f, 1.0f - random.NextSingle());
                    float u2 = 1.0f - random.NextSingle();
                    float z = MathF.Sqrt(-2.0f * MathF.Log(u1)) * MathF.Cos(2.0f * MathF.PI * u2);
                    weights[i, j] = z * std;
                }
            }

            return weights;
        }

        public float[,] Forward(float[,] input, bool[,] mask = null)
        {
            if (input == null) throw new ArgumentNullException(nameof(input));
            ValidateInputWidth(input, nameof(input));

            int seqLen = input.GetLength(0);
            ValidateNonEmptySequence(seqLen, nameof(input));
            ValidateMask(mask, seqLen, seqLen);

            var Q = MatMulWithBias(input, WQ, BiasQ);
            var K = MatMulWithBias(input, WK, BiasK);
            var V = MatMulWithBias(input, WV, BiasV);

            RotaryPositionEmbedding.ApplyInPlace(Q, K, _numHeads);
            return AttentionCore(Q, K, V, mask);
        }

        public float[,] Forward(float[,] query, float[,] keyValue, bool[,] mask = null)
        {
            if (query == null) throw new ArgumentNullException(nameof(query));
            if (keyValue == null) throw new ArgumentNullException(nameof(keyValue));
            ValidateInputWidth(query, nameof(query));
            ValidateInputWidth(keyValue, nameof(keyValue));

            int queryLen = query.GetLength(0);
            int keyLen = keyValue.GetLength(0);
            ValidateNonEmptySequence(queryLen, nameof(query));
            ValidateNonEmptySequence(keyLen, nameof(keyValue));
            ValidateMask(mask, queryLen, keyLen);

            var Q = MatMulWithBias(query, WQ, BiasQ);
            var K = MatMulWithBias(keyValue, WK, BiasK);
            var V = MatMulWithBias(keyValue, WV, BiasV);

            RotaryPositionEmbedding.ApplyInPlace(Q, K, _numHeads);
            return AttentionCore(Q, K, V, mask);
        }

        private float[,] AttentionCore(float[,] Q, float[,] K, float[,] V, bool[,] mask)
        {
            int queryLen = Q.GetLength(0);
            int keyLen = K.GetLength(0);
            var concatenated = new float[queryLen, _embeddingDim];
            float scale = 1.0f / MathF.Sqrt(_headDim);

            for (int head = 0; head < _numHeads; head++)
            {
                int offset = head * _headDim;

                for (int q = 0; q < queryLen; q++)
                {
                    var scores = new float[keyLen];
                    float max = float.NegativeInfinity;

                    for (int k = 0; k < keyLen; k++)
                    {
                        if (mask != null && !mask[q, k])
                        {
                            scores[k] = float.NegativeInfinity;
                            continue;
                        }

                        float dot = 0f;
                        for (int d = 0; d < _headDim; d++)
                            dot += Q[q, offset + d] * K[k, offset + d];

                        float score = dot * scale;
                        scores[k] = score;
                        if (score > max) max = score;
                    }

                    if (float.IsNegativeInfinity(max))
                        continue; // Entire row masked: output remains zero before WO/BiasO.

                    var weights = new float[keyLen];
                    float sum = 0f;
                    for (int k = 0; k < keyLen; k++)
                    {
                        if (float.IsNegativeInfinity(scores[k]))
                        {
                            weights[k] = 0f;
                            continue;
                        }

                        float w = MathF.Exp(scores[k] - max);
                        weights[k] = w;
                        sum += w;
                    }

                    if (sum <= 0f || float.IsNaN(sum) || float.IsInfinity(sum))
                        continue;

                    float invSum = 1.0f / sum;
                    for (int d = 0; d < _headDim; d++)
                    {
                        float value = 0f;
                        for (int k = 0; k < keyLen; k++)
                            value += weights[k] * invSum * V[k, offset + d];

                        concatenated[q, offset + d] = value;
                    }
                }
            }

            return MatMulWithBias(concatenated, WO, BiasO);
        }

        private float[,] MatMulWithBias(float[,] input, float[,] weights, float[] bias)
        {
            if (input == null) throw new ArgumentNullException(nameof(input));
            if (weights == null) throw new ArgumentNullException(nameof(weights));
            if (bias == null) throw new ArgumentNullException(nameof(bias));

            int inputDim = input.GetLength(1);
            int weightInputDim = weights.GetLength(1);
            int outputDim = weights.GetLength(0);

            if (inputDim != weightInputDim)
                throw new ArgumentException($"Input width {inputDim} does not match weight input width {weightInputDim}.");
            if (bias.Length != outputDim)
                throw new ArgumentException($"Bias length {bias.Length} does not match output dimension {outputDim}.", nameof(bias));

            return _accel.MatrixAddBias(_accel.BatchDotProduct(weights, input), bias);
        }

        private void ValidateInputWidth(float[,] matrix, string name)
        {
            if (matrix.GetLength(1) != _embeddingDim)
                throw new ArgumentException($"{name} width must be {_embeddingDim}, got {matrix.GetLength(1)}.", name);
        }

        private static void ValidateNonEmptySequence(int length, string name)
        {
            if (length <= 0)
                throw new ArgumentException("Sequence length must be at least 1.", name);
        }

        private static void ValidateMask(bool[,] mask, int expectedRows, int expectedCols)
        {
            if (mask == null) return;
            if (mask.GetLength(0) != expectedRows || mask.GetLength(1) != expectedCols)
                throw new ArgumentException($"Mask shape must be [{expectedRows},{expectedCols}], got [{mask.GetLength(0)},{mask.GetLength(1)}].", nameof(mask));
        }
    }
}
