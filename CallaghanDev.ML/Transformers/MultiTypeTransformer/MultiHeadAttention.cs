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
            _embeddingDim = embeddingDim;
            _numHeads = numHeads;
            _headDim = embeddingDim / numHeads;
            _accel = accel;

            if (embeddingDim % numHeads != 0)
                throw new ArgumentException("Embedding dimension must be divisible by number of heads");

            if ((_headDim & 1) != 0)
                throw new ArgumentException("RoPE requires an even per-head dimension");

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
            var weights = new float[rows, cols];
            float std = MathF.Sqrt(2.0f / (rows + cols));

            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < cols; j++)
                {
                    float u1 = 1.0f - random.NextSingle();
                    float u2 = 1.0f - random.NextSingle();
                    float z = MathF.Sqrt(-2.0f * MathF.Log(u1)) * MathF.Cos(2.0f * MathF.PI * u2);
                    weights[i, j] = z * std;
                }
            }

            return weights;
        }

        public float[,] Forward(float[,] input, bool[,] mask = null)
        {
            int seqLen = input.GetLength(0);

            var Q = MatMulWithBias(input, WQ, BiasQ);
            var K = MatMulWithBias(input, WK, BiasK);
            var V = MatMulWithBias(input, WV, BiasV);

            RotaryPositionEmbedding.ApplyInPlace(Q, K, _numHeads);

            var output = new float[seqLen, _embeddingDim];

            for (int head = 0; head < _numHeads; head++)
            {
                int startIdx = head * _headDim;

                var Q_head = ExtractHead(Q, startIdx, _headDim);
                var K_head = ExtractHead(K, startIdx, _headDim);
                var V_head = ExtractHead(V, startIdx, _headDim);

                var scores = _accel.MatrixMultiplyTranspose(Q_head, K_head);
                var scaledScores = _accel.MatrixScale(scores, 1.0f / MathF.Sqrt(_headDim));
                var attention = _accel.Softmax(scaledScores, mask);
                var headOutput = _accel.MatrixMultiply(attention, V_head);

                CopyHead(headOutput, output, startIdx, _headDim);
            }

            return MatMulWithBias(output, WO, BiasO);
        }

        public float[,] Forward(float[,] query, float[,] keyValue, bool[,] mask = null)
        {
            int seqLen = query.GetLength(0);

            var Q = MatMulWithBias(query, WQ, BiasQ);
            var K = MatMulWithBias(keyValue, WK, BiasK);
            var V = MatMulWithBias(keyValue, WV, BiasV);

            RotaryPositionEmbedding.ApplyInPlace(Q, K, _numHeads);

            var output = new float[seqLen, _embeddingDim];
            float scale = 1.0f / MathF.Sqrt(_headDim);

            for (int head = 0; head < _numHeads; head++)
            {
                int startIdx = head * _headDim;

                var Q_head = ExtractHead(Q, startIdx, _headDim);
                var K_head = ExtractHead(K, startIdx, _headDim);
                var V_head = ExtractHead(V, startIdx, _headDim);

                var scores = _accel.MatrixMultiplyTranspose(Q_head, K_head);
                scores = _accel.MatrixScale(scores, scale);

                var attention = _accel.Softmax(scores, mask);
                var headOutput = _accel.MatrixMultiply(attention, V_head);

                CopyHead(headOutput, output, startIdx, _headDim);
            }

            return MatMulWithBias(output, WO, BiasO);
        }

        private float[,] MatMulWithBias(float[,] input, float[,] weights, float[] bias)
        {
            int seqLen = input.GetLength(0);
            int outputDim = weights.GetLength(1);
            var result = new float[seqLen, outputDim];

            for (int i = 0; i < seqLen; i++)
            {
                var inputRow = new float[input.GetLength(1)];
                for (int k = 0; k < input.GetLength(1); k++)
                    inputRow[k] = input[i, k];

                var outputRow = _accel.CalculateDotProduct(weights, inputRow);

                for (int j = 0; j < outputDim; j++)
                    result[i, j] = outputRow[j] + bias[j];
            }

            return result;
        }

        private float[,] ExtractHead(float[,] matrix, int startIdx, int headDim)
        {
            int seqLen = matrix.GetLength(0);
            var result = new float[seqLen, headDim];

            for (int i = 0; i < seqLen; i++)
                for (int j = 0; j < headDim; j++)
                    result[i, j] = matrix[i, startIdx + j];

            return result;
        }

        private void CopyHead(float[,] headOutput, float[,] fullOutput, int startIdx, int headDim)
        {
            int seqLen = headOutput.GetLength(0);

            for (int i = 0; i < seqLen; i++)
                for (int j = 0; j < headDim; j++)
                    fullOutput[i, startIdx + j] = headOutput[i, j];
        }
    }
}