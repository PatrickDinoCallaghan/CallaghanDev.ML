using CallaghanDev.ML.Transformers.TACAMT;
using ILGPU;
using ILGPU.Algorithms;
using ILGPU.Runtime;
using System;

namespace CallaghanDev.ML.AccelerationManagers.GPU
{
    public partial class AccelerationGPU : IAccelerationManager, IDisposable
    {
        private Action<Index2D, ArrayView2D<float, Stride2D.DenseX>, float> _matScaleInPlaceKernel;
        private Action<Index1D, ArrayView1D<float, Stride1D.Dense>, float> _vecScaleInPlaceKernel;
        private Action<Index2D, ArrayView2D<float, Stride2D.DenseX>, ArrayView2D<float, Stride2D.DenseX>, float> _matUpdateKernel;
        private Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, float> _vecUpdateKernel;
        private Action<Index2D, ArrayView2D<float, Stride2D.DenseX>, ArrayView1D<int, Stride1D.Dense>, ArrayView2D<float, Stride2D.DenseX>, ArrayView2D<float, Stride2D.DenseX>> _embedTokensKernel;
        private Action<Index2D, ArrayView2D<float, Stride2D.DenseX>, ArrayView1D<float, Stride1D.Dense>, ArrayView2D<float, Stride2D.DenseX>, ArrayView2D<float, Stride2D.DenseX>> _addBiasPosKernel;
        private Action<Index2D, ArrayView2D<float, Stride2D.DenseX>> _sigmoidInPlaceKernel;

        private Action<Index2D, ArrayView2D<float, Stride2D.DenseX>, ArrayView2D<float, Stride2D.DenseX>, ArrayView2D<float, Stride2D.DenseX>, ArrayView1D<float, Stride1D.Dense>> _backpropLinearWeightBiasKernel;
        private Action<Index2D, ArrayView2D<float, Stride2D.DenseX>, ArrayView2D<float, Stride2D.DenseX>, ArrayView2D<float, Stride2D.DenseX>, ArrayView2D<float, Stride2D.DenseX>> _backpropLinearInputKernel;
        private Action<Index2D, ArrayView2D<float, Stride2D.DenseX>, ArrayView2D<float, Stride2D.DenseX>, ArrayView2D<float, Stride2D.DenseX>, ArrayView1D<float, Stride1D.Dense>> _backpropOutputWeightBiasKernel;
        private Action<Index2D, ArrayView2D<float, Stride2D.DenseX>, ArrayView2D<float, Stride2D.DenseX>, ArrayView2D<float, Stride2D.DenseX>> _backpropOutputInputKernel;
        private Action<Index2D, ArrayView2D<float, Stride2D.DenseX>, ArrayView2D<float, Stride2D.DenseX>, ArrayView2D<float, Stride2D.DenseX>> _backpropInputProjectionWeightKernel;
        private Action<Index1D, ArrayView2D<float, Stride2D.DenseX>, ArrayView1D<float, Stride1D.Dense>> _backpropInputProjectionBiasKernel;
        private Action<Index2D, ArrayView2D<float, Stride2D.DenseX>, ArrayView2D<float, Stride2D.DenseX>, ArrayView1D<int, Stride1D.Dense>> _accumulateTokenEmbeddingGradKernel;
        private Action<Index2D, ArrayView2D<float, Stride2D.DenseX>, ArrayView1D<float, Stride1D.Dense>> _matrixSquaredNormKernel;
        private Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>> _vectorSquaredNormKernel;
        private Action<Index1D, ArrayView2D<float, Stride2D.DenseX>, ArrayView1D<int, Stride1D.Dense>, ArrayView2D<float, Stride2D.DenseX>, ArrayView1D<float, Stride1D.Dense>, int> _crossEntropyLossGradientKernel;
        private Action<Index2D, ArrayView2D<float, Stride2D.DenseX>, ArrayView2D<float, Stride2D.DenseX>, ArrayView2D<float, Stride2D.DenseX>, ArrayView1D<float, Stride1D.Dense>, float> _mseLossGradientKernel;
        private Action<Index3D, ArrayView2D<float, Stride2D.DenseX>, ArrayView2D<float, Stride2D.DenseX>, ArrayView2D<float, Stride2D.DenseX>, ArrayView2D<float, Stride2D.DenseX>, ArrayView3D<float, Stride3D.DenseXY>, ArrayView3D<float, Stride3D.DenseXY>, ArrayView2D<float, Stride2D.DenseX>, int> _timeDecayedDWeightsDvKernel;
        private Action<Index2D, ArrayView3D<float, Stride3D.DenseXY>, ArrayView3D<float, Stride3D.DenseXY>, ArrayView2D<float, Stride2D.DenseX>> _timeDecayedDotWeightKernel;
        private Action<Index3D, ArrayView2D<float, Stride2D.DenseX>, ArrayView2D<float, Stride2D.DenseX>, ArrayView3D<float, Stride3D.DenseXY>, ArrayView3D<float, Stride3D.DenseXY>, ArrayView2D<float, Stride2D.DenseX>, ArrayView2D<float, Stride2D.DenseX>, ArrayView2D<float, Stride2D.DenseX>, ArrayView3D<float, Stride3D.DenseXY>, int, float, int> _timeDecayedDqDkBiasKernel;

        private void InitTransformerTrainingKernels()
        {
            _matScaleInPlaceKernel = _accelerator.LoadAutoGroupedStreamKernel<Index2D, ArrayView2D<float, Stride2D.DenseX>, float>(MatScaleInPlaceKernel);
            _vecScaleInPlaceKernel = _accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<float, Stride1D.Dense>, float>(VecScaleInPlaceKernel);
            _matUpdateKernel = _accelerator.LoadAutoGroupedStreamKernel<Index2D, ArrayView2D<float, Stride2D.DenseX>, ArrayView2D<float, Stride2D.DenseX>, float>(MatUpdateKernel);
            _vecUpdateKernel = _accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, float>(VecUpdateKernel);
            _embedTokensKernel = _accelerator.LoadAutoGroupedStreamKernel<Index2D, ArrayView2D<float, Stride2D.DenseX>, ArrayView1D<int, Stride1D.Dense>, ArrayView2D<float, Stride2D.DenseX>, ArrayView2D<float, Stride2D.DenseX>>(EmbedTokensKernel);
            _addBiasPosKernel = _accelerator.LoadAutoGroupedStreamKernel<Index2D, ArrayView2D<float, Stride2D.DenseX>, ArrayView1D<float, Stride1D.Dense>, ArrayView2D<float, Stride2D.DenseX>, ArrayView2D<float, Stride2D.DenseX>>(AddBiasPosKernel);
            _sigmoidInPlaceKernel = _accelerator.LoadAutoGroupedStreamKernel<Index2D, ArrayView2D<float, Stride2D.DenseX>>(SigmoidInPlaceKernel);

            _backpropLinearWeightBiasKernel = _accelerator.LoadAutoGroupedStreamKernel<Index2D, ArrayView2D<float, Stride2D.DenseX>, ArrayView2D<float, Stride2D.DenseX>, ArrayView2D<float, Stride2D.DenseX>, ArrayView1D<float, Stride1D.Dense>>(BackpropLinearWeightBiasKernel);
            _backpropLinearInputKernel = _accelerator.LoadAutoGroupedStreamKernel<Index2D, ArrayView2D<float, Stride2D.DenseX>, ArrayView2D<float, Stride2D.DenseX>, ArrayView2D<float, Stride2D.DenseX>, ArrayView2D<float, Stride2D.DenseX>>(BackpropLinearInputKernel);
            _backpropOutputWeightBiasKernel = _accelerator.LoadAutoGroupedStreamKernel<Index2D, ArrayView2D<float, Stride2D.DenseX>, ArrayView2D<float, Stride2D.DenseX>, ArrayView2D<float, Stride2D.DenseX>, ArrayView1D<float, Stride1D.Dense>>(BackpropOutputWeightBiasKernel);
            _backpropOutputInputKernel = _accelerator.LoadAutoGroupedStreamKernel<Index2D, ArrayView2D<float, Stride2D.DenseX>, ArrayView2D<float, Stride2D.DenseX>, ArrayView2D<float, Stride2D.DenseX>>(BackpropOutputInputKernel);
            _backpropInputProjectionWeightKernel = _accelerator.LoadAutoGroupedStreamKernel<Index2D, ArrayView2D<float, Stride2D.DenseX>, ArrayView2D<float, Stride2D.DenseX>, ArrayView2D<float, Stride2D.DenseX>>(BackpropInputProjectionWeightKernel);
            _backpropInputProjectionBiasKernel = _accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView2D<float, Stride2D.DenseX>, ArrayView1D<float, Stride1D.Dense>>(BackpropInputProjectionBiasKernel);
            _accumulateTokenEmbeddingGradKernel = _accelerator.LoadAutoGroupedStreamKernel<Index2D, ArrayView2D<float, Stride2D.DenseX>, ArrayView2D<float, Stride2D.DenseX>, ArrayView1D<int, Stride1D.Dense>>(AccumulateTokenEmbeddingGradKernel);
            _matrixSquaredNormKernel = _accelerator.LoadAutoGroupedStreamKernel<Index2D, ArrayView2D<float, Stride2D.DenseX>, ArrayView1D<float, Stride1D.Dense>>(MatrixSquaredNormKernel);
            _vectorSquaredNormKernel = _accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>>(VectorSquaredNormKernel);
            _crossEntropyLossGradientKernel = _accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView2D<float, Stride2D.DenseX>, ArrayView1D<int, Stride1D.Dense>, ArrayView2D<float, Stride2D.DenseX>, ArrayView1D<float, Stride1D.Dense>, int>(CrossEntropyLossGradientKernel);
            _mseLossGradientKernel = _accelerator.LoadAutoGroupedStreamKernel<Index2D, ArrayView2D<float, Stride2D.DenseX>, ArrayView2D<float, Stride2D.DenseX>, ArrayView2D<float, Stride2D.DenseX>, ArrayView1D<float, Stride1D.Dense>, float>(MseLossGradientKernel);
            _timeDecayedDWeightsDvKernel = _accelerator.LoadAutoGroupedStreamKernel<Index3D, ArrayView2D<float, Stride2D.DenseX>, ArrayView2D<float, Stride2D.DenseX>, ArrayView2D<float, Stride2D.DenseX>, ArrayView2D<float, Stride2D.DenseX>, ArrayView3D<float, Stride3D.DenseXY>, ArrayView3D<float, Stride3D.DenseXY>, ArrayView2D<float, Stride2D.DenseX>, int>(TimeDecayedDWeightsDvKernel);
            _timeDecayedDotWeightKernel = _accelerator.LoadAutoGroupedStreamKernel<Index2D, ArrayView3D<float, Stride3D.DenseXY>, ArrayView3D<float, Stride3D.DenseXY>, ArrayView2D<float, Stride2D.DenseX>>(TimeDecayedDotWeightKernel);
            _timeDecayedDqDkBiasKernel = _accelerator.LoadAutoGroupedStreamKernel<Index3D, ArrayView2D<float, Stride2D.DenseX>, ArrayView2D<float, Stride2D.DenseX>, ArrayView3D<float, Stride3D.DenseXY>, ArrayView3D<float, Stride3D.DenseXY>, ArrayView2D<float, Stride2D.DenseX>, ArrayView2D<float, Stride2D.DenseX>, ArrayView2D<float, Stride2D.DenseX>, ArrayView3D<float, Stride3D.DenseXY>, int, float, int>(TimeDecayedDqDkBiasKernel);
        }

        private void DisposeTransformerTrainingBuffers()
        {
        }

        #region Transformer training kernels

        private static void MatScaleInPlaceKernel(Index2D idx, ArrayView2D<float, Stride2D.DenseX> mat, float scale)
        {
            mat[idx] *= scale;
        }

        private static void VecScaleInPlaceKernel(Index1D i, ArrayView1D<float, Stride1D.Dense> vec, float scale)
        {
            vec[i] *= scale;
        }

        private static void MatUpdateKernel(Index2D idx, ArrayView2D<float, Stride2D.DenseX> weights, ArrayView2D<float, Stride2D.DenseX> gradients, float learningRate)
        {
            weights[idx] -= learningRate * gradients[idx];
        }

        private static void VecUpdateKernel(Index1D i, ArrayView1D<float, Stride1D.Dense> weights, ArrayView1D<float, Stride1D.Dense> gradients, float learningRate)
        {
            weights[i] -= learningRate * gradients[i];
        }

        private static void EmbedTokensKernel(Index2D idx, ArrayView2D<float, Stride2D.DenseX> tokenEmbedding, ArrayView1D<int, Stride1D.Dense> tokenIds, ArrayView2D<float, Stride2D.DenseX> positionalEncoding, ArrayView2D<float, Stride2D.DenseX> result)
        {
            int i = idx.X;
            int j = idx.Y;
            int tokenId = tokenIds[i];
            result[idx] = tokenEmbedding[tokenId, j] + positionalEncoding[i, j];
        }

        private static void AddBiasPosKernel(Index2D idx, ArrayView2D<float, Stride2D.DenseX> projected, ArrayView1D<float, Stride1D.Dense> bias, ArrayView2D<float, Stride2D.DenseX> positionalEncoding, ArrayView2D<float, Stride2D.DenseX> result)
        {
            int j = idx.Y;
            result[idx] = projected[idx] + bias[j] + positionalEncoding[idx];
        }

        private static void SigmoidInPlaceKernel(Index2D idx, ArrayView2D<float, Stride2D.DenseX> matrix)
        {
            float x = matrix[idx];

            if (x >= 0.0f)
            {
                float ex = XMath.Exp(-x);
                matrix[idx] = 1.0f / (1.0f + ex);
            }
            else
            {
                float ex = XMath.Exp(x);
                matrix[idx] = ex / (1.0f + ex);
            }
        }

        private static void BackpropLinearWeightBiasKernel(Index2D idx, ArrayView2D<float, Stride2D.DenseX> input, ArrayView2D<float, Stride2D.DenseX> dOutput, ArrayView2D<float, Stride2D.DenseX> weightGrad, ArrayView1D<float, Stride1D.Dense> biasGrad)
        {
            int i = idx.X;
            int j = idx.Y;
            int inDim = (int)input.Extent.Y;
            float dOutVal = dOutput[i, j];

            for (int k = 0; k < inDim; k++)
            {
                Atomic.Add(ref weightGrad[j, k], dOutVal * input[i, k]);
            }

            Atomic.Add(ref biasGrad[j], dOutVal);
        }

        private static void BackpropLinearInputKernel(Index2D idx, ArrayView2D<float, Stride2D.DenseX> dOutput, ArrayView2D<float, Stride2D.DenseX> weights, ArrayView2D<float, Stride2D.DenseX> dInput, ArrayView2D<float, Stride2D.DenseX> input)
        {
            int i = idx.X;
            int k = idx.Y;
            int outDim = (int)dOutput.Extent.Y;
            float sum = 0.0f;

            for (int j = 0; j < outDim; j++)
            {
                sum += dOutput[i, j] * weights[j, k];
            }

            dInput[i, k] += sum;
        }

        private static void BackpropOutputWeightBiasKernel(Index2D idx, ArrayView2D<float, Stride2D.DenseX> dLogits, ArrayView2D<float, Stride2D.DenseX> input, ArrayView2D<float, Stride2D.DenseX> weightGrad, ArrayView1D<float, Stride1D.Dense> biasGrad)
        {
            int i = idx.X;
            int v = idx.Y;
            int embeddingDim = (int)input.Extent.Y;
            float dVal = dLogits[i, v];

            for (int e = 0; e < embeddingDim; e++)
            {
                Atomic.Add(ref weightGrad[v, e], input[i, e] * dVal);
            }

            Atomic.Add(ref biasGrad[v], dVal);
        }

        private static void BackpropOutputInputKernel(Index2D idx, ArrayView2D<float, Stride2D.DenseX> dLogits, ArrayView2D<float, Stride2D.DenseX> weights, ArrayView2D<float, Stride2D.DenseX> dX)
        {
            int i = idx.X;
            int e = idx.Y;
            int outputDim = (int)dLogits.Extent.Y;
            float grad = 0.0f;

            for (int v = 0; v < outputDim; v++)
            {
                grad += dLogits[i, v] * weights[v, e];
            }

            dX[i, e] = grad;
        }

        private static void BackpropInputProjectionWeightKernel(Index2D idx, ArrayView2D<float, Stride2D.DenseX> dX, ArrayView2D<float, Stride2D.DenseX> continuousInput, ArrayView2D<float, Stride2D.DenseX> weightGrad)
        {
            int e = idx.X;
            int f = idx.Y;
            int seqLen = (int)dX.Extent.X;
            float sum = 0.0f;

            for (int i = 0; i < seqLen; i++)
            {
                sum += dX[i, e] * continuousInput[i, f];
            }

            weightGrad[e, f] += sum;
        }

        private static void BackpropInputProjectionBiasKernel(Index1D e, ArrayView2D<float, Stride2D.DenseX> dX, ArrayView1D<float, Stride1D.Dense> biasGrad)
        {
            int seqLen = (int)dX.Extent.X;
            float sum = 0.0f;

            for (int i = 0; i < seqLen; i++)
            {
                sum += dX[i, e];
            }

            biasGrad[e] += sum;
        }

        private static void AccumulateTokenEmbeddingGradKernel(Index2D idx, ArrayView2D<float, Stride2D.DenseX> embeddingGrad, ArrayView2D<float, Stride2D.DenseX> dX, ArrayView1D<int, Stride1D.Dense> tokenIds)
        {
            int i = idx.X;
            int j = idx.Y;
            int tokenId = tokenIds[i];
            Atomic.Add(ref embeddingGrad[tokenId, j], dX[i, j]);
        }

        private static void MatrixSquaredNormKernel(Index2D idx, ArrayView2D<float, Stride2D.DenseX> matrix, ArrayView1D<float, Stride1D.Dense> sum)
        {
            float value = matrix[idx];
            Atomic.Add(ref sum[0], value * value);
        }

        private static void VectorSquaredNormKernel(Index1D i, ArrayView1D<float, Stride1D.Dense> vector, ArrayView1D<float, Stride1D.Dense> sum)
        {
            float value = vector[i];
            Atomic.Add(ref sum[0], value * value);
        }

        private static void CrossEntropyLossGradientKernel(Index1D row, ArrayView2D<float, Stride2D.DenseX> logits, ArrayView1D<int, Stride1D.Dense> targets, ArrayView2D<float, Stride2D.DenseX> dLogits, ArrayView1D<float, Stride1D.Dense> lossContrib, int effectiveLen)
        {
            int outputDim = (int)logits.Extent.Y;
            int i = row;
            float maxVal = float.NegativeInfinity;

            for (int j = 0; j < outputDim; j++)
            {
                float value = logits[i, j];
                if (value > maxVal)
                {
                    maxVal = value;
                }
            }

            float sumExp = 0.0f;
            for (int j = 0; j < outputDim; j++)
            {
                float exp = XMath.Exp(logits[i, j] - maxVal);
                dLogits[i, j] = exp;
                sumExp += exp;
            }

            float invSum = sumExp > 0.0f ? 1.0f / sumExp : 0.0f;
            float invLen = 1.0f / effectiveLen;
            int target = targets[i];
            float targetProb = dLogits[i, target] * invSum;
            lossContrib[i] = -XMath.Log(targetProb + 1e-10f);

            for (int j = 0; j < outputDim; j++)
            {
                float probability = dLogits[i, j] * invSum;
                dLogits[i, j] = probability * invLen;

                if (j == target)
                {
                    dLogits[i, j] -= invLen;
                }
            }
        }

        private static void MseLossGradientKernel(Index2D idx, ArrayView2D<float, Stride2D.DenseX> predictions, ArrayView2D<float, Stride2D.DenseX> targets, ArrayView2D<float, Stride2D.DenseX> dOutput, ArrayView1D<float, Stride1D.Dense> lossSum, float invLen)
        {
            float diff = predictions[idx] - targets[idx];
            Atomic.Add(ref lossSum[0], diff * diff);
            dOutput[idx] = 2.0f * diff * invLen;
        }

        private static void TimeDecayedDWeightsDvKernel(Index3D idx, ArrayView2D<float, Stride2D.DenseX> q, ArrayView2D<float, Stride2D.DenseX> k, ArrayView2D<float, Stride2D.DenseX> v, ArrayView2D<float, Stride2D.DenseX> dOutput, ArrayView3D<float, Stride3D.DenseXY> weights, ArrayView3D<float, Stride3D.DenseXY> dWeights, ArrayView2D<float, Stride2D.DenseX> dV, int headDim)
        {
            int h = idx.X;
            int p = idx.Y;
            int s = idx.Z;
            int offset = h * headDim;
            float dot = 0.0f;
            float w = weights[h, p, s];

            for (int d = 0; d < headDim; d++)
            {
                int di = offset + d;
                float dout = dOutput[p, di];
                Atomic.Add(ref dV[s, di], w * dout);
                dot += dout * v[s, di];
            }

            dWeights[h, p, s] = dot;
        }

        private static void TimeDecayedDotWeightKernel(Index2D idx, ArrayView3D<float, Stride3D.DenseXY> weights, ArrayView3D<float, Stride3D.DenseXY> dWeights, ArrayView2D<float, Stride2D.DenseX> dotWeightGrad)
        {
            int h = idx.X;
            int p = idx.Y;
            int keyLen = (int)weights.Extent.Z;
            float dot = 0.0f;

            for (int s = 0; s < keyLen; s++)
            {
                dot += weights[h, p, s] * dWeights[h, p, s];
            }

            dotWeightGrad[h, p] = dot;
        }

        private static void TimeDecayedDqDkBiasKernel(Index3D idx, ArrayView2D<float, Stride2D.DenseX> q, ArrayView2D<float, Stride2D.DenseX> k, ArrayView3D<float, Stride3D.DenseXY> weights, ArrayView3D<float, Stride3D.DenseXY> dWeights, ArrayView2D<float, Stride2D.DenseX> dotWeightGrad, ArrayView2D<float, Stride2D.DenseX> dQ, ArrayView2D<float, Stride2D.DenseX> dK, ArrayView3D<float, Stride3D.DenseXY> dDecayBias, int headDim, float scale, int hasDecayBias)
        {
            int h = idx.X;
            int p = idx.Y;
            int s = idx.Z;
            int offset = h * headDim;
            float dScore = weights[h, p, s] * (dWeights[h, p, s] - dotWeightGrad[h, p]);

            for (int d = 0; d < headDim; d++)
            {
                int di = offset + d;
                Atomic.Add(ref dQ[p, di], dScore * scale * k[s, di]);
                Atomic.Add(ref dK[s, di], dScore * scale * q[p, di]);
            }

            if (hasDecayBias != 0)
            {
                dDecayBias[p, s, h] = dScore;
            }
        }

        #endregion

        #region Transformer training methods

        public void BackpropLinearProjection(float[,] input, float[,] dOutput, float[,] weights, float[,] weightGrad, float[] biasGrad, float[,] dInput)
        {
            if (input == null) throw new ArgumentNullException(nameof(input));
            if (dOutput == null) throw new ArgumentNullException(nameof(dOutput));
            if (weights == null) throw new ArgumentNullException(nameof(weights));
            if (weightGrad == null) throw new ArgumentNullException(nameof(weightGrad));
            if (biasGrad == null) throw new ArgumentNullException(nameof(biasGrad));
            if (dInput == null) throw new ArgumentNullException(nameof(dInput));

            int seqLen = input.GetLength(0);
            int inDim = input.GetLength(1);
            int outDim = dOutput.GetLength(1);

            if (dOutput.GetLength(0) != seqLen || weights.GetLength(0) != outDim || weights.GetLength(1) != inDim || weightGrad.GetLength(0) != outDim || weightGrad.GetLength(1) != inDim || biasGrad.Length < outDim || dInput.GetLength(0) != seqLen || dInput.GetLength(1) != inDim)
            {
                throw new ArgumentException("BackpropLinearProjection dimensions do not match.");
            }

            if (!ShouldUseGpu((long)seqLen * outDim * inDim, GPU_MATMUL_OP_THRESHOLD))
            {
                _mutliThreadCPU.BackpropLinearProjection(input, dOutput, weights, weightGrad, biasGrad, dInput);
                return;
            }

            var bufInput = _accelerator.Allocate2DDenseX<float>(new Index2D(seqLen, inDim));
            var bufDOutput = _accelerator.Allocate2DDenseX<float>(new Index2D(seqLen, outDim));
            var bufWeights = _accelerator.Allocate2DDenseX<float>(new Index2D(outDim, inDim));
            var bufWeightGrad = _accelerator.Allocate2DDenseX<float>(new Index2D(outDim, inDim));
            var bufBiasGrad = _accelerator.Allocate1D<float>(biasGrad.Length);
            var bufDInput = _accelerator.Allocate2DDenseX<float>(new Index2D(seqLen, inDim));

            try
            {
                bufInput.CopyFromCPU(input);
                bufDOutput.CopyFromCPU(dOutput);
                bufWeights.CopyFromCPU(weights);
                bufWeightGrad.CopyFromCPU(weightGrad);
                bufBiasGrad.CopyFromCPU(biasGrad);
                bufDInput.CopyFromCPU(dInput);

                _backpropLinearWeightBiasKernel(new Index2D(seqLen, outDim), bufInput.View, bufDOutput.View, bufWeightGrad.View, bufBiasGrad.View);
                _backpropLinearInputKernel(new Index2D(seqLen, inDim), bufDOutput.View, bufWeights.View, bufDInput.View, bufInput.View);

                bufWeightGrad.CopyToCPU(weightGrad);
                bufBiasGrad.CopyToCPU(biasGrad);
                bufDInput.CopyToCPU(dInput);
            }
            finally
            {
                bufInput.Dispose();
                bufDOutput.Dispose();
                bufWeights.Dispose();
                bufWeightGrad.Dispose();
                bufBiasGrad.Dispose();
                bufDInput.Dispose();
            }
        }

        public float[,] FFNForwardBatch(float[,] input, int seqLen, int outputDim, Func<float[], float[]> forwardPassFn)
        {
            return _mutliThreadCPU.FFNForwardBatch(input, seqLen, outputDim, forwardPassFn);
        }

        public (float loss, float[,] dLogits) CrossEntropyLossAndGradient(float[,] logits, int[] targets, int effectiveLen)
        {
            if (logits == null) throw new ArgumentNullException(nameof(logits));
            if (targets == null) throw new ArgumentNullException(nameof(targets));
            if (effectiveLen <= 0) throw new ArgumentOutOfRangeException(nameof(effectiveLen));

            int rows = logits.GetLength(0);
            int outputDim = logits.GetLength(1);

            if (effectiveLen > rows || effectiveLen > targets.Length)
            {
                throw new ArgumentException("effectiveLen exceeds logits/targets length.");
            }

            for (int i = 0; i < effectiveLen; i++)
            {
                if ((uint)targets[i] >= (uint)outputDim)
                {
                    throw new ArgumentOutOfRangeException(nameof(targets), $"Target index {targets[i]} at row {i} is outside [0,{outputDim}).");
                }
            }

            if (!ShouldUseGpu((long)effectiveLen * outputDim))
            {
                return _mutliThreadCPU.CrossEntropyLossAndGradient(logits, targets, effectiveLen);
            }

            var bufLogits = _accelerator.Allocate2DDenseX<float>(new Index2D(rows, outputDim));
            var bufTargets = _accelerator.Allocate1D<int>(targets.Length);
            var bufDLogits = _accelerator.Allocate2DDenseX<float>(new Index2D(rows, outputDim));
            var bufLoss = _accelerator.Allocate1D<float>(effectiveLen);

            try
            {
                bufLogits.CopyFromCPU(logits);
                bufTargets.CopyFromCPU(targets);
                bufDLogits.CopyFromCPU(new float[rows, outputDim]);
                _crossEntropyLossGradientKernel(new Index1D(effectiveLen), bufLogits.View, bufTargets.View, bufDLogits.View, bufLoss.View, effectiveLen);

                var lossParts = new float[effectiveLen];
                var dLogits = new float[rows, outputDim];
                bufLoss.CopyToCPU(lossParts);
                bufDLogits.CopyToCPU(dLogits);

                float loss = 0.0f;
                for (int i = 0; i < effectiveLen; i++)
                {
                    loss += lossParts[i];
                }

                loss /= effectiveLen;
                return (loss, dLogits);
            }
            finally
            {
                bufLogits.Dispose();
                bufTargets.Dispose();
                bufDLogits.Dispose();
                bufLoss.Dispose();
            }
        }

        public (float loss, float[,] dOutput) MSELossAndGradient(float[,] predictions, float[,] targets, int effectiveLen)
        {
            if (predictions == null) throw new ArgumentNullException(nameof(predictions));
            if (targets == null) throw new ArgumentNullException(nameof(targets));
            if (effectiveLen <= 0) throw new ArgumentOutOfRangeException(nameof(effectiveLen));

            int rows = predictions.GetLength(0);
            int outputDim = predictions.GetLength(1);

            if (targets.GetLength(0) != rows || targets.GetLength(1) != outputDim || effectiveLen > rows)
            {
                throw new ArgumentException("MSE dimensions do not match.");
            }

            if (!ShouldUseGpu((long)effectiveLen * outputDim))
            {
                return _mutliThreadCPU.MSELossAndGradient(predictions, targets, effectiveLen);
            }

            var bufPred = _accelerator.Allocate2DDenseX<float>(new Index2D(effectiveLen, outputDim));
            var bufTargets = _accelerator.Allocate2DDenseX<float>(new Index2D(effectiveLen, outputDim));
            var bufDOutput = _accelerator.Allocate2DDenseX<float>(new Index2D(effectiveLen, outputDim));
            var bufLoss = _accelerator.Allocate1D<float>(1);
            float invLen = 1.0f / (effectiveLen * outputDim);

            try
            {
                var predSlice = new float[effectiveLen, outputDim];
                var targetSlice = new float[effectiveLen, outputDim];

                for (int i = 0; i < effectiveLen; i++)
                {
                    for (int j = 0; j < outputDim; j++)
                    {
                        predSlice[i, j] = predictions[i, j];
                        targetSlice[i, j] = targets[i, j];
                    }
                }

                bufPred.CopyFromCPU(predSlice);
                bufTargets.CopyFromCPU(targetSlice);
                bufLoss.CopyFromCPU(new float[1]);
                _mseLossGradientKernel(new Index2D(effectiveLen, outputDim), bufPred.View, bufTargets.View, bufDOutput.View, bufLoss.View, invLen);

                var lossArray = new float[1];
                var dOutput = new float[rows, outputDim];
                var dOutputSlice = new float[effectiveLen, outputDim];
                bufLoss.CopyToCPU(lossArray);
                bufDOutput.CopyToCPU(dOutputSlice);

                for (int i = 0; i < effectiveLen; i++)
                {
                    for (int j = 0; j < outputDim; j++)
                    {
                        dOutput[i, j] = dOutputSlice[i, j];
                    }
                }

                float loss = lossArray[0] / (effectiveLen * outputDim);
                return (loss, dOutput);
            }
            finally
            {
                bufPred.Dispose();
                bufTargets.Dispose();
                bufDOutput.Dispose();
                bufLoss.Dispose();
            }
        }

        public float[,] BackpropOutputProjection(float[,] dLogits, float[,] input, float[,] weights, float[,] weightGrad, float[] biasGrad, int seqLen, int outputDim, int embeddingDim)
        {
            if (dLogits == null) throw new ArgumentNullException(nameof(dLogits));
            if (input == null) throw new ArgumentNullException(nameof(input));
            if (weights == null) throw new ArgumentNullException(nameof(weights));
            if (weightGrad == null) throw new ArgumentNullException(nameof(weightGrad));
            if (biasGrad == null) throw new ArgumentNullException(nameof(biasGrad));

            if (!ShouldUseGpu((long)seqLen * outputDim * embeddingDim, GPU_MATMUL_OP_THRESHOLD))
            {
                return _mutliThreadCPU.BackpropOutputProjection(dLogits, input, weights, weightGrad, biasGrad, seqLen, outputDim, embeddingDim);
            }

            var bufDLogits = _accelerator.Allocate2DDenseX<float>(new Index2D(seqLen, outputDim));
            var bufInput = _accelerator.Allocate2DDenseX<float>(new Index2D(seqLen, embeddingDim));
            var bufWeights = _accelerator.Allocate2DDenseX<float>(new Index2D(outputDim, embeddingDim));
            var bufWeightGrad = _accelerator.Allocate2DDenseX<float>(new Index2D(outputDim, embeddingDim));
            var bufBiasGrad = _accelerator.Allocate1D<float>(biasGrad.Length);
            var bufDX = _accelerator.Allocate2DDenseX<float>(new Index2D(seqLen, embeddingDim));

            try
            {
                bufDLogits.CopyFromCPU(dLogits);
                bufInput.CopyFromCPU(input);
                bufWeights.CopyFromCPU(weights);
                bufWeightGrad.CopyFromCPU(weightGrad);
                bufBiasGrad.CopyFromCPU(biasGrad);

                _backpropOutputWeightBiasKernel(new Index2D(seqLen, outputDim), bufDLogits.View, bufInput.View, bufWeightGrad.View, bufBiasGrad.View);
                _backpropOutputInputKernel(new Index2D(seqLen, embeddingDim), bufDLogits.View, bufWeights.View, bufDX.View);

                var dX = new float[seqLen, embeddingDim];
                bufDX.CopyToCPU(dX);
                bufWeightGrad.CopyToCPU(weightGrad);
                bufBiasGrad.CopyToCPU(biasGrad);
                return dX;
            }
            finally
            {
                bufDLogits.Dispose();
                bufInput.Dispose();
                bufWeights.Dispose();
                bufWeightGrad.Dispose();
                bufBiasGrad.Dispose();
                bufDX.Dispose();
            }
        }

        public void BackpropInputProjection(float[,] dX, float[,] continuousInput, float[,] weightGrad, float[] biasGrad, int seqLen, int embeddingDim, int inputFeatureDim)
        {
            if (dX == null) throw new ArgumentNullException(nameof(dX));
            if (continuousInput == null) throw new ArgumentNullException(nameof(continuousInput));
            if (weightGrad == null) throw new ArgumentNullException(nameof(weightGrad));
            if (biasGrad == null) throw new ArgumentNullException(nameof(biasGrad));

            if (!ShouldUseGpu((long)seqLen * embeddingDim * inputFeatureDim, GPU_MATMUL_OP_THRESHOLD))
            {
                _mutliThreadCPU.BackpropInputProjection(dX, continuousInput, weightGrad, biasGrad, seqLen, embeddingDim, inputFeatureDim);
                return;
            }

            var bufDX = _accelerator.Allocate2DDenseX<float>(new Index2D(seqLen, embeddingDim));
            var bufContinuous = _accelerator.Allocate2DDenseX<float>(new Index2D(seqLen, inputFeatureDim));
            var bufWeightGrad = _accelerator.Allocate2DDenseX<float>(new Index2D(embeddingDim, inputFeatureDim));
            var bufBiasGrad = _accelerator.Allocate1D<float>(biasGrad.Length);

            try
            {
                bufDX.CopyFromCPU(dX);
                bufContinuous.CopyFromCPU(continuousInput);
                bufWeightGrad.CopyFromCPU(weightGrad);
                bufBiasGrad.CopyFromCPU(biasGrad);

                _backpropInputProjectionWeightKernel(new Index2D(embeddingDim, inputFeatureDim), bufDX.View, bufContinuous.View, bufWeightGrad.View);
                _backpropInputProjectionBiasKernel(new Index1D(embeddingDim), bufDX.View, bufBiasGrad.View);

                bufWeightGrad.CopyToCPU(weightGrad);
                bufBiasGrad.CopyToCPU(biasGrad);
            }
            finally
            {
                bufDX.Dispose();
                bufContinuous.Dispose();
                bufWeightGrad.Dispose();
                bufBiasGrad.Dispose();
            }
        }

        public void AccumulateTokenEmbeddingGrad(float[,] embeddingGrad, float[,] dX, int[] tokenIds, int seqLen, int embeddingDim)
        {
            if (embeddingGrad == null) throw new ArgumentNullException(nameof(embeddingGrad));
            if (dX == null) throw new ArgumentNullException(nameof(dX));
            if (tokenIds == null) throw new ArgumentNullException(nameof(tokenIds));
            if (tokenIds.Length < seqLen) throw new ArgumentException("tokenIds length must be at least seqLen.", nameof(tokenIds));

            int vocabSize = embeddingGrad.GetLength(0);
            for (int i = 0; i < seqLen; i++)
            {
                if (tokenIds[i] < 0 || tokenIds[i] >= vocabSize)
                {
                    throw new ArgumentOutOfRangeException(nameof(tokenIds), $"Token id {tokenIds[i]} at position {i} is outside the embedding gradient rows.");
                }
            }

            if (!ShouldUseGpu((long)seqLen * embeddingDim))
            {
                _mutliThreadCPU.AccumulateTokenEmbeddingGrad(embeddingGrad, dX, tokenIds, seqLen, embeddingDim);
                return;
            }

            var dXSlice = new float[seqLen, embeddingDim];
            for (int i = 0; i < seqLen; i++)
            {
                for (int j = 0; j < embeddingDim; j++)
                {
                    dXSlice[i, j] = dX[i, j];
                }
            }

            var bufEmbeddingGrad = _accelerator.Allocate2DDenseX<float>(new Index2D(vocabSize, embeddingDim));
            var bufDX = _accelerator.Allocate2DDenseX<float>(new Index2D(seqLen, embeddingDim));
            var bufTokenIds = _accelerator.Allocate1D<int>(seqLen);

            try
            {
                var tokenIdSlice = tokenIds;
                if (tokenIds.Length != seqLen)
                {
                    tokenIdSlice = new int[seqLen];
                    Array.Copy(tokenIds, tokenIdSlice, seqLen);
                }

                bufEmbeddingGrad.CopyFromCPU(embeddingGrad);
                bufDX.CopyFromCPU(dXSlice);
                bufTokenIds.CopyFromCPU(tokenIdSlice);
                _accumulateTokenEmbeddingGradKernel(new Index2D(seqLen, embeddingDim), bufEmbeddingGrad.View, bufDX.View, bufTokenIds.View);
                bufEmbeddingGrad.CopyToCPU(embeddingGrad);
            }
            finally
            {
                bufEmbeddingGrad.Dispose();
                bufDX.Dispose();
                bufTokenIds.Dispose();
            }
        }

        public float MatrixSquaredNorm(float[,] matrix)
        {
            if (matrix == null) throw new ArgumentNullException(nameof(matrix));

            int rows = matrix.GetLength(0);
            int cols = matrix.GetLength(1);

            if (!ShouldUseGpu((long)rows * cols))
            {
                return _mutliThreadCPU.MatrixSquaredNorm(matrix);
            }

            var bufMatrix = _accelerator.Allocate2DDenseX<float>(new Index2D(rows, cols));
            var bufSum = _accelerator.Allocate1D<float>(1);

            try
            {
                bufMatrix.CopyFromCPU(matrix);
                bufSum.CopyFromCPU(new float[1]);
                _matrixSquaredNormKernel(new Index2D(rows, cols), bufMatrix.View, bufSum.View);

                var result = new float[1];
                bufSum.CopyToCPU(result);
                return result[0];
            }
            finally
            {
                bufMatrix.Dispose();
                bufSum.Dispose();
            }
        }

        public float VectorSquaredNorm(float[] vector)
        {
            if (vector == null) throw new ArgumentNullException(nameof(vector));

            int n = vector.Length;

            if (!ShouldUseGpu(n))
            {
                return _mutliThreadCPU.VectorSquaredNorm(vector);
            }

            var bufVector = _accelerator.Allocate1D<float>(n);
            var bufSum = _accelerator.Allocate1D<float>(1);

            try
            {
                bufVector.CopyFromCPU(vector);
                bufSum.CopyFromCPU(new float[1]);
                _vectorSquaredNormKernel(new Index1D(n), bufVector.View, bufSum.View);

                var result = new float[1];
                bufSum.CopyToCPU(result);
                return result[0];
            }
            finally
            {
                bufVector.Dispose();
                bufSum.Dispose();
            }
        }

        public void MatrixScaleInPlace(float[,] matrix, float scale)
        {
            int rows = matrix.GetLength(0);
            int cols = matrix.GetLength(1);

            if (!ShouldUseGpu((long)rows * cols))
            {
                _mutliThreadCPU.MatrixScaleInPlace(matrix, scale);
                return;
            }

            var buf = _accelerator.Allocate2DDenseX<float>(new Index2D(rows, cols));
            try
            {
                buf.CopyFromCPU(matrix);
                _matScaleInPlaceKernel(new Index2D(rows, cols), buf.View, scale);
                buf.CopyToCPU(matrix);
            }
            finally
            {
                buf.Dispose();
            }
        }

        public void VectorScaleInPlace(float[] vector, float scale)
        {
            int n = vector.Length;

            if (!ShouldUseGpu(n))
            {
                _mutliThreadCPU.VectorScaleInPlace(vector, scale);
                return;
            }

            var buf = _accelerator.Allocate1D<float>(n);
            try
            {
                buf.CopyFromCPU(vector);
                _vecScaleInPlaceKernel(new Index1D(n), buf.View, scale);
                buf.CopyToCPU(vector);
            }
            finally
            {
                buf.Dispose();
            }
        }

        public void MatrixUpdate(float[,] weights, float[,] gradients, float learningRate)
        {
            int rows = weights.GetLength(0);
            int cols = weights.GetLength(1);

            if (!ShouldUseGpu((long)rows * cols))
            {
                _mutliThreadCPU.MatrixUpdate(weights, gradients, learningRate);
                return;
            }

            var bufW = _accelerator.Allocate2DDenseX<float>(new Index2D(rows, cols));
            var bufG = _accelerator.Allocate2DDenseX<float>(new Index2D(rows, cols));
            try
            {
                bufW.CopyFromCPU(weights);
                bufG.CopyFromCPU(gradients);
                _matUpdateKernel(new Index2D(rows, cols), bufW.View, bufG.View, learningRate);
                bufW.CopyToCPU(weights);
            }
            finally
            {
                bufW.Dispose();
                bufG.Dispose();
            }
        }

        public void VectorUpdate(float[] weights, float[] gradients, float learningRate)
        {
            int n = weights.Length;

            if (!ShouldUseGpu(n))
            {
                _mutliThreadCPU.VectorUpdate(weights, gradients, learningRate);
                return;
            }

            var bufW = _accelerator.Allocate1D<float>(n);
            var bufG = _accelerator.Allocate1D<float>(n);
            try
            {
                bufW.CopyFromCPU(weights);
                bufG.CopyFromCPU(gradients);
                _vecUpdateKernel(new Index1D(n), bufW.View, bufG.View, learningRate);
                bufW.CopyToCPU(weights);
            }
            finally
            {
                bufW.Dispose();
                bufG.Dispose();
            }
        }

        public float[,] EmbedTokensWithPosition(float[,] tokenEmbedding, int[] tokenIds, float[,] positionalEncoding, int seqLen, int embeddingDim)
        {
            if (tokenEmbedding == null) throw new ArgumentNullException(nameof(tokenEmbedding));
            if (tokenIds == null) throw new ArgumentNullException(nameof(tokenIds));
            if (positionalEncoding == null) throw new ArgumentNullException(nameof(positionalEncoding));
            if (seqLen < 0) throw new ArgumentOutOfRangeException(nameof(seqLen));
            if (embeddingDim <= 0) throw new ArgumentOutOfRangeException(nameof(embeddingDim));
            if (tokenEmbedding.GetLength(1) != embeddingDim) throw new ArgumentException("tokenEmbedding column count must equal embeddingDim.", nameof(tokenEmbedding));
            if (positionalEncoding.GetLength(0) < seqLen || positionalEncoding.GetLength(1) < embeddingDim) throw new ArgumentException("positionalEncoding must be at least [seqLen, embeddingDim].", nameof(positionalEncoding));
            if (tokenIds.Length < seqLen) throw new ArgumentException("tokenIds length must be at least seqLen.", nameof(tokenIds));

            int vocabSize = tokenEmbedding.GetLength(0);
            for (int i = 0; i < seqLen; i++)
            {
                if (tokenIds[i] < 0 || tokenIds[i] >= vocabSize)
                {
                    throw new ArgumentOutOfRangeException(nameof(tokenIds), $"Token id {tokenIds[i]} at position {i} is outside tokenEmbedding vocabulary size {vocabSize}.");
                }
            }

            if (!ShouldUseGpu((long)seqLen * embeddingDim))
            {
                var cpuResult = new float[seqLen, embeddingDim];
                for (int i = 0; i < seqLen; i++)
                {
                    int tokenId = tokenIds[i];
                    for (int j = 0; j < embeddingDim; j++)
                    {
                        cpuResult[i, j] = tokenEmbedding[tokenId, j] + positionalEncoding[i, j];
                    }
                }

                return cpuResult;
            }

            var bufEmb = _accelerator.Allocate2DDenseX<float>(new Index2D(vocabSize, embeddingDim));
            var bufIds = _accelerator.Allocate1D<int>(seqLen);
            var bufPos = _accelerator.Allocate2DDenseX<float>(new Index2D(seqLen, embeddingDim));
            var bufOut = _accelerator.Allocate2DDenseX<float>(new Index2D(seqLen, embeddingDim));

            try
            {
                var tokenIdSlice = tokenIds;
                if (tokenIds.Length != seqLen)
                {
                    tokenIdSlice = new int[seqLen];
                    Array.Copy(tokenIds, tokenIdSlice, seqLen);
                }

                var positionalSlice = positionalEncoding;
                if (positionalEncoding.GetLength(0) != seqLen || positionalEncoding.GetLength(1) != embeddingDim)
                {
                    positionalSlice = new float[seqLen, embeddingDim];
                    for (int i = 0; i < seqLen; i++)
                    {
                        for (int j = 0; j < embeddingDim; j++)
                        {
                            positionalSlice[i, j] = positionalEncoding[i, j];
                        }
                    }
                }

                bufEmb.CopyFromCPU(tokenEmbedding);
                bufIds.CopyFromCPU(tokenIdSlice);
                bufPos.CopyFromCPU(positionalSlice);

                _embedTokensKernel(new Index2D(seqLen, embeddingDim), bufEmb.View, bufIds.View, bufPos.View, bufOut.View);

                var result = new float[seqLen, embeddingDim];
                bufOut.CopyToCPU(result);
                return result;
            }
            finally
            {
                bufEmb.Dispose();
                bufIds.Dispose();
                bufPos.Dispose();
                bufOut.Dispose();
            }
        }

        public float[,] AddBiasAndPositionalEncoding(float[,] projected, float[] bias, float[,] positionalEncoding, int seqLen, int embeddingDim)
        {
            if (!ShouldUseGpu((long)seqLen * embeddingDim))
            {
                var cpuResult = new float[seqLen, embeddingDim];
                for (int i = 0; i < seqLen; i++)
                {
                    for (int j = 0; j < embeddingDim; j++)
                    {
                        cpuResult[i, j] = projected[i, j] + bias[j] + positionalEncoding[i, j];
                    }
                }

                return cpuResult;
            }

            var bufProj = _accelerator.Allocate2DDenseX<float>(new Index2D(seqLen, embeddingDim));
            var bufBias = _accelerator.Allocate1D<float>(embeddingDim);
            var bufPos = _accelerator.Allocate2DDenseX<float>(new Index2D(seqLen, embeddingDim));
            var bufOut = _accelerator.Allocate2DDenseX<float>(new Index2D(seqLen, embeddingDim));

            try
            {
                var projectedSlice = projected;
                var positionalSlice = positionalEncoding;
                if (projected.GetLength(0) != seqLen || projected.GetLength(1) != embeddingDim)
                {
                    projectedSlice = new float[seqLen, embeddingDim];
                    for (int i = 0; i < seqLen; i++)
                    {
                        for (int j = 0; j < embeddingDim; j++)
                        {
                            projectedSlice[i, j] = projected[i, j];
                        }
                    }
                }
                if (positionalEncoding.GetLength(0) != seqLen || positionalEncoding.GetLength(1) != embeddingDim)
                {
                    positionalSlice = new float[seqLen, embeddingDim];
                    for (int i = 0; i < seqLen; i++)
                    {
                        for (int j = 0; j < embeddingDim; j++)
                        {
                            positionalSlice[i, j] = positionalEncoding[i, j];
                        }
                    }
                }

                bufProj.CopyFromCPU(projectedSlice);
                bufBias.CopyFromCPU(bias);
                bufPos.CopyFromCPU(positionalSlice);

                _addBiasPosKernel(new Index2D(seqLen, embeddingDim), bufProj.View, bufBias.View, bufPos.View, bufOut.View);

                var result = new float[seqLen, embeddingDim];
                bufOut.CopyToCPU(result);
                return result;
            }
            finally
            {
                bufProj.Dispose();
                bufBias.Dispose();
                bufPos.Dispose();
                bufOut.Dispose();
            }
        }

        public void SigmoidInPlace(float[,] matrix)
        {
            int rows = matrix.GetLength(0);
            int cols = matrix.GetLength(1);

            if (!ShouldUseGpu((long)rows * cols))
            {
                _mutliThreadCPU.SigmoidInPlace(matrix);
                return;
            }

            var buf = _accelerator.Allocate2DDenseX<float>(new Index2D(rows, cols));

            try
            {
                buf.CopyFromCPU(matrix);
                _sigmoidInPlaceKernel(new Index2D(rows, cols), buf.View);
                buf.CopyToCPU(matrix);
            }
            finally
            {
                buf.Dispose();
            }
        }

        public (float[,] dQ, float[,] dK, float[,] dV, float[,,] dDecayBias) BackpropTimeDecayedAttention(
            float[,] q,
            float[,] k,
            float[,] v,
            float[,] dOutput,
            float[][,] attentionWeights,
            float[,] timeDiffs,
            int embeddingDim,
            int numHeads)
        {
            if (q == null) throw new ArgumentNullException(nameof(q));
            if (k == null) throw new ArgumentNullException(nameof(k));
            if (v == null) throw new ArgumentNullException(nameof(v));
            if (dOutput == null) throw new ArgumentNullException(nameof(dOutput));
            if (attentionWeights == null) throw new ArgumentNullException(nameof(attentionWeights));
            if (numHeads <= 0) throw new ArgumentOutOfRangeException(nameof(numHeads));
            if (embeddingDim % numHeads != 0) throw new ArgumentException("embeddingDim must be divisible by numHeads.", nameof(embeddingDim));

            int queryLen = q.GetLength(0);
            int keyLen = k.GetLength(0);
            int headDim = embeddingDim / numHeads;
            float scale = 1.0f / MathF.Sqrt(headDim);
            bool hasDecayBias = timeDiffs != null;

            if (attentionWeights.Length != numHeads)
            {
                throw new ArgumentException("attentionWeights length must match numHeads.", nameof(attentionWeights));
            }

            if (!ShouldUseGpu((long)numHeads * queryLen * keyLen * headDim, GPU_MATMUL_OP_THRESHOLD))
            {
                return _mutliThreadCPU.BackpropTimeDecayedAttention(q, k, v, dOutput, attentionWeights, timeDiffs, embeddingDim, numHeads);
            }

            var packedWeights = new float[numHeads, queryLen, keyLen];
            for (int h = 0; h < numHeads; h++)
            {
                if (attentionWeights[h] == null || attentionWeights[h].GetLength(0) != queryLen || attentionWeights[h].GetLength(1) != keyLen)
                {
                    throw new ArgumentException($"attentionWeights[{h}] must have shape [{queryLen},{keyLen}].", nameof(attentionWeights));
                }

                for (int p = 0; p < queryLen; p++)
                {
                    for (int s = 0; s < keyLen; s++)
                    {
                        packedWeights[h, p, s] = attentionWeights[h][p, s];
                    }
                }
            }

            var bufQ = _accelerator.Allocate2DDenseX<float>(new Index2D(queryLen, embeddingDim));
            var bufK = _accelerator.Allocate2DDenseX<float>(new Index2D(keyLen, embeddingDim));
            var bufV = _accelerator.Allocate2DDenseX<float>(new Index2D(keyLen, embeddingDim));
            var bufDOutput = _accelerator.Allocate2DDenseX<float>(new Index2D(queryLen, embeddingDim));
            var bufWeights = _accelerator.Allocate3DDenseXY<float>(new Index3D(numHeads, queryLen, keyLen));
            var bufDWeights = _accelerator.Allocate3DDenseXY<float>(new Index3D(numHeads, queryLen, keyLen));
            var bufDot = _accelerator.Allocate2DDenseX<float>(new Index2D(numHeads, queryLen));
            var bufDQ = _accelerator.Allocate2DDenseX<float>(new Index2D(queryLen, embeddingDim));
            var bufDK = _accelerator.Allocate2DDenseX<float>(new Index2D(keyLen, embeddingDim));
            var bufDV = _accelerator.Allocate2DDenseX<float>(new Index2D(keyLen, embeddingDim));
            var bufDecay = _accelerator.Allocate3DDenseXY<float>(new Index3D(queryLen, keyLen, numHeads));

            try
            {
                bufQ.CopyFromCPU(q);
                bufK.CopyFromCPU(k);
                bufV.CopyFromCPU(v);
                bufDOutput.CopyFromCPU(dOutput);
                bufWeights.CopyFromCPU(packedWeights);
                bufDQ.CopyFromCPU(new float[queryLen, embeddingDim]);
                bufDK.CopyFromCPU(new float[keyLen, embeddingDim]);
                bufDV.CopyFromCPU(new float[keyLen, embeddingDim]);
                bufDecay.CopyFromCPU(new float[queryLen, keyLen, numHeads]);

                _timeDecayedDWeightsDvKernel(new Index3D(numHeads, queryLen, keyLen), bufQ.View, bufK.View, bufV.View, bufDOutput.View, bufWeights.View, bufDWeights.View, bufDV.View, headDim);
                _timeDecayedDotWeightKernel(new Index2D(numHeads, queryLen), bufWeights.View, bufDWeights.View, bufDot.View);
                _timeDecayedDqDkBiasKernel(new Index3D(numHeads, queryLen, keyLen), bufQ.View, bufK.View, bufWeights.View, bufDWeights.View, bufDot.View, bufDQ.View, bufDK.View, bufDecay.View, headDim, scale, hasDecayBias ? 1 : 0);

                var dQ = new float[queryLen, embeddingDim];
                var dK = new float[keyLen, embeddingDim];
                var dV = new float[keyLen, embeddingDim];
                bufDQ.CopyToCPU(dQ);
                bufDK.CopyToCPU(dK);
                bufDV.CopyToCPU(dV);

                float[,,] dDecayBias = null;
                if (hasDecayBias)
                {
                    dDecayBias = new float[queryLen, keyLen, numHeads];
                    bufDecay.CopyToCPU(dDecayBias);
                }

                return (dQ, dK, dV, dDecayBias);
            }
            finally
            {
                bufQ.Dispose();
                bufK.Dispose();
                bufV.Dispose();
                bufDOutput.Dispose();
                bufWeights.Dispose();
                bufDWeights.Dispose();
                bufDot.Dispose();
                bufDQ.Dispose();
                bufDK.Dispose();
                bufDV.Dispose();
                bufDecay.Dispose();
            }
        }

        #endregion
    }
}
