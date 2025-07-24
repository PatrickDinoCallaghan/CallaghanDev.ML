using System;
using System.Collections.Generic;
using CallaghanDev.ML.Enums;
using ILGPU;
using ILGPU.Algorithms;
using ILGPU.Runtime;
using ILGPU.Runtime.Cuda;
using ILGPU.Runtime.OpenCL;

namespace CallaghanDev.ML.AccelerationManagers
{
    public class AccelerationGPUBatch : IDisposable
    {
       private struct LayerBuffers
        {
            public MemoryBuffer2D<float, Stride2D.DenseX> X, W, A, D;
            public MemoryBuffer1D<float, Stride1D.Dense> B;
            public bool Initialized;
        }

        private readonly Dictionary<int, LayerBuffers> _buffers = new();
        private readonly Accelerator _accelerator;
        private readonly IAccelerationManager _accelerationManagerGPU;

        // Reusable buffer caches
        private readonly Dictionary<(int B, int N), (MemoryBuffer2D<float, Stride2D.DenseX> cost, MemoryBuffer2D<float, Stride2D.DenseX> deriv, MemoryBuffer2D<float, Stride2D.DenseX> output)> _outputGradCache = new();
        private readonly Dictionary<(int B, int Nnext, int Ncur), (MemoryBuffer2D<float, Stride2D.DenseX> w, MemoryBuffer2D<float, Stride2D.DenseX> next, MemoryBuffer2D<float, Stride2D.DenseX> deriv, MemoryBuffer2D<float, Stride2D.DenseX> output)> _hiddenGradCache = new();
        private readonly Dictionary<(int layer, int B, int N, int F), (MemoryBuffer2D<float, Stride2D.DenseX> deltas, MemoryBuffer2D<float, Stride2D.DenseX> acts, MemoryBuffer2D<float, Stride2D.DenseX> grad, MemoryBuffer1D<float, Stride1D.Dense> bias)> _batchUpdateCache = new();

        // Core ILGPU kernels
        private readonly Action<Index2D, ArrayView2D<float, Stride2D.DenseX>, ArrayView2D<float, Stride2D.DenseX>, ArrayView1D<float, Stride1D.Dense>, ArrayView2D<float, Stride2D.DenseX>, ArrayView2D<float, Stride2D.DenseX>, ActivationType> _batchKernel;

        private readonly Action<Index2D, ArrayView2D<float, Stride2D.DenseX>, ArrayView2D<float, Stride2D.DenseX>, float, float> _updateWeightsKernel;

        private readonly Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, float> _updateBiasKernel;

        // Gradient-sum kernels
        private readonly Action<Index2D, ArrayView2D<float, Stride2D.DenseX>, ArrayView2D<float, Stride2D.DenseX>, ArrayView2D<float, Stride2D.DenseX>> _gradSumKernel;

        private readonly Action<Index1D, ArrayView2D<float, Stride2D.DenseX>, ArrayView1D<float, Stride1D.Dense>> _biasSumKernel;

        // Batch gradient kernels
        private readonly Action<Index2D, ArrayView2D<float, Stride2D.DenseX>, ArrayView2D<float, Stride2D.DenseX>, ArrayView2D<float, Stride2D.DenseX>> _outputGradBatchKernel;

        private readonly Action<Index2D, ArrayView2D<float, Stride2D.DenseX>, ArrayView2D<float, Stride2D.DenseX>, ArrayView2D<float, Stride2D.DenseX>, ArrayView2D<float, Stride2D.DenseX>> _hiddenGradBatchKernel;

        public AccelerationGPUBatch(IAccelerationManager accelerationManagerGPU, AccelerationType accelerationType, int deviceId = 0)
        {
            // Create ILGPU context and select accelerator
            var context = Context.Create(builder => builder.EnableAlgorithms().AllAccelerators());
            _accelerator = accelerationType switch
            {
                AccelerationType.GPU => context.CreateCLAccelerator(deviceId),
                AccelerationType.CUDA => context.CreateCudaAccelerator(deviceId),
                _ => throw new ArgumentException("Unsupported AccelerationType. Use GPU or CUDA.")
            };
            _accelerationManagerGPU = accelerationManagerGPU;

            // Load kernels
            _batchKernel = _accelerator.LoadAutoGroupedStreamKernel<
                Index2D,
                ArrayView2D<float, Stride2D.DenseX>,
                ArrayView2D<float, Stride2D.DenseX>,
                ArrayView1D<float, Stride1D.Dense>,
                ArrayView2D<float, Stride2D.DenseX>,
                ArrayView2D<float, Stride2D.DenseX>,
                ActivationType>(BatchKernel);

            _updateWeightsKernel = _accelerator.LoadAutoGroupedStreamKernel<
                Index2D,
                ArrayView2D<float, Stride2D.DenseX>,
                ArrayView2D<float, Stride2D.DenseX>,
                float,
                float>(WeightUpdateKernel);

            _updateBiasKernel = _accelerator.LoadAutoGroupedStreamKernel<
                Index1D,
                ArrayView1D<float, Stride1D.Dense>,
                ArrayView1D<float, Stride1D.Dense>,
                float>(BiasUpdateKernel);

            _gradSumKernel = _accelerator.LoadAutoGroupedStreamKernel<
                Index2D,
                ArrayView2D<float, Stride2D.DenseX>,
                ArrayView2D<float, Stride2D.DenseX>,
                ArrayView2D<float, Stride2D.DenseX>>(GradSumKernel);

            _biasSumKernel = _accelerator.LoadAutoGroupedStreamKernel<
                Index1D,
                ArrayView2D<float, Stride2D.DenseX>,
                ArrayView1D<float, Stride1D.Dense>>(BiasSumKernel);

            _outputGradBatchKernel = _accelerator.LoadAutoGroupedStreamKernel<
                Index2D,
                ArrayView2D<float, Stride2D.DenseX>,
                ArrayView2D<float, Stride2D.DenseX>,
                ArrayView2D<float, Stride2D.DenseX>>(OutputGradBatchKernel);

            _hiddenGradBatchKernel = _accelerator.LoadAutoGroupedStreamKernel<
                Index2D,
                ArrayView2D<float, Stride2D.DenseX>,
                ArrayView2D<float, Stride2D.DenseX>,
                ArrayView2D<float, Stride2D.DenseX>,
                ArrayView2D<float, Stride2D.DenseX>>(HiddenGradBatchKernel);
        }

        /// <summary>
        /// Forward and backward propagate a batch through one layer on the GPU.
        /// </summary>
        public (float[][] activation, float[][] derivative) CalculateBatch(float[][] Xcpu, float[,] Wcpu, float[] Bcpu, ActivationType t, int layerIdx)
        {
            int Bsz = Xcpu.Length;
            int Fsz = Xcpu[0].Length;

            if (_buffers.TryGetValue(layerIdx, out var buf))
            {
                var ext = buf.X.Extent;
                if (ext.X != Bsz || ext.Y != Fsz)
                {
                    DisposeLayerBuffers(layerIdx);
                    buf = default;
                }
            }

            if (!_buffers.TryGetValue(layerIdx, out buf))
            {
                buf = new LayerBuffers
                {
                    X = _accelerator.Allocate2DDenseX<float>(new Index2D(Bsz, Fsz)),
                    W = _accelerator.Allocate2DDenseX<float>(new Index2D(Wcpu.GetLength(0), Fsz)),
                    B = _accelerator.Allocate1D<float>(new Index1D(Bcpu.Length)),
                    A = _accelerator.Allocate2DDenseX<float>(new Index2D(Bsz, Bcpu.Length)),
                    D = _accelerator.Allocate2DDenseX<float>(new Index2D(Bsz, Bcpu.Length)),
                    Initialized = true
                };
                buf.W.CopyFromCPU(Wcpu);
                buf.B.CopyFromCPU(Bcpu);
                _buffers[layerIdx] = buf;
            }

            buf.X.CopyFromCPU(Flatten(Xcpu));
            _batchKernel(new Index2D(Bsz, Bcpu.Length), buf.X.View, buf.W.View, buf.B.View, buf.A.View, buf.D.View, t);

            var act2d = new float[Bsz, Bcpu.Length];
            var der2d = new float[Bsz, Bcpu.Length];
            buf.A.CopyToCPU(act2d);
            buf.D.CopyToCPU(der2d);
            return (Unflatten(act2d), Unflatten(der2d));
        }

        /// <summary>
        /// Batch compute output gradients: out[i,j] = -cost[i,j] * deriv[i,j].
        /// </summary>
        public float[][] CalculateOutputGradients(float[][] costBatch, float[][] derivBatch)
        {
            int B = costBatch.Length, N = costBatch[0].Length;
            var key = (B, N);

            if (!_outputGradCache.TryGetValue(key, out var bufs))
            {
                bufs.cost = _accelerator.Allocate2DDenseX<float>(new Index2D(B, N));
                bufs.deriv = _accelerator.Allocate2DDenseX<float>(new Index2D(B, N));
                bufs.output = _accelerator.Allocate2DDenseX<float>(new Index2D(B, N));
                _outputGradCache[key] = bufs;
            }

            bufs.cost.CopyFromCPU(Flatten(costBatch));
            bufs.deriv.CopyFromCPU(Flatten(derivBatch));

            _outputGradBatchKernel(new Index2D(B, N), bufs.cost.View, bufs.deriv.View, bufs.output.View);

            var result = new float[B, N];
            bufs.output.CopyToCPU(result);
            return Unflatten(result);
        }

        /// <summary>
        /// Batch compute hidden gradients.
        /// </summary>
        public float[][] CalculateHiddenGradients(float[,] weights, float[][] nextBatch, float[][] derivBatch)
        {
            int B = nextBatch.Length;
            int Nnext = weights.GetLength(0), Ncur = weights.GetLength(1);
            var key = (B, Nnext, Ncur);

            if (!_hiddenGradCache.TryGetValue(key, out var bufs))
            {
                bufs.w = _accelerator.Allocate2DDenseX<float>(new Index2D(Nnext, Ncur));
                bufs.next = _accelerator.Allocate2DDenseX<float>(new Index2D(B, Nnext));
                bufs.deriv = _accelerator.Allocate2DDenseX<float>(new Index2D(B, Ncur));
                bufs.output = _accelerator.Allocate2DDenseX<float>(new Index2D(B, Ncur));
                _hiddenGradCache[key] = bufs;
            }

            bufs.w.CopyFromCPU(weights);
            bufs.next.CopyFromCPU(Flatten(nextBatch));
            bufs.deriv.CopyFromCPU(Flatten(derivBatch));

            _hiddenGradBatchKernel(new Index2D(B, Ncur), bufs.next.View, bufs.w.View, bufs.deriv.View, bufs.output.View);

            var result = new float[B, Ncur];
            bufs.output.CopyToCPU(result);
            return Unflatten(result);
        }

        /// <summary>
        /// Apply batched weight & bias updates on GPU (with gradient clipping & L2).
        /// </summary>
        public void ApplyBatchUpdatesGPU(float[,] Wcpu, float[][] deltasCpu, float[][] prevActsCpu, float lr, float lamda, int layerIdx)
        {
            int Bsz = deltasCpu.Length;
            var bufRec = _buffers[layerIdx];
            int N = (int)bufRec.B.Length;
            int Fsz = (int)bufRec.W.Extent.Y;
            var key = (layerIdx, Bsz, N, Fsz);

            if (!_batchUpdateCache.TryGetValue(key, out var bufs))
            {
                bufs.deltas = _accelerator.Allocate2DDenseX<float>(new Index2D(Bsz, N));
                bufs.acts = _accelerator.Allocate2DDenseX<float>(new Index2D(Bsz, Fsz));
                bufs.grad = _accelerator.Allocate2DDenseX<float>(new Index2D(N, Fsz));
                bufs.bias = _accelerator.Allocate1D<float>(new Index1D(N));
                _batchUpdateCache[key] = bufs;
            }

            bufs.deltas.CopyFromCPU(Flatten(deltasCpu));
            bufs.acts.CopyFromCPU(Flatten(prevActsCpu));

            _gradSumKernel(new Index2D(N, Fsz), bufs.deltas.View, bufs.acts.View, bufs.grad.View);
            _updateWeightsKernel(new Index2D(N, Fsz), bufRec.W.View, bufs.grad.View, lr / Bsz, lamda);

            _biasSumKernel(new Index1D(N), bufs.deltas.View, bufs.bias.View);
            _updateBiasKernel(new Index1D(N), bufRec.B.View, bufs.bias.View, lr / Bsz);
        }

        /// <summary>
        /// Copy final weights & biases back to CPU.
        /// </summary>
        public void CopyLayerToCPU(int layerIdx, float[,] weightsCpu, float[] biasesCpu)
        {
            if (!_buffers.TryGetValue(layerIdx, out var buf))
            {
                throw new ArgumentException($"No GPU buffers for layer {layerIdx}");
            }
            buf.W.CopyToCPU(weightsCpu);
            buf.B.CopyToCPU(biasesCpu);
        }

        #region Kernel implementations

        private static void BatchKernel(Index2D idx,
            ArrayView2D<float, Stride2D.DenseX> X,
            ArrayView2D<float, Stride2D.DenseX> W,
            ArrayView1D<float, Stride1D.Dense> B,
            ArrayView2D<float, Stride2D.DenseX> A,
            ArrayView2D<float, Stride2D.DenseX> D,
            ActivationType t)
        {
            int i = idx.X, j = idx.Y;
            float sum = 0;
            for (int k = 0; k < X.Extent.Y; ++k)
                sum += X[i, k] * W[j, k];
            float z = sum + B[j];

            float a, d;
            switch (t)
            {
                case ActivationType.None:
                    a = z; d = 1;
                    break;
                case ActivationType.Sigmoid:
                    var e = XMath.Exp(z);
                    a = e / (1 + e);
                    d = a * (1 - a);
                    break;
                case ActivationType.Tanh:
                    a = XMath.Tanh(z);
                    d = 1 - a * a;
                    break;
                case ActivationType.Relu:
                    a = XMath.Max(0, z);
                    d = z > 0 ? 1 : 0;
                    break;
                case ActivationType.Leakyrelu:
                    a = z > 0 ? z : 0.01f * z;
                    d = z > 0 ? 1 : 0.01f;
                    break;
                default:
                    a = 0; d = 0;
                    break;
            }
            A[i, j] = a;
            D[i, j] = d;
        }

        private static void GradSumKernel(Index2D idx,
            ArrayView2D<float, Stride2D.DenseX> deltas,
            ArrayView2D<float, Stride2D.DenseX> prevActs,
            ArrayView2D<float, Stride2D.DenseX> outGrad)
        {
            int r = idx.X, c = idx.Y;
            float sum = 0;
            for (int i = 0; i < deltas.Extent.X; ++i)
                sum += deltas[i, r] * prevActs[i, c];
            outGrad[r, c] = sum;
        }

        private static void BiasSumKernel(Index1D idx,
            ArrayView2D<float, Stride2D.DenseX> deltas,
            ArrayView1D<float, Stride1D.Dense> outBias)
        {
            int j = idx;
            float sum = 0;
            for (int i = 0; i < deltas.Extent.X; ++i)
                sum += deltas[i, j];
            outBias[j] = sum;
        }

        private static void OutputGradBatchKernel(Index2D idx,
            ArrayView2D<float, Stride2D.DenseX> cost,
            ArrayView2D<float, Stride2D.DenseX> deriv,
            ArrayView2D<float, Stride2D.DenseX> outGrad)
        {
            int i = idx.X, j = idx.Y;
            outGrad[i, j] = -cost[i, j] * deriv[i, j];
        }

        private static void HiddenGradBatchKernel(Index2D idx,
            ArrayView2D<float, Stride2D.DenseX> nextD,
            ArrayView2D<float, Stride2D.DenseX> W,
            ArrayView2D<float, Stride2D.DenseX> deriv,
            ArrayView2D<float, Stride2D.DenseX> outGrad)
        {
            int i = idx.X, j = idx.Y;
            float sum = 0;
            for (int k = 0; k < nextD.Extent.Y; ++k)
                sum += nextD[i, k] * W[k, j];
            outGrad[i, j] = sum * deriv[i, j];
        }

        private static void WeightUpdateKernel(
            Index2D idx,
            ArrayView2D<float, Stride2D.DenseX> W,
            ArrayView2D<float, Stride2D.DenseX> gradSum,
            float scaledLr,
            float lambda)
        {
            int r = idx.X, c = idx.Y;
            float w = W[r, c];
            float g = gradSum[r, c];
            W[r, c] = w - scaledLr * (g + lambda * w);
        }

        private static void BiasUpdateKernel(
            Index1D idx,
            ArrayView1D<float, Stride1D.Dense> B,
            ArrayView1D<float, Stride1D.Dense> biasSum,
            float scaledLr)
        {
            int j = idx;
            B[j] = B[j] - scaledLr * biasSum[j];
        }

        #endregion

        /// <summary>
        /// Flatten jagged float[][] into a 2D array.
        /// </summary>
        /*private static float[,] Flatten(float[][] jagged)
        {
            int rows = jagged.Length;
            int cols = jagged[0].Length;
            var full = new float[rows, cols];
            for (int i = 0; i < rows; i++)
                for (int j = 0; j < cols; j++)
                    full[i, j] = jagged[i][j];
            return full;
        }*/
        private static float[,] Flatten(float[][] jagged)
        {
            int rows = jagged.Length;
            int cols = jagged[0].Length;

            var full = new float[rows, cols];
            int byteCount = cols * sizeof(float);

            Parallel.For(0, rows, i =>
            {
                // copy jagged[i][0..cols] into full[i,0..cols]
                Buffer.BlockCopy(
                    jagged[i],                // source array
                    0,                        // source offset (bytes)
                    full,                     // dest 2D array
                    i * byteCount,            // dest offset = rowIndex * rowSizeBytes
                    byteCount                 // number of bytes to copy
                );
            });

            return full;
        }

        /// <summary>
        /// Unflatten 2D array back to jagged float[][].
        /// </summary>
        /*private static float[][] Unflatten(float[,] arr)
        {
            int rows = arr.GetLength(0);
            int cols = arr.GetLength(1);
            var jagged = new float[rows][];
            for (int i = 0; i < rows; i++)
            {
                jagged[i] = new float[cols];
                for (int j = 0; j < cols; j++)
                    jagged[i][j] = arr[i, j];
            }
            return jagged;
        }*/
        private static float[][] Unflatten(float[,] arr)
        {
            int rows = arr.GetLength(0);
            int cols = arr.GetLength(1);

            var jagged = new float[rows][];
            int byteCount = cols * sizeof(float);

            Parallel.For(0, rows, i =>
            {
                var row = new float[cols];
                Buffer.BlockCopy(
                    arr,                      // source 2D array
                    i * byteCount,            // source offset (bytes)
                    row,                      // dest jagged row
                    0,                        // dest offset
                    byteCount
                );
                jagged[i] = row;
            });

            return jagged;
        }

        public void Dispose()
        {
            _accelerator.Dispose();
            _accelerationManagerGPU.Dispose();
            foreach (var buf in _buffers.Values)
            {
                if (buf.Initialized)
                {
                    buf.X.Dispose();
                    buf.W.Dispose();
                    buf.B.Dispose();
                    buf.A.Dispose();
                    buf.D.Dispose();
                }
            }
        }

        private void DisposeLayerBuffers(int layerIdx)
        {
            if (_buffers.TryGetValue(layerIdx, out var buf) && buf.Initialized)
            {
                buf.X.Dispose();
                buf.W.Dispose();
                buf.B.Dispose();
                buf.A.Dispose();
                buf.D.Dispose();
                _buffers.Remove(layerIdx);
            }
        }

    }
}
