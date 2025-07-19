using CallaghanDev.ML.AccelerationManagers;
using CallaghanDev.ML.Enums;
using ILGPU.Algorithms;
using ILGPU.Runtime;
using ILGPU;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using ILGPU.Runtime.Cuda;
using ILGPU.Runtime.OpenCL;

namespace CallaghanDev.ML.AccelerationManagers
{
    public class AccelerationGPUBatch
    {
        private struct LayerBuffers
        {
            public MemoryBuffer2D<float, Stride2D.DenseX> X, W, A, D;
            public MemoryBuffer1D<float, Stride1D.Dense> B;
            public bool Initialized;
        }
        private readonly Dictionary<int, LayerBuffers> _buffers = new();
        private readonly Action<Index2D, ArrayView2D<float, Stride2D.DenseX>, ArrayView2D<float, Stride2D.DenseX>, float, float> updateWeightsKernel;

        private readonly Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, float> updateBiasKernel;

        private readonly Accelerator _accelerator;
        private readonly Action<Index2D, ArrayView2D<float, Stride2D.DenseX>, ArrayView2D<float, Stride2D.DenseX>, ArrayView1D<float, Stride1D.Dense>, ArrayView2D<float, Stride2D.DenseX>, ArrayView2D<float, Stride2D.DenseX>, ActivationType> kernel;

        private readonly int  _deviceId;

        private readonly AccelerationType _accelerationType;

        private readonly IAccelerationManager _accelerationManagerGPU;

        public AccelerationGPUBatch(IAccelerationManager accelerationManagerGPU, AccelerationType accelerationType, int deviceId = 0)
        {
            var ctx = Context.Create(builder => builder.EnableAlgorithms().AllAccelerators());
            _deviceId = deviceId;
            _accelerationType = accelerationType;

            Context context = Context.Create(builder =>
            {
                builder.EnableAlgorithms();
                builder.AllAccelerators();
            });
            if (accelerationType == AccelerationType.GPU)
            {
                _accelerator = context.CreateCLAccelerator(_deviceId);
                _accelerationManagerGPU = accelerationManagerGPU;
            }
            else if (accelerationType == AccelerationType.CUDA)
            {
                _accelerator = context.CreateCudaAccelerator(_deviceId);
                _accelerationManagerGPU = accelerationManagerGPU;
            }
            else
            {
                throw new Exception("Unsupported Acceleration Type for GPU Batch Processing. Use AccelerationType.GPU or AccelerationType.CUDA.");
            }
            Console.WriteLine($"Using accelerator: {_accelerator.Name} ({_accelerator.AcceleratorType})");

            // in your constructor, after you load `kernel`:
            updateWeightsKernel = _accelerator.LoadAutoGroupedStreamKernel<
                Index2D,
                ArrayView2D<float, Stride2D.DenseX>,
                ArrayView2D<float, Stride2D.DenseX>,
                float,
                float>(WeightUpdateKernel);

            updateBiasKernel = _accelerator.LoadAutoGroupedStreamKernel<
                Index1D,
                ArrayView1D<float, Stride1D.Dense>,
                ArrayView1D<float, Stride1D.Dense>,
                float>(BiasUpdateKernel);
            kernel = _accelerator.LoadAutoGroupedStreamKernel<Index2D, ArrayView2D<float, Stride2D.DenseX>, ArrayView2D<float, Stride2D.DenseX>, ArrayView1D<float, Stride1D.Dense>, ArrayView2D<float, Stride2D.DenseX>, ArrayView2D<float, Stride2D.DenseX>, ActivationType>(BatchKernel);
        }

        public (float[][] activation, float[][] derivative) CalculateBatch(float[][] Xcpu, float[,] Wcpu, float[] Bcpu, ActivationType t, int layerIdx)
        {
            if (!_buffers.TryGetValue(layerIdx, out var buf))
            {
                // first time we see this layer: allocate buffers *and* upload weights & biases
                int Bsz = Xcpu.Length, F = Xcpu[0].Length, N = Bcpu.Length;
                buf = new LayerBuffers
                {
                    X = _accelerator.Allocate2DDenseX<float>(new Index2D(Bsz, F)),
                    W = _accelerator.Allocate2DDenseX<float>(new Index2D(N, F)),
                    B = _accelerator.Allocate1D<float>(new Index1D(N)),
                    A = _accelerator.Allocate2DDenseX<float>(new Index2D(Bsz, N)),
                    D = _accelerator.Allocate2DDenseX<float>(new Index2D(Bsz, N)),
                    // now mark initialized
                    Initialized = true
                };
                // upload the CPU master copy exactly once
                buf.W.CopyFromCPU(Wcpu);
                buf.B.CopyFromCPU(Bcpu);

                _buffers[layerIdx] = buf;
            }

            // every batch we only re‐upload the inputs
            buf.X.CopyFromCPU(Flatten(Xcpu));

            // run the forward kernel
            kernel(new Index2D(Xcpu.Length, Bcpu.Length),
                   buf.X.View, buf.W.View, buf.B.View,
                   buf.A.View, buf.D.View, t);

            // read back activations & derivatives
            var act2d = new float[Xcpu.Length, Bcpu.Length];
            var der2d = new float[Xcpu.Length, Bcpu.Length];
            buf.A.CopyToCPU(act2d);
            buf.D.CopyToCPU(der2d);

            return (Unflatten(act2d), Unflatten(der2d));
        }

        public void ApplyBatchUpdatesGPU(float[,] Wcpu, float[][] deltasCpu, float[][] prevActsCpu, float lr, float λ, int layerIdx)
        {
            var buf = _buffers[layerIdx];
            int Bsz = deltasCpu.Length, N = (int)buf.B.Length, F = (int)buf.W.Extent.Y;

            // sum grads on CPU
            var gradSum = new float[N, F];
            for (int i = 0; i < Bsz; i++)
                for (int r = 0; r < N; r++)
                    for (int c = 0; c < F; c++)
                        gradSum[r, c] += deltasCpu[i][r] * prevActsCpu[i][c];

            var gradBuf = _accelerator.Allocate2DDenseX<float>(new Index2D(N, F));
            gradBuf.CopyFromCPU(gradSum);

            // W[r,c] -= (lr/Bsz)*(gradSum[r,c] + λ*W[r,c])
            updateWeightsKernel(
                new Index2D(N, F),
                buf.W.View,
                gradBuf.View,
                lr / Bsz,
                λ
            );
            gradBuf.Dispose();

            // bias sum
            var biasSum = new float[N];
            for (int i = 0; i < Bsz; i++)
                for (int j = 0; j < N; j++)
                    biasSum[j] += deltasCpu[i][j];

            var biasBuf = _accelerator.Allocate1D<float>(new Index1D(N));
            biasBuf.CopyFromCPU(biasSum);

            // B[j] -= (lr/Bsz)*biasSum[j]
            updateBiasKernel(
                new Index1D(N),
                buf.B.View,
                biasBuf.View,
                lr / Bsz
            );
            biasBuf.Dispose();
        }

        private static void WeightUpdateKernel(Index2D idx, ArrayView2D<float, Stride2D.DenseX> W, ArrayView2D<float, Stride2D.DenseX> gradSum, float scaledLr, float λ)
        {
            int r = idx.X, c = idx.Y;
            float w = W[r, c];
            float g = gradSum[r, c];
            W[r, c] = w - scaledLr * (g + λ * w);
        }
        public void CopyLayerToCPU(int layerIdx, float[,] weightsCpu, float[] biasesCpu)
        {
            if (!_buffers.TryGetValue(layerIdx, out var buf))
            {
                throw new ArgumentException($"No GPU buffers found for layer {layerIdx}");
            }
            // Copy the 2D weight matrix back
            buf.W.CopyToCPU(weightsCpu);

            // Copy the 1D bias vector back
            buf.B.CopyToCPU(biasesCpu);
        }
        private static void BiasUpdateKernel(Index1D j, ArrayView1D<float, Stride1D.Dense> B, ArrayView1D<float, Stride1D.Dense> biasSum, float scaledLr)
        {
            B[j] = B[j] - scaledLr * biasSum[j];
        }
        private static float[,] Flatten(float[][] jagged)
        {
            int n = jagged.Length, m = jagged[0].Length;
            var full = new float[n, m];
            for (int i = 0; i < n; i++)
                for (int j = 0; j < m; j++)
                    full[i, j] = jagged[i][j];
            return full;
        }

        private static float[][] Unflatten(float[,] arr)
        {
            int n = arr.GetLength(0), m = arr.GetLength(1);
            var jagged = new float[n][];
            for (int i = 0; i < n; i++)
            {
                jagged[i] = new float[m];
                for (int j = 0; j < m; j++)
                    jagged[i][j] = arr[i, j];
            }
            return jagged;
        }
        private static void BatchKernel(Index2D idx, ArrayView2D<float, Stride2D.DenseX> X, ArrayView2D<float, Stride2D.DenseX> W, ArrayView1D<float, Stride1D.Dense> B, ArrayView2D<float, Stride2D.DenseX> A, ArrayView2D<float, Stride2D.DenseX> D, ActivationType t)
        {
            int i = idx.X; // sample
            int j = idx.Y; // neuron
            float sum = 0;
            for (int k = 0; k < X.Extent.Y; k++)
            {
                sum += X[i, k] * W[j, k];
            }
            float z = sum + B[j];
            float a, d;
            switch (t)
            {
                case ActivationType.None:
                    a = z; d = 1; break;
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
                    a = XMath.Max(0.0f, z);
                    d = z > 0 ? 1 : 0;
                    break;
                case ActivationType.Leakyrelu:
                    a = z > 0 ? z : 0.01f * z;
                    d = z > 0 ? 1f : 0.01f;
                    break;
                default:
                    var ee = XMath.Exp(z);
                    a = ee / (1 + ee);
                    d = a * (1 - a);
                    break;
            }
            A[i, j] = a;
            D[i, j] = d;
        }

        public float[][] CalculateBatchOutputGradients(float[][] costBatch, float[][] derivBatch)
        {
            int B = costBatch.Length;
            var outL = new float[B][];
            for (int i = 0; i < B; i++)
            {
                outL[i] = _accelerationManagerGPU.CalculateOutputGradients(costBatch[i], derivBatch[i]);
            }
            return outL;
        }

        public float[][] CalculateBatchHiddenGradients(float[,] weights, float[][] nextBatch, float[][] derivBatch)
        {
            int B = nextBatch.Length;
            var outL = new float[B][];

            for (int i = 0; i < B; i++)
            {
                outL[i] = _accelerationManagerGPU.CalculateHiddenGradients(weights, nextBatch[i], derivBatch[i]);
            }
            return outL;
        }

        public void Dispose()
        {
           _accelerator.Dispose();
            _accelerationManagerGPU.Dispose();
        }
    }
}
