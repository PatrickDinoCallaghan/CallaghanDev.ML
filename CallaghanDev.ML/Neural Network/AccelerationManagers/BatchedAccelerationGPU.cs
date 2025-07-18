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
        private readonly Accelerator _accelerator;
        private readonly Action<Index2D, ArrayView2D<double, Stride2D.DenseX>, ArrayView2D<double, Stride2D.DenseX>, ArrayView1D<double, Stride1D.Dense>, ArrayView2D<double, Stride2D.DenseX>, ArrayView2D<double, Stride2D.DenseX>, ActivationType> kernel;

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
           
            kernel = _accelerator.LoadAutoGroupedStreamKernel<Index2D, ArrayView2D<double, Stride2D.DenseX>, ArrayView2D<double, Stride2D.DenseX>, ArrayView1D<double, Stride1D.Dense>, ArrayView2D<double, Stride2D.DenseX>, ArrayView2D<double, Stride2D.DenseX>, ActivationType>(BatchKernel);
        }
        public (double[][] activation, double[][] derivative) CalculateBatch(double[][] X, double[,] W, double[] B, ActivationType t)
        {
            int Bsz = X.Length;
            int F = X[0].Length;
            int N = B.Length;

            var flatX = new double[Bsz, F];
            for (int i = 0; i < Bsz; i++)
            {
                for (int j = 0; j < F; j++)
                {
                    flatX[i, j] = X[i][j];
                }
            }

            var bufX = _accelerator.Allocate2DDenseX<double>(new Index2D(Bsz, F));
            var bufW = _accelerator.Allocate2DDenseX<double>(new Index2D(N, F));
            var bufB = _accelerator.Allocate1D<double>(N);
            var bufA = _accelerator.Allocate2DDenseX<double>(new Index2D(Bsz, N));
            var bufD = _accelerator.Allocate2DDenseX<double>(new Index2D(Bsz, N));

            bufX.CopyFromCPU(flatX);
            bufW.CopyFromCPU(W);
            bufB.CopyFromCPU(B);

            kernel(new Index2D(Bsz, N), bufX.View, bufW.View, bufB.View, bufA.View, bufD.View, t);

            var act2d = new double[Bsz, N];
            var der2d = new double[Bsz, N];
            bufA.CopyToCPU(act2d);
            bufD.CopyToCPU(der2d);

            var acts = new double[Bsz][];
            var ders = new double[Bsz][];
            for (int i = 0; i < Bsz; i++)
            {
                acts[i] = new double[N];
                ders[i] = new double[N];
                for (int j = 0; j < N; j++)
                {
                    acts[i][j] = act2d[i, j];
                    ders[i][j] = der2d[i, j];
                }
            }

            bufX.Dispose(); bufW.Dispose(); bufB.Dispose();
            bufA.Dispose(); bufD.Dispose();

            return (acts, ders);
        }


        private static void BatchKernel(Index2D idx, ArrayView2D<double, Stride2D.DenseX> X, ArrayView2D<double, Stride2D.DenseX> W, ArrayView1D<double, Stride1D.Dense> B, ArrayView2D<double, Stride2D.DenseX> A, ArrayView2D<double, Stride2D.DenseX> D, ActivationType t)
        {
            int i = idx.X; // sample
            int j = idx.Y; // neuron
            double sum = 0;
            for (int k = 0; k < X.Extent.Y; k++)
            {
                sum += X[i, k] * W[j, k];
            }
            double z = sum + B[j];
            double a, d;
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
                    a = XMath.Max(0.0, z);
                    d = z > 0 ? 1 : 0;
                    break;
                case ActivationType.Leakyrelu:
                    a = z > 0 ? z : 0.01 * z;
                    d = z > 0 ? 1 : 0.01;
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

        public double[][] CalculateBatchOutputGradients(double[][] costBatch, double[][] derivBatch)
        {
            int B = costBatch.Length;
            var outL = new double[B][];
            var inner = new AccelerationGPU(_accelerationType, _deviceId);
            for (int i = 0; i < B; i++)
            {
                outL[i] = inner.CalculateOutputGradients(costBatch[i], derivBatch[i]);
            }
            inner.Dispose();
            return outL;
        }

        public double[][] CalculateBatchHiddenGradients(double[,] weights, double[][] nextBatch, double[][] derivBatch)
        {
            int B = nextBatch.Length;
            var outL = new double[B][];

            for (int i = 0; i < B; i++)
            {
                outL[i] = _accelerationManagerGPU.CalculateHiddenGradients(weights, nextBatch[i], derivBatch[i]);
            }
            return outL;
        }

        public double[,] UpdateBatchWeights(double[,] W, double[][] deltas, double[][] prevBatch, double lr, double λ)
        {
            int rows = W.GetLength(0), cols = W.GetLength(1);
            int B = deltas.Length;
            var gradSum = new double[rows, cols];
            for (int i = 0; i < B; i++)
            {
                for (int r = 0; r < rows; r++)
                {
                    for (int c = 0; c < cols; c++)
                    {
                        gradSum[r, c] += deltas[i][r] * prevBatch[i][c];
                    }
                }
            }
            var newW = new double[rows, cols];
            for (int r = 0; r < rows; r++)
            {
                for (int c = 0; c < cols; c++)
                {
                    var g = gradSum[r, c] / B;
                    newW[r, c] = W[r, c] - lr * (g + λ * W[r, c]);
                }
            }
            return newW;
        }

        public double[] UpdateBatchBias(double[] B, double[][] deltas, double lr)
        {
            int n = B.Length, Bsz = deltas.Length;
            var sum = new double[n];
            for (int i = 0; i < Bsz; i++)
            {
                for (int j = 0; j < n; j++) sum[j] += deltas[i][j];

            }
            var outB = new double[n];
            for (int j = 0; j < n; j++)
            {
                outB[j] = B[j] - lr * (sum[j] / Bsz);
            }
            return outB;
        }

        public void Dispose()
        {
           _accelerator.Dispose();
        }
    }
}
