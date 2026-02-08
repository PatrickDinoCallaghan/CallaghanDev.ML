using CallaghanDev.ML.Enums;
using ILGPU;
using ILGPU.Algorithms;
using ILGPU.Runtime;
using ILGPU.Runtime.Cuda;
using ILGPU.Runtime.OpenCL;
using static Microsoft.FSharp.Core.ByRefKinds;

namespace CallaghanDev.ML.AccelerationManagers
{
    public class AccelerationGPU : IAccelerationManager, IDisposable
    {
        private readonly Accelerator _accelerator;

        private readonly Action<Index1D, ArrayView2D<float, Stride2D.DenseX>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>> _dotKernel;
        private readonly Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>> _outGradKernel;
        private readonly Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>> _hidGradKernel;
        private readonly Action<Index2D, ArrayView2D<float, Stride2D.DenseX>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, float, float> _updWKernel;
        private readonly Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, float> _updBKernel;
        private readonly Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView2D<float, Stride2D.DenseX>, ArrayView1D<float, Stride1D.Dense>> _dotTransposedKernel;
        private readonly Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, ActivationType> _actKernel;
        private readonly Action<Index2D, ArrayView2D<float, Stride2D.DenseX>, ArrayView2D<float, Stride2D.DenseX>, ArrayView2D<float, Stride2D.DenseX>> _matMulKernel;
        private readonly Action<Index2D, ArrayView2D<float, Stride2D.DenseX>, ArrayView2D<float, Stride2D.DenseX>, ArrayView2D<float, Stride2D.DenseX>> _matMulTransposeKernel;
        private readonly Action<Index2D, ArrayView2D<float, Stride2D.DenseX>, float> _matScaleKernel;
        private readonly Action<Index2D, ArrayView2D<float, Stride2D.DenseX>, ArrayView2D<float, Stride2D.DenseX>, ArrayView2D<float, Stride2D.DenseX>> _matAddKernel;
        private readonly Action<Index1D, ArrayView2D<float, Stride2D.DenseX>, ArrayView2D<float, Stride2D.DenseX>, ArrayView2D<int, Stride2D.DenseX>> _softmaxKernel;
        private readonly Action<Index1D, ArrayView2D<float, Stride2D.DenseX>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, ArrayView2D<float, Stride2D.DenseX>, float> _layerNormKernel;

        private readonly Dictionary<(int rows, int cols),
            (MemoryBuffer2D<float, Stride2D.DenseX> mat,
             MemoryBuffer1D<float, Stride1D.Dense> vec,
             MemoryBuffer1D<float, Stride1D.Dense> res)> _dotCache
            = new();
        private readonly Dictionary<int,
            (MemoryBuffer1D<float, Stride1D.Dense> cost,
             MemoryBuffer1D<float, Stride1D.Dense> der,
             MemoryBuffer1D<float, Stride1D.Dense> grad)> _outGradCache
            = new();
        private readonly Dictionary<int,
            (MemoryBuffer1D<float, Stride1D.Dense> pre,
             MemoryBuffer1D<float, Stride1D.Dense> der,
             MemoryBuffer1D<float, Stride1D.Dense> delta)> _hidGradCache
            = new();
        private readonly Dictionary<(int rows, int cols), (MemoryBuffer2D<float, Stride2D.DenseX> w, MemoryBuffer1D<float, Stride1D.Dense> d, MemoryBuffer1D<float, Stride1D.Dense> pa)> _updWCache  = new();
        private readonly Dictionary<int,
            (MemoryBuffer1D<float, Stride1D.Dense> b,
             MemoryBuffer1D<float, Stride1D.Dense> d)> _updBCache
            = new();
        private readonly Dictionary<(int rows, int cols),
            (MemoryBuffer2D<float, Stride2D.DenseX> mat,
             MemoryBuffer1D<float, Stride1D.Dense> vec,
             MemoryBuffer1D<float, Stride1D.Dense> res)> _dotTransposedCache
            = new();
        private readonly Dictionary<(int r1, int c1, int c2), (MemoryBuffer2D<float, Stride2D.DenseX> a, MemoryBuffer2D<float, Stride2D.DenseX> b, MemoryBuffer2D<float, Stride2D.DenseX> c)> _matMulCache = new();

        public AccelerationGPU(AccelerationType accelerationType, int deviceIndex = 0)
        {
            Context context = Context.Create(builder =>
            {
                builder.EnableAlgorithms();
                builder.AllAccelerators();
            });
            if (accelerationType == AccelerationType.GPU)
            {
                _accelerator = context.CreateCLAccelerator(deviceIndex);
            }
            else if (accelerationType == AccelerationType.CUDA)
            {
                _accelerator = context.CreateCudaAccelerator(deviceIndex);
            }


            _dotKernel = _accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView2D<float, Stride2D.DenseX>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>>(DotKernel);
            _dotTransposedKernel = _accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView2D<float, Stride2D.DenseX>, ArrayView1D<float, Stride1D.Dense>>(DotTransposedKernel);
            _actKernel = _accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, ActivationType>(ActivateKernel);
            _outGradKernel = _accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>>(OutputGradientKernel);
            _hidGradKernel = _accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>>(HiddenGradientKernel);
            _updWKernel = _accelerator.LoadAutoGroupedStreamKernel<Index2D, ArrayView2D<float, Stride2D.DenseX>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, float, float>(UpdateWeightsKernel);
            _updBKernel = _accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, float>(UpdateBiasKernel);
            _matMulKernel = _accelerator.LoadAutoGroupedStreamKernel<Index2D, ArrayView2D<float, Stride2D.DenseX>, ArrayView2D<float, Stride2D.DenseX>, ArrayView2D<float, Stride2D.DenseX>>(MatMulKernel);
            _matMulTransposeKernel = _accelerator.LoadAutoGroupedStreamKernel<Index2D, ArrayView2D<float, Stride2D.DenseX>, ArrayView2D<float, Stride2D.DenseX>, ArrayView2D<float, Stride2D.DenseX>>(MatMulTransposeKernel);
            _matScaleKernel = _accelerator.LoadAutoGroupedStreamKernel<Index2D, ArrayView2D<float, Stride2D.DenseX>, float>(MatScaleKernel);
            _matAddKernel = _accelerator.LoadAutoGroupedStreamKernel<Index2D, ArrayView2D<float, Stride2D.DenseX>, ArrayView2D<float, Stride2D.DenseX>, ArrayView2D<float, Stride2D.DenseX>>(MatAddKernel);
      
            _softmaxKernel = _accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView2D<float, Stride2D.DenseX>, ArrayView2D<float, Stride2D.DenseX>, ArrayView2D<int, Stride2D.DenseX>>(SoftmaxKernel);
            _layerNormKernel = _accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView2D<float, Stride2D.DenseX>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, ArrayView2D<float, Stride2D.DenseX>, float>(LayerNormKernel);

        }

        public float[] CalculateDotProduct(float[,] matrix, float[] vector)
        {
            int rows = matrix.GetLength(0);
            int cols = matrix.GetLength(1);
            if (vector.Length != cols)
            {
                throw new ArgumentException($"Expected vector of length {cols}, got {vector.Length}");
            }
            if (!_dotCache.TryGetValue((rows, cols), out var bufs))
            {
                bufs = (_accelerator.Allocate2DDenseX<float>(new Index2D(rows, cols)), _accelerator.Allocate1D<float>(cols), _accelerator.Allocate1D<float>(rows));

                _dotCache[(rows, cols)] = bufs;
            }

            bufs.mat.CopyFromCPU(matrix);
            bufs.vec.CopyFromCPU(vector);

            _dotKernel(new Index1D(rows), bufs.mat.View, bufs.vec.View, bufs.res.View);

            var result = new float[rows];
            bufs.res.CopyToCPU(result);
            return result;
        }

        private float[] CalculateDotProductTransposed(float[,] matrix, float[] vector)
        {
            int rows = matrix.GetLength(0);
            int cols = matrix.GetLength(1);
            if (vector.Length != rows)
            {
                throw new ArgumentException($"Expected vector of length {rows}, got {vector.Length}");
            }
            if (!_dotTransposedCache.TryGetValue((rows, cols), out var bufferTemp))
            {
                bufferTemp = (_accelerator.Allocate2DDenseX<float>(new Index2D(rows, cols)), _accelerator.Allocate1D<float>(rows), _accelerator.Allocate1D<float>(cols));
                _dotTransposedCache[(rows, cols)] = bufferTemp;
            }

            bufferTemp.mat.CopyFromCPU(matrix);
            bufferTemp.vec.CopyFromCPU(vector);

            _dotTransposedKernel(new Index1D(cols), bufferTemp.vec.View, bufferTemp.mat.View, bufferTemp.res.View);

            var result = new float[cols];
            bufferTemp.res.CopyToCPU(result);
            return result;
        }

        public (float[] activation, float[] derivative) ActivateLayer(float[] dot, float[] bias, ActivationType t)
        {
            int n = dot.Length;
            if (bias.Length != n)
            {
                throw new ArgumentException($"Expected bias length {n}, got {bias.Length}");
            }
            var dotBuf = _accelerator.Allocate1D<float>(n);
            var biasBuf = _accelerator.Allocate1D<float>(n);
            var actBuf = _accelerator.Allocate1D<float>(n);
            var derBuf = _accelerator.Allocate1D<float>(n);
            try
            {
                dotBuf.CopyFromCPU(dot);
                biasBuf.CopyFromCPU(bias);

                _actKernel(new Index1D(n), dotBuf.View, biasBuf.View, actBuf.View, derBuf.View, t);

                var activation = new float[n];
                var derivative = new float[n];
                actBuf.CopyToCPU(activation);
                derBuf.CopyToCPU(derivative);
                return (activation, derivative);
            }
            finally
            {
                dotBuf.Dispose(); biasBuf.Dispose(); actBuf.Dispose(); derBuf.Dispose();
            }
        }

        public float[] CalculateOutputGradients(float[] cost, float[] derivative)
        {
            int n = cost.Length;
            if (!_outGradCache.TryGetValue(n, out var bufs))
            {
                _outGradCache[n] = bufs = (
                    _accelerator.Allocate1D<float>(n),
                    _accelerator.Allocate1D<float>(n),
                    _accelerator.Allocate1D<float>(n)
                );
            }

            bufs.cost.CopyFromCPU(cost);
            bufs.der.CopyFromCPU(derivative);
            _outGradKernel(new Index1D(n), bufs.cost.View, bufs.der.View, bufs.grad.View);

            var result = new float[n];
            bufs.grad.CopyToCPU(result);
            return result;
        }

        public float[] CalculateHiddenGradients(float[,] weights, float[] nextDeltas, float[] derivative)
        {
            var pre = CalculateDotProductTransposed(weights, nextDeltas);
            int cols = pre.Length;
            if (!_hidGradCache.TryGetValue(cols, out var bufs))
            {
                _hidGradCache[cols] = bufs = (
                    _accelerator.Allocate1D<float>(cols),
                    _accelerator.Allocate1D<float>(cols),
                    _accelerator.Allocate1D<float>(cols)
                );
            }
            bufs.pre.CopyFromCPU(pre);
            bufs.der.CopyFromCPU(derivative);
            _hidGradKernel(new Index1D(cols), bufs.pre.View, bufs.der.View, bufs.delta.View);

            var result = new float[cols];
            bufs.delta.CopyToCPU(result);
            return result;
        }

        public float[,] UpdateWeights(float[,] weights, float[] deltas, float[] prevActivations, float learningRate, float lambda)
        {
            int rows = weights.GetLength(0);
            int cols = weights.GetLength(1);
            if (!_updWCache.TryGetValue((rows, cols), out var bufs))
            {
                _updWCache[(rows, cols)] = bufs = (
                    _accelerator.Allocate2DDenseX<float>(new Index2D(rows, cols)),
                    _accelerator.Allocate1D<float>(rows),
                    _accelerator.Allocate1D<float>(cols)
                );
            }

            bufs.w.CopyFromCPU(weights);
            bufs.d.CopyFromCPU(deltas);
            bufs.pa.CopyFromCPU(prevActivations);

            // 2) Pass lambda into the kernel
            _updWKernel(
                new Index2D(rows, cols),
                bufs.w.View,
                bufs.d.View,
                bufs.pa.View,
                learningRate,
                lambda
            );

            var result = new float[rows, cols];
            bufs.w.CopyToCPU(result);
            return result;
        }
        public float[] UpdateBias(float[] bias, float[] deltas, float learningRate)
        {
            int n = bias.Length;
            if (!_updBCache.TryGetValue(n, out var bufs))
            {
                _updBCache[n] = bufs = (
                    _accelerator.Allocate1D<float>(n),
                    _accelerator.Allocate1D<float>(n)
                );
            }

            bufs.b.CopyFromCPU(bias);
            bufs.d.CopyFromCPU(deltas);
            _updBKernel(new Index1D(n), bufs.b.View, bufs.d.View, learningRate);

            var result = new float[n];
            bufs.b.CopyToCPU(result);
            return result;
        }

        private static void DotKernel(Index1D row, ArrayView2D<float, Stride2D.DenseX> mat, ArrayView1D<float, Stride1D.Dense> vec, ArrayView1D<float, Stride1D.Dense> res)
        {
            float sum = 0;
            int cols = (int)mat.Extent.Y;
            for (int j = 0; j < cols; ++j)
            {
                sum += mat[row, j] * vec[j];
            }
            res[row] = sum;
        }
        private static void DotTransposedKernel(Index1D col, ArrayView1D<float, Stride1D.Dense> vec, ArrayView2D<float, Stride2D.DenseX> mat, ArrayView1D<float, Stride1D.Dense> res)
        {
            float sum = 0;
            int rows = (int)mat.Extent.X;
            for (int i = 0; i < rows; ++i)
            {
                sum += mat[i, col] * vec[i];
            }
            res[col] = sum;
        }

        private static void ActivateKernel(Index1D i,
            ArrayView1D<float, Stride1D.Dense> dot,
            ArrayView1D<float, Stride1D.Dense> bias,
            ArrayView1D<float, Stride1D.Dense> act,
            ArrayView1D<float, Stride1D.Dense> der,
            ActivationType t)
        {
            float z = dot[i] + bias[i];
            float a, d;
            switch (t)
            {
                case ActivationType.None:
                    a = z;
                    d = 1f;
                    break;
                case ActivationType.Sigmoid:
                    a = 1.0f / (1.0f + XMath.Exp(-z));
                    d = a * (1f - a);
                    break;
                case ActivationType.Tanh:
                    a = XMath.Tanh(z);
                    d = 1f - a * a;
                    break;
                case ActivationType.Relu:
                    a = XMath.Max(0.0f, z);
                    d = z >= 0f ? 1f : 0f;
                    break;
                case ActivationType.Leakyrelu:
                    a = z > 0f ? z : 0.01f * z;
                    d = z >= 0f ? 1f : 0.1f;
                    break;
                default:
                    a = 1.0f / (1.0f + XMath.Exp(-z));
                    d = a * (1f - a);
                    break;
            }
            act[i] = a;
            der[i] = d;
        }
        private static void OutputGradientKernel(Index1D i, ArrayView1D<float, Stride1D.Dense> cost,
            ArrayView1D<float, Stride1D.Dense> der,
            ArrayView1D<float, Stride1D.Dense> grad)
        {
            grad[i] = -cost[i] * der[i];
        }
        private static void HiddenGradientKernel(Index1D i,
             ArrayView1D<float, Stride1D.Dense> pre,
             ArrayView1D<float, Stride1D.Dense> der,
             ArrayView1D<float, Stride1D.Dense> delta)
        {
            delta[i] = pre[i] * der[i];
        }

        private static void UpdateWeightsKernel(
             Index2D idx,
             ArrayView2D<float, Stride2D.DenseX> W,
             ArrayView1D<float, Stride1D.Dense> delta,
             ArrayView1D<float, Stride1D.Dense> prevAct,
             float learningRate,
             float lambda)
        {
            int r = idx.X, c = idx.Y;
            float grad = delta[r] * prevAct[c];
            float decay = lambda * W[r, c];
            W[r, c] -= learningRate * (grad + decay);
        }
        private static void UpdateBiasKernel(Index1D i, ArrayView1D<float, Stride1D.Dense> bias, ArrayView1D<float, Stride1D.Dense> delta, float learningRate)
        {
            bias[i] -= learningRate * delta[i];
        }
        public float[,] MatrixMultiply(float[,] A, float[,] B)
        {
            int r1 = A.GetLength(0);
            int c1 = A.GetLength(1);
            int r2 = B.GetLength(0);
            int c2 = B.GetLength(1);

            if (c1 != r2)
                throw new ArgumentException($"Matrix dimensions don't match");

            var key = (r1, c1, c2);
            if (!_matMulCache.TryGetValue(key, out var bufs))
            {
                bufs = (
                    _accelerator.Allocate2DDenseX<float>(new Index2D(r1, c1)),
                    _accelerator.Allocate2DDenseX<float>(new Index2D(r2, c2)),
                    _accelerator.Allocate2DDenseX<float>(new Index2D(r1, c2))
                );
                _matMulCache[key] = bufs;
            }

            bufs.a.CopyFromCPU(A);
            bufs.b.CopyFromCPU(B);
            _matMulKernel(new Index2D(r1, c2), bufs.a.View, bufs.b.View, bufs.c.View);

            var result = new float[r1, c2];
            bufs.c.CopyToCPU(result);
            return result;
        }

        public float[,] MatrixMultiplyTranspose(float[,] A, float[,] B)
        {
            int r1 = A.GetLength(0);
            int c1 = A.GetLength(1);
            int r2 = B.GetLength(0);
            int c2 = B.GetLength(1);

            if (c1 != c2)
                throw new ArgumentException($"Matrix dimensions don't match for A*B^T");

            var bufA = _accelerator.Allocate2DDenseX<float>(new Index2D(r1, c1));
            var bufB = _accelerator.Allocate2DDenseX<float>(new Index2D(r2, c2));
            var bufC = _accelerator.Allocate2DDenseX<float>(new Index2D(r1, r2));

            try
            {
                bufA.CopyFromCPU(A);
                bufB.CopyFromCPU(B);
                _matMulTransposeKernel(new Index2D(r1, r2), bufA.View, bufB.View, bufC.View);

                var result = new float[r1, r2];
                bufC.CopyToCPU(result);
                return result;
            }
            finally
            {
                bufA.Dispose(); bufB.Dispose(); bufC.Dispose();
            }
        }

        public float[,] MatrixScale(float[,] matrix, float scalar)
        {
            int rows = matrix.GetLength(0);
            int cols = matrix.GetLength(1);

            var bufIn = _accelerator.Allocate2DDenseX<float>(new Index2D(rows, cols));
            try
            {
                bufIn.CopyFromCPU(matrix);
                _matScaleKernel(new Index2D(rows, cols), bufIn.View, scalar);

                var result = new float[rows, cols];
                bufIn.CopyToCPU(result);
                return result;
            }
            finally
            {
                bufIn.Dispose();
            }
        }

        public float[,] MatrixAdd(float[,] A, float[,] B)
        {
            int rows = A.GetLength(0);
            int cols = A.GetLength(1);

            var bufA = _accelerator.Allocate2DDenseX<float>(new Index2D(rows, cols));
            var bufB = _accelerator.Allocate2DDenseX<float>(new Index2D(rows, cols));
            var bufC = _accelerator.Allocate2DDenseX<float>(new Index2D(rows, cols));

            try
            {
                bufA.CopyFromCPU(A);
                bufB.CopyFromCPU(B);
                _matAddKernel(new Index2D(rows, cols), bufA.View, bufB.View, bufC.View);

                var result = new float[rows, cols];
                bufC.CopyToCPU(result);
                return result;
            }
            finally
            {
                bufA.Dispose(); bufB.Dispose(); bufC.Dispose();
            }
        }

        public float[,] Softmax(float[,] matrix, bool[,] mask = null)
        {
            int rows = matrix.GetLength(0);
            int cols = matrix.GetLength(1);

            var bufIn = _accelerator.Allocate2DDenseX<float>(new Index2D(rows, cols));
            var bufOut = _accelerator.Allocate2DDenseX<float>(new Index2D(rows, cols));

            // FIXED: Convert bool mask to int mask (ILGPU doesn't support bool)
            var bufMask = mask != null ? _accelerator.Allocate2DDenseX<int>(new Index2D(rows, cols)) : _accelerator.Allocate2DDenseX<int>(new Index2D(1, 1));

            try
            {
                bufIn.CopyFromCPU(matrix);
                if (mask != null)
                {
                    // Convert bool[,] to int[,]
                    var intMask = new int[rows, cols];
                    for (int i = 0; i < rows; i++)
                        for (int j = 0; j < cols; j++)
                            intMask[i, j] = mask[i, j] ? 1 : 0;
                    bufMask.CopyFromCPU(intMask);
                }

                _softmaxKernel(new Index1D(rows), bufIn.View, bufOut.View, bufMask.View);

                var result = new float[rows, cols];
                bufOut.CopyToCPU(result);
                return result;
            }
            finally
            {
                bufIn.Dispose(); bufOut.Dispose(); bufMask.Dispose();
            }
        }

        public float[,] LayerNorm(float[,] input, float[] gamma, float[] beta, float epsilon = 1e-5f)
        {
            int batchSize = input.GetLength(0);
            int features = input.GetLength(1);

            var bufIn = _accelerator.Allocate2DDenseX<float>(new Index2D(batchSize, features));
            var bufGamma = _accelerator.Allocate1D<float>(features);
            var bufBeta = _accelerator.Allocate1D<float>(features);
            var bufOut = _accelerator.Allocate2DDenseX<float>(new Index2D(batchSize, features));

            try
            {
                bufIn.CopyFromCPU(input);
                bufGamma.CopyFromCPU(gamma);
                bufBeta.CopyFromCPU(beta);

                _layerNormKernel(new Index1D(batchSize), bufIn.View, bufGamma.View, bufBeta.View, bufOut.View, epsilon);

                var result = new float[batchSize, features];
                bufOut.CopyToCPU(result);
                return result;
            }
            finally
            {
                bufIn.Dispose(); bufGamma.Dispose(); bufBeta.Dispose(); bufOut.Dispose();
            }
        }

        #region GPU Kernels

        private static void MatMulKernel(Index2D idx, ArrayView2D<float, Stride2D.DenseX> A, ArrayView2D<float, Stride2D.DenseX> B, ArrayView2D<float, Stride2D.DenseX> C)
        {
            int row = idx.X, col = idx.Y;
            float sum = 0.0f;
            int K = (int)A.Extent.Y;
            for (int k = 0; k < K; k++)
                sum += A[row, k] * B[k, col];
            C[row, col] = sum;
        }

        private static void MatMulTransposeKernel(Index2D idx, ArrayView2D<float, Stride2D.DenseX> A, ArrayView2D<float, Stride2D.DenseX> B, ArrayView2D<float, Stride2D.DenseX> C)
        {
            int row = idx.X, col = idx.Y;
            float sum = 0.0f;
            int K = (int)A.Extent.Y;
            for (int k = 0; k < K; k++)
                sum += A[row, k] * B[col, k];  // B[col,k] not B[k,col]
            C[row, col] = sum;
        }

        private static void MatScaleKernel(Index2D idx, ArrayView2D<float, Stride2D.DenseX> mat, float scalar)
        {
            mat[idx] *= scalar;
        }

        private static void MatAddKernel(Index2D idx, ArrayView2D<float, Stride2D.DenseX> A, ArrayView2D<float, Stride2D.DenseX> B, ArrayView2D<float, Stride2D.DenseX> C)
        {
            C[idx] = A[idx] + B[idx];
        }

        private static void SoftmaxKernel(Index1D row, ArrayView2D<float, Stride2D.DenseX> input, ArrayView2D<float, Stride2D.DenseX> output, ArrayView2D<int, Stride2D.DenseX> mask)
        {
            int cols = (int)input.Extent.Y;
            bool hasMask = mask.Extent.Size > 1;

            float max = float.NegativeInfinity;
            for (int j = 0; j < cols; j++)
            {
                // Convert int to bool: 0 = false, non-zero = true
                if (!hasMask || mask[row, j] != 0)
                    max = XMath.Max(max, input[row, j]);
            }

            float sum = 0.0f;
            for (int j = 0; j < cols; j++)
            {
                if (hasMask && mask[row, j] == 0)
                    output[row, j] = 0.0f;
                else
                {
                    output[row, j] = XMath.Exp(input[row, j] - max);
                    sum += output[row, j];
                }
            }

            for (int j = 0; j < cols; j++)
                output[row, j] /= sum;
        }
        private static void LayerNormKernel(Index1D batch, ArrayView2D<float, Stride2D.DenseX> input, ArrayView1D<float, Stride1D.Dense> gamma, ArrayView1D<float, Stride1D.Dense> beta, ArrayView2D<float, Stride2D.DenseX> output, float epsilon)
        {
            int features = (int)input.Extent.Y;

            float mean = 0.0f;
            for (int j = 0; j < features; j++)
                mean += input[batch, j];
            mean /= features;

            float variance = 0.0f;
            for (int j = 0; j < features; j++)
            {
                float diff = input[batch, j] - mean;
                variance += diff * diff;
            }
            variance /= features;

            float stdDev = XMath.Sqrt(variance + epsilon);
            for (int j = 0; j < features; j++)
                output[batch, j] = gamma[j] * (input[batch, j] - mean) / stdDev + beta[j];
        }

        #endregion
        public void Dispose()
        {
            foreach (var v in _matMulCache.Values)
            {
                v.a.Dispose();
                v.b.Dispose();
                v.c.Dispose();
            }
            foreach (var v in _outGradCache.Values) { v.cost.Dispose(); v.der.Dispose(); v.grad.Dispose(); }
            foreach (var v in _hidGradCache.Values) { v.pre.Dispose(); v.der.Dispose(); v.delta.Dispose(); }
            foreach (var v in _updWCache.Values) { v.w.Dispose(); v.d.Dispose(); v.pa.Dispose(); }
            foreach (var v in _updBCache.Values) { v.b.Dispose(); v.d.Dispose(); }
            foreach (var v in _dotTransposedCache.Values) { v.mat.Dispose(); v.vec.Dispose(); v.res.Dispose(); }
            _accelerator.Dispose();
        }
    }
}