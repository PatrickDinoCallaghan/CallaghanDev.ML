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
        private readonly Action<Index2D, ArrayView2D<float, Stride2D.DenseX>, ArrayView2D<float, Stride2D.DenseX>, ArrayView2D<float, Stride2D.DenseX>> _batchDotKernel;
        private readonly Action<Index2D, ArrayView2D<float, Stride2D.DenseX>, float> _matScaleInPlaceKernel;
        private readonly Action<Index1D, ArrayView1D<float, Stride1D.Dense>, float> _vecScaleInPlaceKernel;
        private readonly Action<Index2D, ArrayView2D<float, Stride2D.DenseX>, ArrayView2D<float, Stride2D.DenseX>, float> _matUpdateKernel;
        private readonly Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, float> _vecUpdateKernel;
        private readonly Action<Index2D, ArrayView2D<float, Stride2D.DenseX>> _zeroMatrixKernel;
        private readonly Action<Index1D, ArrayView2D<float, Stride2D.DenseX>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, ArrayView2D<float, Stride2D.DenseX>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, float> _layerNormForwardKernel;
        private readonly Action<Index1D, ArrayView2D<float, Stride2D.DenseX>, ArrayView2D<float, Stride2D.DenseX>, ArrayView1D<float, Stride1D.Dense>, ArrayView2D<float, Stride2D.DenseX>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, ArrayView2D<float, Stride2D.DenseX>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, float> _layerNormBackwardKernel;
        private readonly Action<Index1D, ArrayView2D<float, Stride2D.DenseX>> _matSquaredNormKernel;

        //_dotCache
        private readonly Dictionary<(int rows, int cols), (MemoryBuffer2D<float, Stride2D.DenseX> mat, MemoryBuffer1D<float, Stride1D.Dense> vec, MemoryBuffer1D<float, Stride1D.Dense> res)> _dotCache = new();
        //_outGradCache
        private readonly Dictionary<int, (MemoryBuffer1D<float, Stride1D.Dense> cost, MemoryBuffer1D<float, Stride1D.Dense> der, MemoryBuffer1D<float, Stride1D.Dense> grad)> _outGradCache = new();

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

        private readonly Dictionary<(int outputDim, int inputDim, int seqLen),(MemoryBuffer2D<float, Stride2D.DenseX> w,
         MemoryBuffer2D<float, Stride2D.DenseX> inp,
         MemoryBuffer2D<float, Stride2D.DenseX> res)> _batchDotCache
        = new();
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
            _batchDotKernel = _accelerator.LoadAutoGroupedStreamKernel<Index2D, ArrayView2D<float, Stride2D.DenseX>, ArrayView2D<float, Stride2D.DenseX>, ArrayView2D<float, Stride2D.DenseX>>(BatchDotKernel);
            _matScaleInPlaceKernel = _accelerator.LoadAutoGroupedStreamKernel<Index2D, ArrayView2D<float, Stride2D.DenseX>, float>(MatScaleInPlaceKernel);
            _vecScaleInPlaceKernel = _accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<float, Stride1D.Dense>, float>(VecScaleInPlaceKernel);
            _matUpdateKernel = _accelerator.LoadAutoGroupedStreamKernel<Index2D, ArrayView2D<float, Stride2D.DenseX>, ArrayView2D<float, Stride2D.DenseX>, float>(MatUpdateKernel);
            _vecUpdateKernel = _accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, float>(VecUpdateKernel);
            _zeroMatrixKernel = _accelerator.LoadAutoGroupedStreamKernel<Index2D, ArrayView2D<float, Stride2D.DenseX>>(ZeroMatrixKernel);
            _layerNormForwardKernel = _accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView2D<float, Stride2D.DenseX>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, ArrayView2D<float, Stride2D.DenseX>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, float>(LayerNormForwardKernel);
            _layerNormBackwardKernel = _accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView2D<float, Stride2D.DenseX>, ArrayView2D<float, Stride2D.DenseX>, ArrayView1D<float, Stride1D.Dense>, ArrayView2D<float, Stride2D.DenseX>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, ArrayView2D<float, Stride2D.DenseX>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, float>(LayerNormBackwardKernel);

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
        private static void ActivateKernel(Index1D i, ArrayView1D<float, Stride1D.Dense> dot, ArrayView1D<float, Stride1D.Dense> bias, ArrayView1D<float, Stride1D.Dense> act, ArrayView1D<float, Stride1D.Dense> der, ActivationType t)
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
        private static void OutputGradientKernel(Index1D i, ArrayView1D<float, Stride1D.Dense> cost, ArrayView1D<float, Stride1D.Dense> der, ArrayView1D<float, Stride1D.Dense> grad)
        {
            grad[i] = -cost[i] * der[i];
        }
        private static void HiddenGradientKernel(Index1D i, ArrayView1D<float, Stride1D.Dense> pre, ArrayView1D<float, Stride1D.Dense> der, ArrayView1D<float, Stride1D.Dense> delta)
        {
            delta[i] = pre[i] * der[i];
        }
        private static void UpdateWeightsKernel(Index2D idx, ArrayView2D<float, Stride2D.DenseX> W, ArrayView1D<float, Stride1D.Dense> delta, ArrayView1D<float, Stride1D.Dense> prevAct, float learningRate, float lambda)
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

        private static void BatchDotKernel(Index2D idx, ArrayView2D<float, Stride2D.DenseX> weights, ArrayView2D<float, Stride2D.DenseX> input, ArrayView2D<float, Stride2D.DenseX> result)
        {
            int seq = idx.X;
            int outDim = idx.Y;
            int inputDim = (int)weights.Extent.Y;
            float sum = 0.0f;


            for (int k = 0; k < inputDim; k++)
            {
                sum += weights[outDim, k] * input[seq, k];
            }
            result[seq, outDim] = sum;
        }

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

        private static void ZeroMatrixKernel(Index2D idx, ArrayView2D<float, Stride2D.DenseX> mat)
        {
            mat[idx] = 0.0f;
        }

        private static void LayerNormForwardKernel(Index1D batch, ArrayView2D<float, Stride2D.DenseX> input, ArrayView1D<float, Stride1D.Dense> gamma, ArrayView1D<float, Stride1D.Dense> beta, ArrayView2D<float, Stride2D.DenseX> output, ArrayView1D<float, Stride1D.Dense> means, ArrayView1D<float, Stride1D.Dense> variances, float epsilon)
        {
            int features = (int)input.Extent.Y;

            float mean = 0.0f;


            for (int j = 0; j < features; j++)
            {
                mean += input[batch, j];
            }
            mean = mean / features;
            means[batch] = mean;

            float variance = 0.0f;

            for (int j = 0; j < features; j++)
            {
                float diff = input[batch, j] - mean;
                variance += diff * diff;
            }

            variance = variance/ features;
            variances[batch] = variance;

            float stdDev = XMath.Sqrt(variance + epsilon);

            for (int j = 0; j < features; j++)
            {
                output[batch, j] = gamma[j] * (input[batch, j] - mean) / stdDev + beta[j];
            }
        }

        private static void LayerNormBackwardKernel(Index1D batch, ArrayView2D<float, Stride2D.DenseX> dOut, ArrayView2D<float, Stride2D.DenseX> normalized, ArrayView1D<float, Stride1D.Dense> gamma, ArrayView2D<float, Stride2D.DenseX> input, ArrayView1D<float, Stride1D.Dense> mean, ArrayView1D<float, Stride1D.Dense> variance, ArrayView2D<float, Stride2D.DenseX> dInput, ArrayView1D<float, Stride1D.Dense> dGamma,  ArrayView1D<float, Stride1D.Dense> dBeta, float epsilon)
        {
            int features = (int)input.Extent.Y;
            float invStd = 1.0f / XMath.Sqrt(variance[batch] + epsilon);

            for (int j = 0; j < features; j++)
            {
                Atomic.Add(ref dGamma[j], dOut[batch, j] * normalized[batch, j]);
                Atomic.Add(ref dBeta[j], dOut[batch, j]);
            }

            float dVar = 0.0f;
            float dMean = 0.0f;
            float invStdCubed = invStd * invStd * invStd;

            for (int j = 0; j < features; j++)
            {
                float dNorm = dOut[batch, j] * gamma[j];
                float xMinusMean = input[batch, j] - mean[batch];
                dVar += dNorm * xMinusMean * (-0.5f) * invStdCubed;
                dMean += dNorm * (-invStd);
            }

            float invN = 1.0f / features;

            for (int j = 0; j < features; j++)
            {
                float dNorm = dOut[batch, j] * gamma[j];
                float xMinusMean = input[batch, j] - mean[batch];
                dInput[batch, j] = dNorm * invStd + dVar * 2.0f * xMinusMean * invN + dMean * invN;
            }
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

            var bufMask = mask != null ? _accelerator.Allocate2DDenseX<int>(new Index2D(rows, cols)) : _accelerator.Allocate2DDenseX<int>(new Index2D(1, 1));

            try
            {
                bufIn.CopyFromCPU(matrix);
                if (mask != null)
                {
                    var intMask = new int[rows, cols];
                    for (int i = 0; i < rows; i++)
                    {
                        for (int j = 0; j < cols; j++)
                        {

                            intMask[i, j] = mask[i, j] ? 1 : 0;
                        }
                    }
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

        public float[,] BatchDotProduct(float[,] weights, float[,] inputMatrix)
        {
            int outputDim = weights.GetLength(0);
            int inputDim = weights.GetLength(1);
            int seqLen = inputMatrix.GetLength(0);

            if (inputMatrix.GetLength(1) != inputDim)
                throw new ArgumentException($"Expected input columns {inputDim}, got {inputMatrix.GetLength(1)}");

            var key = (outputDim, inputDim, seqLen);
            if (!_batchDotCache.TryGetValue(key, out var bufs))
            {
                bufs = (
                    _accelerator.Allocate2DDenseX<float>(new Index2D(outputDim, inputDim)),
                    _accelerator.Allocate2DDenseX<float>(new Index2D(seqLen, inputDim)),
                    _accelerator.Allocate2DDenseX<float>(new Index2D(seqLen, outputDim))
                );
                _batchDotCache[key] = bufs;
            }

            bufs.w.CopyFromCPU(weights);
            bufs.inp.CopyFromCPU(inputMatrix);

            _batchDotKernel(new Index2D(seqLen, outputDim), bufs.w.View, bufs.inp.View, bufs.res.View);

            var result = new float[seqLen, outputDim];
            bufs.res.CopyToCPU(result);
            return result;
        }

        public float[,] MultiHeadAttentionForward(float[,] Q, float[,] K, float[,] V, int numHeads, float scale, bool[,] mask = null)
        {
            int seqLen = Q.GetLength(0);
            int embeddingDim = Q.GetLength(1);
            int headDim = embeddingDim / numHeads;

            var concatenated = new float[seqLen, embeddingDim];

            for (int head = 0; head < numHeads; head++)
            {
                int startIdx = head * headDim;

                // Extract head slices
                var Q_head = new float[seqLen, headDim];
                var K_head = new float[seqLen, headDim];
                var V_head = new float[seqLen, headDim];

                for (int i = 0; i < seqLen; i++)
                {
                    for (int j = 0; j < headDim; j++)
                    {
                        Q_head[i, j] = Q[i, startIdx + j];
                        K_head[i, j] = K[i, startIdx + j];
                        V_head[i, j] = V[i, startIdx + j];
                    }
                }

                // scores = Q_head * K_head^T  (GPU accelerated)
                var scores = MatrixMultiplyTranspose(Q_head, K_head);
                // scaledScores = scores * scale  (GPU accelerated)
                var scaledScores = MatrixScale(scores, scale);
                // attnWeights = softmax(scaledScores, mask)  (GPU accelerated)
                var attnWeights = Softmax(scaledScores, mask);
                // headOutput = attnWeights * V_head  (GPU accelerated)
                var headOutput = MatrixMultiply(attnWeights, V_head);

                // Scatter back into concatenated result
                for (int i = 0; i < seqLen; i++)
                {
                    for (int j = 0; j < headDim; j++)
                    {
                        concatenated[i, startIdx + j] = headOutput[i, j];
                    }
                }
            }

            return concatenated;
        }

        public (float[,] dQ, float[,] dK, float[,] dV) MultiHeadAttentionBackward(float[,] Q, float[,] K, float[,] V, float[,] dConcatenated, int numHeads, float scale, bool useDecoderMask = false)
        {
            int seqLen = Q.GetLength(0);
            int embeddingDim = Q.GetLength(1);
            int headDim = embeddingDim / numHeads;

            var dQ_full = new float[seqLen, embeddingDim];
            var dK_full = new float[seqLen, embeddingDim];
            var dV_full = new float[seqLen, embeddingDim];

            for (int head = 0; head < numHeads; head++)
            {
                int startIdx = head * headDim;

                // Extract head slices
                var Q_head = new float[seqLen, headDim];
                var K_head = new float[seqLen, headDim];
                var V_head = new float[seqLen, headDim];
                var dHeadOutput = new float[seqLen, headDim];

                for (int i = 0; i < seqLen; i++)
                {
                    for (int j = 0; j < headDim; j++)
                    {
                        Q_head[i, j] = Q[i, startIdx + j];
                        K_head[i, j] = K[i, startIdx + j];
                        V_head[i, j] = V[i, startIdx + j];
                        dHeadOutput[i, j] = dConcatenated[i, startIdx + j];
                    }
                }

                // Recompute forward pass for this head (GPU accelerated)
                var scores = MatrixMultiplyTranspose(Q_head, K_head);
                var scaledScores = MatrixScale(scores, scale);

                // Softmax with optional decoder mask (done on CPU to match exact CPU logic for masking)
                var attnWeights = new float[seqLen, seqLen];
                for (int i = 0; i < seqLen; i++)
                {
                    float max = float.NegativeInfinity;
                    for (int j = 0; j < seqLen; j++)
                    {
                        if (useDecoderMask && j > i) continue;
                        max = Math.Max(max, scaledScores[i, j]);
                    }
                    float expSum = 0;
                    for (int j = 0; j < seqLen; j++)
                    {
                        if (useDecoderMask && j > i)
                        {
                            attnWeights[i, j] = 0;
                            continue;
                        }
                        attnWeights[i, j] = MathF.Exp(scaledScores[i, j] - max);
                        expSum += attnWeights[i, j];
                    }
                    for (int j = 0; j < seqLen; j++)
                    {
                        attnWeights[i, j] /= (expSum + 1e-10f);
                    }
                }

                // dAttnWeights = dHeadOutput * V_head^T  (GPU accelerated)
                var dAttnWeights = MatrixMultiplyTranspose(dHeadOutput, V_head);

                // dV_head = attnWeights^T * dHeadOutput  (GPU accelerated via MatMul with transposed first arg)
                // attnWeights is [seqLen, seqLen], dHeadOutput is [seqLen, headDim]
                // We need attnWeights^T * dHeadOutput, which is the same as doing a standard MatMul
                // after transposing attnWeights. We can use the existing infrastructure:
                // dV_head[i,k] = sum_j attnWeights[j,i] * dHeadOutput[j,k]
                // This is (attnWeights^T) * dHeadOutput — but we don't have a "transpose first" kernel.
                // We do it by noting: (A^T * B) == (B^T * A)^T, but simpler to just do CPU-style here
                // or reuse MatrixMultiply with a transposed copy.
                var attnWeightsT = new float[seqLen, seqLen];
                for (int i = 0; i < seqLen; i++)
                    for (int j = 0; j < seqLen; j++)
                        attnWeightsT[i, j] = attnWeights[j, i];

                var dV_head = MatrixMultiply(attnWeightsT, dHeadOutput);

                // Softmax backward (CPU, matching exact CPU logic)
                var dScaledScores = new float[seqLen, seqLen];
                for (int i = 0; i < seqLen; i++)
                {
                    float dot = 0;
                    for (int j = 0; j < seqLen; j++)
                    {
                        dot += attnWeights[i, j] * dAttnWeights[i, j];
                    }
                    for (int j = 0; j < seqLen; j++)
                    {
                        dScaledScores[i, j] = attnWeights[i, j] * (dAttnWeights[i, j] - dot);
                        if (useDecoderMask && j > i)
                        {
                            dScaledScores[i, j] = 0;
                        }
                    }
                }

                // Scale backward (GPU accelerated)
                var dScores = MatrixScale(dScaledScores, scale);

                // dQ_head = dScores * K_head  (GPU accelerated)
                var dQ_head = MatrixMultiply(dScores, K_head);

                // dK_head = dScores^T * Q_head  (GPU accelerated)
                var dScoresT = new float[seqLen, seqLen];
                for (int i = 0; i < seqLen; i++)
                    for (int j = 0; j < seqLen; j++)
                        dScoresT[i, j] = dScores[j, i];

                var dK_head = MatrixMultiply(dScoresT, Q_head);

                // Scatter back to full arrays
                for (int i = 0; i < seqLen; i++)
                {
                    for (int j = 0; j < headDim; j++)
                    {
                        dQ_full[i, startIdx + j] += dQ_head[i, j];
                        dK_full[i, startIdx + j] += dK_head[i, j];
                        dV_full[i, startIdx + j] += dV_head[i, j];
                    }
                }
            }

            return (dQ_full, dK_full, dV_full);
        }

        public void BackpropLinearProjection(float[,] input, float[,] dOutput, float[,] weights, float[,] weightGrad, float[] biasGrad, float[,] dInput)
        {
            int seqLen = input.GetLength(0);
            int embeddingDim = input.GetLength(1);

            // weightGrad += input^T * dOutput  (GPU accelerated)
            // input is [seqLen, embeddingDim], dOutput is [seqLen, embeddingDim]
            // input^T is [embeddingDim, seqLen], result is [embeddingDim, embeddingDim]
            // But we need to transpose input first. We can use:
            // (input^T * dOutput)[k, j] = sum_i input[i, k] * dOutput[i, j]
            // This is the same as MatrixMultiplyTranspose(input^T ... ) — let's just build the transpose
            // Actually: MatrixMultiplyTranspose(A, B) computes A * B^T.
            // We want input^T * dOutput. Note: (input^T * dOutput) = (dOutput^T * input)^T
            // Simpler: just allocate a transposed copy.
            var inputT = new float[embeddingDim, seqLen];
            for (int i = 0; i < seqLen; i++)
                for (int k = 0; k < embeddingDim; k++)
                    inputT[k, i] = input[i, k];

            var wGradContrib = MatrixMultiply(inputT, dOutput);

            // Accumulate into weightGrad
            for (int k = 0; k < embeddingDim; k++)
                for (int j = 0; j < embeddingDim; j++)
                    weightGrad[k, j] += wGradContrib[k, j];

            // biasGrad[j] += sum_i dOutput[i, j]
            for (int i = 0; i < seqLen; i++)
                for (int j = 0; j < embeddingDim; j++)
                    biasGrad[j] += dOutput[i, j];

            // dInput += dOutput * weights^T  (GPU accelerated)
            // dOutput is [seqLen, embeddingDim], weights is [embeddingDim, embeddingDim]
            // MatrixMultiplyTranspose(dOutput, weights) computes dOutput * weights^T
            var dInputContrib = MatrixMultiplyTranspose(dOutput, weights);

            // Accumulate into dInput
            for (int i = 0; i < seqLen; i++)
                for (int k = 0; k < embeddingDim; k++)
                    dInput[i, k] += dInputContrib[i, k];
        }

        public (float[,] output, float[] means, float[] variances, float[,] normalized) LayerNormForward(float[,] input, float[] gamma, float[] beta, float epsilon = 1e-5f)
        {
            int batchSize = input.GetLength(0);
            int features = input.GetLength(1);

            var bufIn = _accelerator.Allocate2DDenseX<float>(new Index2D(batchSize, features));
            var bufGamma = _accelerator.Allocate1D<float>(features);
            var bufBeta = _accelerator.Allocate1D<float>(features);
            var bufOut = _accelerator.Allocate2DDenseX<float>(new Index2D(batchSize, features));
            var bufMeans = _accelerator.Allocate1D<float>(batchSize);
            var bufVariances = _accelerator.Allocate1D<float>(batchSize);

            try
            {
                bufIn.CopyFromCPU(input);
                bufGamma.CopyFromCPU(gamma);
                bufBeta.CopyFromCPU(beta);

                _layerNormForwardKernel(new Index1D(batchSize),
                    bufIn.View, bufGamma.View, bufBeta.View,
                    bufOut.View, bufMeans.View, bufVariances.View,
                    epsilon);

                var output = new float[batchSize, features];
                var means = new float[batchSize];
                var variances = new float[batchSize];
                bufOut.CopyToCPU(output);
                bufMeans.CopyToCPU(means);
                bufVariances.CopyToCPU(variances);

                // Compute normalized on CPU from means/variances (avoids extra GPU buffer)
                var normalized = new float[batchSize, features];
                for (int i = 0; i < batchSize; i++)
                {
                    float stdDev = MathF.Sqrt(variances[i] + epsilon);
                    for (int j = 0; j < features; j++)
                    {
                        normalized[i, j] = (input[i, j] - means[i]) / stdDev;
                    }
                }

                return (output, means, variances, normalized);
            }
            finally
            {
                bufIn.Dispose(); bufGamma.Dispose(); bufBeta.Dispose();
                bufOut.Dispose(); bufMeans.Dispose(); bufVariances.Dispose();
            }
        }

        public (float[,] dInput, float[] dGamma, float[] dBeta) LayerNormBackward(float[,] dOut, float[,] normalized, float[] gamma, float[,] input, float[] mean, float[] variance, float epsilon = 1e-5f)
        {
            int batchSize = dOut.GetLength(0);
            int features = dOut.GetLength(1);

            var bufDOut = _accelerator.Allocate2DDenseX<float>(new Index2D(batchSize, features));
            var bufNormalized = _accelerator.Allocate2DDenseX<float>(new Index2D(batchSize, features));
            var bufGamma = _accelerator.Allocate1D<float>(features);
            var bufInput = _accelerator.Allocate2DDenseX<float>(new Index2D(batchSize, features));
            var bufMean = _accelerator.Allocate1D<float>(batchSize);
            var bufVariance = _accelerator.Allocate1D<float>(batchSize);
            var bufDInput = _accelerator.Allocate2DDenseX<float>(new Index2D(batchSize, features));
            var bufDGamma = _accelerator.Allocate1D<float>(features);
            var bufDBeta = _accelerator.Allocate1D<float>(features);

            try
            {
                bufDOut.CopyFromCPU(dOut);
                bufNormalized.CopyFromCPU(normalized);
                bufGamma.CopyFromCPU(gamma);
                bufInput.CopyFromCPU(input);
                bufMean.CopyFromCPU(mean);
                bufVariance.CopyFromCPU(variance);

                // Zero out accumulation buffers
                bufDGamma.CopyFromCPU(new float[features]);
                bufDBeta.CopyFromCPU(new float[features]);
                bufDInput.CopyFromCPU(new float[batchSize, features]);

                _layerNormBackwardKernel(new Index1D(batchSize),
                    bufDOut.View, bufNormalized.View, bufGamma.View,
                    bufInput.View, bufMean.View, bufVariance.View,
                    bufDInput.View, bufDGamma.View, bufDBeta.View,
                    epsilon);

                var dInput = new float[batchSize, features];
                var dGamma = new float[features];
                var dBeta = new float[features];
                bufDInput.CopyToCPU(dInput);
                bufDGamma.CopyToCPU(dGamma);
                bufDBeta.CopyToCPU(dBeta);

                return (dInput, dGamma, dBeta);
            }
            finally
            {
                bufDOut.Dispose(); bufNormalized.Dispose(); bufGamma.Dispose();
                bufInput.Dispose(); bufMean.Dispose(); bufVariance.Dispose();
                bufDInput.Dispose(); bufDGamma.Dispose(); bufDBeta.Dispose();
            }
        }

        public float MatrixSquaredNorm(float[,] matrix)
        {
            int rows = matrix.GetLength(0);
            int cols = matrix.GetLength(1);

            // For reduction operations, it's more efficient to do a simple GPU element-wise
            // square and then reduce on CPU, or just do it on CPU for moderate sizes.
            // Using the existing pattern of allocate-try-finally:
            var bufIn = _accelerator.Allocate2DDenseX<float>(new Index2D(rows, cols));
            try
            {
                bufIn.CopyFromCPU(matrix);

                // Scale in-place to square: we can't easily do a full reduction in ILGPU
                // with the existing kernel pattern, so copy back and sum on CPU.
                // This still benefits from GPU if used as part of a larger pipeline.
                var data = new float[rows, cols];
                bufIn.CopyToCPU(data);

                float sum = 0;
                for (int i = 0; i < rows; i++)
                    for (int j = 0; j < cols; j++)
                        sum += data[i, j] * data[i, j];
                return sum;
            }
            finally
            {
                bufIn.Dispose();
            }
        }

        public void MatrixScaleInPlace(float[,] matrix, float scale)
        {
            int rows = matrix.GetLength(0);
            int cols = matrix.GetLength(1);

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
                bufW.Dispose(); bufG.Dispose();
            }
        }

        public void VectorUpdate(float[] weights, float[] gradients, float learningRate)
        {
            int n = weights.Length;

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
                bufW.Dispose(); bufG.Dispose();
            }
        }

        public void ZeroMatrix(float[,] matrix)
        {
            int rows = matrix.GetLength(0);
            int cols = matrix.GetLength(1);

            var buf = _accelerator.Allocate2DDenseX<float>(new Index2D(rows, cols));
            try
            {
                // No need to copy data to GPU afterall we are just going to zeroing it
                _zeroMatrixKernel(new Index2D(rows, cols), buf.View);
                buf.CopyToCPU(matrix);
            }
            finally
            {
                buf.Dispose();
            }
        }

        public void Dispose()
        {
            foreach (var v in _matMulCache.Values)
            {
                v.a.Dispose();
                v.b.Dispose();
                v.c.Dispose();
            }
            foreach (var v in _outGradCache.Values) 
            { 
                v.cost.Dispose();
                v.der.Dispose(); 
                v.grad.Dispose();
            }
            foreach (var v in _hidGradCache.Values) 
            { 
                v.pre.Dispose();
                v.der.Dispose();
                v.delta.Dispose(); 
            }
            foreach (var v in _updWCache.Values) 
            {
                v.w.Dispose();
                v.d.Dispose();
                v.pa.Dispose();
            }
            foreach (var v in _updBCache.Values) 
            { 
                v.b.Dispose();
                v.d.Dispose(); 
            }
            foreach (var v in _dotTransposedCache.Values) 
            {
                v.mat.Dispose(); 
                v.vec.Dispose(); 
                v.res.Dispose();
            }
            _accelerator.Dispose();
        }

    }
}
