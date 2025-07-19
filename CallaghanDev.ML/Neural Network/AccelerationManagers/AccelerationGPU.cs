using CallaghanDev.ML.Enums;
using ILGPU.Algorithms;
using ILGPU.Runtime;
using ILGPU;
using ILGPU.Runtime.OpenCL;
using ILGPU.Runtime.Cuda;

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
        private readonly Dictionary<(int rows, int cols),
            (MemoryBuffer2D<float, Stride2D.DenseX> w,
             MemoryBuffer1D<float, Stride1D.Dense> d,
             MemoryBuffer1D<float, Stride1D.Dense> pa)> _updWCache
            = new();
        private readonly Dictionary<int,
            (MemoryBuffer1D<float, Stride1D.Dense> b,
             MemoryBuffer1D<float, Stride1D.Dense> d)> _updBCache
            = new();
        private readonly Dictionary<(int rows, int cols),
            (MemoryBuffer2D<float, Stride2D.DenseX> mat,
             MemoryBuffer1D<float, Stride1D.Dense> vec,
             MemoryBuffer1D<float, Stride1D.Dense> res)> _dotTransposedCache
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
                    d = 1;
                    break;
                case ActivationType.Sigmoid:
                    float e = XMath.Exp(z);
                    a = e / (1 + e);
                    d = a * (1 - a);
                    break;
                case ActivationType.Tanh:
                    a = XMath.Tanh(z);
                    d = 1 - z * z;  // match CPU: derivative = 1 - z^2
                    break;
                case ActivationType.Relu:
                    a = XMath.Max(0.0f, z);
                    d = z >= 0 ? 1 : 0;
                    break;
                case ActivationType.Leakyrelu:
                    a = z > 0f ? z : 0.01f * z;
                    d = z >= 0f ? 1 : 0.1f;  // match CPU: derivative negative slope = 0.1f
                    break;
                default:
                    float ee = XMath.Exp(z);
                    a = ee / (1 + ee);
                    d = a * (1 - a);
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
        private static void UpdateBiasKernel(Index1D i,
       ArrayView1D<float, Stride1D.Dense> bias,
       ArrayView1D<float, Stride1D.Dense> delta,
       float learningRate)
        {
            bias[i] -= learningRate * delta[i];
        }

        public void Dispose()
        {
            foreach (var v in _outGradCache.Values) { v.cost.Dispose(); v.der.Dispose(); v.grad.Dispose(); }
            foreach (var v in _hidGradCache.Values) { v.pre.Dispose(); v.der.Dispose(); v.delta.Dispose(); }
            foreach (var v in _updWCache.Values) { v.w.Dispose(); v.d.Dispose(); v.pa.Dispose(); }
            foreach (var v in _updBCache.Values) { v.b.Dispose(); v.d.Dispose(); }
            foreach (var v in _dotTransposedCache.Values) { v.mat.Dispose(); v.vec.Dispose(); v.res.Dispose(); }
            _accelerator.Dispose();
        }
    }
}