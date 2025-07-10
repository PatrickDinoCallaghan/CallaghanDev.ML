using CallaghanDev.ML.Enums;
using ILGPU.Algorithms;
using ILGPU.Runtime;
using ILGPU;
using ILGPU.Runtime.OpenCL;

namespace CallaghanDev.ML.AccelerationManagers
{
    public class AccelerationGPU : IAccelerationManager, IDisposable
    {
        private readonly Accelerator _accelerator;

        private readonly Action<Index1D, ArrayView2D<double, Stride2D.DenseX>, ArrayView1D<double, Stride1D.Dense>, ArrayView1D<double, Stride1D.Dense>> _dotKernel;
        private readonly Action<Index1D, ArrayView1D<double, Stride1D.Dense>, ArrayView1D<double, Stride1D.Dense>, ArrayView1D<double, Stride1D.Dense>> _outGradKernel;
        private readonly Action<Index1D, ArrayView1D<double, Stride1D.Dense>, ArrayView1D<double, Stride1D.Dense>, ArrayView1D<double, Stride1D.Dense>> _hidGradKernel;
        private readonly Action<Index2D, ArrayView2D<double, Stride2D.DenseX>, ArrayView1D<double, Stride1D.Dense>, ArrayView1D<double, Stride1D.Dense>, double> _updWKernel;
        private readonly Action<Index1D, ArrayView1D<double, Stride1D.Dense>, ArrayView1D<double, Stride1D.Dense>, double> _updBKernel;
        private readonly Action<Index1D, ArrayView1D<double, Stride1D.Dense>, ArrayView2D<double, Stride2D.DenseX>, ArrayView1D<double, Stride1D.Dense>> _dotTransposedKernel;
        private readonly Action<Index1D, ArrayView1D<double, Stride1D.Dense>, ArrayView1D<double, Stride1D.Dense>, ArrayView1D<double, Stride1D.Dense>, ArrayView1D<double, Stride1D.Dense>, ActivationType> _actKernel;

        private readonly Dictionary<(int rows, int cols),
            (MemoryBuffer2D<double, Stride2D.DenseX> mat,
             MemoryBuffer1D<double, Stride1D.Dense> vec,
             MemoryBuffer1D<double, Stride1D.Dense> res)> _dotCache
            = new();
        private readonly Dictionary<int,
            (MemoryBuffer1D<double, Stride1D.Dense> cost,
             MemoryBuffer1D<double, Stride1D.Dense> der,
             MemoryBuffer1D<double, Stride1D.Dense> grad)> _outGradCache
            = new();
        private readonly Dictionary<int,
            (MemoryBuffer1D<double, Stride1D.Dense> pre,
             MemoryBuffer1D<double, Stride1D.Dense> der,
             MemoryBuffer1D<double, Stride1D.Dense> delta)> _hidGradCache
            = new();
        private readonly Dictionary<(int rows, int cols),
            (MemoryBuffer2D<double, Stride2D.DenseX> w,
             MemoryBuffer1D<double, Stride1D.Dense> d,
             MemoryBuffer1D<double, Stride1D.Dense> pa)> _updWCache
            = new();
        private readonly Dictionary<int,
            (MemoryBuffer1D<double, Stride1D.Dense> b,
             MemoryBuffer1D<double, Stride1D.Dense> d)> _updBCache
            = new();
        private readonly Dictionary<(int rows, int cols),
            (MemoryBuffer2D<double, Stride2D.DenseX> mat,
             MemoryBuffer1D<double, Stride1D.Dense> vec,
             MemoryBuffer1D<double, Stride1D.Dense> res)> _dotTransposedCache
            = new();

        public AccelerationGPU()
        {
            var context = Context.Create(builder => builder.AllAccelerators());
            _accelerator = context.CreateCLAccelerator(0);

            _dotKernel = _accelerator.LoadAutoGroupedStreamKernel<
                Index1D,
                ArrayView2D<double, Stride2D.DenseX>,
                ArrayView1D<double, Stride1D.Dense>,
                ArrayView1D<double, Stride1D.Dense>>(
                DotKernel);

            _dotTransposedKernel = _accelerator.LoadAutoGroupedStreamKernel<
                Index1D,
                ArrayView1D<double, Stride1D.Dense>,
                ArrayView2D<double, Stride2D.DenseX>,
                ArrayView1D<double, Stride1D.Dense>>(
                DotTransposedKernel);

            _actKernel = _accelerator.LoadAutoGroupedStreamKernel<
                Index1D,
                ArrayView1D<double, Stride1D.Dense>,
                ArrayView1D<double, Stride1D.Dense>,
                ArrayView1D<double, Stride1D.Dense>,
                ArrayView1D<double, Stride1D.Dense>,
                ActivationType>(ActivateKernel);

            _outGradKernel = _accelerator.LoadAutoGroupedStreamKernel<
                Index1D,
                ArrayView1D<double, Stride1D.Dense>,
                ArrayView1D<double, Stride1D.Dense>,
                ArrayView1D<double, Stride1D.Dense>>(
                OutputGradientKernel);

            _hidGradKernel = _accelerator.LoadAutoGroupedStreamKernel<
                Index1D,
                ArrayView1D<double, Stride1D.Dense>,
                ArrayView1D<double, Stride1D.Dense>,
                ArrayView1D<double, Stride1D.Dense>>(
                HiddenGradientKernel);

            _updWKernel = _accelerator.LoadAutoGroupedStreamKernel<
                Index2D,
                ArrayView2D<double, Stride2D.DenseX>,
                ArrayView1D<double, Stride1D.Dense>,
                ArrayView1D<double, Stride1D.Dense>,
                double>(UpdateWeightsKernel);

            _updBKernel = _accelerator.LoadAutoGroupedStreamKernel<
                Index1D,
                ArrayView1D<double, Stride1D.Dense>,
                ArrayView1D<double, Stride1D.Dense>,
                double>(UpdateBiasKernel);
        }

        public double[] CalculateDotProduct(double[,] matrix, double[] vector)
        {
            int rows = matrix.GetLength(0);
            int cols = matrix.GetLength(1);
            if (vector.Length != cols)
            {
                throw new ArgumentException($"Expected vector of length {cols}, got {vector.Length}");
            }
            if (!_dotCache.TryGetValue((rows, cols), out var bufs))
            {
                bufs = (_accelerator.Allocate2DDenseX<double>(new Index2D(rows, cols)), _accelerator.Allocate1D<double>(cols), _accelerator.Allocate1D<double>(rows));

                _dotCache[(rows, cols)] = bufs;
            }

            bufs.mat.CopyFromCPU(matrix);
            bufs.vec.CopyFromCPU(vector);

            _dotKernel(new Index1D(rows), bufs.mat.View, bufs.vec.View, bufs.res.View);

            var result = new double[rows];
            bufs.res.CopyToCPU(result);
            return result;
        }

        private double[] CalculateDotProductTransposed(double[,] matrix, double[] vector)
        {
            int rows = matrix.GetLength(0);
            int cols = matrix.GetLength(1);
            if (vector.Length != rows)
            {
                throw new ArgumentException($"Expected vector of length {rows}, got {vector.Length}");
            }
            if (!_dotTransposedCache.TryGetValue((rows, cols), out var bufferTemp))
            {
                bufferTemp = (_accelerator.Allocate2DDenseX<double>(new Index2D(rows, cols)), _accelerator.Allocate1D<double>(rows), _accelerator.Allocate1D<double>(cols));
                _dotTransposedCache[(rows, cols)] = bufferTemp;
            }

            bufferTemp.mat.CopyFromCPU(matrix);
            bufferTemp.vec.CopyFromCPU(vector);

            _dotTransposedKernel(new Index1D(cols), bufferTemp.vec.View, bufferTemp.mat.View, bufferTemp.res.View);

            var result = new double[cols];
            bufferTemp.res.CopyToCPU(result);
            return result;
        }

        public (double[] activation, double[] derivative) ActivateLayer(double[] dot, double[] bias, ActivationType t)
        {
            int n = dot.Length;
            if (bias.Length != n)
            {
                throw new ArgumentException($"Expected bias length {n}, got {bias.Length}");
            }
            var dotBuf = _accelerator.Allocate1D<double>(n);
            var biasBuf = _accelerator.Allocate1D<double>(n);
            var actBuf = _accelerator.Allocate1D<double>(n);
            var derBuf = _accelerator.Allocate1D<double>(n);
            try
            {
                dotBuf.CopyFromCPU(dot);
                biasBuf.CopyFromCPU(bias);

                _actKernel(new Index1D(n), dotBuf.View, biasBuf.View, actBuf.View, derBuf.View, t);

                var activation = new double[n];
                var derivative = new double[n];
                actBuf.CopyToCPU(activation);
                derBuf.CopyToCPU(derivative);
                return (activation, derivative);
            }
            finally
            {
                dotBuf.Dispose(); biasBuf.Dispose(); actBuf.Dispose(); derBuf.Dispose();
            }
        }

        public double[] CalculateOutputGradients(double[] cost, double[] derivative)
        {
            int n = cost.Length;
            if (!_outGradCache.TryGetValue(n, out var bufs))
            {
                _outGradCache[n] = bufs = (
                    _accelerator.Allocate1D<double>(n),
                    _accelerator.Allocate1D<double>(n),
                    _accelerator.Allocate1D<double>(n)
                );
            }

            bufs.cost.CopyFromCPU(cost);
            bufs.der.CopyFromCPU(derivative);
            _outGradKernel(new Index1D(n), bufs.cost.View, bufs.der.View, bufs.grad.View);

            var result = new double[n];
            bufs.grad.CopyToCPU(result);
            return result;
        }

        public double[] CalculateHiddenGradients(double[,] weights, double[] nextDeltas, double[] derivative)
        {
            var pre = CalculateDotProductTransposed(weights, nextDeltas);
            int cols = pre.Length;
            if (!_hidGradCache.TryGetValue(cols, out var bufs))
            {
                _hidGradCache[cols] = bufs = (
                    _accelerator.Allocate1D<double>(cols),
                    _accelerator.Allocate1D<double>(cols),
                    _accelerator.Allocate1D<double>(cols)
                );
            }
            bufs.pre.CopyFromCPU(pre);
            bufs.der.CopyFromCPU(derivative);
            _hidGradKernel(new Index1D(cols), bufs.pre.View, bufs.der.View, bufs.delta.View);

            var result = new double[cols];
            bufs.delta.CopyToCPU(result);
            return result;
        }

        public double[,] UpdateWeights(double[,] weights, double[] deltas, double[] prevActivations, double learningRate)
        {
            int rows = weights.GetLength(0);
            int cols = weights.GetLength(1);
            if (!_updWCache.TryGetValue((rows, cols), out var bufs))
            {
                _updWCache[(rows, cols)] = bufs = (
                    _accelerator.Allocate2DDenseX<double>(new Index2D(rows, cols)),
                    _accelerator.Allocate1D<double>(rows),
                    _accelerator.Allocate1D<double>(cols)
                );
            }

            bufs.w.CopyFromCPU(weights);
            bufs.d.CopyFromCPU(deltas);
            bufs.pa.CopyFromCPU(prevActivations);
            _updWKernel(new Index2D(rows, cols), bufs.w.View, bufs.d.View, bufs.pa.View, learningRate);

            var result = new double[rows, cols];
            bufs.w.CopyToCPU(result);
            return result;
        }

        public double[] UpdateBias(double[] bias, double[] deltas, double learningRate)
        {
            int n = bias.Length;
            if (!_updBCache.TryGetValue(n, out var bufs))
            {
                _updBCache[n] = bufs = (
                    _accelerator.Allocate1D<double>(n),
                    _accelerator.Allocate1D<double>(n)
                );
            }

            bufs.b.CopyFromCPU(bias);
            bufs.d.CopyFromCPU(deltas);
            _updBKernel(new Index1D(n), bufs.b.View, bufs.d.View, learningRate);

            var result = new double[n];
            bufs.b.CopyToCPU(result);
            return result;
        }

        private static void DotKernel(Index1D row, ArrayView2D<double, Stride2D.DenseX> mat, ArrayView1D<double, Stride1D.Dense> vec, ArrayView1D<double, Stride1D.Dense> res)
        {
            double sum = 0;
            int cols = (int)mat.Extent.Y;
            for (int j = 0; j < cols; ++j)
            {
                sum += mat[row, j] * vec[j];
            }
            res[row] = sum;
        }

        private static void DotTransposedKernel(Index1D col, ArrayView1D<double, Stride1D.Dense> vec, ArrayView2D<double, Stride2D.DenseX> mat, ArrayView1D<double, Stride1D.Dense> res)
        {
            double sum = 0;
            int rows = (int)mat.Extent.X;
            for (int i = 0; i < rows; ++i)
            {
                sum += mat[i, col] * vec[i];
            }
            res[col] = sum;
        }

        private static void ActivateKernel(Index1D i, ArrayView1D<double, Stride1D.Dense> dot,  ArrayView1D<double, Stride1D.Dense> bias,  ArrayView1D<double, Stride1D.Dense> act, ArrayView1D<double, Stride1D.Dense> der, ActivationType t)
        {
            double z = dot[i] + bias[i];
            double a, d;
            switch (t)
            {
                case ActivationType.Sigmoid:
                    double e = XMath.Exp(z);
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
                    double ee = XMath.Exp(z);
                    a = ee / (1 + ee);
                    d = a * (1 - a);
                    break;
            }
            act[i] = a;
            der[i] = d;
        }

        private static void OutputGradientKernel(Index1D i, ArrayView1D<double, Stride1D.Dense> cost, ArrayView1D<double, Stride1D.Dense> der, ArrayView1D<double, Stride1D.Dense> grad)
        {
            grad[i] = -cost[i] * der[i];
        }

        private static void HiddenGradientKernel(Index1D i, ArrayView1D<double, Stride1D.Dense> pre,  ArrayView1D<double, Stride1D.Dense> der, ArrayView1D<double, Stride1D.Dense> delta)
        {
            delta[i] = pre[i] * der[i];
        }

        private static void UpdateWeightsKernel(Index2D idx, ArrayView2D<double, Stride2D.DenseX> W, ArrayView1D<double, Stride1D.Dense> delta, ArrayView1D<double, Stride1D.Dense> prevAct, double learningRate)
        {
            int r = idx.X, c = idx.Y;
            W[r, c] -= learningRate * delta[r] * prevAct[c];
        }

        private static void UpdateBiasKernel(Index1D i, ArrayView1D<double, Stride1D.Dense> bais, ArrayView1D<double, Stride1D.Dense> delta, double learningRate)
        {
            bais[i] -= learningRate * delta[i];
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
