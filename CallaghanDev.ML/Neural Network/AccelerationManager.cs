using CallaghanDev.ML.NN;
using ILGPU.Runtime;
using ILGPU;
using ILGPU.Runtime.OpenCL;
using ILGPU.Runtime.CPU;
using ILGPU.Algorithms;

namespace CallaghanDev.ML.Neural_Network
{
    public interface IAccelerationManager
    {
        public double[] CalculateDotProduct(double[,] matrix, double[] vector);
    }
    public class AccelerationManager : IAccelerationManager
    {
        private Accelerator accelerator;

        private readonly Action<Index1D,
                            ArrayView1D<double, Stride1D.Dense>,
                            ArrayView2D<double, Stride2D.DenseX>,
                            ArrayView1D<double, Stride1D.Dense>> kernel;

        private readonly Dictionary<(int rows, int cols), (MemoryBuffer1D<double, Stride1D.Dense> vec, MemoryBuffer2D<double, Stride2D.DenseX> mat, MemoryBuffer1D<double, Stride1D.Dense> res)> bufferCache = new();



        public AccelerationManager(Parameters parameters) 
        {
            InitializeAccelerator(parameters);
            kernel = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<double, Stride1D.Dense>, ArrayView2D<double, Stride2D.DenseX>, ArrayView1D<double, Stride1D.Dense>>(VectorMatrixMultiplyAcceleratedKernel);
        }
        private void InitializeAccelerator(Parameters parameters)
        {
            Context context = Context.Create(builder => builder.AllAccelerators());

            if (parameters.AccelerationType == AccelerationType.CPU)
            {
                accelerator = context.CreateCPUAccelerator(0);
            }
            else
            {
                accelerator = context.CreateCLAccelerator(0);
            }
        }


        private static void VectorMatrixMultiplyAcceleratedKernel(
           Index1D index,
           ArrayView1D<double, Stride1D.Dense> vectorView,
           ArrayView2D<double, Stride2D.DenseX> matrixView,
           ArrayView1D<double, Stride1D.Dense> resultView)
        {
            double sum = 0.0d;
            for (var i = 0; i < vectorView.Length; i++)
            {
                sum += vectorView[i] * matrixView[i, index];
            }
            resultView[index] = sum;
        }

      
        public double[] CalculateDotProduct(double[,] matrix, double[] vector)
        {
            int rows = matrix.GetLength(0),
                cols = matrix.GetLength(1);

            // get or create the buffers for this shape
            if (!bufferCache.TryGetValue((rows, cols), out var bufs))
            {
                // var infers the correct MemoryBuffer<…,Stride> types
                var vec = accelerator.Allocate1D<double>(rows);
                var mat = accelerator.Allocate2DDenseX<double>(new Index2D(rows, cols));
                var res = accelerator.Allocate1D<double>(cols);

                bufs = (vec, mat, res);
                bufferCache[(rows, cols)] = bufs;
            }

            // now just copy data and launch
            bufs.vec.CopyFromCPU(vector);
            bufs.mat.CopyFromCPU(matrix);

            kernel(bufs.res.Extent.ToIntIndex(),
                   bufs.vec.View,
                   bufs.mat.View,
                   bufs.res.View);

            double[] result = new double[cols];
            bufs.res.CopyToCPU(result);
            return result;
        }

    }
}
