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

        private Action<Index1D, ArrayView1D<double, Stride1D.Dense>, ArrayView2D<double, Stride2D.DenseX>, ArrayView1D<double, Stride1D.Dense>> double_MatrixVectorKernel;

        public AccelerationManager(Parameters parameters) 
        {

            InitializeAccelerator(parameters);
            LoadBackpropergationKernel();
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
        private void LoadBackpropergationKernel()
        {
            double_MatrixVectorKernel = accelerator.LoadAutoGroupedStreamKernel<
                     Index1D,
                     ArrayView1D<double, Stride1D.Dense>,
                     ArrayView2D<double, Stride2D.DenseX>,
                     ArrayView1D<double, Stride1D.Dense>>(
                     VectorMatrixMultiplyAcceleratedKernel);
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
            var vectorLength = vector.Length;
            var matrixRows = matrix.GetLength(0);
            var matrixCols = matrix.GetLength(1);

            if (vectorLength != matrixRows)
                throw new ArgumentException($"Vector length {vectorLength} does not match matrix rows {matrixRows}");



            using var resultBuffer = accelerator.Allocate1D<double>(matrixCols);
            using var vectorBuffer = accelerator.Allocate1D<double>(vectorLength);
            using var matrixBuffer = accelerator.Allocate2DDenseX<double>(new Index2D(matrixRows, matrixCols));

            vectorBuffer.CopyFromCPU(vector);
            matrixBuffer.CopyFromCPU(matrix);

            double_MatrixVectorKernel(resultBuffer.Extent.ToIntIndex(), vectorBuffer.View, matrixBuffer.View, resultBuffer.View);

            double[] result = new double[matrixCols];

            resultBuffer.CopyToCPU(result);

            return result;
        }

    }
}
