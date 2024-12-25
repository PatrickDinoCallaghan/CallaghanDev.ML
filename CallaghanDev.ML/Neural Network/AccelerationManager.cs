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

    }
    public class AccelerationManager
    {
        private Accelerator accelerator;

        private Action<Index1D, ArrayView2D<double, Stride2D.DenseX>, ArrayView1D<double, Stride1D.Dense>, ArrayView1D<double, Stride1D.Dense>, ArrayView1D<double, Stride1D.Dense>, ArrayView1D<double, Stride1D.Dense>, ArrayView1D<double, Stride1D.Dense>, ArrayView2D<double, Stride2D.DenseX>, double, double> double_GPUBackpropResult;

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

            double_GPUBackpropResult = accelerator.LoadAutoGroupedStreamKernel<
                Index1D,
                ArrayView2D<double, Stride2D.DenseX>,
                ArrayView1D<double, Stride1D.Dense>,
                ArrayView1D<double, Stride1D.Dense>,
                ArrayView1D<double, Stride1D.Dense>,
                ArrayView1D<double, Stride1D.Dense>,
                ArrayView1D<double, Stride1D.Dense>,
                ArrayView2D<double, Stride2D.DenseX>,
                double,
            double>(BackpropagationKernel);


            double_MatrixVectorKernel = accelerator.LoadAutoGroupedStreamKernel<
                     Index1D,
                     ArrayView1D<double, Stride1D.Dense>,
                     ArrayView2D<double, Stride2D.DenseX>,
                     ArrayView1D<double, Stride1D.Dense>>(
                     VectorMatrixMultiplyAcceleratedKernel);
        }


       static void VectorMatrixMultiplyAcceleratedKernel(
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

        static void BackpropagationKernel(
            Index1D neuronIndex,
            ArrayView2D<double, Stride2D.DenseX> weightsMatrix,
            ArrayView1D<double, Stride1D.Dense> deltasView,
            ArrayView1D<double, Stride1D.Dense> activationsView,
            ArrayView1D<double, Stride1D.Dense> activationDerivativeView,
            ArrayView1D<double, Stride1D.Dense> biasesView,
            ArrayView1D<double, Stride1D.Dense> updatedDeltasView,
            ArrayView2D<double, Stride2D.DenseX> updatedWeightsView,
            double learningRate,
            double GradientClippingThreshold)
        {
            long precedingActivationsNo = activationsView.Extent.X;
            long nextNeurons = deltasView.Extent.X;

            // Calculate the sum of weighted deltas for this neuron
            double sumOfWeightedDeltas = 0.0;
            for (int i = 0; i < nextNeurons; i++)
            {
                sumOfWeightedDeltas += weightsMatrix[i, neuronIndex] * deltasView[i];
            }

            // Calculate the L2 norm of the gradients
            double normSquared = 0.0;
            for (int i = 0; i < nextNeurons; i++)
            {
                double gradient = weightsMatrix[i, neuronIndex] * deltasView[i];
                normSquared += gradient * gradient;
            }
            double norm = XMath.Sqrt(normSquared);

            // Apply norm-based gradient clipping
            double scale = 1.0;
            if (norm > GradientClippingThreshold)
            {
                scale = GradientClippingThreshold / norm;
            }

            // Scale the sum of weighted deltas
            sumOfWeightedDeltas *= scale;

            // Calculate the delta using the activation derivative
            double delta = sumOfWeightedDeltas * activationDerivativeView[neuronIndex];
            updatedDeltasView[neuronIndex] = delta;

            // Update weights and biases
            for (int i = 0; i < precedingActivationsNo; i++)
            {
                updatedWeightsView[i, neuronIndex] = learningRate * delta * activationsView[i];
            }

            biasesView[neuronIndex] -= learningRate * delta;
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
        public (double[] updatedBiases, double[,] updatedWeights, double[] updatedDeltas) CalculateBackpropagationValues(
            int numNeurons_CurrentLayer,
            int NumNeurons_PreviousLayer,
            double[,] weightsMatrix,
            double[] deltas,
            double[] activations,
            double[] activationDerivatives,
            double[] biases,
            double learningRate,
            double GradientClippingThreshold)
        {
            double[,] updatedWeights = new double[NumNeurons_PreviousLayer, numNeurons_CurrentLayer];
            double[] updatedBiases = new double[numNeurons_CurrentLayer];
            double[] updatedDeltas = new double[numNeurons_CurrentLayer];

            // Allocate GPU buffers with accurate sizes
            using var weightsMatrixBuffer = accelerator.Allocate2DDenseX<double>(new Index2D(weightsMatrix.GetLength(0), weightsMatrix.GetLength(1)));
            using var deltasBuffer = accelerator.Allocate1D<double>(deltas.Length);
            using var activationsBuffer = accelerator.Allocate1D<double>(activations.Length);
            using var activationDerivativesBuffer = accelerator.Allocate1D<double>(activationDerivatives.Length);
            using var biasesBuffer = accelerator.Allocate1D<double>(biases.Length);
            using var updatedDeltasBuffer = accelerator.Allocate1D<double>(numNeurons_CurrentLayer);
            using var updatedWeightsView = accelerator.Allocate2DDenseX<double>(new Index2D(NumNeurons_PreviousLayer, numNeurons_CurrentLayer));

            // Copy input data to GPU buffers sequentially
            weightsMatrixBuffer.CopyFromCPU(weightsMatrix);
            deltasBuffer.CopyFromCPU(deltas);
            activationsBuffer.CopyFromCPU(activations);
            activationDerivativesBuffer.CopyFromCPU(activationDerivatives);
            biasesBuffer.CopyFromCPU(biases);

            // Launch the kernel function with the appropriate parameters
            double_GPUBackpropResult(
                numNeurons_CurrentLayer,
                weightsMatrixBuffer.View,
                deltasBuffer.View,
                activationsBuffer.View,
                activationDerivativesBuffer.View,
                biasesBuffer.View,
                updatedDeltasBuffer.View,
                updatedWeightsView.View,
                learningRate,
                GradientClippingThreshold);

            // Synchronize the GPU to ensure the kernel execution is complete
            accelerator.Synchronize();
            updatedWeightsView.CopyToCPU(updatedWeights);
            biasesBuffer.CopyToCPU(updatedBiases);
            updatedDeltasBuffer.CopyToCPU(updatedDeltas);

            return (updatedBiases, updatedWeights, updatedDeltas);
        }


    }
}
