using CallaghanDev.ML.Neural_Network.Exceptions;
using CallaghanDev.Utilities.ConsoleHelper;
using CallaghanDev.Utilities.MathTools;
using ILGPU;
using ILGPU.Algorithms;
using ILGPU.Runtime;
using ILGPU.Runtime.CPU;
using ILGPU.Runtime.OpenCL;
using Newtonsoft.Json;
using System.Collections.Concurrent;
using System.Diagnostics;
using System.Threading.Tasks;

namespace CallaghanDev.ML
{
    /// <summary>
    /// Represents a neural network designed for parallel processing and GPU acceleration using ILGPU.
    /// 
    /// Techniques and Features:
    /// - **Data Structure**: 
    ///   - Uses a `Matrix` to store neurons and their connections.
    ///   - `NeuriteTensor` and `NeuriteTensorT` matrices to represent connections between neurons in different layers.
    ///
    /// - **Task Management**: 
    ///   - Utilizes a `TaskManager` with multiple `TaskContainer<T>` instances for asynchronous task management.
    ///   - Efficiently manages task execution and synchronization.
    ///
    /// - **Initialization**: 
    ///   - Initializes sensory neurons, hidden layers, and motor (output) layers.
    ///   - Loads backpropagation and matrix-vector multiplication kernels for GPU execution.
    ///   - Initializes calculation tensors and task containers for neural network operations.
    ///
    /// - **GPU Acceleration**:
    ///   - Uses ILGPU for parallel processing of matrix-vector operations and backpropagation.
    ///   - Allocates and manages GPU buffers for efficient data transfer and computation.
    ///   - Custom kernels for backpropagation and matrix-vector multiplication.
    ///
    /// - **Training and Prediction**:
    ///   - Forward propagation method to calculate neuron activations.
    ///   - Backpropagation methods for updating weights and biases based on gradients.
    ///   - Training method to learn from input data over multiple epochs.
    ///   - Predict method for inference on new input data.
    ///
    /// - **Cost Function**:
    ///   - Supports Mean Squared Error (MSE) cost function.
    ///   - Calculates gradients for backpropagation using the derivative of the cost function.
    ///
    /// - **Utility Methods**:
    ///   - Methods for saving and loading the neural network state to and from files.
    ///   - Methods to handle changes in data and reinitialize necessary components.
    ///
    /// Aims:
    /// - **Efficiency**: Leveraging parallel processing and GPU acceleration to enhance the performance of neural network training and inference.
    /// - **Modularity**: Structured in a way that allows for easy extension and modification of the neural network architecture.
    /// - **Scalability**: Designed to handle large datasets and complex neural network structures by utilizing modern hardware capabilities.
    /// </summary>

    public class NeuralNetwork
    {
        private Matrix<INeuron> Data = new Matrix<INeuron>();

        // This structure arranges each element in the array of matrices to represent a matrix associated with a specific layer. Starting at the first hidden layer from the input layer
        // Each row corresponds to a neuron's dendrites in the hidden layer, indicating these neurons originate from the current layer, and connect to each neuron in the next layer.
        // Transposing this matrix switches the focus from source neurons in the current layer to target neurons in the subsequent layer, and their connections to each neuron in the subsequent layer.
        private Matrix<Neurite>[] NeuriteTensor;
        private Matrix<Neurite>[] NeuriteTensorT;

        private int _NumberOfInputs;
        private int _NumberOfOutputs;

        private int _HiddenLayerWidth;
        private int _NoHiddenLayers;

        private float _clippingLimit_Upper;
        private float _clippingLimit_Lower;

        private double _L2RegulationLamda = 0;

        float _HuberLossDelta = 1;

        //See how the bottom two are related and try and remove one. Its unneeded overhead
        private INeuron[][] neuronsJaggedArray;
        private INeuron[][] Neurons;
        private Random random;
        private ParallelOptions options;
        private Accelerator accelerator;
        private ActivationType _DefaultActivationType;
        CostFunctionType _costFunction;


        private CostFunction CostFunctionDeriv { get; set; }
        private delegate double CostFunction(double value, double prediction);
        [JsonIgnore]
        private CostFunction costFunction { get; set; }


        #region ctor

        AccelerationType _AccelerationType;

        public NeuralNetwork(AccelerationType accelerationType, SensoryNeuron[] sensoryNeurons, int NoHiddenLayers, int HiddenLayerWidth, int NumberOfOutputs, ActivationType DefaultActivationType, CostFunctionType costFunction, double l2RegulationLamda = 0, float clippingLimit_Upper = 1, float clippingLimit_Lower = -1)
        {
            _L2RegulationLamda = l2RegulationLamda;
            _clippingLimit_Upper = clippingLimit_Upper;
            _clippingLimit_Lower = clippingLimit_Lower;
            _costFunction = costFunction;
            _DefaultActivationType = DefaultActivationType;
            _NoHiddenLayers = NoHiddenLayers;
            _NumberOfInputs = sensoryNeurons.Count();
            _NumberOfOutputs = NumberOfOutputs;
            _HiddenLayerWidth = HiddenLayerWidth;
            _AccelerationType = accelerationType;
            InitSensoryNeurons(sensoryNeurons);
            Initialize();
        }
        public NeuralNetwork(AccelerationType accelerationType, SensoryNeuron[] sensoryNeurons, int NoHiddenLayers, int HiddenLayerWidth, int NumberOfOutputs, ActivationType DefaultActivationType, CostFunctionType costFunction, float HuberLossDelta, double l2RegulationLamda = 0, float clippingLimit_Upper = 1, float clippingLimit_Lower = -1)
        {

            if (costFunction != CostFunctionType.huberLoss)
            {
                throw new Exception("CostFunction must be huberloss if huberloss deleta is selected");
            }
            _L2RegulationLamda = l2RegulationLamda;
            _clippingLimit_Upper = clippingLimit_Upper;
            _clippingLimit_Lower = clippingLimit_Lower;
            _costFunction = costFunction;
            _DefaultActivationType = DefaultActivationType;
            _NoHiddenLayers = NoHiddenLayers;
            _NumberOfInputs = sensoryNeurons.Count();
            _NumberOfOutputs = NumberOfOutputs;
            _HiddenLayerWidth = HiddenLayerWidth;
            _HuberLossDelta = HuberLossDelta;
            _AccelerationType = accelerationType;
            InitSensoryNeurons(sensoryNeurons);
            Initialize();
        }

        public static SensoryNeuron[] GetSensoryNeurons(double[][] TrainingData)
        {
            int TrainingDataLength = TrainingData.First().Length;

            if (TrainingData.Any(r => r.Length != TrainingDataLength))
            {
                throw new Exception("All training data must have the same number of values");
            }

            Tuple<double, double>[] MinMaxArray = new Tuple<double, double>[TrainingDataLength];

            for (int i = 0; i < TrainingDataLength; i++)
            {
                MinMaxArray[i] = new Tuple<double, double>(double.MaxValue, double.MinValue);
            }

            foreach (var datapoint in TrainingData)
            {
                for (int i = 0; i < TrainingDataLength; i++)
                {
                    if (datapoint[i] < MinMaxArray[i].Item1)
                    {
                        Tuple<double, double> NewTuple = new Tuple<double, double>(datapoint[i], Math.Floor(MinMaxArray[i].Item2));

                        MinMaxArray[i] = NewTuple;
                    }
                    if (datapoint[i] > MinMaxArray[i].Item2)
                    {
                        Tuple<double, double> NewTuple = new Tuple<double, double>(MinMaxArray[i].Item1, Math.Ceiling(datapoint[i]));
                        MinMaxArray[i] = NewTuple;
                    }
                }
            }

            SensoryNeuron[] inputNeurons = new SensoryNeuron[TrainingDataLength];

            for (int i = 0; i < TrainingDataLength; i++)  // XOR problem has 2 inputs
            {
                inputNeurons[i] = (new SensoryNeuron(MinMaxArray[i].Item2, MinMaxArray[i].Item1));
            }

            return inputNeurons;
        }

        #endregion

        #region initialization methods

        private Action<Index1D, ArrayView2D<double, Stride2D.DenseX>, ArrayView1D<double, Stride1D.Dense>, ArrayView1D<double, Stride1D.Dense>, ArrayView1D<double, Stride1D.Dense>, ArrayView1D<double, Stride1D.Dense>, ArrayView1D<double, Stride1D.Dense>, ArrayView2D<double, Stride2D.DenseX>, double, double, double> double_GPUBackpropResult;

        private Action<Index1D, ArrayView1D<double, Stride1D.Dense>, ArrayView2D<double, Stride2D.DenseX>, ArrayView1D<double, Stride1D.Dense>> double_MatrixVectorKernel;

        public void Initialize()
        {
            Context context = Context.Create(builder => builder.AllAccelerators());
            if (_AccelerationType == AccelerationType.CPU)
            {
                accelerator = context.CreateCPUAccelerator(0);
            }
            else
            {
                accelerator = context.CreateCLAccelerator(0);
            }

            options = new ParallelOptions
            {
                MaxDegreeOfParallelism = Environment.ProcessorCount
            };

            random = new Random();

            InitCostFunction(_costFunction);

            NeuriteTensor = new Matrix<Neurite>[_NoHiddenLayers + 1];

            InitHiddenLayer();
            InitMotorLayers();
            InitCalculationTensors();
            LoadBackpropergationKernel();
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
                double,
                double>(BackpropagationKernel);


            double_MatrixVectorKernel = accelerator.LoadAutoGroupedStreamKernel<
                     Index1D,
                     ArrayView1D<double, Stride1D.Dense>,
                     ArrayView2D<double, Stride2D.DenseX>,
                     ArrayView1D<double, Stride1D.Dense>>(
                     VectorMatrixMultiplyAcceleratedKernel);
        }
        private void InitSensoryNeurons(SensoryNeuron[] sensoryNeurons)
        {
            for (int i = 0; i < _NumberOfInputs; i++)
            {
                sensoryNeurons[i].LayerIndex = 0;
                Data[(int)i, 0] = sensoryNeurons[i];
            }
        }
        private void InitHiddenLayer()
        {
            for (int j = 0; j < _NoHiddenLayers; j++)
            {
                NeuriteTensor[j] = new Matrix<Neurite>();

                int NumberOfNeronsInpreviousLayer = (int)Data.Column(j).Length;
                for (int h = 0; h < _HiddenLayerWidth; h++)
                {
                    // if the next layer is the motor layer and it has fewer neurons this should be fewer 
                    AssociativeNeuron neuron = new AssociativeNeuron(_DefaultActivationType);
                    neuron.LayerIndex = j + 1;
                    Data[h, j + 1] = neuron;

                    for (int i = 0; i < NumberOfNeronsInpreviousLayer; i++)
                    {
                        //double weight = GetRandomdouble(random,-1, 1);
                        double weight = LeakyReLUInitializer(NumberOfNeronsInpreviousLayer);

                        Neurite neurite = new Neurite(Data[i, j], weight);

                        NeuriteTensor[j][h, i] = neurite;

                        neuron.Dendrites.Add(neurite);
                    }
                }

            }
        }
        private void InitMotorLayers()
        {
            int PreviousColIndex = Data.ColumnCount() - 1;

            int NumberOfNeronsInpreviousLayer = Data.Column(PreviousColIndex).Length;
            if (NumberOfNeronsInpreviousLayer != _HiddenLayerWidth)
            {
                throw new Exception();
            }
            NeuriteTensor[_NoHiddenLayers] = new Matrix<Neurite>();

            for (int m = 0; m < _NumberOfOutputs; m++)
            {
                MotorNeuron neuron = new MotorNeuron(_DefaultActivationType);
                neuron.LayerIndex = PreviousColIndex + 1;
                Data[m, PreviousColIndex + 1] = neuron;
                for (int i = 0; i < NumberOfNeronsInpreviousLayer; i++)
                {
                    //double weight = GetRandomdouble(random, -1,1);
                    double weight = LeakyReLUInitializer(NumberOfNeronsInpreviousLayer);
                    Neurite neurite = new Neurite(Data[i, PreviousColIndex], weight);

                    NeuriteTensor[_NoHiddenLayers][m, i] = neurite;

                    neuron.Dendrites.Add(neurite);
                }
            }
        }
        private double LeakyReLUInitializer(int incomingNeurites)
        {
            double standardDeviation = (double)Math.Sqrt((double)(2.0f / incomingNeurites));
            return (double)(random.NextDouble() * 2 - 1) * standardDeviation;
        }

        private void InitCalculationTensors()
        {
            NeuriteTensorT = NeuriteTensor.Select(r => r.Transpose()).ToArray();
            Neurons = Data.ConvertToColumnArray();

            int colCount = Data.ColumnCount() - 2;
            neuronsJaggedArray = new INeuron[colCount + 2][];

            for (int layerIndex = colCount; layerIndex >= 0; layerIndex--)
            {
                neuronsJaggedArray[layerIndex + 1] = Neurons[layerIndex + 1].Where(r => r != null).ToArray();
                neuronsJaggedArray[layerIndex] = Neurons[layerIndex].Where(r => r != null).ToArray();
            }

        }

        #endregion

        #region Forward Propagate

        public void ForwardPropagate()
        {
            int columnCount = Data.ColumnCount();
            for (int j = 1; j < columnCount; j++)
            {
                int previousLayerNeuronCount = Neurons[j - 1].Length;
                int currentLayerNeuronCount = Neurons[j].Length;

                // Initialize arrays
                double[] dotProduct = new double[currentLayerNeuronCount];

                Task<double[]> sourceNeuronsActivationsTask = Task.Run(() => Neurons[j - 1].Where(r => r != null).Select(r => r.Activation).ToArray());
                Task<double[,]> neuritesWeightsTask = Task.Run(() => NeuriteTensorT[j - 1].Select(r => r.Weight).ToArray());

                // Wait for all tasks to complete
                Task.WaitAll(sourceNeuronsActivationsTask, neuritesWeightsTask);


                dotProduct = CalculateDotProduct(neuritesWeightsTask.Result, sourceNeuronsActivationsTask.Result);

                Parallel.For(0, dotProduct.Length, c =>
                {
                    Neurons[j][c].Activation = Neurons[j][c].activationFunction(dotProduct[c] + Neurons[j][c].Bias);

                    if (double.IsNaN(Neurons[j][c].Activation))
                    {
                        throw new InfinityException($"NaN detected in forward propagation at layer {j}, neuron {c}, type:{Neurons[j][c].GetType().Name}");
                    }
                    if (double.IsInfinity(Neurons[j][c].Activation))
                    {
                        throw new NaNException($"Infinity detected in forward propagation at layer {j}, neuron {c}, type:{Neurons[j][c].GetType().Name}");
                    }
                });
            }
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

        #endregion

        public void UpdateOutputLayerGradients(double[] costs, double learningRate, double lambda = 0)
        {
            INeuron[] MotorNeurons = Data.Column(Data.ColumnCount() - 1).ToArray();
            int motorNeuronCount = MotorNeurons.Length;

            // Preallocate variables to avoid repeated allocations
            double activationDerivative;
            double gradient;

            Parallel.For(0, motorNeuronCount, i =>
            {
                INeuron motorNeuron = MotorNeurons[i];
                activationDerivative = motorNeuron.activationFunctionDeriv(motorNeuron.Activation);
                gradient = -costs[i] * activationDerivative;
                gradient = Math.Max(_clippingLimit_Lower, Math.Min(_clippingLimit_Upper, gradient));
                motorNeuron.Delta = gradient;

                Neurite[] dendrites = motorNeuron.Dendrites.ToArray();
                int dendriteCount = dendrites.Length;

                for (int j = 0; j < dendriteCount; j++)
                {
                    Neurite connection = dendrites[j];
                    double weightUpdate = learningRate * (gradient * connection.SourceNeuron.Activation + lambda * connection.Weight);
                    connection.Weight -= weightUpdate;

                    if (double.IsNaN(connection.Weight))
                    {
                        throw new NaNException();
                    }
                    if (double.IsInfinity(connection.Weight))
                    {
                        throw new InfinityException();
                    }
                }

                motorNeuron.Bias -= learningRate * gradient;
            });
        }

        #region Back propagation
        // Utility function to extract the weight matrix of a specific layer
        public double[,] GetWeightsMatrix(int layerIndex)
        {

            double[,] Weights = new double[NeuriteTensor[layerIndex].RowCount(), NeuriteTensor[layerIndex].ColumnCount()];// NeuriteTensor[layerIndex].Select(r => r.Weight).ToArray();

            int rowCount = NeuriteTensor[layerIndex].RowCount();
            int colCount = NeuriteTensor[layerIndex].ColumnCount();


            if (rowCount > colCount)
            {
                Parallel.For(0, rowCount, i =>
                {
                    for (int j = 0; j < colCount; j++)
                    {
                        Weights[i, j] = NeuriteTensor[layerIndex][i, j].Weight;
                    }
                });
            }
            else
            {
                Parallel.For(0, colCount, j =>
                {
                    for (int i = 0; i < rowCount; i++)
                    {
                        Weights[i, j] = NeuriteTensor[layerIndex][i, j].Weight;
                    }
                });
            }
            return Weights;
        }

        public void UpdateHiddenLayerGradients(double learningRate)
        {
            int colCount = _NoHiddenLayers;

            for (int layerIndex = colCount; layerIndex > 0; layerIndex--)
            {
                INeuron[] precedingLayerNeurons = neuronsJaggedArray[layerIndex - 1];

                INeuron[] hiddenNeurons = neuronsJaggedArray[layerIndex];

                INeuron[] nextLayerNeurons = neuronsJaggedArray[layerIndex + 1];


                if (nextLayerNeurons.Length == 0 || hiddenNeurons.Length == 0 || precedingLayerNeurons.Length == 0)
                {
                    throw new IndexOutOfRangeException("Mismatch in neuron array sizes.");
                }

                double[,] weightsMatrix = GetWeightsMatrix(layerIndex);

                int numCurrentNeurons = hiddenNeurons.Length;
                int numPrecedingNeurons = precedingLayerNeurons.Length;

                Task<double[]> taskDeltasArray = Task.Run(() => nextLayerNeurons.Select(neuron => neuron.Delta).ToArray());
                Task<double[]> taskPrecedingLayerNeuronsActivations = Task.Run(() => precedingLayerNeurons.Select(r => r.Activation).ToArray());
                Task<double[]> taskHiddenNeuronsActivationFunctionDeriv = Task.Run(() => hiddenNeurons.Select(r => r.activationFunctionDeriv(r.Activation)).ToArray());
                Task<double[]> taskHiddenNeuronsBiases = Task.Run(() => hiddenNeurons.Select(r => r.Bias).ToArray());

                // Wait for all tasks to complete
                Task.WaitAll(taskDeltasArray, taskPrecedingLayerNeuronsActivations, taskHiddenNeuronsActivationFunctionDeriv, taskHiddenNeuronsBiases);

                // Retrieve results
                double[] deltasArray = taskDeltasArray.Result;
                double[] precedingLayerNeuronsActivations = taskPrecedingLayerNeuronsActivations.Result;
                double[] hiddenNeurons_activationFunctionDeriv = taskHiddenNeuronsActivationFunctionDeriv.Result;
                double[] hiddenNeurons_Biases = taskHiddenNeuronsBiases.Result;


                var GPUBackpropResult = LaunchBackpropagationKernel(
                    accelerator,
                    numCurrentNeurons,
                    numPrecedingNeurons,
                    weightsMatrix,
                    deltasArray,
                    precedingLayerNeuronsActivations,
                    hiddenNeurons_activationFunctionDeriv,
                    hiddenNeurons_Biases,
                    learningRate,
                    _clippingLimit_Lower,
                    _clippingLimit_Upper);

                Parallel.For(0, numCurrentNeurons, i =>
                {
                    hiddenNeurons[i].Delta = GPUBackpropResult.updatedDeltas[i];

                    for (int k = 0; k < numPrecedingNeurons; k++)
                    {
                        NeuriteTensor[layerIndex - 1][i, k].Weight -= GPUBackpropResult.updatedWeights[k, i]; // CopyToCPU transposes the data into column major order so the result will have k,i instead of i,k
                    }
                    hiddenNeurons[i].Bias = GPUBackpropResult.updatedBiases[i];
                });


            }
        }
        public (double[] updatedBiases, double[,] updatedWeights, double[] updatedDeltas) LaunchBackpropagationKernel(
            Accelerator accelerator,
            int numNeurons_CurrentLayer,
            int NumNeurons_PreviousLayer,
            double[,] weightsMatrix,
            double[] deltas,
            double[] activations,
            double[] activationDerivatives,
            double[] biases,
            double learningRate,
            double clippingLowerLimit,
            double clippingUpperLimit)
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


            // Copy input data to GPU buffers      
            weightsMatrixBuffer.CopyFromCPU(weightsMatrix);
            deltasBuffer.CopyFromCPU(deltas);
            activationsBuffer.CopyFromCPU(activations);
            activationDerivativesBuffer.CopyFromCPU(activationDerivatives);
            biasesBuffer.CopyFromCPU(biases);

            Task task1 = Task.Run(() => weightsMatrixBuffer.CopyFromCPU(weightsMatrix));
            Task task2 = Task.Run(() => deltasBuffer.CopyFromCPU(deltas));
            Task task3 = Task.Run(() => activationsBuffer.CopyFromCPU(activations));
            Task task4 = Task.Run(() => activationDerivativesBuffer.CopyFromCPU(activationDerivatives));
            Task task5 = Task.Run(() => biasesBuffer.CopyFromCPU(biases));

            Task.WaitAll(task1, task2, task3, task4, task5);


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
                clippingLowerLimit,
                clippingUpperLimit);

            // Synchronize the GPU to ensure the kernel execution is complete
            accelerator.Synchronize();
            updatedWeightsView.CopyToCPU(updatedWeights);
            biasesBuffer.CopyToCPU(updatedBiases);
            updatedDeltasBuffer.CopyToCPU(updatedDeltas);




            return (updatedBiases, updatedWeights, updatedDeltas);
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
    double clippingLowerLimit,
    double clippingUpperLimit)
        {
            long precedingActivationsNo = activationsView.Extent.X;
            long nextNeurons = deltasView.Extent.X;

            // Calculate the sum of weighted deltas for this neuron
            double sumOfWeightedDeltas = 0.0;
            for (int i = 0; i < nextNeurons; i++)
            {
                sumOfWeightedDeltas += weightsMatrix[i, neuronIndex] * deltasView[i];
            }

            // Apply gradient clipping
            sumOfWeightedDeltas = XMath.Max(clippingLowerLimit, XMath.Min(sumOfWeightedDeltas, clippingUpperLimit));

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

        #endregion

        #region Cost Function Methods

        private void InitCostFunction(CostFunctionType costFunctionType)
        {
            switch (costFunctionType)
            {
                case CostFunctionType.mse:
                    costFunction = UtilityFunctions.mse;
                    CostFunctionDeriv = UtilityFunctions.FirstDerivative.mse;
                    break;
                case CostFunctionType.ZeroWeightedMSE:
                    costFunction = UtilityFunctions.ZeroWeightedMSE;
                    CostFunctionDeriv = UtilityFunctions.FirstDerivative.ZeroWeightedMSEDerivative;
                    break;
                case CostFunctionType.binaryCrossEntropy:
                    costFunction = UtilityFunctions.binaryCrossEntropy;
                    CostFunctionDeriv = UtilityFunctions.FirstDerivative.binaryCrossEntropy;
                    break;
                case CostFunctionType.mae:
                    costFunction = UtilityFunctions.mae;
                    CostFunctionDeriv = UtilityFunctions.FirstDerivative.mae;
                    break;
                case CostFunctionType.huberLoss:
                    costFunction = (actual, predicted) => UtilityFunctions.huberLoss(actual, predicted, _HuberLossDelta);
                    CostFunctionDeriv = (actual, predicted) => UtilityFunctions.FirstDerivative.huberLoss(actual, predicted, _HuberLossDelta);
                    break;
                case CostFunctionType.categoricalCrossEntropy:
                    costFunction = UtilityFunctions.categoricalCrossEntropy;
                    CostFunctionDeriv = UtilityFunctions.FirstDerivative.categoricalCrossEntropy;
                    break;
                default:
                    throw new ArgumentException("Unsupported cost function type");
            }
        }
        private double[] CalculateCost(double[] expectedOutputValues)
        {
            int columnIndex = Data.ColumnCount() - 1;
            INeuron[] MotorNeurons = Data.Column(columnIndex);
            double[] costDifferences = new double[_NumberOfOutputs];

            Parallel.For(0, _NumberOfOutputs, i =>
            {
                double nnOutputValue = MotorNeurons[i].Activation;
                costDifferences[i] = CostFunctionDeriv(nnOutputValue, expectedOutputValues[i]);
            });

            return costDifferences;
        }
        #endregion

        #region Training
        public void Train(double[][] trainingDataCollection, double[][] ExpectedResults, double LearningRate, int epochs, bool Silent = false)
        {
            long MaxVal = (long)epochs * (long)trainingDataCollection.Length;
            long Counter = 0;
            for (int e = 0; e < epochs; e++)
            {
                for (int i = 0; i < trainingDataCollection.Length; i++)
                {
                    if (!Silent)
                    {
                        cnsl.DisplayProgressBar(Counter, MaxVal, "Progress");
                    }
                    Learn(trainingDataCollection[i], ExpectedResults[i], LearningRate);
                    Counter++;
                }
            }
        }
        private void Learn(double[] trainingData, double[] ExpectedResult, double LearningRate)
        {
            SetSensoryNeuronsValues(trainingData);
            ForwardPropagate();
            BackPropagate(LearningRate, ExpectedResult);
        }
        public void SetSensoryNeuronsValues(double[] inputValues)
        {
            Parallel.For(0, _NumberOfInputs, i =>
           {
               Data[(int)i, 0].Activation = inputValues[i];

           });
        }
        private void BackPropagate(double learningRate, double[] expectedOutputValues)
        {
            double[] costs = CalculateCost(expectedOutputValues);

            UpdateOutputLayerGradients(costs, learningRate, _L2RegulationLamda);
            UpdateHiddenLayerGradients(learningRate);
        }
        #endregion

        #region GetSettings

        public double Evaluate(double[][] inputs, double[][] outputs)
        {
            double totalError = 0.0;
            List<double[]> predictions = new List<double[]>();

            for (int i = 0; i < inputs.Length; i++)
            {
                double[] prediction = Predict(inputs[i]);
                predictions.Add(prediction);
                totalError += CalculateError(prediction, outputs[i]);
            }
            double InputSensitive = InputSensitivity(inputs, predictions.ToArray());

            return (inputs.Length / totalError) * InputSensitivity(inputs, predictions.ToArray());
        }

        public double InputSensitivity(double[][] inputs, double[][] outputs)
        {
            double totalError = 0.0;

            List<double[]> CheckedArray = new List<double[]>();
            List<double> SensitivityList = new List<double>();

            for (int i = 1; i < inputs.Length; i++)
            {
                if (CheckedArray.Where(r => ArrayIsEqual(r, inputs[i]) == true).Count() == 0)
                {
                    double Dec = 0;
                    double SensitivityCurrent = 0;

                    for (int j = 0; j < outputs[i].Length; j++)
                    {
                        SensitivityCurrent += Math.Abs(outputs[i][j] - outputs[i - 1][j]);
                        Dec += outputs[i][j];
                    }

                    SensitivityList.Add(SensitivityCurrent / Dec);
                }
            }

            double Sensitivity = SensitivityList.Average();
            return Sensitivity;
        }

        private bool ArrayIsEqual(double[] array1, double[] array2)
        {
            if (array1.Length != array2.Length)
            {
                return false;
            }
            else
            {
                for (int i = 0; i < array1.Length; i++)
                {
                    if (array1[i] != array2[i])
                    {
                        return false;
                    }
                }
                return true;
            }
        }

        private double CalculateError(double[] prediction, double[] actual)
        {
            double error = 0.0;
            for (int i = 0; i < prediction.Length; i++)
            {
                error += Math.Pow(prediction[i] - actual[i], 2);
            }
            return error;
        }


        public static Tuple<Dictionary<string, object>, NeuralNetwork> GetBestSettings(double TrainingRate, double[][] trainingInputs, double[][] trainingOutputs, double[][] validationInputs, double[][] validationOutputs, int epochs = 1000)
        {
            Console.WriteLine("Get Best Settings");
            var bestSettings = new ConcurrentDictionary<string, object>();
            double bestPerformance = double.MaxValue;
            object bestPerformanceLock = new object();

            var activationTypes = Enum.GetValues(typeof(ActivationType)).Cast<ActivationType>().ToList();
            var costFunctionTypes = Enum.GetValues(typeof(CostFunctionType)).Cast<CostFunctionType>().ToList();

            var huberLossDeltas = new float[] { 0.1f, 0.5f, 1.0f, 1.5f, 2.0f, 2.5f, 3.0f };
            var l2RegulationLambdas = new double[] { 0, 0.001, 0.01, 0.1, 1.0, 10.0 };
            var clippingLimits = new float[] { 0.1f, 0.5f, 1.0f, 1.5f, 2.0f, 2.5f, 3.0f };

            long counter = 0;
            long maxVal = costFunctionTypes.Count * costFunctionTypes.Count * huberLossDeltas.Length * l2RegulationLambdas.Length * clippingLimits.Length;

            Console.WriteLine($"Number of trainable networks {maxVal}");

            NeuralNetwork bestNeuralNetwork = null;
            object bestNetworkLock = new object();

            var options = new ParallelOptions { MaxDegreeOfParallelism = Environment.ProcessorCount };


                     //for (int i = 0; i < activationTypes.Count(); i++)

            Parallel.For(0, activationTypes.Count(), i =>
            {
                foreach (var costFunction in costFunctionTypes)
                {
                    foreach (var l2Lambda in l2RegulationLambdas)
                    {
                        foreach (var upperLimit in clippingLimits)
                        {
                            foreach (var lowerLimit in clippingLimits)
                            {
                                try
                                {
                                    if (costFunction == CostFunctionType.huberLoss)
                                    {
                                        foreach (var huberDelta in huberLossDeltas)
                                        {
                                            var nn = new NeuralNetwork(AccelerationType.GPU, GetSensoryNeurons(trainingInputs), 5, trainingInputs[0].Length, trainingOutputs[0].Length, activationTypes[(int)i], costFunction, huberDelta, l2Lambda, upperLimit, -lowerLimit);
                                            nn.Train(trainingInputs, trainingOutputs, TrainingRate, epochs, true);
                                            var performance = nn.Evaluate(validationInputs, validationOutputs);

                                            lock (bestPerformanceLock)
                                            {
                                                if (performance > bestPerformance)
                                                {
                                                    bestPerformance = performance;
                                                    bestSettings["ActivationType"] = activationTypes[(int)i];
                                                    bestSettings["CostFunction"] = costFunction;
                                                    bestSettings["L2RegulationLambda"] = l2Lambda;
                                                    bestSettings["ClippingLimit_Upper"] = upperLimit;
                                                    bestSettings["ClippingLimit_Lower"] = lowerLimit;
                                                    bestSettings["HuberLossDelta"] = huberDelta;

                                                    Debug.WriteLine($"Best Performance: {bestPerformance}");

                                                    foreach (var setting in bestSettings)
                                                    {
                                                        Debug.WriteLine($"{setting.Key}: {setting.Value}");
                                                    }

                                                    lock (bestNetworkLock)
                                                    {
                                                        bestNeuralNetwork = nn;
                                                    }
                                                }
                                            }

                                            Interlocked.Increment(ref counter);
                                            cnsl.DisplayProgressBar(counter, maxVal, " Progress");
                                        }
                                    }
                                    else
                                    {
                                        var nn = new NeuralNetwork(AccelerationType.GPU, GetSensoryNeurons(trainingInputs), 5, trainingInputs[0].Length, trainingOutputs[0].Length, activationTypes[(int)i], costFunction, l2Lambda, upperLimit, -lowerLimit);
                                        nn.Train(trainingInputs, trainingOutputs, TrainingRate, epochs, true);
                                        var performance = nn.Evaluate(validationInputs, validationOutputs);

                                        lock (bestPerformanceLock)
                                        {
                                            if (performance > bestPerformance)
                                            {
                                                bestPerformance = performance;
                                                bestSettings["ActivationType"] = activationTypes[(int)i];
                                                bestSettings["CostFunction"] = costFunction;
                                                bestSettings["L2RegulationLambda"] = l2Lambda;
                                                bestSettings["ClippingLimit_Upper"] = upperLimit;
                                                bestSettings["ClippingLimit_Lower"] = lowerLimit;

                                                Debug.WriteLine($"Best Performance: {bestPerformance}");
                                                foreach (var setting in bestSettings)
                                                {
                                                    Debug.WriteLine($"{setting.Key}: {setting.Value}");
                                                }

                                                lock (bestNetworkLock)
                                                {
                                                    bestNeuralNetwork = nn;
                                                }
                                            }
                                        }

                                        Interlocked.Increment(ref counter);
                                        cnsl.DisplayProgressBar(counter, maxVal, " Progress");
                                    }
                                }
                                catch (NaNException nanEx)
                                {
                                    Debug.WriteLine($"{nanEx.Message} \n activationType:{activationTypes[(int)i]}, costFunction:{costFunction.ToString()}, clippingLimits:[{lowerLimit},{upperLimit}], l2Lambda:{l2Lambda}");

                                    continue; // Skip the rest of the loop body and move to the next iteration
                                }
                                catch (InfinityException infinityEx)
                                {
                                    continue; // Skip the rest of the loop body and move to the next iteration
                                    Debug.WriteLine($"{infinityEx.Message} \n activationType:{activationTypes[(int)i]}, costFunction:{costFunction.ToString()}, clippingLimits:[{lowerLimit},{upperLimit}], l2Lambda:{l2Lambda}");
                                }
                                catch (Exception ex)
                                {
                                    continue; // Skip the rest of the loop body and move to the next iteration
                                    Debug.WriteLine(ex.Message);
                                }
                            }
                        }
                    }
                }
            });

            Console.WriteLine($"Best Performance: {bestPerformance}");
            foreach (var setting in bestSettings)
            {
                Console.WriteLine($"{setting.Key}: {setting.Value}");
            }

            return new Tuple<Dictionary<string, object>, NeuralNetwork>(new Dictionary<string, object>(bestSettings), bestNeuralNetwork);
        }
        
        #endregion

        public double[] Predict(double[] inputValues)
        {
            SetSensoryNeuronsValues(inputValues);
            ForwardPropagate();

            INeuron[] outputNeurons = Data.Column(Data.ColumnCount() - 1);

            var prediction = outputNeurons.Select(neuron => neuron.Activation).ToArray();

            return outputNeurons.Select(neuron => neuron.Activation).ToArray();
        }
        public void Load(string FileName)
        {
            Data = Matrix<INeuron>.LoadFromFile(FileName);
        }
        public void Save(string FileName)
        {
            Data.ExportToFile(FileName);
        }

        /// <summary>
        /// When the data has been changed this method needs to be called.
        /// </summary>
        public void DataChanged()
        {
            InitCalculationTensors();
        }
    }
}