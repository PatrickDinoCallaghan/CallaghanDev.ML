using CallaghanDev.Utilities.Code;
using CallaghanDev.Utilities.ConsoleHelper;
using CallaghanDev.Utilities.MathTools;
using ILGPU;
using ILGPU.Algorithms;
using ILGPU.Runtime;
using ILGPU.Runtime.OpenCL;
using Newtonsoft.Json;
using System.Diagnostics;

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

        #region Task Containers

        TaskManager taskManager = new TaskManager();

        TaskContainer<double[]> deltasArrayTaskContainer ;
        TaskContainer<double[]> precedingLayerNeuronsActivationsTaskContainer;
        TaskContainer<double[]> hiddenNeuronsActivationFunctionDerivTaskContainer;
        TaskContainer<double[]> hiddenNeuronsBiasesTaskContainer;

        #endregion

        #region ctor


        public NeuralNetwork(SensoryNeuron[] sensoryNeurons, int NoHiddenLayers, int HiddenLayerWidth, int NumberOfOutputs, ActivationType DefaultActivationType, CostFunctionType costFunction, double l2RegulationLamda= 0, float clippingLimit_Upper = 1, float clippingLimit_Lower = -1)
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

            InitSensoryNeurons(sensoryNeurons);
            Initialize();
        }
        public NeuralNetwork(SensoryNeuron[] sensoryNeurons, int NoHiddenLayers, int HiddenLayerWidth, int NumberOfOutputs, ActivationType DefaultActivationType, CostFunctionType costFunction, float HuberLossDelta , double l2RegulationLamda = 0, float clippingLimit_Upper = 1, float clippingLimit_Lower = -1)
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
                inputNeurons[i] = (new SensoryNeuron( MinMaxArray[i].Item2, MinMaxArray[i].Item1));
            }

            return inputNeurons;
        }

        // Helper method to evaluate the performance of the network on validation data
        public double Evaluate(double[][] inputs, double[][] outputs)
        {
            double totalError = 0.0;
            for (int i = 0; i < inputs.Length; i++)
            {
                var prediction = Predict(inputs[i]);
                totalError += CalculateError(prediction, outputs[i]);
            }
            return totalError / inputs.Length;
        }

        // Helper method to calculate the error between prediction and actual output
        private double CalculateError(double[] prediction, double[] actual)
        {
            double error = 0.0;
            for (int i = 0; i < prediction.Length; i++)
            {
                error += Math.Pow(prediction[i] - actual[i], 2);
            }
            return error;
        }
        public static Dictionary<string, object> GetBestSettings(double[][] trainingInputs, double[][] trainingOutputs, double[][] validationInputs, double[][] validationOutputs, int epochs = 100)
        {
            Console.WriteLine("Get Best Settings");
            var bestSettings = new Dictionary<string, object>();
            double bestPerformance = double.MaxValue;

            var activationTypes = Enum.GetValues(typeof(ActivationType)).Cast<ActivationType>().ToList();
            var costFunctionTypes = Enum.GetValues(typeof(CostFunctionType)).Cast<CostFunctionType>().ToList();

           // var huberLossDeltas = new float[] { 0.1f, 0.5f, 1.0f, 1.5f, 2.0f, 2.5f, 3.0f };
           // var l2RegulationLambdas = new double[] { 0, 0.001, 0.01, 0.1, 1.0, 10.0 };
            //var clippingLimits = new float[] { 0.1f, 0.5f, 1.0f, 1.5f, 2.0f, 2.5f, 3.0f };

            var huberLossDeltas = new float[] { 0.1f, 0.5f, 1.0f };
            var l2RegulationLambdas = new double[] { 0, 0.01, 0.1 };
            var clippingLimits = new float[] { 0.5f, 1.0f, 1.5f };

            long Counter = 0;

            long MaxVal = costFunctionTypes.Count() * costFunctionTypes.Count() * huberLossDeltas.Length * l2RegulationLambdas.Length * clippingLimits.Length;
            Console.WriteLine($"Number of trainable networks {MaxVal}");
            foreach (var activationType in activationTypes)
            {
                foreach (var costFunction in costFunctionTypes)
                {
                    foreach (var l2Lambda in l2RegulationLambdas)
                    {
                        foreach (var upperLimit in clippingLimits)
                        {
                            foreach (var lowerLimit in clippingLimits)
                            {
                                if (costFunction == CostFunctionType.huberLoss)
                                {
                                    foreach (var huberDelta in huberLossDeltas)
                                    {
                                        var nn = new NeuralNetwork(GetSensoryNeurons(trainingInputs), 5, trainingInputs[0].Length, trainingOutputs[0].Length, activationType, costFunction, huberDelta, l2Lambda, upperLimit, -lowerLimit);
                                        nn.Train(trainingInputs, trainingOutputs, 0.01, epochs);
                                        var performance = nn.Evaluate(validationInputs, validationOutputs);

                                        if (performance < bestPerformance)
                                        {
                                            bestPerformance = performance;
                                            bestSettings["ActivationType"] = activationType;
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
                                        }


                                        cnsl.DisplayProgressBar(++Counter, MaxVal, " Progress");
                                    }
                                }
                                else
                                {
                                    var nn = new NeuralNetwork(GetSensoryNeurons(trainingInputs), 5, trainingInputs[0].Length, trainingOutputs[0].Length, activationType, costFunction, l2Lambda, upperLimit, -lowerLimit);
                                    nn.Train(trainingInputs, trainingOutputs, 0.01, epochs);
                                    var performance = nn.Evaluate(validationInputs, validationOutputs);

                                    if (performance < bestPerformance)
                                    {
                                        bestPerformance = performance;
                                        bestSettings["ActivationType"] = activationType;
                                        bestSettings["CostFunction"] = costFunction;
                                        bestSettings["L2RegulationLambda"] = l2Lambda;
                                        bestSettings["ClippingLimit_Upper"] = upperLimit;
                                        bestSettings["ClippingLimit_Lower"] = lowerLimit;

                                        Debug.WriteLine($"Best Performance: {bestPerformance}");
                                        foreach (var setting in bestSettings)
                                        {
                                            Debug.WriteLine($"{setting.Key}: {setting.Value}");
                                        }
                                    }
                                    cnsl.DisplayProgressBar(++Counter, MaxVal, " Progress");
                                }
                            }
                        }
                    }
                }
            }

            Debug.WriteLine($"Best Performance: {bestPerformance}");
            foreach (var setting in bestSettings)
            {
                Debug.WriteLine($"{setting.Key}: {setting.Value}");
            }
            return bestSettings;
        }
        #endregion

        #region initialization methods

        private Action<Index1D, ArrayView2D<double, Stride2D.DenseX>, ArrayView1D<double, Stride1D.Dense>, ArrayView1D<double, Stride1D.Dense>, ArrayView1D<double, Stride1D.Dense>, ArrayView1D<double, Stride1D.Dense>, ArrayView1D<double, Stride1D.Dense>, ArrayView2D<double, Stride2D.DenseX>, double, double, double> double_GPUBackpropResult;

        private Action<Index1D, ArrayView1D<double, Stride1D.Dense>, ArrayView2D<double, Stride2D.DenseX>, ArrayView1D<double, Stride1D.Dense>> double_MatrixVectorKernel;

        public void Initialize()
        {
            Context context = Context.Create(builder => builder.AllAccelerators());

            accelerator = context.CreateCLAccelerator(0);

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
            InitTaskContainers();
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
        private void InitTaskContainers()
        {
            deltasArrayTaskContainer = taskManager.GetOrCreateContainer<double[]>("deltasArrayTask");
            precedingLayerNeuronsActivationsTaskContainer = taskManager.GetOrCreateContainer<double[]>("precedingLayerNeuronsActivationsTask");
            hiddenNeuronsActivationFunctionDerivTaskContainer = taskManager.GetOrCreateContainer<double[]>("hiddenNeuronsActivationFunctionDerivTask");
            hiddenNeuronsBiasesTaskContainer = taskManager.GetOrCreateContainer<double[]>("hiddenNeuronsBiasesTask");


        }
        #endregion

        #region Forward Propagate

        public void ForwardPropagate()
        {
            for (int j = 1; j < Data.ColumnCount(); j++)
            {
                double[] SourceNeurons_Activations = Neurons[j - 1].Where(r => r != null).Select(r => r.Activation).ToArray();
                double[,] neurites_Weights = NeuriteTensorT[j - 1].Select(r => r.Weight).ToArray();

                double[] dotproduct = CalculateDotProduct(neurites_Weights, SourceNeurons_Activations);

                Parallel.For(0, dotproduct.Length, c =>
                {
                    Neurons[j][c].Activation = Neurons[j][c].activationFunction(dotproduct[c] + Neurons[j][c].Bias);

                    if (double.IsNaN(Neurons[j][c].Activation))
                    {
                        throw new Exception($"NaN detected in forward propagation at layer {j}, neuron {c}, type:{Neurons[j][c].GetType().Name}");
                    }
                    if ( double.IsInfinity(Neurons[j][c].Activation))
                    {
                        throw new Exception($"Infinity detected in forward propagation at layer {j}, neuron {c}, type:{Neurons[j][c].GetType().Name}");
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

            for (int i = 0; i < MotorNeurons.Length; i++)
            {
                // Calculate the derivative of the activation function for each neuron's current activation level.
                double activationDerivative = MotorNeurons[i].activationFunctionDeriv(MotorNeurons[i].Activation);

                // Compute the gradient of the loss with respect to the neuron's output (Delta).
                // This is derived from the derivative of the cost function with respect to the output
                // multiplied by the derivative of the activation function.
                double gradient = -costs[i] * activationDerivative;
                gradient = Math.Max(_clippingLimit_Lower, Math.Min(_clippingLimit_Upper, gradient));
                MotorNeurons[i].Delta = gradient;



                Parallel.ForEach(MotorNeurons[i].Dendrites, connection =>
                {
                    // Calculate the weight update, which is a product of the learning rate, the delta, and
                    // the activation of the neuron from which this dendrite receives input.
                    // double weightUpdate = learningRate * (gradient * connection.SourceNeuron.Activation + lambda * connection.Weight);

                    // Calculate the weight update, which is a product of the learning rate, the delta, and
                    // the activation of the neuron from which this dendrite receives input, plus the regularization term.
                    // The regularization term (lambda * connection.Weight) helps to prevent overfitting by penalizing large weights.
                    // Lambda is a hyperparameter that controls the strength of this penalty: 
                    // a higher lambda value increases the penalty, encouraging smaller weights and thus simpler models.
                    double weightUpdate = learningRate * (gradient * connection.SourceNeuron.Activation + lambda * connection.Weight);




                    // Apply the update: since we are minimizing the loss, we subtract the update.
                    connection.Weight -= weightUpdate;


                    if (double.IsNaN(connection.Weight))
                    {
                        throw new Exception($"NaN detected");
                    }
                    if (double.IsInfinity(connection.Weight))
                    {
                        throw new Exception($"Infinity detected");
                    }
                });

                // Update the bias of the neuron as well, using the learning rate and the calculated delta.
                MotorNeurons[i].Bias -= learningRate * gradient;
            }
        }

        #region Back propagation
        // Utility function to extract the weight matrix of a specific layer
        public double[,] GetWeightsMatrix(int layerIndex)
        {

            double[,] Weights = new double[NeuriteTensor[layerIndex].RowCount(), NeuriteTensor[layerIndex].ColumnCount()];// NeuriteTensor[layerIndex].Select(r => r.Weight).ToArray();

            Parallel.ForEach(NeuriteTensor[layerIndex], item =>
            {
                Weights[item.Key.Row, item.Key.Column] = item.Value.Weight;
            });

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

                taskManager.SetOrCreateTask(deltasArrayTaskContainer, () => nextLayerNeurons.Select(neuron => neuron.Delta).ToArray());
                taskManager.SetOrCreateTask(precedingLayerNeuronsActivationsTaskContainer, () => precedingLayerNeurons.Select(r => r.Activation).ToArray());
                taskManager.SetOrCreateTask(hiddenNeuronsActivationFunctionDerivTaskContainer, () => hiddenNeurons.Select(r => r.activationFunctionDeriv(r.Activation)).ToArray());
                taskManager.SetOrCreateTask(hiddenNeuronsBiasesTaskContainer, () => hiddenNeurons.Select(r => r.Bias).ToArray());

                // Wait for all tasks to complete
                taskManager.WaitForAll(deltasArrayTaskContainer, precedingLayerNeuronsActivationsTaskContainer, hiddenNeuronsActivationFunctionDerivTaskContainer, hiddenNeuronsBiasesTaskContainer);


                // Retrieve the results
                double[] deltasArray = deltasArrayTaskContainer.Task.Result;
                double[] precedingLayerNeuronsActivations = precedingLayerNeuronsActivationsTaskContainer.Task.Result;
                double[] hiddenNeurons_activationFunctionDeriv = hiddenNeuronsActivationFunctionDerivTaskContainer.Task.Result;
                double[] hiddenNeurons_Biases = hiddenNeuronsBiasesTaskContainer.Task.Result;

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
            /*
            taskManager.SetOrCreateTask<int>(weightsMatrixBufferTaskContainer, () => { weightsMatrixBuffer.CopyFromCPU(weightsMatrix); return 0; });
            taskManager.SetOrCreateTask<int>(deltasBuffersTaskContainer, () => { deltasBuffer.CopyFromCPU(deltas); return 0; });
            taskManager.SetOrCreateTask<int>(activationsBufferTaskContainer, () => { activationsBuffer.CopyFromCPU(activations); return 0; });
            taskManager.SetOrCreateTask<int>(activationDerivativesBufferTaskContainer, () => { activationDerivativesBuffer.CopyFromCPU(activationDerivatives); return 0; });
            taskManager.SetOrCreateTask<int>(biasesBufferTaskContainer, () => { biasesBuffer.CopyFromCPU(biases); return 0; });
            // Wait for all tasks to complete
            taskManager.WaitForAll(weightsMatrixBufferTaskContainer, deltasBuffersTaskContainer, activationsBufferTaskContainer, activationDerivativesBufferTaskContainer, biasesBufferTaskContainer);
            */
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
            INeuron[] MotorNeurons = Data.Column(Data.ColumnCount() - 1).Select(r => r).ToArray();

            double[] NNOutputValues = MotorNeurons.Select(r => r.Activation).ToArray();

            double[] costDifferences = new double[expectedOutputValues.Length];

            for (int i = 0; i < expectedOutputValues.Length; i++)
            {
                costDifferences[i] = CostFunctionDeriv( NNOutputValues[i], expectedOutputValues[i]);
            }

            return costDifferences;
        }
        #endregion

        #region Training
        public void Train(double[][] trainingDataCollection, double[][] ExpectedResults, double LearningRate, int epochs)
        {
            long MaxVal = (long)epochs * (long)trainingDataCollection.Length;
            long Counter = 0;
            for (int e = 0; e < epochs; e++)
            {
                for (int i = 0; i < trainingDataCollection.Length; i++)
                {
                    cnsl.DisplayProgressBar(Counter, MaxVal, "Progress");
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

