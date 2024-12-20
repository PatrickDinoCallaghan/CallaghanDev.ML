using CallaghanDev.Common.Math;
using CallaghanDev.Utilities.MathTools;
using DocumentFormat.OpenXml.Spreadsheet;
using DocumentFormat.OpenXml.Wordprocessing;

namespace CallaghanDev.ML.NN
{
    public interface IDataManager
    {
        //TODO: You have too many collections representing the same data
        // Remove Neurons[][] and find a good way to extract this infor from Data directly
        public Matrix<INeuron> Data { get; set; }
    }
    public class DataManager : IDisposable
    {
        public Matrix<INeuron> Data = new Matrix<INeuron>();

        public INeuron[][] Neurons;

        // This structure arranges each element in the array of matrices to represent a matrix associated with a specific layer. Starting at the first hidden layer from the input layer
        // Each row corresponds to a neuron's dendrites in the hidden layer, indicating these neurons originate from the current layer, and connect to each neuron in the next layer.
        // Transposing this matrix switches the focus from source neurons in the current layer to target neurons in the subsequent layer, and their connections to each neuron in the subsequent layer.
        public Matrix<Neurite>[] NeuriteTensor;


        private Parameters _parameters;
        private Random random;
        public DataManager()
        {

        }

        public void InitializeData(Parameters parameters, SensoryNeuron[] sensoryNeurons, bool LoadingFromFile = false)
        {
            _parameters = parameters;
            InitSensoryNeurons(sensoryNeurons);

            random = new Random();

            if (!LoadingFromFile)
            {
                NeuriteTensor = new Matrix<Neurite>[parameters.NoHiddenLayers + 1];
                InitHiddenLayer(parameters);
                InitMotorLayers(parameters);
            }
            InitCalculationTensors();
        }

        private void InitSensoryNeurons(SensoryNeuron[] sensoryNeurons)
        {
            for (int i = 0; i < sensoryNeurons.Length; i++)
            {
                sensoryNeurons[i].LayerIndex = 0;
                Data[(int)i, 0] = sensoryNeurons[i];
            }
        }
        private void InitHiddenLayer(Parameters parameters)
        {
            for (int j = 0; j < parameters.NoHiddenLayers; j++)
            {
                NeuriteTensor[j] = new Matrix<Neurite>();

                int NumberOfNeuronsInPreviousLayer = (int)Data.Column(j).Count();
                int NumberOfNeuronsInNextLayer = j == parameters.NoHiddenLayers - 1 ? parameters.NumberOfOutputs : parameters.HiddenLayerWidth;
                List<double> LayerWeights = new List<double>();

                for (int h = 0; h < parameters.HiddenLayerWidth; h++)
                {
                    // if the next layer is the motor layer and it has fewer neurons this should be fewer 
                    AssociativeNeuron neuron = new AssociativeNeuron(parameters.DefaultActivationType);
                    neuron.LayerIndex = j + 1;
                    Data[h, j + 1] = neuron;

                    for (int i = 0; i < NumberOfNeuronsInPreviousLayer; i++)
                    {
                        double weight = Initializer(parameters.DefaultActivationType, NumberOfNeuronsInPreviousLayer, NumberOfNeuronsInNextLayer);
                        LayerWeights.Add(weight);
                        Neurite neurite = new Neurite(weight);

                        NeuriteTensor[j][h, i] = neurite;
                    }
                }
            }
        }

        private void InitMotorLayers(Parameters parameters)
        {
            int PreviousColIndex = Data.ColumnCount() - 1;

            int NumberOfNeuronsInpreviousLayer = Data.Column(PreviousColIndex).Count();
        

            NeuriteTensor[parameters.NoHiddenLayers] = new Matrix<Neurite>();
            List<double> LayerWeights = new List<double>();

            for (int m = 0; m < parameters.NumberOfOutputs; m++)
            {
                MotorNeuron neuron = new MotorNeuron(parameters.DefaultActivationType);
                neuron.LayerIndex = PreviousColIndex + 1;
                Data[m, PreviousColIndex + 1] = neuron;
                for (int i = 0; i < NumberOfNeuronsInpreviousLayer; i++)
                {
                    double weight = Initializer(parameters.DefaultActivationType, NumberOfNeuronsInpreviousLayer, 0);
                    LayerWeights.Add(weight);
                    Neurite neurite = new Neurite(weight);

                    NeuriteTensor[parameters.NoHiddenLayers][m, i] = neurite;
                }
            }
        }

        public void InitCalculationTensors()
        {

            INeuron[][] Neurons_WithNulls = Data.ConvertToColumnArray();

            int colCount = Data.ColumnCount();
            Neurons = new INeuron[colCount][];

            for (int layerIndex = colCount - 1; layerIndex >= 0; layerIndex--)
            {
                Neurons[layerIndex] = Neurons_WithNulls[layerIndex].Where(r => r != null).ToArray();
            }
        }
        private double Initializer(ActivationType activationType, int incomingNeurites, int outgoingNeurites)
        {
            switch (activationType)
            {
                case ActivationType.Tanh:
                    return XavierGlorotInitializer(incomingNeurites, outgoingNeurites);
                case ActivationType.Sigmoid:
                    return XavierGlorotInitializer(incomingNeurites, outgoingNeurites);
                case ActivationType.Leakyrelu:
                    return LeakyReLUInitializer(incomingNeurites);
                case ActivationType.Relu:
                    return LeakyReLUInitializer(incomingNeurites);
                default:
                    return GetRandomDouble(random, -1, 1);
            }
        }
        /// <summary>
        /// He Initialization : ReLU / LeakyReLU Linear
        /// </summary>
        /// <param name="incomingNeurites"></param>
        /// <returns></returns>
        private double LeakyReLUInitializer(int incomingNeurites)
        {
            double standardDeviation = Math.Sqrt(2.0 / incomingNeurites);
            // Generate a normally distributed random value
            double u1 = 1.0 - random.NextDouble(); // Uniform (0,1] random double
            double u2 = 1.0 - random.NextDouble(); // Uniform (0,1] random double
            double z = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Sin(2.0 * Math.PI * u2); // Box-Muller transform
            return z * standardDeviation; // Scale by standard deviation


        }
        /// <summary>
        /// Generates a weight using Xavier/Glorot Initialization. : Sigmoid/Tanh:
        /// </summary>
        /// <param name="incomingNeurites">Number of incoming connections (neurons in the previous layer).</param>
        /// <param name="outgoingNeurites">Number of outgoing connections (neurons in the next layer).</param>
        /// <returns>A weight value initialized using Xavier/Glorot method.</returns>
        public double XavierGlorotInitializer(int incomingNeurites, int outgoingNeurites)
        {
            if (incomingNeurites <= 0 || outgoingNeurites <= 0)
                throw new ArgumentException("Neurites count must be positive.");

            // Compute the standard deviation for Xavier Initialization
            double standardDeviation = Math.Sqrt(2.0 / (incomingNeurites + outgoingNeurites));

            // Box-Muller transform to generate normally distributed random value
            double u1 = 1.0 - random.NextDouble(); // Uniform (0,1] random double
            double u2 = 1.0 - random.NextDouble(); // Uniform (0,1] random double
            double z = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2); // Normally distributed random value

            // Scale the value by the standard deviation
            return z * standardDeviation;
        }
        private double GetRandomDouble(Random random, double min, double max)
        {
            if (min >= max)
            {
                throw new ArgumentException("The minimum value must be less than the maximum value.");
            }

            // Generate a random double between 0.0 (inclusive) and 1.0 (exclusive)
            double randomValue = random.NextDouble();

            // Scale and shift the value to the specified range
            return min + (randomValue * (max - min));
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
                        Tuple<double, double> NewTuple = new Tuple<double, double>(Math.Floor(datapoint[i]), (MinMaxArray[i].Item2));
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

            for (int i = 0; i < TrainingDataLength; i++) 
            {
                inputNeurons[i] = (new SensoryNeuron(MinMaxArray[i].Item1, MinMaxArray[i].Item2));
            }

            return inputNeurons;
        }


        private bool disposed = false;

        public void Dispose()
        {
            Dispose(true);
            GC.SuppressFinalize(this);
        }

        protected virtual void Dispose(bool disposing)
        {
            if (!disposed)
            {
                if (disposing)
                {
                    Data.Dispose();
                    // Dispose other managed resources if necessary.

                    foreach (var item in NeuriteTensor)
                    {
                        item.Dispose();
                    }
                }

                // Dispose unmanaged resources here, if any.

                disposed = true;
            }
        }

        ~DataManager()
        {
            Dispose(false);
        }
    }
}
