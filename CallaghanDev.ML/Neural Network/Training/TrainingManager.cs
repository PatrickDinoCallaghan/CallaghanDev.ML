using CallaghanDev.ML.Neural_Network;

namespace CallaghanDev.ML.NN.Training
{
    public abstract class TrainingManagerBase
    {
        protected CostFunctionManager _costFunctionManager { get; set; }
        protected DataManager _dataManager { get; set; }
        protected AccelerationManager _accelerationManager { get; set; }
        protected Parameters _parameters { get; set; }
        protected Action _ForwardPropagate { get; set; }

        protected TrainingManagerBase(CostFunctionManager costFunctionManager, DataManager dataManager, AccelerationManager accelerationManager, Parameters parameters, Action forwardPropagate)
        {
            _costFunctionManager = costFunctionManager;
            _dataManager = dataManager;
            _accelerationManager = accelerationManager;
            _parameters = parameters;
            _ForwardPropagate = forwardPropagate;
        }
        public abstract void Train(double[][] trainingDataCollection, double[][] ExpectedResults, double LearningRate, int epochs, bool Silent = false);
        public abstract void Train(AccelerationType accelerationType, double[][] trainingDataCollection, double[][] ExpectedResults, double LearningRate, int epochs, bool Silent = false);

        public void SetSensoryNeuronsValues(double[] inputValues)
        {
            Parallel.For(0, _parameters.SensoryNeurons.Length, i =>
            {
                _dataManager.Data[(int)i, 0].Activation = inputValues[i];

            });
        }

        public EpochFinished OnEpochFinished;

        public delegate void EpochFinished(int Epoch);
    }


}