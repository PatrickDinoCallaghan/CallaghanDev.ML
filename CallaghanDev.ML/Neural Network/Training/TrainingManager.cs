﻿using CallaghanDev.ML.Neural_Network;

namespace CallaghanDev.ML.NN.Training
{
    public abstract class TrainingManagerBase
    {
        protected CostFunctionManager _costFunctionManager { get; set; }
        protected DataManager _dataManager { get; set; }
        protected IAccelerationManager _accelerationManager { get; set; }
        protected Parameters _parameters { get; set; }

        protected TrainingManagerBase(CostFunctionManager costFunctionManager, DataManager dataManager, IAccelerationManager accelerationManager, Parameters parameters)
        {
            _costFunctionManager = costFunctionManager;
            _dataManager = dataManager;
            _accelerationManager = accelerationManager;
            _parameters = parameters;
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