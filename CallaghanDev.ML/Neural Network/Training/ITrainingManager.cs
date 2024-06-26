﻿namespace CallaghanDev.ML.NN.Training
{
    public interface ITrainingManager
    {
        public void Train(double[][] trainingDataCollection, double[][] ExpectedResults, double LearningRate, int epochs, bool Silent = false);
        public void Train(AccelerationType accelerationType, double[][] trainingDataCollection, double[][] ExpectedResults, double LearningRate, int epochs, bool Silent = false);
        public void SetSensoryNeuronsValues(double[] inputValues);
    }
}
