using CallaghanDev.ML.Neural_Network;
using CallaghanDev.ML.NN;
using CallaghanDev.ML.NN.Training;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace CallaghanDev.ML.NN.Training
{
    public class NeuroevolutionManager : TrainingManagerBase, ITrainingManager
    {
        public NeuroevolutionManager(CostFunctionManager costFunctionManager, DataManager dataManager, AccelerationManager accelerationManager, Parameters parameters, Action forwardPropagate) : base(costFunctionManager, dataManager, accelerationManager, parameters, forwardPropagate)
        {

        }

        public override void Train(double[][] trainingDataCollection, double[][] ExpectedResults, double LearningRate, int epochs, bool Silent = false)
        {
            throw new NotImplementedException();
        }

        public override void Train(AccelerationType accelerationType, double[][] trainingDataCollection, double[][] ExpectedResults, double LearningRate, int epochs, bool Silent = false)
        {
            throw new NotImplementedException();
        }
    }
}
