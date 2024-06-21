using CallaghanDev.ML.NN;
using CallaghanDev.Utilities.MathTools;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using static System.Runtime.InteropServices.JavaScript.JSType;

namespace CallaghanDev.ML.Neural_Network
{
    public class CostFunctionManager
    {

        private CostFunction CostFunctionDeriv { get; set; }
        private delegate double CostFunction(double value, double prediction);
        private CostFunction costFunction { get; set; }

        Matrix<INeuron> _Data;
        Parameters _parameters;
        public CostFunctionManager(Matrix<INeuron> Data, Parameters parameters)
        {
            _Data = Data;
            _parameters = parameters;

            Init(parameters.CostFunction);
        }

        public void Init(CostFunctionType costFunctionType)
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
                    costFunction = (actual, predicted) => UtilityFunctions.huberLoss(actual, predicted, _parameters.HuberLossDelta);
                    CostFunctionDeriv = (actual, predicted) => UtilityFunctions.FirstDerivative.huberLoss(actual, predicted, _parameters.HuberLossDelta);
                    break;
                case CostFunctionType.categoricalCrossEntropy:
                    costFunction = UtilityFunctions.categoricalCrossEntropy;
                    CostFunctionDeriv = UtilityFunctions.FirstDerivative.categoricalCrossEntropy;
                    break;
                default:
                    throw new ArgumentException("Unsupported cost function type");
            }
        }
        public double[] CalculateCost(double[] expectedOutputValues)
        {
            int columnIndex = _Data.ColumnCount() - 1;
            INeuron[] MotorNeurons = _Data.Column(columnIndex);
            double[] costDifferences = new double[_parameters.NumberOfOutputs];

            Parallel.For(0, _parameters.NumberOfOutputs, i =>
            {
                double nnOutputValue = MotorNeurons[i].Activation;
                costDifferences[i] = CostFunctionDeriv(nnOutputValue, expectedOutputValues[i]);
            });

            return costDifferences;
        }
    }
}
