using CallaghanDev.ML.Enums;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace CallaghanDev.ML.AccelerationManagers
{
    public interface IAccelerationManager
    {
        double[] CalculateDotProduct(double[,] matrix, double[] vector);
        (double[] activation, double[] derivative) ActivateLayer(double[] dot, double[] bias, ActivationType activationType);
        double[] CalculateOutputGradients(double[] cost, double[] derivative);
        double[] CalculateHiddenGradients(double[,] weights, double[] nextDeltas, double[] derivative);
        double[,] UpdateWeights(double[,] weights, double[] deltas, double[] prevActivations, double learningRate, double lambda);
        double[] UpdateBias(double[] bias, double[] deltas, double learningRate);
    }
}
