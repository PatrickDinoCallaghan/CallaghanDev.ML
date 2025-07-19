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
        float[] CalculateDotProduct(float[,] matrix, float[] vector);
        (float[] activation, float[] derivative) ActivateLayer(float[] dot, float[] bias, ActivationType activationType);
        float[] CalculateOutputGradients(float[] cost, float[] derivative);
        float[] CalculateHiddenGradients(float[,] weights, float[] nextDeltas, float[] derivative);
        float[,] UpdateWeights(float[,] weights, float[] deltas, float[] prevActivations, float learningRate, float lambda);
        float[] UpdateBias(float[] bias, float[] deltas, float learningRate);
        void Dispose();
    }
}
