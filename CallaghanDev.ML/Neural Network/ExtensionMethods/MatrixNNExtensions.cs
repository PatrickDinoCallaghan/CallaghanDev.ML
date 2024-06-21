using CallaghanDev.ML.Neural_Network;
using CallaghanDev.ML.NN;
using CallaghanDev.Utilities.MathTools;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using System.Security.Cryptography;
using System.Text;
using System.Threading.Tasks;
using static System.Runtime.InteropServices.JavaScript.JSType;

namespace CallaghanDev.ML
{
    public static class MatrixNNExtensions
    {
        public static NeuralNetworkDto MapToDto(this NeuralNetwork neuralNetworkInstance)
        {
            Type type = neuralNetworkInstance.GetType();



            // Access private fields using reflection
            //FieldInfo DataField = type.GetField("Data", BindingFlags.NonPublic | BindingFlags.Instance);
            FieldInfo dataManagerField = type.GetField("dataManager", BindingFlags.NonPublic | BindingFlags.Instance);
            FieldInfo parametersField = type.GetField("parameters", BindingFlags.NonPublic | BindingFlags.Instance);

            // Read the values of the fields

            Parameters parameters = ((Parameters)parametersField.GetValue(neuralNetworkInstance));

            DataManager dataManager = ((DataManager)dataManagerField.GetValue(neuralNetworkInstance));

            return new NeuralNetworkDto()
            {
                NoHiddenLayers = parameters.NoHiddenLayers,
                HiddenLayerWidth = parameters.HiddenLayerWidth,
                NumberOfOutputs = parameters.NumberOfOutputs,
                DefaultActivationType = parameters.DefaultActivationType,
                costFunction = parameters.CostFunction,
                l2RegulationLamda = parameters.L2RegulationLamda,
                clippingLimit_Upper = parameters.ClippingLimitUpper,
                clippingLimit_Lower = parameters.ClippingLimitLower,
                HuberLossDelta = parameters.HuberLossDelta,
                Data = dataManager.Data,
                NeuriteTensor = dataManager.NeuriteTensor
            };
        }
    }
}
