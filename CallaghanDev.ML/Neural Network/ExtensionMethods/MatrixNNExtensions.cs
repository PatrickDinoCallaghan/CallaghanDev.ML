using CallaghanDev.ML.Neural_Network;
using CallaghanDev.Utilities.MathTools;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
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
            FieldInfo DataField = type.GetField("Data", BindingFlags.NonPublic | BindingFlags.Instance);
            FieldInfo neuriteTensorTField = type.GetField("NeuriteTensorT", BindingFlags.NonPublic | BindingFlags.Instance);
            FieldInfo numberOfInputsField = type.GetField("_NumberOfInputs", BindingFlags.NonPublic | BindingFlags.Instance);
            FieldInfo numberOfOutputsField = type.GetField("_NumberOfOutputs", BindingFlags.NonPublic | BindingFlags.Instance);
            FieldInfo hiddenLayerWidthField = type.GetField("_HiddenLayerWidth", BindingFlags.NonPublic | BindingFlags.Instance);
            FieldInfo noHiddenLayersField = type.GetField("_NoHiddenLayers", BindingFlags.NonPublic | BindingFlags.Instance);
            FieldInfo clippingLimitUpperField = type.GetField("_clippingLimit_Upper", BindingFlags.NonPublic | BindingFlags.Instance);
            FieldInfo clippingLimitLowerField = type.GetField("_clippingLimit_Lower", BindingFlags.NonPublic | BindingFlags.Instance);
            FieldInfo l2RegulationLamdaField = type.GetField("_L2RegulationLamda", BindingFlags.NonPublic | BindingFlags.Instance);
            FieldInfo huberLossDeltaField = type.GetField("_HuberLossDelta", BindingFlags.NonPublic | BindingFlags.Instance);
            FieldInfo defaultActivationTypeField = type.GetField("_DefaultActivationType", BindingFlags.NonPublic | BindingFlags.Instance);
            FieldInfo costFunctionField = type.GetField("_costFunction", BindingFlags.NonPublic | BindingFlags.Instance);

            // Read the values of the fields


            return new NeuralNetworkDto()
            {
                sensoryNeurons = ((Matrix<INeuron>)DataField.GetValue(neuralNetworkInstance)).Where(r => r.Value.GetType() == typeof(SensoryNeuron)).Select(r => (SensoryNeuron)r.Value).ToArray(),
                NoHiddenLayers = (int)noHiddenLayersField.GetValue(neuralNetworkInstance),
                HiddenLayerWidth = (int)hiddenLayerWidthField.GetValue(neuralNetworkInstance),
                NumberOfOutputs = (int)numberOfOutputsField.GetValue(neuralNetworkInstance),
                DefaultActivationType = (ActivationType)defaultActivationTypeField.GetValue(neuralNetworkInstance),
                costFunction = (CostFunctionType)costFunctionField.GetValue(neuralNetworkInstance),
                l2RegulationLamda = (double)l2RegulationLamdaField.GetValue(neuralNetworkInstance),
                clippingLimit_Upper = (float)clippingLimitUpperField.GetValue(neuralNetworkInstance),
                clippingLimit_Lower = (float)clippingLimitLowerField.GetValue(neuralNetworkInstance),
                HuberLossDelta = (float)huberLossDeltaField.GetValue(neuralNetworkInstance),
                Data = (Matrix<INeuron>)DataField.GetValue(neuralNetworkInstance)
            };
        }
    }
}
