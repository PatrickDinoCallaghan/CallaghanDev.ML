using CallaghanDev.Common.Math;
using CallaghanDev.Utilities.Math;
using CallaghanDev.Utilities.MathTools;
using Newtonsoft.Json;

namespace CallaghanDev.ML.Neural_Network
{
    [JsonObject(MemberSerialization.OptIn)]
    public class NeuralNetworkDto
    {
        [JsonProperty]
        public int NoHiddenLayers;
        [JsonProperty]
        public int HiddenLayerWidth;
        [JsonProperty]
        public int NumberOfOutputs;
        [JsonProperty]
        public ActivationType DefaultActivationType;
        [JsonProperty]
        public CostFunctionType costFunction;
        [JsonProperty]
        public double l2RegulationLamda = 0;
        [JsonProperty]
        public float GradientClippingThreshold =5;
        [JsonProperty]
        public float HuberLossDelta;

        [JsonProperty]
        [JsonConverter(typeof(MatrixJsonConverter<INeuron>))]
        public Matrix<INeuron> Data;


        [JsonProperty]
        [JsonConverter(typeof(MatrixArrayJsonConverter<Neurite>))]
        public Matrix<Neurite>[] NeuriteTensor;

    }

    
}
