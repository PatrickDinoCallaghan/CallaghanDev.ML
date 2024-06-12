using CallaghanDev.Utilities.Math;
using CallaghanDev.Utilities.MathTools;
using Newtonsoft.Json;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace CallaghanDev.ML.Neural_Network
{
    [JsonObject(MemberSerialization.OptIn)]
    public class NeuralNetworkDto
    {
        [JsonProperty]
        public SensoryNeuron[] sensoryNeurons;
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
        public float clippingLimit_Upper = 1;
        [JsonProperty]
        public float clippingLimit_Lower = -1;
        [JsonProperty]
        public float HuberLossDelta;

        [JsonProperty]
        [JsonConverter(typeof(MatrixJsonConverter<INeuron>))]
        public Matrix<INeuron> Data;

    }

    
}
