using Newtonsoft.Json;

namespace CallaghanDev.ML
{
    [JsonObject(MemberSerialization.OptIn)]
    public class MotorNeuron : NeuronBase, INeuron
    {
        [JsonProperty]
        public int LayerIndex { get; set; }

        public MotorNeuron(ActivationType activationType) : base(activationType)
        {
            this.Bias = 0;
        }
    }
}
