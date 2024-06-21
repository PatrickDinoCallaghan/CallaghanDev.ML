using Newtonsoft.Json;

namespace CallaghanDev.ML
{
    [JsonObject(MemberSerialization.OptIn)]
    public class AssociativeNeuron : NeuronBase, INeuron
    {
        [JsonProperty]
        public int LayerIndex { get; set; }

        public AssociativeNeuron(ActivationType activationType) : base(activationType)
        {

        }
    }

}
