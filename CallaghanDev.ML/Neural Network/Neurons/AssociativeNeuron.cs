using CallaghanDev.Utilities.MathTools;
using Newtonsoft.Json;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

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
