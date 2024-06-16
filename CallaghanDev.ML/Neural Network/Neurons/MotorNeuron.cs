using CallaghanDev.Utilities.MathTools;
using Newtonsoft.Json;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

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
