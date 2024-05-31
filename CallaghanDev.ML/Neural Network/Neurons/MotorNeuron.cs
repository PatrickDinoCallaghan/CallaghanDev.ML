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
    public class MotorNeuron : NeuronBase, INeuron
    {


        public int LayerIndex { get; set; }

        public MotorNeuron(ActivationType activationType) : base(activationType)
        {

            _Dendrites = new List<Neurite>();
            this.Bias = 0;
        }
    }
}
