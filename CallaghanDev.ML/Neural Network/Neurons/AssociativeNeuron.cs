using CallaghanDev.Utilities.MathTools;
using Newtonsoft.Json;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace CallaghanDev.ML
{
    public class AssociativeNeuron : NeuronBase, INeuron
    {

        public int LayerIndex { get; set; }


        public AssociativeNeuron(ActivationType activationType) : base(activationType)
        {
            _Dendrites = new List<Neurite>();
        }

    }
}
