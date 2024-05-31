using Newtonsoft.Json;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace CallaghanDev.ML
{
    public class Neurite
    {
        #region Property
        [JsonIgnore]
        private INeuron _SourceNeuron;
        [JsonIgnore]
        public INeuron SourceNeuron
        {
            get
            {
                return _SourceNeuron;
            }
        }
        [JsonProperty]
        public double Weight { get; set; }
        #endregion

        public Neurite(INeuron SourceNeuron, double InitialWeight)
        {
            _SourceNeuron = SourceNeuron;
            Weight = InitialWeight;
        }

    }
}
