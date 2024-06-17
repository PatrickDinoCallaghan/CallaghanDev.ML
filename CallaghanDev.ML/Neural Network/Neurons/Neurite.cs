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

        [JsonProperty]
        public double Weight { get; set; }
        #endregion

        public Neurite(double InitialWeight)
        {
            Weight = InitialWeight;
        }
        public Neurite()
        {

        }
    }
}
