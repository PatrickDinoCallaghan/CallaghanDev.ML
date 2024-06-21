using Newtonsoft.Json;
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
