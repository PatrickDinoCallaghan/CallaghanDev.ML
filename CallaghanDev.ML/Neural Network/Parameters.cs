using CallaghanDev.ML.Enums;
using Newtonsoft.Json;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace CallaghanDev.ML
{
    public class Parameters
    {
        public Parameters() { }

        [JsonProperty("HuberLossDelta")]
        private float _HuberLossDelta = 1;

        public AccelerationType AccelerationType { get; set; }
        public int AccelerationDeviceId { get; set; } = 0;

        public CostFunctionType CostFunction { get; set; }
        public ActivationDistribution ActivationDistribution { get; set; }

        public float L2RegulationLamda { get; set; } = 0;
        public float GradientClippingThreshold { get; set; } = 1;
        
        [JsonIgnore]
        public float HuberLossDelta
        {
            get
            {
                return _HuberLossDelta;
            }
            set
            {
                if (CostFunction == CostFunctionType.huberLoss)
                {

                    _HuberLossDelta = value;
                }
            }
        }

        public float GradientExplosionThreshold { get; set; } = 1e3f;
        public float GradientVanishingThreshold { get; set; } = 1e-5f;

        public List<int> LayerWidths { get; set; } = new List<int>();
        public List<ActivationType> LayerActivations { get; set; } = new List<ActivationType>();

        [JsonProperty]
        internal float[] inputActivationMin { get; set; }

        [JsonProperty]
        internal float[] inputActivationMax { get; set; }

        public float? OptimalLearningRate { get; set; } =null;
        public int EpochsTrainedOn { get; set; } = 10;

        public override string ToString()
        {
            var sb = new StringBuilder();
            sb.AppendLine("Parameters:");
            sb.AppendLine($"  AccelerationType: {AccelerationType}");
            sb.AppendLine($"  ActivationType: {AccelerationType}");
            sb.AppendLine($"  CostFunction: {CostFunction}");
            sb.AppendLine($"  L2RegulationLamda: {L2RegulationLamda}");
            sb.AppendLine($"  GradientClippingThreshold: {GradientClippingThreshold}");
            if (CostFunction == CostFunctionType.huberLoss)
            {
                sb.AppendLine($"  HuberLossDelta: {HuberLossDelta}");
            }
            return sb.ToString();
        }
        public Parameters Clone()
        {
            return new Parameters
            {
                AccelerationType = this.AccelerationType,
                AccelerationDeviceId = this.AccelerationDeviceId,
                CostFunction = this.CostFunction,
                ActivationDistribution = this.ActivationDistribution,
                L2RegulationLamda = this.L2RegulationLamda,
                GradientClippingThreshold = this.GradientClippingThreshold,
                GradientExplosionThreshold = this.GradientExplosionThreshold,
                GradientVanishingThreshold = this.GradientVanishingThreshold,
                LayerWidths = new List<int>(this.LayerWidths),
                LayerActivations = new List<ActivationType>(this.LayerActivations),
                inputActivationMin = this.inputActivationMin?.ToArray(),
                inputActivationMax = this.inputActivationMax?.ToArray(),
                HuberLossDelta = this.HuberLossDelta
            };
        }

        /// <summary>
        /// Saves the current instance to a JSON file.
        /// </summary>
        public void SaveToFile(string filePath)
        {
            var json = JsonConvert.SerializeObject(this, Formatting.Indented);
            File.WriteAllText(filePath, json);
        }

        /// <summary>
        /// Loads a Parameters instance from a JSON file.
        /// </summary>
        public static Parameters LoadFromFile(string filePath)
        {
            if (!File.Exists(filePath))
                throw new FileNotFoundException("Parameters file not found.", filePath);

            var json = File.ReadAllText(filePath);
            return JsonConvert.DeserializeObject<Parameters>(json);
        }
    }

}
