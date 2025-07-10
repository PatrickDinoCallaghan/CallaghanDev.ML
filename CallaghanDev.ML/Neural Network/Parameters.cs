using CallaghanDev.ML.Enums;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace CallaghanDev.ML
{
    public class Parameters
    {
        private float _HuberLossDelta = 1;

        public AccelerationType AccelerationType { get; set; }
        public CostFunctionType CostFunction { get; set; }
        public ActivationDistribution ActivationDistribution { get; set; }

        public double L2RegulationLamda { get; set; } = 0;
        public float GradientClippingThreshold { get; set; } = 1;
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

        public double GradientExplosionThreshold { get; set; } = 1e3;
        public double GradientVanishingThreshold { get; set; } = 1e-5;

        public List<int> LayerWidths { get; set; } = new List<int>();
        public List<ActivationType> LayerActivations { get; set; } = new List<ActivationType>();

        internal double[] inputActivationMin { get; set; }
        internal double[] inputActivationMax { get; set; }

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
    }

}
