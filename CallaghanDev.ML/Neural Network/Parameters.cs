using System.Text;

namespace CallaghanDev.ML.NN
{
    public class Parameters
    {
        private float _HuberLossDelta = 1;
        private AccelerationType _AccelerationType;
        public AccelerationType AccelerationType { get { return _AccelerationType; } set { _AccelerationType = value; } }
        public ActivationType DefaultActivationType { get; set; }
        public CostFunctionType CostFunction { get; set; }
        public ActivationDistribution ActivationDistribution { get; set; }
        public SensoryNeuron[] SensoryNeurons { get; set; }
        public int NoHiddenLayers { get; set; }
        public int HiddenLayerWidth { get; set; }
        public int NumberOfOutputs { get; set; }
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
                if (CostFunction != CostFunctionType.huberLoss)
                {
                    throw new Exception("CostFunction must be huberloss if huberloss deleta is selected");
                }
                else
                {
                    _HuberLossDelta = value;
                }
            }
        }

        public double GradientExplosionThreshold { get; set; } = 1e3;
        public double GradientVanishingThreshold { get; set; } = 1e-5;
        public override string ToString()
        {
            var sb = new StringBuilder();
            sb.AppendLine("NNParameters:");
            sb.AppendLine($"  AccelerationType: {AccelerationType}");
            sb.AppendLine($"  SensoryNeurons: {SensoryNeurons.Length} neurons");
            sb.AppendLine($"  NoHiddenLayers: {NoHiddenLayers}");
            sb.AppendLine($"  HiddenLayerWidth: {HiddenLayerWidth}");
            sb.AppendLine($"  NumberOfOutputs: {NumberOfOutputs}");
            sb.AppendLine($"  DefaultActivationType: {DefaultActivationType}");
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
