﻿using System.Diagnostics;
using System.Text;

namespace CallaghanDev.ML.NN
{
    public class Parameters
    {
        private float _HuberLossDelta = 1;

        private AccelerationType _AccelerationType;

        public AccelerationType AccelerationType { get { return _AccelerationType; } set { _AccelerationType = value; Debug.WriteLine(value.ToString()); } }
        public ActivationType DefaultActivationType { get; set; }
        public CostFunctionType CostFunction { get; set; }
        public SensoryNeuron[] SensoryNeurons { get; set; }
        public int NoHiddenLayers { get; set; }
        public int HiddenLayerWidth { get; set; }
        public int NumberOfOutputs { get; set; }
        public double L2RegulationLamda { get; set; } = 0;
        public float ClippingLimitUpper { get; set; } = float.MaxValue;
        public float ClippingLimitLower { get; set; } = float.MinValue;
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
            sb.AppendLine($"  ClippingLimitUpper: {ClippingLimitUpper}");
            sb.AppendLine($"  ClippingLimitLower: {ClippingLimitLower}");
            if (CostFunction == CostFunctionType.huberLoss)
            {
                sb.AppendLine($"  HuberLossDelta: {HuberLossDelta}");
            }
            return sb.ToString();
        }
    }
}
