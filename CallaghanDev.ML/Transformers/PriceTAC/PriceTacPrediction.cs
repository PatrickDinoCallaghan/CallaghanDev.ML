namespace CallaghanDev.ML.Transformers.PriceTAC
{
    public sealed class PriceTacPrediction
    {
        public float High { get; set; }
        public float Low { get; set; }
        public float Close { get; set; }
        public float Range { get; set; }
        public float Quality { get; set; }
        public float DirectionProb { get; set; }
        public float MidWindowDirectionProb { get; set; }
        public float Confidence { get; set; } = 1f;
    }
}
