
namespace CallaghanDev.ML
{
    public struct NNInputSettings
    {
        private float _MaxValue;
        private float _MinValue;
        public NNInputSettings(float InMaxValue, float InMinValue)
        {
            _MinValue = InMinValue;
            _MaxValue = InMaxValue;
        }

        public float MaxValue { get { return _MaxValue; } }
        public float MinValue { get { return _MinValue; } }
    }
}
