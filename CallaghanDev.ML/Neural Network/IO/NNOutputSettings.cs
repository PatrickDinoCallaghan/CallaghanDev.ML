namespace CallaghanDev.ML
{
    public struct NNOutputSettings
    {
        private string _Label;
        public NNOutputSettings(string InLabel)
        {
            _Label = InLabel;
        }
        public string Label { get { return _Label; } }
    }
}
