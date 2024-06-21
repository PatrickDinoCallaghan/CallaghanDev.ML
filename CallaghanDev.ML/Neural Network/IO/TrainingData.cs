
namespace CallaghanDev.ML
{
    public class TrainingData
    {

        private float[] _InputData;
        public float[] InputData { get { return _InputData; } }

        private float[] _OutputData;
        public float[] OutputData { get { return _OutputData; } }

        public TrainingData(float[] inputData, float[] outputData)
        {
            _InputData = inputData;
            _OutputData = outputData;
        }
    }
}
