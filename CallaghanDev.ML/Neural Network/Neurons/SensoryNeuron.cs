using Newtonsoft.Json;
using System.Diagnostics;

namespace CallaghanDev.ML
{
    [JsonObject(MemberSerialization.OptIn)]
    public class SensoryNeuron : NeuronBase, INeuron
    {
        [JsonProperty]
        private double _MaxValue;

        [JsonProperty]
        private double _MinValue;


        private double _Value;


        [JsonIgnore]
        public new double Bias { get { return 1; } set { throw new WrongNeuronException("Cannot set the Bias function of an Sensory neuron."); } }

        [JsonProperty]
        public int LayerIndex { get; set; }

        [JsonIgnore]
        public new double Activation
        {
            get
            {
                return (_MaxValue - _MinValue) == 0 ? 0 : ((_Value - _MinValue) / (_MaxValue - _MinValue));
            }

            set
            {
                _Value = value;
            }
        }
        [JsonConstructor]
        public SensoryNeuron( ) : base(ActivationType.None)
        {

        }

        public SensoryNeuron(double MinValue, double MaxValue) : base(ActivationType.None)
        {
            if (MaxValue <= MinValue)
            {
               throw new ArgumentOutOfRangeException();
            }
            _MaxValue = MaxValue;
            _MinValue = MinValue;

            SetTrainingDataRange(_MaxValue, _MinValue);
        }

        private void SetTrainingDataRange(double Max, double Min)
        {
            _MaxValue = Max;
            _MinValue = Min;
        }
    }
}
