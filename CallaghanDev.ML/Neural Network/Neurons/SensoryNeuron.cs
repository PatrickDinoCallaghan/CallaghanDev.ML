using CallaghanDev.Utilities.MathTools;
using Newtonsoft.Json;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

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

        public SensoryNeuron(double MaxValue, double MinValue) : base(ActivationType.None)
        {
            if (_MaxValue < _MinValue)
            {
                throw new Exception();
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
