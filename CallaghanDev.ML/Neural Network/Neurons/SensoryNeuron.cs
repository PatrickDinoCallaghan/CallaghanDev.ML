using CallaghanDev.Utilities.MathTools;
using Newtonsoft.Json;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace CallaghanDev.ML
{
    public class SensoryNeuron : NeuronBase, INeuron
    {
        private double _MaxValue;
        private double _MinValue;

        private double _Value;

        [JsonIgnore]
        public new List<Neurite> Dendrites { get { throw new WrongNeuronException("Sensory Neuron contains no Dendrites as it has no source neurons"); } set { } }

        public new double Bias { get { return 1; } set { throw new WrongNeuronException("Cannot set the Bias function of an Sensory neuron."); } }

        public int LayerIndex { get; set; }
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
