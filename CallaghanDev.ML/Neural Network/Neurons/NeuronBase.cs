using CallaghanDev.Utilities.MathTools;
using Newtonsoft.Json;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace CallaghanDev.ML
{
    public class NeuronBase
    {
        #region Activation Function Methods


        [JsonIgnore]
        public INeuron.ActivationFunction _activationFunction;
        [JsonIgnore]
        public INeuron.ActivationFunction _activationFunctionDeriv;

        public NeuronBase(ActivationType activationType) 
        {

            this.activationType = activationType; 
            this.activationFunction = GetActivationFunction(activationType);
            activationFunctionDeriv = GetActivationFunctionDeriv(activationType);
        }

        [JsonIgnore]
        public INeuron.ActivationFunction activationFunction 
        {
            get
            {
                if (_activationFunction == null)
                {
                    _activationFunction = GetActivationFunction(activationType);
                }

                return _activationFunction;
            }
            set
            {
                _activationFunction = value;
            }
        }
        [JsonIgnore]
        public INeuron.ActivationFunction activationFunctionDeriv
        {
            get 
            {
                if (_activationFunctionDeriv == null)
                {
                    _activationFunctionDeriv = GetActivationFunctionDeriv(activationType);
                }

                return _activationFunctionDeriv;
            }
            set
            {
                _activationFunctionDeriv = value;
            }
        
        }
        internal INeuron.ActivationFunction GetActivationFunction(ActivationType type)
        {
            switch (type)
            {
                case ActivationType.Sigmoid:
                    return x => Function.sigmoid(x);
                case ActivationType.Tanh:
                    return x => Function.tanh(x);
                case ActivationType.Relu:
                    return x => Function.relu(x);
                case ActivationType.Leakyrelu:
                    return x => Function.leakyrelu(x);
                default:
                    return x => Function.sigmoid(x);
            }
        }
        [JsonProperty]
        public ActivationType activationType { get; set; }
        internal INeuron.ActivationFunction GetActivationFunctionDeriv(ActivationType type)
        {
            switch (type)
            {
                case ActivationType.Sigmoid:
                    return x => Function.FirstDerivative.sigmoid(x);
                case ActivationType.Tanh:
                    return x => Function.FirstDerivative.tanh(x);
                case ActivationType.Relu:
                    return x => Function.FirstDerivative.relu(x);
                case ActivationType.Leakyrelu:
                    return x => Function.FirstDerivative.leakyrelu(x);
                default:
                    return x => Function.FirstDerivative.sigmoid(x);
            }
        }

        #endregion


        
        public double Activation { get; set; }
        public void Dispose()
        {
            throw new NotImplementedException();
        }

        [JsonProperty]
        public double Bias { get; set; }

        //Dendrites receive signals from the other neurons to the cell body
        public List<Neurite> Dendrites
        {
            get
            {
                return _Dendrites;
            }
        }

        [JsonProperty]
        internal List<Neurite> _Dendrites;

        public double Delta { get; set; }

    }
}
