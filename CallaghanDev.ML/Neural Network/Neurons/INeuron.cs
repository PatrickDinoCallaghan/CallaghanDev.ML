namespace CallaghanDev.ML
{
    public interface INeuron : IDisposable
    {
        double Bias { get; set; }

        double Activation { get; set; }


        public delegate double ActivationFunction(double x);
        public INeuron.ActivationFunction activationFunction { get; set; }

        public INeuron.ActivationFunction activationFunctionDeriv { get; set; }

        public double Delta { get; set; }

        public int LayerIndex { get; set; }
    }
}
