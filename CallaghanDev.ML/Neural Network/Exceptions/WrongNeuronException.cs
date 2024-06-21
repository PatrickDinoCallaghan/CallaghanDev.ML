namespace CallaghanDev.ML
{
    public class WrongNeuronException : Exception
    {
        public WrongNeuronException()
            : base("The neuron does not have the expected property and has been used incorrectly.")
        {
        }

        public WrongNeuronException(string message)
            : base($"Error: {message}. The neuron does not support the attempted operation or access to a non-existent property.")
        {
        }

        public WrongNeuronException(string message, Exception inner)
            : base($"Error: {message}. The neuron's usage is improper due to a missing property or incorrect application.", inner)
        {
        }
    }
}
