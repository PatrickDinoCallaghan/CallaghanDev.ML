
namespace CallaghanDev.ML.TestConsoleApp
{
    public class Program
    {
        public static void Main()
        {
            NNtest();
        }
        public static void NNtest()
        {
            TestNN neuralNetwork_Tests  = new TestNN();

            //Doesnt work every time but good progress I think.
            neuralNetwork_Tests.NeuralNetworkXorTestPolynomial();
            neuralNetwork_Tests.NeuralNetworkOrTest();
            neuralNetwork_Tests.NeuralNetworkAndCPUTest();
            neuralNetwork_Tests.NeuralNetworkBatchXorTestCUDA();
            neuralNetwork_Tests.NeuralNetworkAndGPUTest();
            neuralNetwork_Tests.NeuralNetworkXorTestAutoTuneTest();
            Console.ReadLine();
        }

    }
}

