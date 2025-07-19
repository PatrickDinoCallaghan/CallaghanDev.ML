
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

            neuralNetwork_Tests.NeuralNetworkAndGPUTest();
            neuralNetwork_Tests.NeuralNetworkBatchXorTestCUDA();
            neuralNetwork_Tests.NeuralNetworkXorTest();
            neuralNetwork_Tests.NeuralNetworkOrTest();
            neuralNetwork_Tests.NeuralNetworkAndCPUTest();
            Console.ReadLine();
        }

    }
}

