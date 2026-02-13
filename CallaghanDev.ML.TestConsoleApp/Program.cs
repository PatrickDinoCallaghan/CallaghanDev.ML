
namespace CallaghanDev.ML.TestConsoleApp
{
    public class Program
    {
        public static void Main()
        {
            Tests tt = new Tests();
            tt.RunAllTests();
            RunNeuralNetworkBooleanLogicTests();
        }
        public static void RunNeuralNetworkBooleanLogicTests()
        {
            NeuralNetworkBooleanLogicTests neuralNetwork_Tests = new NeuralNetworkBooleanLogicTests();

            neuralNetwork_Tests.NeuralNetworkXorTestPolynomial();
            neuralNetwork_Tests.NeuralNetworkOrTest();
            neuralNetwork_Tests.NeuralNetworkAndCPUTest();
            neuralNetwork_Tests.NeuralNetworkBatchXorTestCUDA();

            //neuralNetwork_Tests.NeuralNetworkAndGPUTest();

            neuralNetwork_Tests.NeuralNetworkXorTestAutoTuneTest();

            Console.ReadLine();
        }
    }
}

