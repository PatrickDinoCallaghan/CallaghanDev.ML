
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
            TestFNN neuralNetwork_Tests  = new TestFNN();

            TransformerTrainer_Tests transformerTrainer_Tests = new TransformerTrainer_Tests();
            transformerTrainer_Tests.RunAllTests();
            Transformer_Tests transformer_Tests = new Transformer_Tests();
            transformer_Tests.RunAllTests();
            //Doesnt work every time but good progress I think.
            neuralNetwork_Tests.NeuralNetworkXorTestPolynomial();
            neuralNetwork_Tests.NeuralNetworkOrTest();
            neuralNetwork_Tests.NeuralNetworkAndCPUTest();
                neuralNetwork_Tests.NeuralNetworkBatchXorTestCUDA(); 
           // neuralNetwork_Tests.NeuralNetworkAndGPUTest();
            neuralNetwork_Tests.NeuralNetworkXorTestAutoTuneTest();
            Console.ReadLine();
        }

    }
}

