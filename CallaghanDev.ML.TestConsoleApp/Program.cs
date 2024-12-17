
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

            // Handle Ctrl+C to close the application gracefully
            Console.CancelKeyPress += (sender, e) =>
            {
                e.Cancel = true; // Prevent the process from terminating.
                Console.WriteLine("\nApplication is closing...");
                Environment.Exit(0);
            };


            TestNN neuralNetwork_Tests  = new TestNN();

            //TestNN.Run();
            for (int i = 0; i < 100; i++)
            {

                TestNN neuralNetwork_Tests2 = new TestNN();
                neuralNetwork_Tests2.NeuralNetworkXorTest();
            }
            neuralNetwork_Tests.NeuralNetworkXorTest();
            neuralNetwork_Tests.NeuralNetworkXorTest();
            neuralNetwork_Tests.NeuralNetworkOrTest();
            neuralNetwork_Tests.NeuralNetworkAndCPUTest();
            neuralNetwork_Tests.NeuralNetworkRegressionTest();


            Console.ReadLine();
        }

    }
}

