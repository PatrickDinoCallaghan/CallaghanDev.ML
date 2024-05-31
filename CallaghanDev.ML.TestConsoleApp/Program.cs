namespace CallaghanDev.ML.TestConsoleApp
{
    public class Program
    {
        public static void Main(string[] args)
        {

            // Handle Ctrl+C to close the application gracefully
            Console.CancelKeyPress += (sender, e) =>
            {
                e.Cancel = true; // Prevent the process from terminating.
                Console.WriteLine("\nApplication is closing...");
                Environment.Exit(0);
            };


            TestNN neuralNetwork_Tests = new TestNN();

            neuralNetwork_Tests.NeuralNetworkXorTest();
            neuralNetwork_Tests.NeuralNetworkOrTest();
            neuralNetwork_Tests.NeuralNetworkAndTest();
            neuralNetwork_Tests.NeuralNetworkRegressionTest();


            Console.WriteLine("End");
            Console.ReadLine();
        }

    }
}