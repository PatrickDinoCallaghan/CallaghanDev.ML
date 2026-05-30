using CallaghanDev.ML.Enums;
using CallaghanDev.ML.TestConsoleApp.Tests;
using ILGPU.Runtime;

namespace CallaghanDev.ML.TestConsoleApp
{
    public static class Program
    {
        public static void Main(string[] args)
        {
            new AccelerationManagerParityTests().RunAllTests();

            TransformerTestSuite.Run(Enums.AccelerationType.MultiThreadCPU, 0);
            TransformerTestSuite.Run(Enums.AccelerationType.CPU, 0);
            TransformerTestSuite.Run(Enums.AccelerationType.GPU, 0);

            Console.ReadKey();

            Console.WriteLine();

            foreach (var accelerationType in Enum.GetValues<Enums.AccelerationType>())
            {
                Console.WriteLine(accelerationType);
                TransformerTestSuite.Run(accelerationType, 0);
            }

            Console.WriteLine();
            Console.ForegroundColor = ConsoleColor.Cyan;
            Console.WriteLine("  All test suites complete. Press any key to exit.");
            Console.ResetColor();
            Console.ReadKey();
        }
    }

    public static class TransformerTestSuite
    {
        public static void Run(AccelerationType accelerationType, int deviceId)
        {
            Console.OutputEncoding = System.Text.Encoding.UTF8;

            PrintBanner("TRANSFORMER TEST SUITE", '=');

            new MmtacTests(accelerationType, deviceId).RunAllTests();
            new BPETokenizerTests(accelerationType).RunAllTests();
            new RotaryPositionEmbeddingTests(accelerationType).RunAllTests();
            new CrossAttentionMultimodalTests(accelerationType).RunAllTests();
            new TacmatTests(accelerationType).RunAllTests();
            new MultiTypeTransformerTests(accelerationType).RunAllTests();
            new NeuralNetworkTests(accelerationType).RunAllTests();
        }

        internal static void PrintBanner(string title, char ch = '-')
        {
            string line = new string(ch, 70);
            Console.WriteLine();
            Console.ForegroundColor = ConsoleColor.Cyan;
            Console.WriteLine($"  {line}");
            Console.WriteLine($"  {title}");
            Console.WriteLine($"  {line}");
            Console.ResetColor();
        }
    }
}