using CallaghanDev.ML.Enums;
using CallaghanDev.ML.TestConsoleApp.Tests;
using ILGPU.Runtime;
using System.Diagnostics;

namespace CallaghanDev.ML.TestConsoleApp
{
    public static class Program
    {
        public static void Main(string[] args)
        {

            var sw = Stopwatch.StartNew();
            TransformerTestSuite.Run(Enums.AccelerationType.MultiThreadCPU, 0);
            sw.Stop();
            Console.WriteLine($"{Enums.AccelerationType.MultiThreadCPU} Tests finished in {sw.Elapsed:hh\\:mm\\:ss\\.fff}.Any key to clear for next test");
            Console.ReadKey();

            sw = Stopwatch.StartNew();
            TransformerTestSuite.Run(Enums.AccelerationType.CPU, 0);
            sw.Stop();
            Console.WriteLine($"{Enums.AccelerationType.CPU} Tests finished in {sw.Elapsed:hh\\:mm\\:ss\\.fff}.Any key to clear for next test");
            Console.ReadKey();


            sw = Stopwatch.StartNew();
            TransformerTestSuite.Run(Enums.AccelerationType.GPU, 0);
            sw.Stop();
            Console.WriteLine($"{Enums.AccelerationType.GPU} Tests finished in {sw.Elapsed:hh\\:mm\\:ss\\.fff}.Any key to clear for next test");
            Console.ReadKey();

            Console.ForegroundColor = ConsoleColor.Cyan;
            Console.WriteLine("  All test suites complete. Press any key to exit.");
            Console.ResetColor();
            Console.ReadKey();


            new AccelerationManagerParityTests().RunAllTests();

            Console.WriteLine("Finished. Any key to clear for next test");
            Console.ReadKey();
        }
    }

    public static class TransformerTestSuite
    {
        public static void Run(AccelerationType accelerationType, int deviceId)
        {
            Console.OutputEncoding = System.Text.Encoding.UTF8;

            PrintBanner("TRANSFORMER TEST SUITE", '=');
            new PriceTacTests(accelerationType, deviceId).RunAllTests();
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