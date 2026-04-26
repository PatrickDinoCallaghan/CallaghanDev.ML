using CallaghanDev.ML.TestConsoleApp.Tests;

namespace CallaghanDev.ML.TestConsoleApp
{
    public static class Program
    {
        public static void Main(string[] args)
        {
            TransformerTestSuite.Run();
        }
    }

    public static class TransformerTestSuite
    {
        public static void Run()
        {
            Console.OutputEncoding = System.Text.Encoding.UTF8;

            PrintBanner("TRANSFORMER TEST SUITE", '=');

            new RotaryPositionEmbeddingTests().RunAllTests();
            new BPETokenizerTests().RunAllTests();
            new MmtacTests().RunAllTests();
            new CrossAttentionMultimodalTests().RunAllTests();
            new TacmatTests().RunAllTests();
            new MultiTypeTransformerTests().RunAllTests();

            Console.WriteLine();
            Console.ForegroundColor = ConsoleColor.Cyan;
            Console.WriteLine("  All test suites complete. Press any key to exit.");
            Console.ResetColor();
            Console.ReadKey();
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