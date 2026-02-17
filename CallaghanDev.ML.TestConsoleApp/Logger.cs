using System;
using System.IO;

namespace CallaghanDev.ML.TestConsoleApp
{
    public class Logger : ILogger
    {
        private readonly string _outputDirectory;

        public Logger(string OutputDirectory = "/LOGS")
        {
            _outputDirectory = OutputDirectory;
        }

        public void Info(string message, ConsoleColor consoleColor = ConsoleColor.White, bool SendEmail = false)
        {
            WriteToConsole("INFO", message, consoleColor);
        }

        public void WriteLine(string message) => Info(message);

        public void Warning(string message, bool SendEmail = false)
        {
            WriteToConsole("WARNING", message, ConsoleColor.Yellow);
        }

        public void Error(string message, bool SendEmail = false)
        {
            WriteToConsole("ERROR", message, ConsoleColor.Red);
            WriteToFile(message);
        }

        private void WriteToFile(string logEntry)
        {
            try
            {
                string logDirectory = _outputDirectory;

                if (!Directory.Exists(logDirectory))
                    Directory.CreateDirectory(logDirectory);

                string logFilePath = Path.Combine(
                    logDirectory,
                    $"log_{DateTime.Now:yyyy-MM-dd}.txt"
                );

                File.AppendAllText(logFilePath, logEntry + Environment.NewLine);
            }
            catch (Exception ex)
            {
                Console.ForegroundColor = ConsoleColor.Red;
                Console.WriteLine($"[ERROR] Failed to write log to file: {ex.Message}");
                Console.ResetColor();
            }
        }

        private void WriteToConsole(string logType, string message, ConsoleColor color)
        {
            var originalColor = Console.ForegroundColor;

            try
            {
                Console.ForegroundColor = color;
                Console.WriteLine($"[{DateTime.Now}] {logType}: {message}");
            }
            finally
            {
                Console.ResetColor();
            }
        }
    }

}
