
namespace CallaghanDev.ML
{
    public interface ILogger
    {
        public void WriteLine(string message);
        public void Info(string message, ConsoleColor consoleColor = ConsoleColor.White, bool SendEmail = false);
        public void Warning(string message, bool SendEmail = false);
        public void Error(string message, bool SendEmail = false);
    }
}