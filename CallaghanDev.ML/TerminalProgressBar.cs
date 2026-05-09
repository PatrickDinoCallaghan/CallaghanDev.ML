using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Text;

namespace CallaghanDev.ML
{
    /// <summary>
    /// Small dependency-free terminal progress renderer for long-running console tasks.
    /// It falls back to start/finish lines when stdout is redirected.
    /// </summary>
    public class TerminalProgressBar : IDisposable
    {
        private static readonly object ConsoleGate = new object();
        private static readonly string[] SpinnerFrames = { "⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏" };

        private readonly string _title;
        private readonly long? _total;
        private readonly bool _enabled;
        private readonly Stopwatch _stopwatch = Stopwatch.StartNew();
        private readonly TimeSpan _renderThrottle;
        private readonly object _stateGate = new object();

        private Timer? _timer;
        private DateTime _lastRenderUtc = DateTime.MinValue;
        private long _current;
        private string _message = string.Empty;
        private int _spinnerIndex;
        private bool _completed;
        private bool _disposed;

        public TerminalProgressBar(string title, long? total = null, TimeSpan? renderThrottle = null)
        {
            _title = string.IsNullOrWhiteSpace(title) ? "Progress" : title.Trim();
            _total = total.HasValue && total.Value > 0 ? total.Value : null;
            _renderThrottle = renderThrottle ?? TimeSpan.FromMilliseconds(125);
            _enabled = !Console.IsOutputRedirected;

            if (_enabled)
            {
                Render(force: true);

                if (!_total.HasValue)
                {
                    _timer = new Timer(_ => Render(force: true), null, _renderThrottle, _renderThrottle);
                }
            }
            else
            {
                Console.WriteLine($"{_title} started...");
            }
        }

        public static TerminalProgressBar StartIndeterminate(string title, string? message = null)
        {
            var progress = new TerminalProgressBar(title);
            if (!string.IsNullOrWhiteSpace(message))
            {
                progress.ReportMessage(message);
            }

            return progress;
        }

        public void Increment(long amount = 1, string? message = null)
        {
            if (amount < 0)
            {
                amount = 0;
            }

            lock (_stateGate)
            {
                if (_completed || _disposed)
                {
                    return;
                }

                _current += amount;
                if (_total.HasValue && _current > _total.Value)
                {
                    _current = _total.Value;
                }

                if (!string.IsNullOrWhiteSpace(message))
                {
                    _message = message.Trim();
                }
            }

            Render(force: false);
        }

        public void Report(long current, string? message = null)
        {
            lock (_stateGate)
            {
                if (_completed || _disposed)
                {
                    return;
                }

                _current = Math.Max(0, current);
                if (_total.HasValue && _current > _total.Value)
                {
                    _current = _total.Value;
                }

                if (!string.IsNullOrWhiteSpace(message))
                {
                    _message = message.Trim();
                }
            }

            Render(force: false);
        }

        public void ReportMessage(string message)
        {
            lock (_stateGate)
            {
                if (_completed || _disposed)
                {
                    return;
                }

                _message = message?.Trim() ?? string.Empty;
            }

            Render(force: false);
        }

        public void Complete(string? message = null)
        {
            Timer? timerToDispose = null;

            lock (_stateGate)
            {
                if (_completed)
                {
                    return;
                }

                if (_total.HasValue)
                {
                    _current = _total.Value;
                }

                if (!string.IsNullOrWhiteSpace(message))
                {
                    _message = message.Trim();
                }

                _completed = true;
                timerToDispose = _timer;
                _timer = null;
            }

            timerToDispose?.Dispose();

            if (_enabled)
            {
                lock (ConsoleGate)
                {
                    ClearCurrentLine();
                    Console.ForegroundColor = ConsoleColor.Green;
                    Console.Write("done. ");
                    Console.ResetColor();
                    Console.WriteLine(BuildCompletionLine());
                }
            }
            else
            {
                Console.WriteLine(BuildCompletionLine());
            }
        }

        public void Dispose()
        {
            if (_disposed)
            {
                return;
            }

            _disposed = true;
            Complete();
        }

        private void Render(bool force)
        {
            if (!_enabled)
            {
                return;
            }

            DateTime now = DateTime.UtcNow;
            if (!force && now - _lastRenderUtc < _renderThrottle)
            {
                return;
            }

            _lastRenderUtc = now;

            string line;
            lock (_stateGate)
            {
                if (_completed || _disposed)
                {
                    return;
                }

                line = BuildRenderLine();
            }

            lock (ConsoleGate)
            {
                ClearCurrentLine();
                Console.Write(TrimToConsoleWidth(line));
            }
        }

        private string BuildRenderLine()
        {
            string elapsed = FormatElapsed(_stopwatch.Elapsed);
            string suffix = string.IsNullOrWhiteSpace(_message) ? string.Empty : $" | {_message}";

            if (_total.HasValue)
            {
                double ratio = _total.Value <= 0 ? 0.0 : Math.Clamp((double)_current / _total.Value, 0.0, 1.0);
                int consoleWidth = 96;
                try
                {
                    consoleWidth = Math.Max(1, Console.WindowWidth);
                }
                catch
                {
                    // Keep rendering safe on unusual terminal hosts.
                }

                int width = Math.Max(10, Math.Min(34, consoleWidth / 4));
                int filled = (int)Math.Round(width * ratio);
                string bar = new string('█', filled) + new string('░', width - filled);
                return $"{_title} [{bar}] {ratio * 100,6:0.0}% {_current:N0}/{_total.Value:N0} | {elapsed}{suffix}";
            }

            string frame = SpinnerFrames[_spinnerIndex++ % SpinnerFrames.Length];
            return $"{frame} {_title} | {elapsed}{suffix}";
        }

        private string BuildCompletionLine()
        {
            string suffix = string.IsNullOrWhiteSpace(_message) ? string.Empty : $" | {_message}";

            if (_total.HasValue)
            {
                return $"{_title} complete: {_current:N0}/{_total.Value:N0} in {FormatElapsed(_stopwatch.Elapsed)}{suffix}";
            }

            return $"{_title} complete in {FormatElapsed(_stopwatch.Elapsed)}{suffix}";
        }

        private static string FormatElapsed(TimeSpan elapsed)
        {
            if (elapsed.TotalHours >= 1)
            {
                return elapsed.ToString(@"hh\:mm\:ss");
            }

            return elapsed.ToString(@"mm\:ss");
        }

        private static void ClearCurrentLine()
        {
            int width;
            try
            {
                width = Math.Max(1, Console.WindowWidth);
            }
            catch
            {
                width = 120;
            }

            Console.Write('\r');
            Console.Write(new string(' ', Math.Max(0, width - 1)));
            Console.Write('\r');
        }

        private static string TrimToConsoleWidth(string value)
        {
            int width;
            try
            {
                width = Math.Max(20, Console.WindowWidth - 1);
            }
            catch
            {
                width = 120;
            }

            if (value.Length <= width)
            {
                return value;
            }

            return value.Substring(0, Math.Max(0, width - 1));
        }
    }
}