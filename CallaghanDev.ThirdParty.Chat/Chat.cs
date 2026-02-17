using LLama.Common;
using LLama;
using LLama.Sampling;
using System.Text;

namespace CallaghanDev.ML.LLMChat
{
    // Can be found here https://huggingface.co/itlwas/Mistral-7B-Instruct-v0.1-Q4_K_M-GGUF
    public class Chat : IDisposable
    {
        private readonly LLamaWeights _model;
        private readonly object _context;
        private readonly InteractiveExecutor _executor;
        private readonly ChatSession _session;
        private readonly InferenceParams _inferenceParams;

        public Chat(bool useGpuCompute, string modelPath, string SystemPrompt, uint ContextSize = 2048, InferenceParams customInferenceParams = null)
        {
            // Toggle GPU vs CPU.
            // For this example, if using GPU we assume offloading 32 layers, otherwise 0 layers.
            int gpuLayerCount = useGpuCompute ? 32 : 0;
            var parameters = new ModelParams(modelPath)
            {
                ContextSize = ContextSize,              // Maximum context length.
                GpuLayerCount = gpuLayerCount,          // Offload layers if GPU enabled.
                MainGpu = useGpuCompute ? 0 : -1,       // Use GPU index 0 if enabled, else CPU-only.
                SplitMode = null,                       // No GPU splitting.
            };

            // Load model and create the context.
            _model = LLamaWeights.LoadFromFile(parameters);
            _context = _model.CreateContext(parameters);
            _executor = new InteractiveExecutor((LLamaContext)_context);

            // Set up the initial chat history.
            var chatHistory = new ChatHistory();
            chatHistory.AddMessage(AuthorRole.System, SystemPrompt);
            _session = new ChatSession(_executor, chatHistory);

            // Set up inference parameters.
            _inferenceParams = customInferenceParams ?? new InferenceParams()
            {
                MaxTokens = 2048,
                AntiPrompts = new List<string> { "User:" },
                SamplingPipeline = new DefaultSamplingPipeline()
                {
                    Temperature = 0.3f,
                    TopP = 0.9f,
                },
            };
        }
        public static int CalculateContextSize(string systemPrompt, IEnumerable<string> expectedInputs, int maxOutputTokens = 512)
        {
            int systemPromptTokens = systemPrompt.Length / 4;

            int maxInputTokens = 0;
            if (expectedInputs != null && expectedInputs.Any())
            {
                maxInputTokens = expectedInputs.Max(s => s.Length) / 4;
            }

            // Calculate required tokens with safety buffer
            int requiredTokens = systemPromptTokens + maxInputTokens + maxOutputTokens + 512;

            int contextSize = RoundToCommonContextSize(requiredTokens);

            Console.WriteLine($"Calculated context size: {contextSize} tokens");
            Console.WriteLine($"  System prompt: ~{systemPromptTokens} tokens");
            Console.WriteLine($"  Max input: ~{maxInputTokens} tokens");
            Console.WriteLine($"  Max output: {maxOutputTokens} tokens");
            Console.WriteLine($"  Safety buffer: 512 tokens");

            return contextSize;
        }

        private  static int RoundToCommonContextSize(int requiredTokens)
        {
            int[] commonSizes = { 512, 1024, 2048, 4096, 8192, 16384, 32768 };

            foreach (int size in commonSizes)
            {
                if (size >= requiredTokens)
                    return size;
            }

            return ((requiredTokens / 1024) + 1) * 1024;
        }
        public async IAsyncEnumerable<string> ChatAsync(string userInput)
        {
            var message = new ChatHistory.Message(AuthorRole.User, userInput);
            await foreach (var text in _session.ChatAsync(message, _inferenceParams))
            {
                yield return text;
            }
        }

        public async Task<string> GetResponse(string Input)
        {
            StringBuilder sb = new StringBuilder();

            await foreach (var text in ChatAsync(Input))
            {
                Console.ForegroundColor = ConsoleColor.White;
                sb.Append(text);
            }

            return sb.ToString();
        }
        public async Task PrintResponseToConsole(string Input)
        {
            string userInput = Console.ReadLine() ?? "";

            while (userInput != "exit")
            {
                await foreach (var text in ChatAsync(userInput))
                {
                    Console.ForegroundColor = ConsoleColor.White;
                    Console.Write(text);
                }
                Console.ForegroundColor = ConsoleColor.Green;
                userInput = Console.ReadLine() ?? "";
            }
        }
        public void Dispose()
        {
            if (_context is IDisposable disposableContext)
            {
                disposableContext.Dispose();
            }
            _model?.Dispose();
        }
    }
}
