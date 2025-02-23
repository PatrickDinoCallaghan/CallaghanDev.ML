using LLama.Common;
using LLama;
using LLama.Sampling;
using System.Text;

namespace CallaghanDev.ML.LLMChat
{
    //This is programmed to use mistral-7b-instruct-v0.1.Q4_K_M.gguf
    // Can be found here https://huggingface.co/itlwas/Mistral-7B-Instruct-v0.1-Q4_K_M-GGUF
    public class Chat : IDisposable
    {
        private readonly LLamaWeights _model;
        private readonly object _context;
        private readonly InteractiveExecutor _executor;
        private readonly ChatSession _session;
        private readonly InferenceParams _inferenceParams;

        public Chat(bool useGpuCompute, string modelPath, string SystemPrompt)
        {

            // Toggle GPU vs CPU.
            // For this example, if using GPU we assume offloading 32 layers, otherwise 0 layers.
            int gpuLayerCount = useGpuCompute ? 32 : 0;
            var parameters = new ModelParams(modelPath)
            {
                ContextSize = 2048,               // Maximum context length.
                GpuLayerCount = gpuLayerCount,      // Offload layers if GPU enabled.
                MainGpu = useGpuCompute ? 0 : -1,     // Use GPU index 0 if enabled, else CPU-only.
                SplitMode = null,                 // No GPU splitting.
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
            _inferenceParams = new InferenceParams()
            {
                MaxTokens = 2048,  // Maximum output tokens.
                AntiPrompts = new List<string> { "User:" },
                SamplingPipeline = new DefaultSamplingPipeline(),
            };

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
