using System.Text;
using Newtonsoft.Json;
using System.Net.Http.Headers;

namespace CallaghanDev.ML.OpenAI
{
    public class GPTClient
    {
        private readonly string _apiKey;
        private readonly HttpClient _client;
        public readonly OpenAIModel Model;

        public string ModelName => Model.ToTagName();
        public int MaxTokens => Model.GetMaxTokens();

        public GPTClient(string apiKey, OpenAIModel model, string baseUrl = "https://api.openai.com")
        {
            _apiKey = apiKey ?? throw new ArgumentNullException(nameof(apiKey));
            Model = model;
            _client = new HttpClient { BaseAddress = new Uri(baseUrl) };
            _client.Timeout = TimeSpan.FromSeconds(60);
        }

        public async Task<string> Run(string systemContext, string message) => await RunSingleRequest(systemContext, message);

        /// <summary>
        /// Make a single request to OpenAI's /v1/chat/completions endpoint.
        /// </summary>
        /// <param name="systemContext">System role context (behavior hint for assistant).</param>
        /// <param name="userMessage">The user's message.</param>
        public async Task<string> RunSingleRequest(string systemContext, string userMessage)
        {
            var requestData = new
            {
                model = Model.ToTagName(),
                max_tokens = MaxTokens,
                messages = new[]
                {
                new { role = "system", content = systemContext },
                new { role = "user", content = userMessage }
            }
            };

            var requestJson = JsonConvert.SerializeObject(requestData);
            var requestContent = new StringContent(requestJson, Encoding.UTF8, "application/json");

            _client.DefaultRequestHeaders.Clear();
            _client.DefaultRequestHeaders.Authorization = new AuthenticationHeaderValue("Bearer", _apiKey);

            var response = await _client.PostAsync("/v1/chat/completions", requestContent);

            if (!response.IsSuccessStatusCode)
            {
                var errorBody = await response.Content.ReadAsStringAsync();
                throw new Exception($"API Error {response.StatusCode}: {errorBody}");
            }
            else
            {

                var responseBody = await response.Content.ReadAsStringAsync();
                var openAIResponse = JsonConvert.DeserializeObject<OpenAIChatResponse>(responseBody);

                // You can modify to return the full object if needed
                return openAIResponse?.choices?[0]?.message?.content ?? "(no reply)";
            }
        }

        /// <summary>
        /// Print the info for this client/model.
        /// </summary>
        public void PrintModelInfo()
        {
            Console.WriteLine($"Model: {Model.ToTagName()}");
            Console.WriteLine($"Max tokens: {Model.GetMaxTokens()}");
            Console.WriteLine($"Input price: {Model.GetInputPrice()}");
            Console.WriteLine($"Cached input price: {Model.GetCachedInputPrice()}");
            Console.WriteLine($"Output price: {Model.GetOutputPrice()}");
        }
    }

}
