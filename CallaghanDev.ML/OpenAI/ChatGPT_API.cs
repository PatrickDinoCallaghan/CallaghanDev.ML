using System.Text;
using Newtonsoft.Json;
using System.Net.Http.Headers;

namespace CallaghanDev.ML.OpenAI
{
    public class ChatGPT_API 
    {
        private const int Max_Tokens = 4096;
        private string _API_URL;
        private string _API_KEY;
        private HttpClient _client;

        public ChatGPT_API(string GPTBaseUrl, string APIKey)
        {
            _client = new HttpClient();
            _API_URL = GPTBaseUrl;
            _API_KEY = APIKey;
            _client.BaseAddress = new Uri(_API_URL);
        }
        public ChatGPT_API(string APIKey)
        {
            _client = new HttpClient();
            _API_URL = @"https://api.openai.com/v1/assistants"; 
            _API_KEY = APIKey;
            _client.BaseAddress = new Uri(_API_URL);
        }

        public async Task<OpenAIResponse> Run(string SystemContext, string message, string SessionToken = null)
        {
            return await RunSingleRequest(SystemContext, message) ;
        }

        /// <summary>
        /// Calls the gpt 4 language model. 
        /// </summary>
        /// <param name="SystemContext">Define how you want gpt to respond to the question. Example: "You are a poetic assistant, skilled in explaining complex programming concepts with creative flair."</param>
        /// <param name="message">The question you want a response too. Example: "Compose a poem that explains the concept of recursion in programming." </param>
        /// <returns></returns>
        public async Task<OpenAIResponse> RunSingleRequest(string SystemContext, string message)
        {
            _client.DefaultRequestHeaders.Clear();
            _client.DefaultRequestHeaders.Add("Authorization", $"Bearer {_API_KEY}");
            _client.DefaultRequestHeaders.Add("User-Agent", "ChatGPT Client");
            try
            {
                var requestData = CreateRequestData("gpt-4-1106-preview", Max_Tokens, SystemContext, message);


                var response = await _client.PostAsync("/v1/chat/completions", new StringContent(JsonConvert.SerializeObject(requestData), Encoding.UTF8, "application/json"));
                var responseBody = await response.Content.ReadAsStringAsync();
                Console.WriteLine(responseBody);

                OpenAIResponse rtn = JsonConvert.DeserializeObject<OpenAIResponse>(responseBody);

                if (rtn?.Usage?.TotalTokens >= requestData.max_tokens)
                {
                    Console.WriteLine("Total tokens exeeded limit.");
                }

                return rtn;
            }
            catch (Exception ex)
            {
                HandleException(ex);
                throw ex;  
            }
        }
        private dynamic CreateRequestData(string Model, int In_max_Tokens, string SystemContext, string message, List<string> PreviousMessageResponses = null)
        {
            if (PreviousMessageResponses?.Count > 0)
            {
                message = "Go on...";
                return new
                {
                    model = Model,
                    max_tokens = In_max_Tokens,
                    messages = new[]
                    {
                        new { role = "system", content = SystemContext },
                        new { role = "assistant", content =  string.Join("", PreviousMessageResponses)},
                        new { role = "user", content = message }
                    }
                };
            }
            else
            {

                return new
                {
                    model = Model,
                    max_tokens = In_max_Tokens,
                    messages = new[]
                    {
                        new { role = "system", content = SystemContext },
                        new { role = "user", content = message }
                    }
                };
            }

    
        }
        
      
        public int EstimateTokens(string text)
        {
            // Split by spaces to count words
            int wordCount = text.Split(new[] { ' ', '\n', '\r' }, StringSplitOptions.RemoveEmptyEntries).Length;

            // Count punctuation and special characters as individual tokens
            int punctuationCount = text.Count(char.IsPunctuation);

            return wordCount + punctuationCount;
        }
        public async Task<string> GetUsage( DateTime date)
        {
            // Set up the client with the appropriate headers
            _client.DefaultRequestHeaders.Clear();
            _client.DefaultRequestHeaders.Accept.Add(new MediaTypeWithQualityHeaderValue("application/json"));

            // Add Authorization header
            _client.DefaultRequestHeaders.Authorization = new AuthenticationHeaderValue("Bearer", _API_KEY);

            // Send GET request
            HttpResponseMessage response = await _client.GetAsync($"/v1/usage?date={date.ToString("yyyy-MM-dd")}");

            // Handle the response
            if (response.IsSuccessStatusCode)
            {
                return await response.Content.ReadAsStringAsync();
            }
            else
            {
                throw new HttpRequestException($"Request failed: {response.StatusCode}, {await response.Content.ReadAsStringAsync()}");
            }
        }

        public async Task<string> GetBallance()
        {
            // Set up the client with the appropriate headers
            _client.DefaultRequestHeaders.Clear();
            _client.DefaultRequestHeaders.Accept.Add(new MediaTypeWithQualityHeaderValue("application/json"));

            // Add Basic Authorization header
            // Assuming _USERNAME and _PASSWORD are your basic auth credentials
            var byteArray = Encoding.ASCII.GetBytes($"{"patrick@callaghandev.com"}:{_API_KEY}");
            _client.DefaultRequestHeaders.Authorization = new AuthenticationHeaderValue("Basic", Convert.ToBase64String(byteArray));

            // Send GET request
            HttpResponseMessage response = await _client.GetAsync("/dashboard/billing/credit_grants");

            // Handle the response
            if (response.IsSuccessStatusCode)
            {
                return await response.Content.ReadAsStringAsync();
            }
            else
            {
                throw new HttpRequestException($"Request failed: {response.StatusCode}, {await response.Content.ReadAsStringAsync()}");
            }
        }

        private void HandleException(Exception ex)
        {
            if (ex is HttpRequestException httpRequestException)
            {
                Console.WriteLine($"Request error: {httpRequestException.Message}");
            }
            else if (ex is JsonSerializationException jsonEx)
            {
                Console.WriteLine($"JSON Serialization error: {jsonEx.Message}");
            }
            else if (ex is TaskCanceledException taskCanceledEx)
            {
                Console.WriteLine($"Task canceled: {taskCanceledEx.Message}");
            }
            else if (ex.Message.Contains("rate_limit_exceeded"))
            {
                Console.WriteLine("Rate limit exceeded. Please wait and try again.");
            }
            else if (ex.Message.Contains("model_not_found"))
            {
                Console.WriteLine("The specified model was not found.");
            }
            else if (ex.Message.Contains("payment_required"))
            {
                Console.WriteLine("Payment is required. Please check your OpenAI account.");
            }
            else
            {
                Console.WriteLine($"An unexpected error occurred: {ex.Message}");
            }
        }
    }

}
