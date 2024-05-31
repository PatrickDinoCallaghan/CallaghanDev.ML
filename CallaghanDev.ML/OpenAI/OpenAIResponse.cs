using Newtonsoft.Json;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace CallaghanDev.ML.OpenAI
{
    public class OpenAIResponse
    {
        public string Id { get; set; }
        [JsonProperty("object")]
        public string ObjectType { get; set; }
        public long Created { get; set; }
        public string Model { get; set; }
        public Choice[] Choices { get; set; }
        public Usage Usage { get; set; }
        public string ToStringLong()
        {
            var choicesString = Choices != null ? string.Join(", ", Choices.Select(c => c.ToString())) : "null";
            var usageString = Usage != null ? Usage.ToString() : "null";

            return $"Id: {Id}, ObjectType: {ObjectType}, Created: {Created}, Model: {Model}, Choices: [{choicesString}], Usage: {usageString}";
        }
        public override string ToString()
        {
            return GetAllChoiceMessages();
        }
        private string GetAllChoiceMessages()
        {
            var messageContents = new StringBuilder();
            foreach (var choice in Choices)
            {
                if (choice != null && choice.Message != null)
                {
                    messageContents.AppendLine(choice.Message.Content);
                }
            }

            return messageContents.ToString();
        }
    }

    public class Choice
    {
        public int Index { get; set; }
        public Message Message { get; set; }
        [JsonProperty("finish_reason")]
        public string FinishReason { get; set; }
        public override string ToString()
        {
            var messageString = Message != null ? Message.ToString() : "null";
            return $"Index: {Index}, Message: {messageString}, FinishReason: {FinishReason}";
        }
    }

    public class Message
    {
        public string Role { get; set; }
        public string Content { get; set; }
        public override string ToString()
        {
            return $"Role: {Role}, Content: {Content}";
        }
    }

    public class Usage
    {
        [JsonProperty("prompt_tokens")]
        public int PromptTokens { get; set; } //This represents the number of tokens used in the prompt that you sent to the model.
        [JsonProperty("completion_tokens")]
        public int CompletionTokens { get; set; } //This represents the number of tokens used in the response (or completion) returned by the model.
        [JsonProperty("total_tokens")]
        public int TotalTokens { get; set; } //This represents the total number of tokens used in both the prompt and the completion.
        public override string ToString()
        {
            return $"PromptTokens: {PromptTokens}, CompletionTokens: {CompletionTokens}, TotalTokens: {TotalTokens}";
        }
    }
}
