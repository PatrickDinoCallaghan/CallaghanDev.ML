using Newtonsoft.Json;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace CallaghanDev.ML.OpenAI
{
    public class OpenAIChatResponse
    {
        public List<OpenAIChoice> choices { get; set; }
    }
    public class OpenAIChoice
    {
        public OpenAIMessage message { get; set; }
    }
    public class OpenAIMessage
    {
        public string role { get; set; }
        public string content { get; set; }
    }
}
