using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace CallaghanDev.ML.OpenAI
{
    [AttributeUsage(AttributeTargets.Field, AllowMultiple = false)]
    public class ModelNameAttribute : Attribute
    {
        public string ModelName { get; }
        public ModelNameAttribute(string modelName) => ModelName = modelName;
    }

    [AttributeUsage(AttributeTargets.Field, AllowMultiple = false)]
    public class PricingAttribute : Attribute
    {
        public string Input { get; }
        public string CachedInput { get; }
        public string Output { get; }
        public PricingAttribute(string input, string cachedInput, string output)
        {
            Input = input;
            CachedInput = cachedInput;
            Output = output;
        }
    }

    [AttributeUsage(AttributeTargets.Field, AllowMultiple = false)]
    public class MaxTokensAttribute : Attribute
    {
        public int MaxTokens { get; }
        public MaxTokensAttribute(int maxTokens) => MaxTokens = maxTokens;
    }
    public enum OpenAIModel
    {
        [ModelName("gpt-4.1-2025-04-14"), Pricing("$2.00", "$0.50", "$8.00"), MaxTokens(128000)]
        GPT_4_1,

        [ModelName("gpt-4.1-mini"), Pricing("$0.40", "$0.10", "$1.60"), MaxTokens(32000)]
        GPT_4_1_Mini,

        [ModelName("gpt-4.1-nano-2025-04-14"), Pricing("$0.10", "$0.025", "$0.40"), MaxTokens(8192)]
        GPT_4_1_Nano,

        [ModelName("gpt-4.5-preview-2025-02-27"), Pricing("$75.00", "$37.50", "$150.00"), MaxTokens(128000)]
        GPT_4_5_Preview,

        [ModelName("gpt-4o-2024-08-06"), Pricing("$2.50", "$1.25", "$10.00"), MaxTokens(16384)]
        GPT_4o,

        [ModelName("gpt-4o-audio-preview-2024-12-17"), Pricing("$2.50", "-", "$10.00"), MaxTokens(128000)]
        GPT_4o_Audio_Preview,

        [ModelName("gpt-4o-realtime-preview-2024-12-17"), Pricing("$5.00", "$2.50", "$20.00"), MaxTokens(128000)]
        GPT_4o_Realtime_Preview,

        [ModelName("gpt-4o-mini-2024-07-18"), Pricing("$0.15", "$0.075", "$0.60"), MaxTokens(128000)]
        GPT_4o_Mini,

        [ModelName("gpt-4o-mini-audio-preview-2024-12-17"), Pricing("$0.15", "-", "$0.60"), MaxTokens(32000)]
        GPT_4o_Mini_Audio_Preview,

        [ModelName("gpt-4o-mini-realtime-preview-2024-12-17"), Pricing("$0.60", "$0.30", "$2.40"), MaxTokens(32000)]
        GPT_4o_Mini_Realtime_Preview,

        [ModelName("o1-2024-12-17"), Pricing("$15.00", "$7.50", "$60.00"), MaxTokens(128000)]
        O1,

        [ModelName("o1-pro-2025-03-19"), Pricing("$150.00", "-", "$600.00"), MaxTokens(128000)]
        O1_Pro,

        [ModelName("o3-2025-04-16"), Pricing("$10.00", "$2.50", "$40.00"), MaxTokens(128000)]
        O3,

        [ModelName("o4-mini-2025-04-16"), Pricing("$1.10", "$0.275", "$4.40"), MaxTokens(32000)]
        O4_Mini,

        [ModelName("o3-mini-2025-01-31"), Pricing("$1.10", "$0.55", "$4.40"), MaxTokens(32000)]
        O3_Mini,

        [ModelName("o1-mini-2024-09-12"), Pricing("$1.10", "$0.55", "$4.40"), MaxTokens(32000)]
        O1_Mini,

        [ModelName("codex-mini-latest"), Pricing("$1.50", "$0.375", "$6.00"), MaxTokens(8192)]
        Codex_Mini_Latest,

        [ModelName("gpt-4o-mini-search-preview-2025-03-11"), Pricing("$0.15", "-", "$0.60"), MaxTokens(32000)]
        GPT_4o_Mini_Search_Preview,

        [ModelName("gpt-4o-search-preview-2025-03-11"), Pricing("$2.50", "-", "$10.00"), MaxTokens(128000)]
        GPT_4o_Search_Preview,

        [ModelName("computer-use-preview-2025-03-11"), Pricing("$3.00", "-", "$12.00"), MaxTokens(32000)]
        Computer_Use_Preview,

        [ModelName("gpt-image-1"), Pricing("$5.00", "$1.25", "-"), MaxTokens(32000)]
        GPT_Image_1,
    }
}
