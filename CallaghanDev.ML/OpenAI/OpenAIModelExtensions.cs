using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Reflection;

namespace CallaghanDev.ML.OpenAI
{

    public static class OpenAIModelExtensions
    {
        public static string ToTagName(this OpenAIModel model)
            => model.GetAttribute<ModelNameAttribute>()?.ModelName ?? model.ToString();

        public static string GetInputPrice(this OpenAIModel model)
            => model.GetAttribute<PricingAttribute>()?.Input ?? "-";

        public static string GetCachedInputPrice(this OpenAIModel model)
            => model.GetAttribute<PricingAttribute>()?.CachedInput ?? "-";

        public static string GetOutputPrice(this OpenAIModel model)
            => model.GetAttribute<PricingAttribute>()?.Output ?? "-";

        public static int GetMaxTokens(this OpenAIModel model)
            => model.GetAttribute<MaxTokensAttribute>()?.MaxTokens ?? 4096;

        private static T GetAttribute<T>(this OpenAIModel model) where T : Attribute
        {
            var type = model.GetType();
            var memInfo = type.GetMember(model.ToString());
            var attrs = memInfo[0].GetCustomAttributes(typeof(T), false);
            return (T)(attrs.Length > 0 ? attrs[0] : null);
        }
    }

}
