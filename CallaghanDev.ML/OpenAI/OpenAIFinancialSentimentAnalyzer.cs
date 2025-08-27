using CallaghanDev.Utilities;
using CallaghanDev.Utilities.String;
using Newtonsoft.Json.Linq;
using Newtonsoft.Json;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using System.Linq;
using System.Text;
using System.Text.RegularExpressions;
using System.Threading.Tasks;

namespace CallaghanDev.ML.OpenAI
{
    public class OpenAIAssetSignal
    {
        public string Name { get; set; }
        public float Signal { get; set; }
        public int NewsId { get; set; }

        public override string ToString()
        {
            return $"OpenAIAssetSignal:{Name}: {Signal:F2}";
        }
    }

    public class OpenAINewsInput
    {
        public OpenAINewsInput(string InStory, long InNewsId)
        {
            Story = InStory;
            NewsId = InNewsId;
        }
        public string Story { get; set; }
        public long NewsId { get; set; }

    }
    public class OpenAIFinancialSentimentAnalyzer
    {
        private readonly ILogger _logger;
        private readonly GPTClient _gptClient;
        private readonly Similarity _similarity;
        private readonly string _openAiApiKey;
        private string GetSystemPrompt() => @"
You are a financial analysis assistant. 
Read the following news story and select from this list of assets only those which the news is likely to affect:

HighGradeCopper, Copper, Zinc, Aluminium, Lead, IronOre, Nickel, UKLongGilt, USUltraTreasuryBond, US10YearTNote, USTreasuryBond, GermanBund, GermanBuxl, JapaneseGovernmentBond, ItalianBTP,
US5YearTNote, USInflationIndex, Oil_USCrude, NaturalGas, Oil_BrentCrude, NaturalGasUk, HeatingOil, Gasoline, London_GasOil, US500, Germany40, Japan225, HongKongHS50, France40, ChinaA50,
EU600, Spain50, SwitzerlandBlueChip, Sweden30, EmergingMarketsIndex, Netherlands25, SouthAfrica40, FTSE250, SingaporeBlueChip, Denmark25, Gold, Silver, Palladium, Platinum, Cocoa_NewYork
Wheat_Chicago, ArabicaCoffee_NewYork, Cocoa_London, Sugar_NewYork, Corn, SoyBeanOil, Cotton, RobustaCoffee_London, SoyBeans, Sugar_London, SoyBeanMeal, GBPUSD, USDJPY, EURUSD, EURJPY,
AUDUSD, EURGBP, USDCAD, USDCHF, EURCHF, GBPEUR, GBPJPY, CHFJPY, GBPCHF, CADJPY, EURCAD, GBPCAD, CADCHF, USDZAR, GBPZAR, EURZAR, USDSGD, ZARJPY, EURSGD, INRJPY, SGDJPY, GBPSGD, USDHKD,
NZDUSD, AUDJPY, GBPAUD, AUDCAD, EURAUD, NZDJPY, AUDCHF, GBPNZD, AUDNZD, NZDCAD, EURNZD, NZDCHF, AUDSGD, AUDEUR, AUDGBP, NZDAUD, NZDEUR, NZDGBP, USDSEK, USDNOK, GBPSEK, GBPNOK, EURNOK,
EURSEK, NOKSEK, CADNOK, CHFNOK, EURDKK, GBPDKK, NOKJPY, SEKJPY, USDDKK, MXNJPY, USDMXN, USDTRY, TRYJPY, EURMXN, GBPTRY, EURTRY, USDILS, EURHUF, GBPHUF, USDHUF, EURPLN, GBPILS, GBPPLN,
EURILS, USDPLN, PLNJPY, CHFHUF, EURCZK, GBPCZK, GBPMXN, USDCZK, USDCNH, GBPINR, USDBRL, USDTHB, BRLJPY, USDPHP, USDIDR, CNHJPY, USDKRW, GBPCNH, USDINR, USDTWD, AUDCNH, CADCNH, EURCNH,
NZDCNH

For each relevant asset, output a JSON array of (asset, signal) pairs where:
- The first value is the asset name as given.
- The second value must be a number between -1 and 1 (inclusive), where -1 represents a definitive sell, +1 represents a definitive buy, and values in between represent less certain signals.
- Do not include assets not directly affected by the news.

Example response:
[
  [""Oil_USCrude"", 1],
  [""US500"", -1]
]

Only output the JSON array, nothing else.
";
        private List<string> _TradableAssetNames { get; set; }
        public OpenAIFinancialSentimentAnalyzer(string openAiApiKey, List<string> TradableAssetNames, ILogger logger)
        {
            _logger = logger;
            _similarity =new Similarity();
            _gptClient = new GPTClient(openAiApiKey, OpenAIModel.GPT_4_1_Mini);
            _openAiApiKey = openAiApiKey;
            _TradableAssetNames = TradableAssetNames;

        }

        #region OpenAI Sentiment Anyalsis
        public async Task<List<OpenAIAssetSignal>> GetSentimentAnalysis(OpenAINewsInput openAINewsInputs, int Attempt = 0)
        {
            string resultjson = "";
            GPTClient gptClient = new GPTClient(_openAiApiKey, OpenAIModel.GPT_4_1_Mini);
            try
            {
                resultjson = await gptClient.Run(GetSystemPrompt(), openAINewsInputs.Story);
            }
            catch (Exception ex)
            {
                _logger?.Error($"error calling OpenAI: {ex.Message}");
            }

            List<OpenAIAssetSignal> AssetSignals = GetAssetSignals(resultjson, 0);


            if (!(AssetSignals.Count() == 0 || AssetSignals == null))
            {

                foreach (var item in AssetSignals)
                {
                    string AssetNameResponse = item.Name;
                    string AssetMostSimilar = _similarity.StringMostSimilar(AssetNameResponse, _TradableAssetNames);

                    item.Name = AssetMostSimilar;

                }
            }

            return AssetSignals;
        }
        private List<OpenAIAssetSignal> GetAssetSignals(string resultjson, int Attempt = 0)
        {
            string cleaned = "";
            try
            {
                cleaned = System.Text.RegularExpressions.Regex.Replace(resultjson, @"^```[a-zA-Z]*\s*|```$", "", System.Text.RegularExpressions.RegexOptions.Multiline).Trim();


                int firstBracket = cleaned.IndexOf('[');
                int lastBracket = cleaned.LastIndexOf(']');
                if (firstBracket == -1 || lastBracket == -1 || lastBracket <= firstBracket)
                    throw new Exception("Could not find JSON array in response: " + resultjson);

                cleaned = cleaned.Substring(firstBracket, lastBracket - firstBracket + 1);

                if (cleaned == "[]" || string.IsNullOrEmpty(cleaned))
                {
                    return new List<OpenAIAssetSignal>();
                }

                Exception lastEx = null;

                return ParseAssetSignalJson(cleaned);


            }
            catch (Exception ex)
            {
                if (Attempt < 3)
                {
                    return GetAssetSignals(resultjson, Attempt + 1);
                }
                else
                {
                    _logger.Error($"Failed to deserialize OpenAI response after {Attempt + 1} attempts. Last response: {resultjson}, how it looked cleaned:{cleaned}");

                    return new List<OpenAIAssetSignal>();
                }
            }
        }
        private List<OpenAIAssetSignal> ParseAssetSignalJson(string json)
        {
            Exception lastEx = null;
            if (string.IsNullOrWhiteSpace(json)) throw new ArgumentException("Input JSON is empty.");

            // 1. Try List<List<object>>
            try
            {
                var arrayResult = JsonConvert.DeserializeObject<List<List<object>>>(json);
                if (arrayResult != null && arrayResult.All(x => x.Count == 2))
                {
                    return arrayResult.Select(arr => new OpenAIAssetSignal
                    {
                        Name = arr[0]?.ToString() ?? "",
                        Signal = Convert.ToSingle(arr[1], CultureInfo.InvariantCulture)
                    }).ToList();
                }
            }
            catch (Exception ex) { lastEx = ex; }

            // 2. Try List<object[]>
            try
            {
                var arrayResult = JsonConvert.DeserializeObject<List<object[]>>(json);
                if (arrayResult != null && arrayResult.All(x => x.Length == 2))
                {
                    return arrayResult.Select(arr => new OpenAIAssetSignal
                    {
                        Name = arr[0]?.ToString() ?? "",
                        Signal = Convert.ToSingle(arr[1], CultureInfo.InvariantCulture)
                    }).ToList();
                }
            }
            catch (Exception ex) { lastEx = ex; }

            // 3. Try List<List<string>> (numbers as string)
            try
            {
                var arrayResult = JsonConvert.DeserializeObject<List<List<string>>>(json);
                if (arrayResult != null && arrayResult.All(x => x.Count == 2))
                {
                    return arrayResult.Select(arr => new OpenAIAssetSignal
                    {
                        Name = arr[0],
                        Signal = Convert.ToSingle(arr[1], CultureInfo.InvariantCulture)
                    }).ToList();
                }
            }
            catch (Exception ex) { lastEx = ex; }

            // 4. Try List<Tuple<string, float>>
            try
            {
                var tupleResult = JsonConvert.DeserializeObject<List<Tuple<string, float>>>(json);
                if (tupleResult != null && tupleResult.All(x => x.Item1 != null))
                {
                    return tupleResult.Select(t => new OpenAIAssetSignal
                    {
                        Name = t.Item1,
                        Signal = t.Item2
                    }).ToList();
                }
            }
            catch (Exception ex) { lastEx = ex; }

            // 5. Try List<OpenAIAssetSignal> (array of objects with correct property names)
            try
            {
                var objResult = JsonConvert.DeserializeObject<List<OpenAIAssetSignal>>(json);
                if (objResult != null && objResult.All(x => x.Name != null))
                {
                    return objResult;
                }
            }
            catch (Exception ex) { lastEx = ex; }

            // 6. Try List<Dictionary<string, object>> (generic objects with unknown keys)
            try
            {
                var dicts = JsonConvert.DeserializeObject<List<Dictionary<string, object>>>(json);
                if (dicts != null && dicts.All(d => d.Count >= 2))
                {
                    var list = new List<OpenAIAssetSignal>();
                    foreach (var dict in dicts)
                    {
                        // Try to find key for name
                        string nameKey = dict.Keys.FirstOrDefault(k => k.Equals("Item1", StringComparison.OrdinalIgnoreCase) ||
                                                                       k.ToLower().Contains("asset") || k.ToLower().Contains("name"));
                        // Try to find key for value
                        string valueKey = dict.Keys.FirstOrDefault(k => k.Equals("Item2", StringComparison.OrdinalIgnoreCase) ||
                                                                        k.ToLower().Contains("signal") || k.ToLower().Contains("value"));
                        if (nameKey != null && valueKey != null)
                        {
                            var name = dict[nameKey]?.ToString() ?? "";
                            var valObj = dict[valueKey];
                            float val = 0.0f;
                            if (valObj is float d) val = d;
                            else if (valObj is float f) val = (float)f;
                            else if (valObj is int i) val = i;
                            else if (valObj is long l) val = l;
                            else if (valObj is decimal m) val = (float)m;
                            else if (valObj != null) val = Convert.ToSingle(valObj, CultureInfo.InvariantCulture);

                            list.Add(new OpenAIAssetSignal { Name = name, Signal = val });
                        }
                    }
                    if (list.Count > 0) return list;
                }
            }
            catch (Exception ex) { lastEx = ex; }

            // 7. Try single quotes replaced by float quotes
            try
            {
                var alt = json.Replace('\'', '"');
                var arrayResult = JsonConvert.DeserializeObject<List<List<object>>>(alt);
                if (arrayResult != null && arrayResult.All(x => x.Count == 2))
                {
                    return arrayResult.Select(arr => new OpenAIAssetSignal
                    {
                        Name = arr[0]?.ToString() ?? "",
                        Signal = Convert.ToSingle(arr[1], CultureInfo.InvariantCulture)
                    }).ToList();
                }
            }
            catch (Exception ex) { lastEx = ex; }

            // 8. Try parsing as JArray and extracting pairs
            try
            {
                var jarr = JArray.Parse(json);
                var list = new List<OpenAIAssetSignal>();
                foreach (var token in jarr)
                {
                    // If array [name, value]
                    if (token is JArray innerArr && innerArr.Count == 2)
                    {
                        var name = innerArr[0]?.ToString();
                        float value = Convert.ToSingle(innerArr[1]?.ToString(), CultureInfo.InvariantCulture);
                        list.Add(new OpenAIAssetSignal { Name = name, Signal = value });
                    }
                    // If object { ... }
                    else if (token is JObject obj)
                    {
                        var props = obj.Properties().ToList();
                        string name = null;
                        float value = 0.0f;
                        foreach (var prop in props)
                        {
                            if (prop.Name.Equals("Item1", StringComparison.OrdinalIgnoreCase) ||
                                prop.Name.ToLower().Contains("asset") ||
                                prop.Name.ToLower().Contains("name"))
                            {
                                name = prop.Value.ToString();
                            }
                            if (prop.Name.Equals("Item2", StringComparison.OrdinalIgnoreCase) ||
                                prop.Name.ToLower().Contains("signal") ||
                                prop.Name.ToLower().Contains("value"))
                            {
                                value = Convert.ToSingle(prop.Value.ToString(), CultureInfo.InvariantCulture);
                            }
                        }
                        if (name != null)
                        {
                            list.Add(new OpenAIAssetSignal { Name = name, Signal = value });
                        }
                    }
                }
                if (list.Count > 0) return list;
            }
            catch (Exception ex) { lastEx = ex; }
            // Fix unquoted strings in array-of-arrays: [USDCNH, 0.5] => ["USDCNH", 0.5]
            try
            {
                var pattern = @"\[\s*([A-Za-z_][A-Za-z0-9_]*),";
                // Replace [USDCNH, 0.5] with ["USDCNH", 0.5]
                var fixedJson = System.Text.RegularExpressions.Regex.Replace(json, pattern, m =>
                {
                    var symbol = m.Groups[1].Value;
                    return $"[\"{symbol}\",";
                });
                // If we changed anything, try parsing as List<List<object>>
                if (!ReferenceEquals(fixedJson, json))
                {
                    var arrayResult = JsonConvert.DeserializeObject<List<List<object>>>(fixedJson);
                    if (arrayResult != null && arrayResult.All(x => x.Count == 2))
                    {
                        return arrayResult.Select(arr => new OpenAIAssetSignal
                        {
                            Name = arr[0]?.ToString() ?? "",
                            Signal = Convert.ToSingle(arr[1], CultureInfo.InvariantCulture)
                        }).ToList();
                    }
                }
            }
            catch (Exception ex) { lastEx = ex; }
            // 9. SUPER-GENERIC: Parse each line and split by "," into two values
            try
            {
                // Remove any outer brackets and whitespace
                var lines = json
                    .Trim()
                    .Trim('[', ']')
                    .Split(new[] { '\n', '\r' }, StringSplitOptions.RemoveEmptyEntries);

                var list = new List<OpenAIAssetSignal>();

                foreach (var line in lines)
                {
                    var l = line.Trim();
                    // Skip empty lines and those not resembling a pair
                    if (string.IsNullOrWhiteSpace(l) || (!l.Contains(",") && !l.Contains(" "))) continue;

                    // Remove leading/trailing brackets/quotes
                    l = l.Trim().TrimStart('[').TrimEnd(']').Trim();

                    // Try splitting on comma (could fall back to whitespace split if needed)
                    string[] parts = l.Split(',');
                    if (parts.Length == 2)
                    {
                        string name = Regex.Replace(parts[0].Trim().Trim('"'), @"[^A-Za-z0-9_]", "");
                        string valString = Regex.Replace(parts[1].Trim().Trim('"'), @"[^0-9\.\-eE]", "");

                        if (!string.IsNullOrEmpty(name) && !string.IsNullOrEmpty(valString))
                        {
                            if (float.TryParse(valString, NumberStyles.Any, CultureInfo.InvariantCulture, out float value))
                            {
                                list.Add(new OpenAIAssetSignal { Name = name, Signal = value });
                            }
                        }
                    }
                }
                if (list.Count > 0) return list;
            }
            catch (Exception ex) { lastEx = ex; }

            throw new Exception("Deserialization failed: Could not parse OpenAI response.\nInput: " + json, lastEx);
        }
        #endregion

    }
}
