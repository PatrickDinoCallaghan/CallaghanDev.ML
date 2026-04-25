using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Text.RegularExpressions;
using System.IO;
using System.Text.Json;
using CallaghanDev.ML.AccelerationManagers;
using static CallaghanDev.ML.AccelerationManagers.AccelerationCPU;

namespace CallaghanDev.ML.Transformers
{
    public class BPETokenizer
    {
        public const string PAD_TOKEN = "<|pad|>";
        public const string START_TOKEN = "<|startoftext|>";
        public const string END_TOKEN = "<|endoftext|>";
        public const string UNK_TOKEN = "<|unk|>";

        public int PadTokenId { get; private set; }
        public int StartTokenId { get; private set; }
        public int EndTokenId { get; private set; }
        public int UnkTokenId { get; private set; }

        private Dictionary<string, int> _vocabToId;
        private Dictionary<int, string> _idToVocab;
        private List<(string, string)> _merges;
        private Dictionary<(string, string), int> _mergePriority;

        private Dictionary<string, List<int>> _encodeCache;
        private const int MAX_CACHE_SIZE = 10000;

        public int VocabSize => _vocabToId.Count;
        public int MaxTokenLength { get; set; } = 100;
        public bool LowerCase { get; set; } = false; 
        
        private IAccelerationManager _accel;
        public BPETokenizer(IAccelerationManager accelerationManager)
        {
            _vocabToId = new Dictionary<string, int>();
            _idToVocab = new Dictionary<int, string>();
            _merges = new List<(string, string)>();
            _mergePriority = new Dictionary<(string, string), int>();
            _encodeCache = new Dictionary<string, List<int>>();
            _accel = accelerationManager;
            InitializeSpecialTokens();
        }

        public BPETokenizer()
        {
        }

        private void InitializeSpecialTokens()
        {
            AddSpecialToken(PAD_TOKEN, 0);
            AddSpecialToken(START_TOKEN, 1);
            AddSpecialToken(END_TOKEN, 2);
            AddSpecialToken(UNK_TOKEN, 3);

            PadTokenId = 0;
            StartTokenId = 1;
            EndTokenId = 2;
            UnkTokenId = 3;
        }

        private void AddSpecialToken(string token, int id)
        {
            _vocabToId[token] = id;
            _idToVocab[id] = token;
        }

        private string[] PreTokenize(string text)
        {
            return _accel.PreTokenize(text);
            /*var tokens = new List<string>();
            var currentToken = new StringBuilder();

            for (int i = 0; i < text.Length; i++)
            {
                char c = text[i];

                if (char.IsLetterOrDigit(c))
                {
                    currentToken.Append(c);
                }
                else
                {
                    if (currentToken.Length > 0)
                    {
                        tokens.Add(currentToken.ToString());
                        currentToken.Clear();
                    }

                    if (!char.IsWhiteSpace(c))
                    {
                        tokens.Add(c.ToString());
                    }
                }
            }

            if (currentToken.Length > 0)
            {
                tokens.Add(currentToken.ToString());
            }

            return tokens.ToArray();*/
        }

        public void Train(string[] texts, int vocabSize = 50000, int minFrequency = 10)
        {
            if (texts == null || texts.Length == 0)
            {
                throw new ArgumentException("Cannot train tokenizer on empty corpus.", nameof(texts));
            }

            if (vocabSize <= 4)
            {
                throw new ArgumentOutOfRangeException(nameof(vocabSize), "vocabSize must be greater than special-token count.");
            }

            if (minFrequency <= 0)
            {
                throw new ArgumentOutOfRangeException(nameof(minFrequency), "minFrequency must be positive.");
            }

            var specialTokens = _vocabToId.Where(kv => kv.Key.StartsWith("<|")).ToList();
            _vocabToId.Clear();
            _idToVocab.Clear();
            _merges.Clear();
            _mergePriority.Clear();
            _encodeCache.Clear();

            foreach (var (token, id) in specialTokens)
            {
                _vocabToId[token] = id;
                _idToVocab[id] = token;
            }

            var wordFreqs = GetWordFrequencies(texts);
            var charVocab = BuildCharacterVocabulary(wordFreqs);

            int nextId = _vocabToId.Count;
            foreach (var ch in charVocab.OrderBy(c => c))
            {
                if (!_vocabToId.ContainsKey(ch))
                {
                    _vocabToId[ch] = nextId;
                    _idToVocab[nextId] = ch;
                    nextId++;
                }
            }

            LearnBPEMerges(wordFreqs, vocabSize - nextId, minFrequency);
        }

        private Dictionary<string, int> GetWordFrequencies(string[] texts)
        {

            return _accel.GetWordFrequencies(texts, LowerCase);
            /*
            var wordFreqs = new Dictionary<string, int>();

            foreach (var text in texts)
            {
                var processedText = LowerCase ? text.ToLowerInvariant() : text;
                var words = PreTokenize(processedText);

                foreach (var word in words)
                {
                    if (string.IsNullOrWhiteSpace(word))
                    {
                        continue;
                    }
                    var charSeq = string.Join(" ", word.Select(c => c.ToString()));

                    if (wordFreqs.ContainsKey(charSeq))
                    {
                        wordFreqs[charSeq]++;
                    }
                    else
                    {
                        wordFreqs[charSeq] = 1;
                    }
                }
            }

            return wordFreqs;*/
        }

        private HashSet<string> BuildCharacterVocabulary(Dictionary<string, int> wordFreqs)
        {
            return _accel.BuildCharacterVocabulary(wordFreqs);

            /*
            var chars = new HashSet<string>();

            foreach (var word in wordFreqs.Keys)
            {
                foreach (var ch in word.Split(' '))
                {
                    if (!string.IsNullOrEmpty(ch))
                    {
                        chars.Add(ch);
                    }
                }
            }

            return chars;*/
        }

        private void LearnBPEMerges(Dictionary<string, int> wordFreqs,  int numMerges,  int minFrequency)
        {
            var words = wordFreqs.ToDictionary(
                kv => kv.Key.Split(' ').ToList(),
                kv => kv.Value,
                new ListEqualityComparer<string>());

            for (int mergeIdx = 0; mergeIdx < numMerges; mergeIdx++)
            {
                var pairCounts = _accel.CountPairFrequencies(words);

                var (bestPair, frequency) = _accel.SelectBestPair(pairCounts, minFrequency);

                if (frequency == 0)
                    break;

                var (left, right) = bestPair;
                var merged = left + right;

                _merges.Add((left, right));
                _mergePriority[(left, right)] = mergeIdx;

                if (!_vocabToId.ContainsKey(merged))
                {
                    int id = _vocabToId.Count;
                    _vocabToId[merged] = id;
                    _idToVocab[id] = merged;
                }

                words = _accel.ApplyMergeToVocabulary(words, left, right);
            }
        }
        private List<string> ApplyMerge(List<string> word, string left, string right)
        {
            return _accel.ApplyMerge(word, left, right);
           /*
            var result = new List<string>();
            int i = 0;

            while (i < word.Count)
            {
                if (i < word.Count - 1 && word[i] == left && word[i + 1] == right)
                {
                    result.Add(left + right);
                    i += 2;
                }
                else
                {
                    result.Add(word[i]);
                    i++;
                }
            }

            return result;*/
        }

        public int[] Encode(string text, bool addSpecialTokens = true)
        {
            if (string.IsNullOrEmpty(text))
            {
                return addSpecialTokens ? new[] { StartTokenId, EndTokenId } : Array.Empty<int>();
            }

            var processedText = LowerCase ? text.ToLowerInvariant() : text;

            var words = PreTokenize(processedText);
            var tokens = new List<int>();

            if (addSpecialTokens)
            {
                tokens.Add(StartTokenId);
            }

            foreach (var word in words)
            {
                if (string.IsNullOrWhiteSpace(word))
                {
                    continue;
                } 
                if (_encodeCache.TryGetValue(word, out var cachedTokens))
                {
                    tokens.AddRange(cachedTokens);
                }
                else
                {
                    var wordTokens = EncodeWord(word);
                    tokens.AddRange(wordTokens);

                    if (_encodeCache.Count < MAX_CACHE_SIZE)
                    {
                        _encodeCache[word] = wordTokens;
                    }
                }
            }

            if (addSpecialTokens)
            {
                tokens.Add(EndTokenId);
            }

            return tokens.ToArray();
        }

        private List<int> EncodeWord(string word)
        {
            return _accel.EncodeWord(
                word,
                _mergePriority,
                _vocabToId,
                UnkTokenId);
        }

        public string Decode(int[] tokenIds, bool skipSpecialTokens = true)
        {
            return _accel.DecodeTokens(tokenIds, _idToVocab, UNK_TOKEN, skipSpecialTokens);
        }

        public int[][] EncodeBatch(string[] texts, bool addSpecialTokens = true,int? maxLength = null)
        {
            var encoded = new int[texts.Length][];

            for (int i = 0; i < texts.Length; i++)
            {
                encoded[i] = Encode(texts[i], addSpecialTokens);

                if (maxLength.HasValue)
                {
                    encoded[i] = _accel.PadOrTruncate(
                        encoded[i],
                        maxLength.Value,
                        addSpecialTokens,
                        PadTokenId,
                        EndTokenId);
                }
            }

            return encoded;
        }
        public void Save(string directory)
        {
            if (!Directory.Exists(directory))
            {
                Directory.CreateDirectory(directory);
            }

            var vocabPath = Path.Combine(directory, "vocab.json");
            var vocabJson = JsonSerializer.Serialize(_vocabToId, new JsonSerializerOptions
            {
                WriteIndented = true
            });
            File.WriteAllText(vocabPath, vocabJson);

            var mergesPath = Path.Combine(directory, "merges.txt");
            using (var writer = new StreamWriter(mergesPath))
            {
                writer.WriteLine("#version: 0.2");
                foreach (var (left, right) in _merges)
                {
                    writer.WriteLine($"{left} {right}");
                }
            }

            var configPath = Path.Combine(directory, "tokenizer_config.json");
            var config = new Dictionary<string, object>
            {
                ["vocab_size"] = VocabSize,
                ["max_token_length"] = MaxTokenLength,
                ["lowercase"] = LowerCase
            };
            var configJson = JsonSerializer.Serialize(config, new JsonSerializerOptions
            {
                WriteIndented = true
            });
            File.WriteAllText(configPath, configJson);

            Console.WriteLine($"Tokenizer saved to {directory}");
        }
        public static BPETokenizer Load(string directory)
        {
            var tokenizer = new BPETokenizer(new AccelerationCPU());

            var vocabPath = Path.Combine(directory, "vocab.json");
            var vocabJson = File.ReadAllText(vocabPath);

            tokenizer._vocabToId =
                JsonSerializer.Deserialize<Dictionary<string, int>>(vocabJson)
                ?? throw new InvalidOperationException("Failed to deserialize tokenizer vocab.");

            tokenizer._idToVocab = tokenizer._vocabToId.ToDictionary(kv => kv.Value, kv => kv.Key);

            tokenizer.PadTokenId = tokenizer._vocabToId[PAD_TOKEN];
            tokenizer.StartTokenId = tokenizer._vocabToId[START_TOKEN];
            tokenizer.EndTokenId = tokenizer._vocabToId[END_TOKEN];
            tokenizer.UnkTokenId = tokenizer._vocabToId[UNK_TOKEN];

            tokenizer._merges = new List<(string, string)>();
            tokenizer._mergePriority = new Dictionary<(string, string), int>();
            tokenizer._encodeCache = new Dictionary<string, List<int>>();

            var mergesPath = Path.Combine(directory, "merges.txt");
            var lines = File.ReadAllLines(mergesPath);

            int priority = 0;

            foreach (var line in lines)
            {
                if (line.StartsWith("#") || string.IsNullOrWhiteSpace(line))
                    continue;

                var parts = line.Split(' ', StringSplitOptions.RemoveEmptyEntries);

                if (parts.Length >= 2)
                {
                    var merge = (parts[0], parts[1]);
                    tokenizer._merges.Add(merge);
                    tokenizer._mergePriority[merge] = priority++;
                }
            }

            var configPath = Path.Combine(directory, "tokenizer_config.json");

            if (File.Exists(configPath))
            {
                var configJson = File.ReadAllText(configPath);
                var config = JsonSerializer.Deserialize<Dictionary<string, JsonElement>>(configJson);

                if (config != null)
                {
                    if (config.TryGetValue("max_token_length", out var mtl))
                        tokenizer.MaxTokenLength = mtl.GetInt32();

                    if (config.TryGetValue("lowercase", out var lc))
                        tokenizer.LowerCase = lc.GetBoolean();
                }
            }

            return tokenizer;
        }

        public string IdToToken(int id)
        {
            return _idToVocab.TryGetValue(id, out var token) ? token : UNK_TOKEN;
        }

        public int TokenToId(string token)
        {
            return _vocabToId.TryGetValue(token, out var id) ? id : UnkTokenId;
        }
        public List<string> GetVocabulary()
        {
            return _idToVocab.OrderBy(kv => kv.Key).Select(kv => kv.Value).ToList();
        }
    }

}