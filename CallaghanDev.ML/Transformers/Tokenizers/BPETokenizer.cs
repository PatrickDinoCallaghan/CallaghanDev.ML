using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Text.RegularExpressions;
using System.IO;
using System.Text.Json;

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

        public BPETokenizer()
        {
            _vocabToId = new Dictionary<string, int>();
            _idToVocab = new Dictionary<int, string>();
            _merges = new List<(string, string)>();
            _mergePriority = new Dictionary<(string, string), int>();
            _encodeCache = new Dictionary<string, List<int>>();

            InitializeSpecialTokens();
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
            var tokens = new List<string>();
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

            return tokens.ToArray();
        }

        public void Train(string[] texts, int vocabSize = 5000, int minFrequency = 2)
        {
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

            Console.WriteLine($"Character vocabulary: {charVocab.Count} characters");

            Console.WriteLine($"Step 2: Learning BPE merges (target: {vocabSize - nextId} merges)...");
            LearnBPEMerges(wordFreqs, vocabSize - nextId, minFrequency);

            Console.WriteLine($"Training complete!");
            Console.WriteLine($"Final vocabulary size: {VocabSize}");
            Console.WriteLine($"Number of merges: {_merges.Count}");
        }

        private Dictionary<string, int> GetWordFrequencies(string[] texts)
        {
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

            return wordFreqs;
        }

        private HashSet<string> BuildCharacterVocabulary(Dictionary<string, int> wordFreqs)
        {
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

            return chars;
        }

        private void LearnBPEMerges(Dictionary<string, int> wordFreqs, int numMerges, int minFrequency)
        {
            var words = wordFreqs.ToDictionary(kv => kv.Key.Split(' ').ToList(), kv => kv.Value, new ListEqualityComparer<string>());

            for (int mergeIdx = 0; mergeIdx < numMerges; mergeIdx++)
            {
                var pairCounts = new Dictionary<(string, string), int>();

                foreach (var (word, freq) in words)
                {
                    for (int i = 0; i < word.Count - 1; i++)
                    {
                        var pair = (word[i], word[i + 1]);
                        if (pairCounts.ContainsKey(pair))
                        {
                            pairCounts[pair] += freq;
                        }
                        else
                        {
                            pairCounts[pair] = freq;
                        }
                    }
                }

                if (pairCounts.Count == 0)
                {
                    break;
                }

                var bestPair = pairCounts.Where(kv => kv.Value >= minFrequency)
                    .OrderByDescending(kv => kv.Value)
                    .ThenBy(kv => kv.Key.Item1 + kv.Key.Item2)
                    .FirstOrDefault();

                if (bestPair.Value == 0)
                {
                    break;
                }

                var (left, right) = bestPair.Key;
                var merged = left + right;

                _merges.Add((left, right));
                _mergePriority[(left, right)] = mergeIdx;

                if (!_vocabToId.ContainsKey(merged))
                {
                    int id = _vocabToId.Count;
                    _vocabToId[merged] = id;
                    _idToVocab[id] = merged;
                }

                var newWords = new Dictionary<List<string>, int>(new ListEqualityComparer<string>());
                foreach (var (word, freq) in words)
                {
                    var newWord = ApplyMerge(word, left, right);
                    if (newWords.ContainsKey(newWord))
                    {
                        newWords[newWord] += freq;
                    }
                    else
                    {
                        newWords[newWord] = freq;
                    }
                }
                words = newWords;

                if ((mergeIdx + 1) % 100 == 0)
                {
                    Console.Write($"\r  Learned {mergeIdx + 1}/{numMerges} merges...");
                }
            }

            Console.WriteLine($"\r  Learned {_merges.Count} merges       ");
        }

        private List<string> ApplyMerge(List<string> word, string left, string right)
        {
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

            return result;
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
            var parts = word.Select(c => c.ToString()).ToList();

            while (parts.Count > 1)
            {
                (string, string)? bestMerge = null;
                int bestPriority = int.MaxValue;
                int bestPos = -1;

                for (int i = 0; i < parts.Count - 1; i++)
                {
                    var pair = (parts[i], parts[i + 1]);
                    if (_mergePriority.TryGetValue(pair, out int priority))
                    {
                        if (priority < bestPriority)
                        {
                            bestMerge = pair;
                            bestPriority = priority;
                            bestPos = i;
                        }
                    }
                }

                if (bestMerge == null)
                {
                    break;
                }

                var (left, right) = bestMerge.Value;
                parts[bestPos] = left + right;
                parts.RemoveAt(bestPos + 1);
            }

            var tokenIds = new List<int>();
            foreach (var part in parts)
            {
                if (_vocabToId.TryGetValue(part, out int id))
                {
                    tokenIds.Add(id);
                }
                else
                {
                    foreach (var ch in part)
                    {
                        var chStr = ch.ToString();
                        if (_vocabToId.TryGetValue(chStr, out int chId))
                        {
                            tokenIds.Add(chId);
                        }
                        else
                        {
                            tokenIds.Add(UnkTokenId);
                        }
                    }
                }
            }

            return tokenIds;
        }

        public string Decode(int[] tokenIds, bool skipSpecialTokens = true)
        {
            if (tokenIds == null || tokenIds.Length == 0)
            {
                return string.Empty;
            }

            var sb = new StringBuilder();

            foreach (var id in tokenIds)
            {
                if (!_idToVocab.TryGetValue(id, out var token))
                {
                    token = UNK_TOKEN;
                }

                if (skipSpecialTokens && token.StartsWith("<|") && token.EndsWith("|>"))
                {
                    continue;
                }
                sb.Append(token);
            }

            return sb.ToString();
        }

        public int[][] EncodeBatch(string[] texts, bool addSpecialTokens = true, int? maxLength = null)
        {
            var encoded = texts.Select(text => Encode(text, addSpecialTokens)).ToArray();

            if (maxLength.HasValue)
            {
                for (int i = 0; i < encoded.Length; i++)
                {
                    if (encoded[i].Length > maxLength.Value)
                    {
                        var truncated = new int[maxLength.Value];
                        Array.Copy(encoded[i], truncated, maxLength.Value);
                        if (addSpecialTokens)
                        {
                            truncated[maxLength.Value - 1] = EndTokenId;
                        }
                        encoded[i] = truncated;
                    }
                    else if (encoded[i].Length < maxLength.Value)
                    {
                        var padded = new int[maxLength.Value];
                        Array.Copy(encoded[i], padded, encoded[i].Length);
                        for (int j = encoded[i].Length; j < maxLength.Value; j++)
                        {
                            padded[j] = PadTokenId;
                        }
                        encoded[i] = padded;
                    }
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
            var tokenizer = new BPETokenizer();

            var vocabPath = Path.Combine(directory, "vocab.json");
            var vocabJson = File.ReadAllText(vocabPath);
            tokenizer._vocabToId = JsonSerializer.Deserialize<Dictionary<string, int>>(vocabJson);
            tokenizer._idToVocab = tokenizer._vocabToId.ToDictionary(kv => kv.Value, kv => kv.Key);

            tokenizer.PadTokenId = tokenizer._vocabToId[PAD_TOKEN];
            tokenizer.StartTokenId = tokenizer._vocabToId[START_TOKEN];
            tokenizer.EndTokenId = tokenizer._vocabToId[END_TOKEN];
            tokenizer.UnkTokenId = tokenizer._vocabToId[UNK_TOKEN];

            var mergesPath = Path.Combine(directory, "merges.txt");
            tokenizer._merges.Clear();
            tokenizer._mergePriority.Clear();

            var lines = File.ReadAllLines(mergesPath);
            int priority = 0;
            foreach (var line in lines)
            {
                if (line.StartsWith("#") || string.IsNullOrWhiteSpace(line))
                {
                    continue;
                }

                var parts = line.Split(' ');
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

                if (config.TryGetValue("max_token_length", out var mtl))
                {
                    tokenizer.MaxTokenLength = mtl.GetInt32();
                }
                if (config.TryGetValue("lowercase", out var lc))
                {
                    tokenizer.LowerCase = lc.GetBoolean();
                }
            }

            Console.WriteLine($"Tokenizer loaded from {directory}");
            Console.WriteLine($"Vocabulary size: {tokenizer.VocabSize}");
            Console.WriteLine($"Number of merges: {tokenizer._merges.Count}");

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

    internal class ListEqualityComparer<T> : IEqualityComparer<List<T>>
    {
        public bool Equals(List<T> x, List<T> y)
        {
            if (x == null || y == null)
            {
                return x == y;
            }

            if (x.Count != y.Count)
            {
                return false;
            }

            for (int i = 0; i < x.Count; i++)
            {
                if (!EqualityComparer<T>.Default.Equals(x[i], y[i]))
                {
                    return false;
                }
            }

            return true;
        }

        public int GetHashCode(List<T> obj)
        {
            if (obj == null)
            {
                return 0;
            }

            unchecked
            {
                int hash = 17;
                foreach (var item in obj)
                {
                    hash = hash * 31 + (item?.GetHashCode() ?? 0);
                }
                return hash;
            }
        }
    }
}