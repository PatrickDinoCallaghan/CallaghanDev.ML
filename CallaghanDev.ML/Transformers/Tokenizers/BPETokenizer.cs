using CallaghanDev.ML;
using CallaghanDev.ML.AccelerationManagers;
using CallaghanDev.ML.Enums;
using CallaghanDev.ML.Transformers.Configuration;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.Json;
using System.Threading.Tasks;
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
        private readonly object _encodeCacheLock = new object();

        private const int MAX_CACHE_SIZE = 10000;

        private IAccelerationManager _accel;
        private AccelerationType _accelerationType = AccelerationType.CPU;

        public int VocabSize => _vocabToId.Count;
        public int MaxTokenLength { get; set; } = 100;
        public bool LowerCase { get; set; } = false;

        /// <summary>
        /// Controls tokenizer training progress output. You can also override per Train(...) call.
        /// </summary>
        public bool ShowTrainingProgress { get; set; } = true;

        public BPETokenizer(RuntimeConfig runtime)
        {
            Init(runtime);
        }

        private BPETokenizer()
        {
        }

        private void Init(RuntimeConfig runtime)
        {
            if (runtime == null)
            {
                throw new ArgumentNullException(nameof(runtime));
            }

            _vocabToId = new Dictionary<string, int>();
            _idToVocab = new Dictionary<int, string>();
            _merges = new List<(string, string)>();
            _mergePriority = new Dictionary<(string, string), int>();
            _encodeCache = new Dictionary<string, List<int>>();

            _accelerationType = runtime.AccelerationType;
            _accel = AccelerationFactory.Create(runtime);

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
            return _accel.PreTokenize(text);
        }

        /// <summary>
        /// Trains a BPE tokenizer.
        /// Existing callers still work because showProgress has a default value.
        /// </summary>
        public void Train(string[] texts, int vocabSize = 50000, int minFrequency = 10, bool? showProgress = null)
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

            bool progressEnabled = showProgress ?? ShowTrainingProgress;

            ResetVocabularyKeepingSpecialTokens();

            Dictionary<string, int> wordFreqs;

            if (progressEnabled)
            {
                using (var progress = TerminalProgressBar.StartIndeterminate("Tokenizer: counting word frequencies", $"{texts.Length:N0} texts | target vocab {vocabSize:N0} | min freq {minFrequency:N0}"))
                {
                    wordFreqs = GetWordFrequencies(texts);
                    progress.Complete($"{wordFreqs.Count:N0} unique pre-tokenized words");
                }
            }
            else
            {
                wordFreqs = GetWordFrequencies(texts);
            }

            HashSet<string> charVocab;

            if (progressEnabled)
            {
                using (var progress = TerminalProgressBar.StartIndeterminate("Tokenizer: building character vocabulary", $"{wordFreqs.Count:N0} unique words"))
                {
                    charVocab = BuildCharacterVocabulary(wordFreqs);
                    progress.Complete($"{charVocab.Count:N0} base characters");
                }
            }
            else
            {
                charVocab = BuildCharacterVocabulary(wordFreqs);
            }

            int nextId = _vocabToId.Count;

            foreach (var ch in charVocab.OrderBy(c => c, StringComparer.Ordinal))
            {
                if (!_vocabToId.ContainsKey(ch))
                {
                    _vocabToId[ch] = nextId;
                    _idToVocab[nextId] = ch;
                    nextId++;
                }
            }

            int requestedMerges = Math.Max(0, vocabSize - nextId);
            int learnedMerges;

            if (progressEnabled && requestedMerges > 0)
            {
                using (var progress = new TerminalProgressBar("Tokenizer: learning BPE merges", requestedMerges))
                {
                    learnedMerges = LearnBPEMerges(wordFreqs, requestedMerges, minFrequency, progress);

                    progress.Complete($"learned {learnedMerges:N0} merges | vocab {VocabSize:N0}/{vocabSize:N0}");
                }

            }
            else
            {
                learnedMerges = LearnBPEMerges(wordFreqs, requestedMerges, minFrequency, progress: null);
            }

            ClearEncodeCache();

            if (progressEnabled)
            {
                Console.WriteLine($"Tokenizer ready: vocab={VocabSize:N0}, merges={learnedMerges:N0}, texts={texts.Length:N0}, uniqueWords={wordFreqs.Count:N0}");
            }
        }

        private void ResetVocabularyKeepingSpecialTokens()
        {
            var specialTokens = _vocabToId
                .Where(kv => kv.Key.StartsWith("<|", StringComparison.Ordinal))
                .OrderBy(kv => kv.Value)
                .ToList();

            _vocabToId.Clear();
            _idToVocab.Clear();
            _merges.Clear();
            _mergePriority.Clear();
            ClearEncodeCache();

            foreach (var kv in specialTokens)
            {
                _vocabToId[kv.Key] = kv.Value;
                _idToVocab[kv.Value] = kv.Key;
            }

            PadTokenId = _vocabToId[PAD_TOKEN];
            StartTokenId = _vocabToId[START_TOKEN];
            EndTokenId = _vocabToId[END_TOKEN];
            UnkTokenId = _vocabToId[UNK_TOKEN];
        }

        private Dictionary<string, int> GetWordFrequencies(string[] texts)
        {
            return _accel.GetWordFrequencies(texts, LowerCase);
        }

        private HashSet<string> BuildCharacterVocabulary(Dictionary<string, int> wordFreqs)
        {
            return _accel.BuildCharacterVocabulary(wordFreqs);
        }

        private int LearnBPEMerges(Dictionary<string, int> wordFreqs, int numMerges, int minFrequency, TerminalProgressBar progress)
        {
            if (numMerges <= 0)
            {
                return 0;
            }

            var words = wordFreqs.ToDictionary(
                kv => kv.Key.Split(' ').ToList(),
                kv => kv.Value, new ListEqualityComparer<string>());

            int learned = 0;

            for (int mergeIdx = 0; mergeIdx < numMerges; mergeIdx++)
            {
                progress?.Report(mergeIdx, $"merge {mergeIdx + 1:N0}/{numMerges:N0} | vocab {VocabSize:N0}");

                var pairCounts = _accel.CountPairFrequencies(words);
                var (bestPair, frequency) = _accel.SelectBestPair(pairCounts, minFrequency);

                if (frequency == 0)
                {
                    progress?.Report(mergeIdx, $"stopped early: no pair frequency >= {minFrequency:N0}");

                    break;
                }

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
                learned++;

                if (mergeIdx == 0 || mergeIdx % 25 == 0 || mergeIdx == numMerges - 1)
                {
                    progress?.Report(mergeIdx + 1, $"freq {frequency:N0} | '{TrimTokenForStatus(left)}' + '{TrimTokenForStatus(right)}' -> '{TrimTokenForStatus(merged)}'");
                }
                else
                {
                    progress?.Increment();
                }
            }

            return learned;
        }

        public int[] Encode(string text, bool addSpecialTokens = true)
        {
            if (string.IsNullOrEmpty(text))
            {
                return addSpecialTokens ? new[] { StartTokenId, EndTokenId }  : Array.Empty<int>();
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

                if (TryGetCachedWord(word, out var cachedTokens))
                {
                    tokens.AddRange(cachedTokens);
                    continue;
                }

                var wordTokens = EncodeWord(word);
                tokens.AddRange(wordTokens);
                AddWordToCache(word, wordTokens);
            }

            if (addSpecialTokens)
            {
                tokens.Add(EndTokenId);
            }

            return tokens.ToArray();
        }

        private bool TryGetCachedWord(string word, out List<int> tokenIds)
        {
            lock (_encodeCacheLock)
            {
                if (_encodeCache.TryGetValue(word, out var cached))
                {
                    tokenIds = cached;
                    return true;
                }
            }

            tokenIds = null;
            return false;
        }

        private void AddWordToCache(string word, List<int> tokenIds)
        {
            if (string.IsNullOrEmpty(word) || tokenIds == null)
            {
                return;
            }

            lock (_encodeCacheLock)
            {
                if (_encodeCache.Count >= MAX_CACHE_SIZE || _encodeCache.ContainsKey(word))
                {
                    return;
                }

                _encodeCache[word] = tokenIds;
            }
        }

        private void ClearEncodeCache()
        {
            lock (_encodeCacheLock)
            {
                _encodeCache.Clear();
            }
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
            if (tokenIds == null || tokenIds.Length == 0)
            {
                return string.Empty;
            }

            return _accel.DecodeTokens(tokenIds, _idToVocab, UNK_TOKEN, skipSpecialTokens);
        }

        /// <summary>
        /// Encodes a batch of texts.
        /// Set showProgress=true for large one-off batches.
        /// Set parallel=true only for CPU/MultiThreadCPU. GPU/CUDA paths are kept sequential.
        /// </summary>
        public int[][] EncodeBatch(string[] texts, bool addSpecialTokens = true, int? maxLength = null, bool parallel = false, bool showProgress = false)
        {
            if (texts == null)
            {
                throw new ArgumentNullException(nameof(texts));
            }

            var encoded = new int[texts.Length][];

            if (texts.Length == 0)
            {
                return encoded;
            }

            bool canParallelize =
                parallel
                && texts.Length >= 64
                && (_accelerationType == AccelerationType.CPU
                    || _accelerationType == AccelerationType.MultiThreadCPU);

            TerminalProgressBar progress = null;

            try
            {
                if (showProgress)
                {
                    progress = new TerminalProgressBar(
                        canParallelize
                            ? "Tokenizer: encoding batch parallel"
                            : "Tokenizer: encoding batch",
                        texts.Length);
                }

                if (canParallelize)
                {
                    var options = new ParallelOptions
                    {
                        MaxDegreeOfParallelism = Math.Max(1, Environment.ProcessorCount - 1)
                    };

                    Parallel.For(
                        0,
                        texts.Length,
                        options,
                        i =>
                        {
                            encoded[i] = EncodeAndPad(texts[i], addSpecialTokens, maxLength);
                            progress?.Increment(1, $"encoded {i + 1:N0}/{texts.Length:N0}");
                        });
                }
                else
                {
                    for (int i = 0; i < texts.Length; i++)
                    {
                        encoded[i] = EncodeAndPad(texts[i], addSpecialTokens, maxLength);

                        if (showProgress && (i == 0 || i % 50 == 0 || i == texts.Length - 1))
                        {
                            progress?.Report(i + 1, $"encoded {i + 1:N0}/{texts.Length:N0}");
                        }
                        else
                        {
                            progress?.Increment();
                        }
                    }
                }

                progress?.Complete($"{texts.Length:N0} texts encoded");
                return encoded;
            }
            finally
            {
                progress?.Dispose();
            }
        }

        private int[] EncodeAndPad(string text, bool addSpecialTokens, int? maxLength)
        {
            var result = Encode(text, addSpecialTokens);

            if (maxLength.HasValue)
            {
                result = _accel.PadOrTruncate(result, maxLength.Value, addSpecialTokens, PadTokenId, EndTokenId);
            }

            return result;
        }

        #region Save/Load

        public void Save(string directory)
        {
            if (string.IsNullOrWhiteSpace(directory))
            {
                throw new ArgumentException("Tokenizer save directory is required.", nameof(directory));
            }

            Directory.CreateDirectory(directory);

            using (var progress = TerminalProgressBar.StartIndeterminate(
                "Tokenizer: saving",
                directory))
            {
                var vocabPath = Path.Combine(directory, "vocab.json");
                var vocabJson = JsonSerializer.Serialize(
                    _vocabToId,
                    new JsonSerializerOptions { WriteIndented = true });

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

                var configJson = JsonSerializer.Serialize(
                    config,
                    new JsonSerializerOptions { WriteIndented = true });

                File.WriteAllText(configPath, configJson);

                progress.Complete($"vocab {VocabSize:N0} | merges {_merges.Count:N0}");
            }
        }

        public static BPETokenizer Load(string directory, AccelerationType accelerationType, int AccelerationDeviceId = 0)
        {
            if (string.IsNullOrWhiteSpace(directory))
            {
                throw new ArgumentException("Tokenizer directory is required.", nameof(directory));
            }

            if (!Directory.Exists(directory))
            {
                throw new DirectoryNotFoundException($"Tokenizer directory not found: {directory}");
            }

            var tokenizer = new BPETokenizer();

            var runtime = new RuntimeConfig
            {
                AccelerationType = accelerationType,
                AccelerationDeviceId = AccelerationDeviceId
            };

            tokenizer.Init(runtime);

            using (var progress = TerminalProgressBar.StartIndeterminate("Tokenizer: loading", directory))
            {
                var vocabPath = Path.Combine(directory, "vocab.json");

                if (!File.Exists(vocabPath))
                {
                    throw new FileNotFoundException("Tokenizer vocab.json was not found.", vocabPath);
                }

                var vocabJson = File.ReadAllText(vocabPath);

                tokenizer._vocabToId = JsonSerializer.Deserialize<Dictionary<string, int>>(vocabJson) ?? throw new InvalidOperationException("Failed to deserialize tokenizer vocab.");

                tokenizer._idToVocab = tokenizer._vocabToId.ToDictionary(kv => kv.Value, kv => kv.Key);

                tokenizer.PadTokenId = tokenizer._vocabToId[PAD_TOKEN];
                tokenizer.StartTokenId = tokenizer._vocabToId[START_TOKEN];
                tokenizer.EndTokenId = tokenizer._vocabToId[END_TOKEN];
                tokenizer.UnkTokenId = tokenizer._vocabToId[UNK_TOKEN];

                tokenizer._merges = new List<(string, string)>();
                tokenizer._mergePriority = new Dictionary<(string, string), int>();
                tokenizer._encodeCache = new Dictionary<string, List<int>>();

                var mergesPath = Path.Combine(directory, "merges.txt");

                if (!File.Exists(mergesPath))
                {
                    throw new FileNotFoundException("Tokenizer merges.txt was not found.", mergesPath);
                }

                var lines = File.ReadAllLines(mergesPath);

                int priority = 0;

                foreach (var line in lines)
                {
                    if (line.StartsWith("#", StringComparison.Ordinal) || string.IsNullOrWhiteSpace(line))
                    {
                        continue;
                    }

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
                        {
                            tokenizer.MaxTokenLength = mtl.GetInt32();
                        }

                        if (config.TryGetValue("lowercase", out var lc))
                        {
                            tokenizer.LowerCase = lc.GetBoolean();
                        }
                    }
                }

                progress.Complete($"vocab {tokenizer.VocabSize:N0} | merges {tokenizer._merges.Count:N0}");
            }

            return tokenizer;
        }

        #endregion

        public string IdToToken(int id)
        {
            return _idToVocab.TryGetValue(id, out var token)
                ? token
                : UNK_TOKEN;
        }

        public int TokenToId(string token)
        {
            return _vocabToId.TryGetValue(token, out var id)
                ? id
                : UnkTokenId;
        }

        public List<string> GetVocabulary()
        {
            return _idToVocab
                .OrderBy(kv => kv.Key)
                .Select(kv => kv.Value)
                .ToList();
        }

        private static string TrimTokenForStatus(string token)
        {
            if (string.IsNullOrEmpty(token))
            {
                return string.Empty;
            }

            const int max = 18;

            token = token
                .Replace("\r", "\\r")
                .Replace("\n", "\\n")
                .Replace("\t", "\\t");

            return token.Length <= max
                ? token
                : token.Substring(0, max - 1) + "…";
        }
    }
}