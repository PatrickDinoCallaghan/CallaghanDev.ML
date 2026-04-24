using CallaghanDev.ML.AccelerationManagers;
using CallaghanDev.ML.AccelerationManagers.GPU;
using CallaghanDev.ML.Enums;
using CallaghanDev.ML.Extensions;
using CallaghanDev.ML.Transformers.Cache;
using CallaghanDev.ML.Transformers.Configuration;
using CallaghanDev.ML.Transformers.CrossAttentionMultimodal;
using CallaghanDev.ML.Transformers.MultiTypeTransformer;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text.Json;
using TransformerBlock = CallaghanDev.ML.Transformers.MultiTypeTransformer.TransformerBlock;

namespace CallaghanDev.ML.Transformers.TACAMT
{
    public class Model
    {
        private const int NewsMemoryFormatVersionWithTokenIds = 2;
        private const int NewsMemoryVersionMarkerWithTokenIds = -NewsMemoryFormatVersionWithTokenIds;

        /// <summary>
        /// Learned type embedding for context entries in cross-attention.
        /// Shape: [2, PriceEmbeddingDim]
        /// Index 0 = news context, Index 1 = price memory context.
        /// Added to each context hidden state before it enters the price decoder's
        /// cross-attention and decay network, giving the model an explicit signal
        /// to learn different decay/attention dynamics per source type.
        /// </summary>
        public float[,] ContextTypeEmbedding { get; set; }


        private readonly MultimodalTransformerConfig _config;
        private readonly Random _random;
        private readonly IAccelerationManager _accel;

        public MultimodalTransformerConfig Config => _config;
        public IAccelerationManager AccelerationManager => _accel;

        public float[,] TextTokenEmbedding { get; set; }
        public TransformerBlock[] TextBlocks { get; set; }

        public float[,] PriceInputProjection { get; set; }
        public float[] PriceInputProjectionBias { get; set; }
        public TacamtBlock[] PriceBlocks { get; set; }

        public float[,] OutputProjection { get; set; }
        public float[] OutputBias { get; set; }
        public float[,] ConfidenceProjection { get; set; }
        public float[] ConfidenceBias { get; set; }

        public List<NewsMemoryEntry> NewsMemory { get; set; } = new List<NewsMemoryEntry>();
        public List<PriceMemoryEntry> PriceMemory { get; set; } = new List<PriceMemoryEntry>();
        public double LastPriceTimestamp { get; set; } = 0;

        /// <summary>
        /// Optional tokenizer for encoding raw text into token IDs.
        /// Must match _config.TextVocabSize. Set via SetTokenizer() or loaded from disk.
        /// </summary>
        public BPETokenizer Tokenizer { get; private set; }

        /// <summary>
        /// Configuration for attention-score-based memory pruning.
        /// </summary>
        public MemoryPruningConfig PruningConfig { get; set; } = new MemoryPruningConfig();

        public Model(MultimodalTransformerConfig config, Random random = null)
        {
            config.Validate();
            _config = config;
            _random = random ?? new Random();
            if (_config.Runtime.AccelerationType == AccelerationType.GPU || _config.Runtime.AccelerationType == AccelerationType.CUDA)
            {
                _accel = new AccelerationGPU(_config.Runtime.AccelerationType, _config.Runtime.AccelerationDeviceId);
            }
            else if (_config.Runtime.AccelerationType == AccelerationType.CPU)
            {
                _accel = new AccelerationCPU();
            }
            else if (_config.Runtime.AccelerationType == AccelerationType.MultiThreadCPU)
            {
                _accel = new AccelerationMutliThreadCPU();
            }
            else
            {
                throw new Exception("Unsupported AccelerationType");
            }

            InitTextEncoder();
            InitPriceDecoder();
            InitOutputHead();

            int ed = _config.Price.EmbeddingDim;
            ContextTypeEmbedding = new float[2, ed];
            float typeStd = MathF.Sqrt(2.0f / ed) * 0.1f;

            for (int t = 0; t < 2; t++)
            {
                for (int d = 0; d < ed; d++)
                {
                    ContextTypeEmbedding[t, d] = SampleGaussian() * typeStd;
                }
            }
        }

        /// <summary>
        /// Attach a trained BPE tokenizer. Validates that vocab size does not exceed config.
        /// A tokenizer with fewer tokens than TextVocabSize is allowed - the embedding
        /// table is pre-allocated at TextVocabSize but only trained tokens get meaningful
        /// representations. A tokenizer larger than TextVocabSize is rejected because
        /// token IDs would index out of bounds in the embedding lookup.
        /// </summary>
        public void SetTokenizer(BPETokenizer tokenizer)
        {
            if (tokenizer.VocabSize > _config.Text.VocabSize)
            {
                throw new ArgumentException($"Tokenizer vocab size ({tokenizer.VocabSize}) exceeds config TextVocabSize ({_config.Text.VocabSize}). Either increase TextVocabSize in the config or reduce the tokenizer target vocab size.");
            }
            Tokenizer = tokenizer;
        }

        /// <summary>
        /// Train a BPE tokenizer from raw article texts and attach it to the model.
        /// Must be called before TokenizeStories() can be used.
        /// Safe to call on a freshly created model or after Load() if no tokenizer was saved.
        /// </summary>
        /// <param name="texts">Raw news article strings to train on.</param>
        /// <param name="minFrequency">Minimum pair frequency for BPE merges. Default: 2.</param>
        public void TrainTokenizer(string[] texts, int minFrequency = 10)
        {
            if (texts == null || texts.Length == 0)
            {
                throw new ArgumentException("Cannot train tokenizer on empty text corpus.");
            }

            var tokenizer = new BPETokenizer();
            tokenizer.Train(texts, _config.Text.VocabSize, minFrequency);

            // BPETokenizer.Train may produce fewer tokens than requested if the corpus is small.
            // That is fine - the embedding table has room for TextVocabSize entries but only
            // the trained subset will be used. We bypass SetTokenizer's upper-bound check
            // since we know the tokenizer was trained with TextVocabSize as the target.
            Tokenizer = tokenizer;

        }

        /// <summary>
        /// Convenience: encode raw text strings into NewsStory objects using the attached tokenizer.
        /// Throws if no tokenizer is set.
        /// </summary>
        public NewsStory[] TokenizeStories(string[] texts, float[] arrivalTimes)
        {
            if (Tokenizer == null)
            {
                throw new InvalidOperationException("No tokenizer attached. Call SetTokenizer() first or use Load() with a saved tokenizer directory.");
            }
            if (texts.Length != arrivalTimes.Length)
            {
                throw new ArgumentException("texts and arrivalTimes must have the same length.");
            }

            var stories = new NewsStory[texts.Length];
            for (int i = 0; i < texts.Length; i++)
            {
                stories[i] = new NewsStory(Tokenizer.Encode(texts[i]), arrivalTimes[i]);
            }
            return stories;
        }

        public (float[,] predictions, float[,] confidence) Forward(NewsStory[] stories, float[,] priceSequence)
        {
            float[,] sh = null; float[] st = null;

            if (stories != null && stories.Length > 0)
            {
                (sh, st) = EncodeStories(stories);
            }

            if (sh != null)
            {
                int numNews = sh.GetLength(0);
                int ed = _config.Price.EmbeddingDim;
                var typeIndices = new int[numNews];
                _accel.ApplyContextTypeEmbedding(sh, ContextTypeEmbedding, typeIndices);
            }

            return ProjectToOutput(ForwardPriceDecoder(priceSequence, sh, st));
        }

        public (float[,] predictions, float[,] confidence) ForwardWithCache(NewsStory[] stories, float[,] priceSequence, int rowStart, int rowCount, MultimodalForwardCache cache, bool isTraining = false, Random dropoutRng = null)
        {
            if (priceSequence == null)
                throw new ArgumentNullException(nameof(priceSequence));

            if (cache == null)
                throw new ArgumentNullException(nameof(cache));

            if (rowStart < 0)
                throw new ArgumentOutOfRangeException(nameof(rowStart));

            if (rowCount <= 0)
                throw new ArgumentOutOfRangeException(nameof(rowCount));

            if (rowStart + rowCount > priceSequence.GetLength(0))
                throw new ArgumentException("Invalid slice bounds.");

            cache.NumNewsContext = 0;
            cache.NumPriceContext = 0;
            cache.PriceContextHidden = null;

            float[,] storyHidden = null;
            float[] storyTimes = null;

            if (stories != null && stories.Length > 0)
            {
                (storyHidden, storyTimes) = EncodeStoriesWithCache(stories, cache);

                // story ArrivalTime is relative to the full priceSequence row 0.
                // If we decode a sliced window starting at rowStart, shift key times
                // into the sliced decoder coordinate system.
                if (rowStart != 0 && storyTimes != null)
                {
                    var shiftedTimes = new float[storyTimes.Length];
                    for (int i = 0; i < storyTimes.Length; i++)
                        shiftedTimes[i] = storyTimes[i] - rowStart;

                    storyTimes = shiftedTimes;
                }
            }
            else
            {
                cache.TextFinalHidden = null;
                cache.TextTokenIds = null;
                cache.StoryCaches = new List<MultimodalForwardCache>();
                cache.StoryTokenCounts = Array.Empty<int>();
            }

            cache.StoryArrivalTimes = storyTimes;

            if (storyHidden != null && storyHidden.GetLength(0) > 0)
            {
                int numNews = storyHidden.GetLength(0);
                var typeIndices = new int[numNews]; // all 0 = news

                _accel.ApplyContextTypeEmbedding(storyHidden, ContextTypeEmbedding, typeIndices);

                cache.TextFinalHidden = storyHidden;
                cache.NumNewsContext = numNews;
                cache.NumPriceContext = 0;
            }
            else
            {
                storyHidden = null;
                storyTimes = null;
            }

            var priceHidden = ForwardPriceDecoderWithCache(
                priceSequence,
                rowStart,
                rowCount,
                storyHidden,
                storyTimes,
                cache,
                isTraining,
                dropoutRng);

            cache.PriceFinalHidden = priceHidden;

            return ProjectToOutput(priceHidden);
        }
        public (float[,] predictions, float[,] confidence) ForwardWithCache(NewsStory[] stories, float[,] priceSequence, MultimodalForwardCache cache, bool isTraining = false, Random dropoutRng = null)
        {
            int sl = priceSequence.GetLength(0);

            return ForwardWithCache(stories, priceSequence, rowStart: 0, rowCount: sl, cache, isTraining, dropoutRng);
        }
        /*
      


        public (float[,] predictions, float[,] confidence) ForwardWithCache(NewsStory[] stories, float[,] priceSequence, int rowStart, int rowCount, MultimodalForwardCache cache, bool isTraining = false, Random dropoutRng = null)
        {
            // --- Correctness guards ---
            if (priceSequence == null)
            {
                throw new ArgumentNullException(nameof(priceSequence));
            }

            if (rowStart < 0 || rowCount < 0)
            {
                throw new ArgumentOutOfRangeException();

            }
            if (rowStart + rowCount > priceSequence.GetLength(0))
            {
                throw new ArgumentException("Invalid slice bounds.");
            }

            float[,] sh = null;
            float[] st = null;

            if (stories != null && stories.Length > 0)
            {
                (sh, st) = EncodeStoriesWithCache(stories, cache);
            }
            else
            {
                cache.TextFinalHidden = null;
                cache.TextTokenIds = null;
            }

            cache.StoryArrivalTimes = st;

            // Apply context type embedding (identical logic)
            if (sh != null)
            {
                int numNews = sh.GetLength(0);

                // SPEEDUP: reuse buffer instead of allocating each call (optional)
                var typeIndices = new int[numNews];  // still safe, tiny cost

                _accel.ApplyContextTypeEmbedding(sh, ContextTypeEmbedding, typeIndices);
            }

            var ph = ForwardPriceDecoderWithCache(
                priceSequence,
                rowStart,
                rowCount,
                sh,
                st,
                cache,
                isTraining,
                dropoutRng);

            cache.PriceFinalHidden = ph;

            return ProjectToOutput(ph);
        }
        */

        public (float[] prediction, float confidence) PredictNext(NewsStory[] stories, float[,] priceSequence)
        {
            var (pred, conf) = Forward(stories, priceSequence);
            int last = pred.GetLength(0) - 1;
            var p = new float[_config.Output.OutputDim];

            for (int j = 0; j < _config.Output.OutputDim; j++)
            {
                p[j] = pred[last, j];
            }
            return (p, _config.Output.UseConfidenceHead ? conf[last, 0] : 1.0f);
        }

        public (float[], float) PredictNext(int[] t, float[,] p)
        {
            return t != null && t.Length > 0 ? PredictNext(new[] { new NewsStory(t, 0f) }, p) : PredictNext((NewsStory[])null, p);
        }


        /// <summary>
        /// Convenience: predict from raw text strings using the attached tokenizer.
        /// </summary>
        public (float[] prediction, float confidence) PredictNext(string[] texts, float[] arrivalTimes, float[,] priceSequence)
        {
            var stories = TokenizeStories(texts, arrivalTimes);
            return PredictNext(stories, priceSequence);
        }

        public (float[] prediction, float confidence) PredictWithMemory(
            NewsStory[] newStories,
            float[,] priceSequence,
            double currentAbsoluteTimestamp,
            double timeUnitsPerPosition = 1.0,
            int maxNewsMemorySize = 100,
            int maxPriceMemorySize = 200)
        {
            if (priceSequence == null)
                throw new ArgumentNullException(nameof(priceSequence));

            if (priceSequence.GetLength(0) <= 0)
                throw new ArgumentException("priceSequence must contain at least one row.", nameof(priceSequence));

            if (timeUnitsPerPosition <= 0.0)
                throw new ArgumentOutOfRangeException(nameof(timeUnitsPerPosition));

            int embeddingDim = _config.Price.EmbeddingDim;
            float invTime = (float)(1.0 / timeUnitsPerPosition);

            RemoveInvalidMemoryEntries(embeddingDim);

            PruneNewsMemory(maxNewsMemorySize);
            PricePruneMemory(maxPriceMemorySize);

            NewsMemory = (NewsMemory ?? new List<NewsMemoryEntry>())
                .OrderBy(e => e.AbsoluteTimestamp)
                .ToList();

            PriceMemory = (PriceMemory ?? new List<PriceMemoryEntry>())
                .OrderBy(e => e.AbsoluteTimestamp)
                .ToList();

            int storedNewsCount = NewsMemory.Count;
            int storedPriceCount = PriceMemory.Count;

            NewsStory[] validNewStories = null;
            float[,] freshNewsHidden = null;
            float[] freshNewsTimes = null;
            int freshNewsCount = 0;

            if (newStories != null && newStories.Length > 0)
            {
                validNewStories = newStories
                    .Where(story => story?.TokenIds != null && story.TokenIds.Length > 0)
                    .Select(story => new NewsStory((int[])story.TokenIds.Clone(), story.ArrivalTime))
                    .ToArray();

                if (validNewStories.Length > 0)
                {
                    (freshNewsHidden, freshNewsTimes) = EncodeStories(validNewStories);
                    freshNewsCount = freshNewsHidden.GetLength(0);

                    if (freshNewsHidden.GetLength(1) != embeddingDim)
                    {
                        throw new InvalidOperationException(
                            $"Encoded story dim ({freshNewsHidden.GetLength(1)}) does not match price embedding dim ({embeddingDim}).");
                    }
                }
            }

            int totalContext = storedNewsCount + freshNewsCount + storedPriceCount;

            float[,] combinedHidden = null;
            float[] combinedTimes = null;

            if (totalContext > 0)
            {
                combinedHidden = new float[totalContext, embeddingDim];
                combinedTimes = new float[totalContext];

                var typeIndices = new int[totalContext];
                int ctx = 0;

                for (int i = 0; i < storedNewsCount; i++, ctx++)
                {
                    var entry = NewsMemory[i];

                    CopyHiddenRow(entry.HiddenState, combinedHidden, ctx, embeddingDim);

                    combinedTimes[ctx] = (float)((entry.AbsoluteTimestamp - currentAbsoluteTimestamp) * invTime);
                    typeIndices[ctx] = 0;
                }

                for (int i = 0; i < freshNewsCount; i++, ctx++)
                {
                    for (int d = 0; d < embeddingDim; d++)
                        combinedHidden[ctx, d] = freshNewsHidden[i, d];

                    combinedTimes[ctx] = freshNewsTimes[i];
                    typeIndices[ctx] = 0;
                }

                for (int i = 0; i < storedPriceCount; i++, ctx++)
                {
                    var entry = PriceMemory[i];

                    CopyHiddenRow(entry.HiddenState, combinedHidden, ctx, embeddingDim);

                    combinedTimes[ctx] = (float)((entry.AbsoluteTimestamp - currentAbsoluteTimestamp) * invTime);
                    typeIndices[ctx] = 1;
                }

                _accel.ApplyContextTypeEmbedding(combinedHidden, ContextTypeEmbedding, typeIndices);
            }

            var cache = new MultimodalForwardCache(_config.Text.NumLayers, _config.Price.NumLayers)
            {
                NumNewsContext = storedNewsCount + freshNewsCount,
                NumPriceContext = storedPriceCount,
                TextFinalHidden = combinedHidden,
                StoryArrivalTimes = combinedTimes,
                PriceContextHidden = null
            };

            float[,] priceHidden = ForwardPriceDecoderWithCache(
                priceSequence,
                rowStart: 0,
                rowCount: priceSequence.GetLength(0),
                storyHidden: combinedHidden,
                storyTimes: combinedTimes,
                cache: cache,
                isTraining: false,
                dropoutRng: null);

            cache.PriceFinalHidden = priceHidden;

            var (predictions, confidences) = ProjectToOutput(priceHidden);

            int last = predictions.GetLength(0) - 1;
            var prediction = new float[_config.Output.OutputDim];

            for (int j = 0; j < _config.Output.OutputDim; j++)
                prediction[j] = predictions[last, j];

            float confidence =
                _config.Output.UseConfidenceHead && confidences != null
                    ? confidences[last, 0]
                    : 1.0f;

            if (combinedHidden != null)
            {
                bool updatedFromActualAttention = UpdateMemoryAttentionScoresFromCache(
                    cache,
                    storedNewsCount,
                    storedPriceCount);

                if (!updatedFromActualAttention)
                {
                    UpdateMemoryAttentionScores(
                        priceHidden,
                        combinedHidden,
                        storedNewsCount,
                        storedPriceCount);
                }
            }

            if (validNewStories != null && validNewStories.Length > 0)
                UpdateNewsMemory(validNewStories, currentAbsoluteTimestamp, timeUnitsPerPosition, maxNewsMemorySize);

            if (maxPriceMemorySize > 0)
            {
                int priceRows = priceHidden.GetLength(0);

                for (int t = 0; t < priceRows; t++)
                {
                    var hidden = new float[embeddingDim];

                    for (int d = 0; d < embeddingDim; d++)
                        hidden[d] = priceHidden[t, d];

                    PriceMemory.Add(new PriceMemoryEntry
                    {
                        HiddenState = hidden,
                        AbsoluteTimestamp = currentAbsoluteTimestamp + t * timeUnitsPerPosition
                    });
                }

                PricePruneMemory(maxPriceMemorySize);
            }

            LastPriceTimestamp =
                currentAbsoluteTimestamp +
                Math.Max(0, priceSequence.GetLength(0) - 1) * timeUnitsPerPosition;

            PruneNewsMemory(maxNewsMemorySize);

            return (prediction, confidence);
        }

        private void RemoveInvalidMemoryEntries(int embeddingDim)
        {
            if (NewsMemory != null)
            {
                NewsMemory = NewsMemory
                    .Where(e => e != null && HasValidHidden(e.HiddenState, embeddingDim))
                    .OrderBy(e => e.AbsoluteTimestamp)
                    .ToList();
            }

            if (PriceMemory != null)
            {
                PriceMemory = PriceMemory
                    .Where(e => e != null && HasValidHidden(e.HiddenState, embeddingDim))
                    .OrderBy(e => e.AbsoluteTimestamp)
                    .ToList();
            }
        }

        private static bool HasValidHidden(float[] hidden, int embeddingDim)
        {
            return hidden != null && hidden.Length == embeddingDim;
        }

        private static void CopyHiddenRow(float[] source, float[,] destination, int row, int embeddingDim)
        {
            if (!HasValidHidden(source, embeddingDim))
                throw new ArgumentException("Memory hidden state has an invalid shape.", nameof(source));

            for (int d = 0; d < embeddingDim; d++)
                destination[row, d] = source[d];
        }

        public void UpdateNewsMemory(NewsStory[] stories, double ts, double tpp = 1.0, int max = 100)
        {
            if (stories == null || stories.Length == 0)
                return;

            if (tpp <= 0.0)
                throw new ArgumentOutOfRangeException(nameof(tpp));
            int embDim = _config.Price.EmbeddingDim;

            var (encodedStories, _) = EncodeStories(stories);

            if (encodedStories.GetLength(1) != embDim)
            {
                throw new InvalidOperationException(
                    $"Encoded story dim ({encodedStories.GetLength(1)}) does not match price embedding dim ({embDim}).");
            }

            for (int i = 0; i < stories.Length; i++)
            {
                if (stories[i]?.TokenIds == null || stories[i].TokenIds.Length == 0)
                    throw new ArgumentException($"Story {i} has no token IDs.", nameof(stories));

                var hidden = new float[embDim];

                for (int d = 0; d < embDim; d++)
                    hidden[d] = encodedStories[i, d];

                NewsMemory.Add(new NewsMemoryEntry
                {
                    TokenIds = (int[])stories[i].TokenIds.Clone(),
                    HiddenState = hidden,
                    AbsoluteTimestamp = ts + stories[i].ArrivalTime * tpp
                });
            }

            PruneNewsMemory(max);
        }
        /// <summary>
        /// Updates the running attention scores for all memory entries by computing
        /// approximate attention weights from the last price position to each context entry.
        /// This uses a simplified dot-product similarity (without full multi-head attention)
        /// to avoid the cost of a full forward pass, but captures the essential signal:
        /// entries that the model's queries are similar to get high scores.
        /// </summary>
        internal void UpdateMemoryAttentionScores(float[,] priceHidden, float[,] contextHidden, int totalCtx)
        {
            int newsRows = Math.Min(NewsMemory?.Count ?? 0, Math.Max(0, totalCtx));
            int priceRows = Math.Min(PriceMemory?.Count ?? 0, Math.Max(0, totalCtx - newsRows));

            UpdateMemoryAttentionScores(priceHidden, contextHidden, newsRows, priceRows);
        }

        /// <summary>
        /// Updates running memory attention scores using a cheap fallback similarity score.
        /// The first newsMemoryRows context rows must be stored news memory; the last
        /// priceMemoryRows context rows must be stored price memory. Rows in between
        /// (for example fresh news in the current call) are not mapped to persisted memory.
        /// Prefer UpdateMemoryAttentionScoresFromCache when a forward cache is available.
        /// </summary>
        internal void UpdateMemoryAttentionScores(
            float[,] priceHidden,
            float[,] contextHidden,
            int newsMemoryRows,
            int priceMemoryRows)
        {
            if (priceHidden == null || contextHidden == null)
                return;

            int totalCtx = contextHidden.GetLength(0);

            if (priceHidden.GetLength(0) <= 0 || totalCtx <= 0)
                return;

            int embeddingDim = _config.Price.EmbeddingDim;
            int lastPos = priceHidden.GetLength(0) - 1;
            float scale = 1.0f / MathF.Sqrt(embeddingDim);

            var scores = _accel.ComputeMemoryAttentionScores(
                priceHidden,
                lastPos,
                contextHidden,
                totalCtx,
                scale);

            NormalizeScoresInPlace(scores, totalCtx, preserveNonNegativeWeights: false);
            ApplyMemoryAttentionScores(scores, totalCtx, newsMemoryRows, priceMemoryRows);
        }

        /// <summary>
        /// Updates memory scores from the actual cross-attention weights captured during
        /// the forward pass. Returns false when the backend/cache did not provide usable
        /// cross-attention weights, so callers can fall back to approximate similarity.
        /// </summary>
        internal bool UpdateMemoryAttentionScoresFromCache(
            MultimodalForwardCache cache,
            int newsMemoryRows,
            int priceMemoryRows)
        {
            if (cache?.TextFinalHidden == null || cache.PriceBlockCaches == null)
                return false;

            int totalCtx = cache.TextFinalHidden.GetLength(0);

            if (totalCtx <= 0)
                return false;

            var scores = new float[totalCtx];
            int contributors = 0;

            foreach (var blockCache in cache.PriceBlockCaches)
            {
                var weightsByHead = blockCache?.CrossAttentionWeights;

                if (weightsByHead == null)
                    continue;

                foreach (var weights in weightsByHead)
                {
                    if (weights == null)
                        continue;

                    int queryRows = weights.GetLength(0);
                    int keyRows = weights.GetLength(1);

                    if (queryRows <= 0 || keyRows <= 0)
                        continue;

                    int queryRow = queryRows - 1;
                    int usableKeys = Math.Min(totalCtx, keyRows);

                    for (int k = 0; k < usableKeys; k++)
                        scores[k] += weights[queryRow, k];

                    contributors++;
                }
            }

            if (contributors <= 0)
                return false;

            float inv = 1.0f / contributors;

            for (int i = 0; i < totalCtx; i++)
                scores[i] *= inv;

            NormalizeScoresInPlace(scores, totalCtx, preserveNonNegativeWeights: true);
            ApplyMemoryAttentionScores(scores, totalCtx, newsMemoryRows, priceMemoryRows);

            return true;
        }

        private void ApplyMemoryAttentionScores(
            float[] scores,
            int totalCtx,
            int newsMemoryRows,
            int priceMemoryRows)
        {
            if (scores == null || totalCtx <= 0)
                return;

            float alpha = PruningConfig?.AttentionScoreAlpha ?? 1.0f;

            int newsRows = Math.Clamp(
                newsMemoryRows,
                0,
                Math.Min(NewsMemory?.Count ?? 0, totalCtx));

            for (int i = 0; i < newsRows; i++)
            {
                var entry = NewsMemory[i];
                entry.AttentionScore = alpha * scores[i] + (1f - alpha) * entry.AttentionScore;
                entry.QueryCount++;
            }

            int priceRows = Math.Clamp(
                priceMemoryRows,
                0,
                Math.Min(PriceMemory?.Count ?? 0, totalCtx));

            int priceOffset = totalCtx - priceRows;

            for (int i = 0; i < priceRows; i++)
            {
                int ctxIdx = priceOffset + i;

                if (ctxIdx < 0 || ctxIdx >= totalCtx)
                    continue;

                var entry = PriceMemory[i];
                entry.AttentionScore = alpha * scores[ctxIdx] + (1f - alpha) * entry.AttentionScore;
                entry.QueryCount++;
            }
        }

        private static void NormalizeScoresInPlace(float[] scores, int count, bool preserveNonNegativeWeights)
        {
            if (scores == null || count <= 0)
                return;

            if (preserveNonNegativeWeights)
            {
                bool validWeights = true;
                float weightSum = 0f;

                for (int i = 0; i < count; i++)
                {
                    if (!float.IsFinite(scores[i]) || scores[i] < 0f)
                    {
                        validWeights = false;
                        break;
                    }

                    weightSum += scores[i];
                }

                if (validWeights && weightSum > 0f && float.IsFinite(weightSum))
                {
                    float inv = 1.0f / weightSum;

                    for (int i = 0; i < count; i++)
                        scores[i] *= inv;

                    return;
                }
            }

            float maxScore = float.NegativeInfinity;

            for (int i = 0; i < count; i++)
            {
                if (scores[i] > maxScore)
                    maxScore = scores[i];
            }

            if (!float.IsFinite(maxScore))
            {
                float uniform = 1.0f / count;

                for (int i = 0; i < count; i++)
                    scores[i] = uniform;

                return;
            }

            float sumExp = 0f;

            for (int i = 0; i < count; i++)
            {
                scores[i] = MathF.Exp(scores[i] - maxScore);
                sumExp += scores[i];
            }

            if (sumExp > 0f && float.IsFinite(sumExp))
            {
                float inv = 1.0f / sumExp;

                for (int i = 0; i < count; i++)
                    scores[i] *= inv;
            }
            else
            {
                float uniform = 1.0f / count;

                for (int i = 0; i < count; i++)
                    scores[i] = uniform;
            }
        }

        /// <summary>
        /// Prune news memory using attention-based strategy.
        /// Entries with consistently low attention scores are pruned first,
        /// unless they haven't been queried enough times (cold-start protection).
        /// A fraction of slots is reserved for the newest entries to prevent
        /// immediate pruning of fresh information.
        /// </summary>
        internal void PruneNewsMemory(int maxSize)
        {
            if (NewsMemory == null)
                return;

            if (maxSize <= 0)
            {
                NewsMemory.Clear();
                return;
            }

            if (NewsMemory.Count <= maxSize)
                return;

            if (PruningConfig == null || !PruningConfig.UseAttentionBasedPruning)
            {
                NewsMemory = NewsMemory
                    .OrderByDescending(e => e.AbsoluteTimestamp)
                    .Take(maxSize)
                    .OrderBy(e => e.AbsoluteTimestamp)
                    .ToList();

                return;
            }

            float reserveFraction = Math.Clamp(PruningConfig.NewEntryReserveFraction, 0f, 1f);

            int reserveCount = (int)(maxSize * reserveFraction);

            if (reserveFraction > 0f && reserveCount == 0)
                reserveCount = 1;

            reserveCount = Math.Clamp(reserveCount, 0, maxSize);

            int scoreKeepCount = maxSize - reserveCount;

            var byNewest = NewsMemory
                .OrderByDescending(e => e.AbsoluteTimestamp)
                .ToList();

            var reserved = byNewest.Take(reserveCount).ToList();
            var reservedSet = reserved.ToHashSet();

            var candidates = byNewest
                .Where(e => !reservedSet.Contains(e))
                .ToList();

            var keptByScore = candidates
                .OrderByDescending(e =>
                    e.QueryCount >= PruningConfig.MinQueryCountForPruning
                        ? e.AttentionScore
                        : float.MaxValue)
                .ThenByDescending(e => e.AbsoluteTimestamp)
                .Take(scoreKeepCount)
                .ToList();

            NewsMemory = reserved
                .Concat(keptByScore)
                .OrderBy(e => e.AbsoluteTimestamp)
                .ToList();
        }
        /// <summary>
        /// Prune price memory using attention-based strategy. Same logic as news pruning.
        /// </summary>
        internal void PricePruneMemory(int maxSize)
        {
            if (PriceMemory == null)
                return;

            if (maxSize <= 0)
            {
                PriceMemory.Clear();
                return;
            }

            if (PriceMemory.Count <= maxSize)
                return;

            if (PruningConfig == null || !PruningConfig.UseAttentionBasedPruning)
            {
                PriceMemory = PriceMemory
                    .OrderByDescending(e => e.AbsoluteTimestamp)
                    .Take(maxSize)
                    .OrderBy(e => e.AbsoluteTimestamp)
                    .ToList();

                return;
            }

            float reserveFraction = Math.Clamp(PruningConfig.NewEntryReserveFraction, 0f, 1f);

            int reserveCount = (int)(maxSize * reserveFraction);

            if (reserveFraction > 0f && reserveCount == 0)
                reserveCount = 1;

            reserveCount = Math.Clamp(reserveCount, 0, maxSize);

            int scoreKeepCount = maxSize - reserveCount;

            var byNewest = PriceMemory
                .OrderByDescending(e => e.AbsoluteTimestamp)
                .ToList();

            var reserved = byNewest.Take(reserveCount).ToList();
            var reservedSet = reserved.ToHashSet();

            var candidates = byNewest
                .Where(e => !reservedSet.Contains(e))
                .ToList();

            var keptByScore = candidates
                .OrderByDescending(e =>
                    e.QueryCount >= PruningConfig.MinQueryCountForPruning
                        ? e.AttentionScore
                        : float.MaxValue)
                .ThenByDescending(e => e.AbsoluteTimestamp)
                .Take(scoreKeepCount)
                .ToList();

            PriceMemory = reserved
                .Concat(keptByScore)
                .OrderBy(e => e.AbsoluteTimestamp)
                .ToList();
        }


        public void ClearAllMemory()
        {
            NewsMemory.Clear();
            PriceMemory.Clear();
            LastPriceTimestamp = 0;
        }
        public void ClearNewsMemory()
        {
            NewsMemory.Clear();
        }
        public void ClearPriceMemory() { PriceMemory.Clear(); }

        public (float[,], float[]) EncodeStories(NewsStory[] stories)
        {
            if (stories == null)
                throw new ArgumentNullException(nameof(stories));

            int storyCount = stories.Length;
            int embeddingDim = _config.Text.EmbeddingDim;

            var storyHidden = new float[storyCount, embeddingDim];
            var arrivalTimes = new float[storyCount];

            for (int s = 0; s < storyCount; s++)
            {
                if (stories[s]?.TokenIds == null || stories[s].TokenIds.Length == 0)
                    throw new ArgumentException($"Story {s} has no token IDs.", nameof(stories));

                var tokenHidden = ForwardTextEncoder(stories[s].TokenIds);
                int tokenCount = tokenHidden.GetLength(0);

                if (tokenCount <= 0)
                    throw new InvalidOperationException($"Story {s} produced an empty text hidden state.");

                float invTokenCount = 1.0f / tokenCount;

                for (int d = 0; d < embeddingDim; d++)
                {
                    float sum = 0f;

                    for (int t = 0; t < tokenCount; t++)
                        sum += tokenHidden[t, d];

                    storyHidden[s, d] = sum * invTokenCount;
                }

                arrivalTimes[s] = stories[s].ArrivalTime;
            }

            return (storyHidden, arrivalTimes);
        }
        internal (float[,], float[]) EncodeStoriesWithCache(NewsStory[] stories, MultimodalForwardCache cache)
        {
            if (stories == null)
                throw new ArgumentNullException(nameof(stories));

            if (cache == null)
                throw new ArgumentNullException(nameof(cache));

            int storyCount = stories.Length;
            int embeddingDim = _config.Text.EmbeddingDim;

            var storyHidden = new float[storyCount, embeddingDim];
            var arrivalTimes = new float[storyCount];

            cache.StoryCaches = new List<MultimodalForwardCache>(storyCount);
            cache.StoryTokenCounts = new int[storyCount];

            for (int s = 0; s < storyCount; s++)
            {
                if (stories[s]?.TokenIds == null || stories[s].TokenIds.Length == 0)
                    throw new ArgumentException($"Story {s} has no token IDs.", nameof(stories));

                var storyCache = new MultimodalForwardCache(_config.Text.NumLayers, 0);
                var tokenHidden = ForwardTextEncoderWithCache(stories[s].TokenIds, storyCache);

                cache.StoryCaches.Add(storyCache);

                int tokenCount = tokenHidden.GetLength(0);

                if (tokenCount <= 0)
                    throw new InvalidOperationException($"Story {s} produced an empty text hidden state.");

                cache.StoryTokenCounts[s] = tokenCount;

                float invTokenCount = 1.0f / tokenCount;

                for (int d = 0; d < embeddingDim; d++)
                {
                    float sum = 0f;

                    for (int t = 0; t < tokenCount; t++)
                        sum += tokenHidden[t, d];

                    storyHidden[s, d] = sum * invTokenCount;
                }

                arrivalTimes[s] = stories[s].ArrivalTime;
            }

            cache.TextFinalHidden = storyHidden;
            cache.TextTokenIds = stories.SelectMany(s => s.TokenIds).ToArray();
            cache.NumNewsContext = storyCount;
            cache.NumPriceContext = 0;
            cache.PriceContextHidden = null;

            return (storyHidden, arrivalTimes);
        }
        private float[,] ForwardPriceDecoder(float[,] priceSequence, float[,] ctx, float[] ctxTimes)
        {
            if (priceSequence == null)
                throw new ArgumentNullException(nameof(priceSequence));

            int seqLen = priceSequence.GetLength(0);

            if (seqLen <= 0)
                throw new ArgumentException("priceSequence must contain at least one row.", nameof(priceSequence));

            int embeddingDim = _config.Price.EmbeddingDim;

            bool hasContext = ctx != null && ctx.GetLength(0) > 0;

            if (hasContext)
            {
                if (ctx.GetLength(1) != embeddingDim)
                    throw new ArgumentException($"Context embedding dim must be {embeddingDim}.", nameof(ctx));

                if (ctxTimes != null && ctxTimes.Length != ctx.GetLength(0))
                    throw new ArgumentException("ctxTimes length must match context row count.", nameof(ctxTimes));
            }
            else
            {
                ctx = null;
                ctxTimes = null;
            }

            var x = EmbedPriceSequence(priceSequence, seqLen);
            bool[,] mask = _config.Price.UseDecoderOnly ? CreateCausalMask(seqLen) : null;

            float[,] timeDiffs = null;
            float[] keyTimes = null;

            if (hasContext && ctxTimes != null)
            {
                timeDiffs = _accel.ComputeTimeDiffMatrix(seqLen, ctxTimes);
                keyTimes = ctxTimes;
            }

            foreach (var block in PriceBlocks)
            {
                block.SetContext(ctx);
                block.SetTimeData(timeDiffs, keyTimes, _config.DecayNetwork.Enabled);
                block.SetTraining(false);

                x = block.Forward(x, mask);
            }

            return x;
        }

        /// <summary>
        /// Main forward pass (recommended clean usage)
        /// </summary>
        public float[,] Forward(
            float[,] x,
            float[,] context = null,
            bool[,] selfMask = null,
            float[,] timeDiffs = null,
            float[] keyTimes = null,
            bool isTraining = false,
            Random rng = null)
        {
            foreach (var block in PriceBlocks)
            {
                block.SetContext(context);
                block.SetTimeData(timeDiffs, keyTimes, _config.DecayNetwork.Enabled);
                block.SetTraining(isTraining, rng);

                x = block.Forward(x, selfMask);
            }

            return x;
        }

        /// <summary>
        /// Optional: cleaner composable usage (step-by-step)
        /// </summary>
        public float[,] ForwardWithState(float[,] x, bool[,] selfMask = null)
        {
            foreach (var block in PriceBlocks)
            {
                x = block.Forward(x, selfMask);
            }

            return x;
        }
        private static bool[] ComputeCrossAttentionHasValidKey(float[,] timeDiffs, int queryLen, int keyLen)
        {
            var hasValidKey = new bool[queryLen];

            if (timeDiffs == null)
            {
                for (int q = 0; q < queryLen; q++)
                    hasValidKey[q] = keyLen > 0;

                return hasValidKey;
            }

            for (int q = 0; q < queryLen; q++)
            {
                for (int s = 0; s < keyLen; s++)
                {
                    if (timeDiffs[q, s] >= 0f)
                    {
                        hasValidKey[q] = true;
                        break;
                    }
                }
            }

            return hasValidKey;
        }

        private float[,] ComputeProjectionWithOptionalRows(float[,] input, float[,] w, float[] b, bool[] includeRows)
        {
            var p = _accel.BatchDotProduct(w, input);
            int rows = p.GetLength(0);
            int cols = p.GetLength(1);
            var res = new float[rows, cols];

            for (int i = 0; i < rows; i++)
            {
                if (includeRows != null && !includeRows[i])
                    continue;

                for (int j = 0; j < cols; j++)
                    res[i, j] = p[i, j] + b[j];
            }

            return res;
        }



        /// <summary>
        /// Helper: precompute time-difference matrix using accel
        /// </summary>
        public float[,] ComputeTimeDiffs(int seqLen, float[] keyTimes)
        {
            return _accel.ComputeTimeDiffMatrix(seqLen, keyTimes);
        }

        // Existing signature remains for compatibility.
        // It now delegates to the new overload.
        internal float[,] ForwardPriceDecoderWithCache(float[,] priceSequence, float[,] storyHidden, float[] storyTimes, MultimodalForwardCache cache, bool isTraining = true, Random dropoutRng = null)
        {
            int sl = priceSequence.GetLength(0);
            return ForwardPriceDecoderWithCache(
                priceSequence,
                rowStart: 0,
                rowCount: sl,
                storyHidden: storyHidden,
                storyTimes: storyTimes,
                cache: cache,
                isTraining: isTraining,
                dropoutRng: dropoutRng);
        }
        internal float[,] ForwardPriceDecoderWithCache(
            float[,] priceSequence,
            int rowStart,
            int rowCount,
            float[,] storyHidden,
            float[] storyTimes,
            MultimodalForwardCache cache,
            bool isTraining = true,
            Random dropoutRng = null)
        {
            if (priceSequence == null)
                throw new ArgumentNullException(nameof(priceSequence));

            if (cache == null)
                throw new ArgumentNullException(nameof(cache));

            if (rowStart < 0)
                throw new ArgumentOutOfRangeException(nameof(rowStart));

            if (rowCount <= 0)
                throw new ArgumentOutOfRangeException(nameof(rowCount));

            if (rowStart + rowCount > priceSequence.GetLength(0))
                throw new ArgumentException("rowStart + rowCount exceeds priceSequence row count.");

            int seqLen = rowCount;
            int embeddingDim = _config.Price.EmbeddingDim;
            int numHeads = _config.Price.NumHeads;
            int headDim = embeddingDim / numHeads;
            float scale = 1.0f / MathF.Sqrt(headDim);

            bool hasContext = storyHidden != null && storyHidden.GetLength(0) > 0;

            if (hasContext)
            {
                if (storyHidden.GetLength(1) != embeddingDim)
                    throw new ArgumentException($"storyHidden embedding dim must be {embeddingDim}.", nameof(storyHidden));

                if (storyTimes != null && storyTimes.Length != storyHidden.GetLength(0))
                    throw new ArgumentException("storyTimes length must match storyHidden row count.", nameof(storyTimes));
            }
            else
            {
                storyHidden = null;
                storyTimes = null;
            }

            var emb = EmbedPriceSequence(priceSequence, rowStart, rowCount);

            cache.PriceEmbedded = emb;
            cache.PriceContinuousInput =
                (rowStart == 0)
                    ? priceSequence
                    : _accel.SliceRows(priceSequence, rowStart, rowStart + rowCount);

            bool[,] selfMask = _config.Price.UseDecoderOnly ? CreateCausalMask(seqLen) : null;
            var x = emb;

            for (int layer = 0; layer < _config.Price.NumLayers; layer++)
            {
                var block = PriceBlocks[layer];
                var blockCache = cache.PriceBlockCaches[layer];

                blockCache.BlockInput = x;

                var selfQ = ComputeProjection(x, block.SelfAttention.WQ, block.SelfAttention.BiasQ);
                var selfK = ComputeProjection(x, block.SelfAttention.WK, block.SelfAttention.BiasK);
                var selfV = ComputeProjection(x, block.SelfAttention.WV, block.SelfAttention.BiasV);

                RotaryPositionEmbedding.ApplyInPlace(selfQ, selfK, numHeads);

                blockCache.SelfQ = selfQ;
                blockCache.SelfK = selfK;
                blockCache.SelfV = selfV;

                var selfAttnOutput = _accel.MultiHeadAttentionForward(selfQ, selfK, selfV, numHeads, scale, selfMask);

                blockCache.SelfAttnOutput = selfAttnOutput;

                var selfProjected = ComputeProjection(selfAttnOutput, block.SelfAttention.WO, block.SelfAttention.BiasO);
                var selfResidual = _accel.MatrixAdd(x, selfProjected);

                blockCache.SelfResidualInput = selfResidual;

                var (normedSelf, selfMean, selfVariance, selfNormalized) =
                    _accel.LayerNormForward(selfResidual, block.LNSelfGamma, block.LNSelfBeta);

                blockCache.LNSelfCache.Input = selfResidual;
                blockCache.LNSelfCache.Mean = selfMean;
                blockCache.LNSelfCache.Variance = selfVariance;
                blockCache.LNSelfCache.Normalized = selfNormalized;
                blockCache.NormedSelf = normedSelf;

                float[,] crossNormed;

                if (hasContext)
                {
                    float[,] timeDiffs = null;
                    float[] keyTimes = null;

                    if (storyTimes != null)
                    {
                        timeDiffs = _accel.ComputeTimeDiffMatrix(seqLen, storyTimes);
                        keyTimes = storyTimes;
                    }

                    block.SetTimeData(timeDiffs, keyTimes, _config.DecayNetwork.Enabled);

                    blockCache.TimeDiffs = timeDiffs;
                    blockCache.KeyTimesFromRef = keyTimes;
                    blockCache.CrossAttentionHasValidKey = ComputeCrossAttentionHasValidKey(timeDiffs, seqLen, storyHidden.GetLength(0));
                    blockCache.DecayCache = null;
                    blockCache.CrossAttentionWeights = null;
                    blockCache.CrossScoresPreSoftmax = null;

                    var crossQ = ComputeProjection(normedSelf, block.CrossAttention.WQ, block.CrossAttention.BiasQ);
                    var crossK = ComputeProjection(storyHidden, block.CrossAttention.WK, block.CrossAttention.BiasK);
                    var crossV = ComputeProjection(storyHidden, block.CrossAttention.WV, block.CrossAttention.BiasV);

                    RotaryPositionEmbedding.ApplyInPlace(crossQ, crossK, numHeads);

                    blockCache.CrossQ = crossQ;
                    blockCache.CrossK = crossK;
                    blockCache.CrossV = crossV;

                    var crossAttnOutput = ContentAwareCrossAttentionWithCache(
                        crossQ,
                        crossK,
                        crossV,
                        timeDiffs,
                        keyTimes,
                        normedSelf,
                        storyHidden,
                        block,
                        blockCache,
                        isTraining,
                        dropoutRng);

                    blockCache.CrossAttnOutput = crossAttnOutput;

                    var crossProjected = ComputeProjectionWithOptionalRows(
                        crossAttnOutput,
                        block.CrossAttention.WO,
                        block.CrossAttention.BiasO,
                        blockCache.CrossAttentionHasValidKey);

                    var crossResidual = _accel.MatrixAdd(normedSelf, crossProjected);

                    blockCache.CrossResidualInput = crossResidual;

                    var (normedCross, crossMean, crossVariance, crossNormalized) =
                        _accel.LayerNormForward(crossResidual, block.LnCrossGamma, block.LnCrossBeta);

                    blockCache.LNCrossCache.Input = crossResidual;
                    blockCache.LNCrossCache.Mean = crossMean;
                    blockCache.LNCrossCache.Variance = crossVariance;
                    blockCache.LNCrossCache.Normalized = crossNormalized;
                    blockCache.NormedCross = normedCross;

                    crossNormed = normedCross;
                }
                else
                {
                    blockCache.CrossQ = null;
                    blockCache.CrossK = null;
                    blockCache.CrossV = null;
                    blockCache.CrossAttnOutput = null;
                    blockCache.CrossAttentionWeights = null;
                    blockCache.CrossScoresPreSoftmax = null;
                    blockCache.CrossAttentionHasValidKey = null;
                    blockCache.TimeDiffs = null;
                    blockCache.KeyTimesFromRef = null;
                    blockCache.DecayCache = null;

                    var (normedCross, crossMean, crossVariance, crossNormalized) =
                        _accel.LayerNormForward(normedSelf, block.LnCrossGamma, block.LnCrossBeta);

                    blockCache.LNCrossCache.Input = normedSelf;
                    blockCache.LNCrossCache.Mean = crossMean;
                    blockCache.LNCrossCache.Variance = crossVariance;
                    blockCache.LNCrossCache.Normalized = crossNormalized;
                    blockCache.NormedCross = normedCross;

                    crossNormed = normedCross;
                }

                var ffnInputRows = new float[seqLen][];

                for (int i = 0; i < seqLen; i++)
                {
                    var row = new float[embeddingDim];

                    for (int j = 0; j < embeddingDim; j++)
                        row[j] = crossNormed[i, j];

                    ffnInputRows[i] = row;
                }

                var ffnOutput = _accel.FFNForwardBatch(
                    crossNormed,
                    seqLen,
                    embeddingDim,
                    block.FeedForwardNetwork.ForwardPassOnly);

                blockCache.FFNInputRows = ffnInputRows;
                blockCache.FFNOutput = ffnOutput;

                var ffnResidual = _accel.MatrixAdd(crossNormed, ffnOutput);

                blockCache.FFNResidualInput = ffnResidual;

                var (normedFfn, ffnMean, ffnVariance, ffnNormalized) =
                    _accel.LayerNormForward(ffnResidual, block.LNFFNGamma, block.LNFFNBeta);

                blockCache.LNFFNCache.Input = ffnResidual;
                blockCache.LNFFNCache.Mean = ffnMean;
                blockCache.LNFFNCache.Variance = ffnVariance;
                blockCache.LNFFNCache.Normalized = ffnNormalized;

                x = normedFfn;
            }

            return x;
        }
        private float[,] ContentAwareCrossAttentionWithCache(
      float[,] Q,
      float[,] K,
      float[,] V,
      float[,] timeDiffs,
      float[] keyTimesFromRef,
      float[,] queryEmbeddings,
      float[,] keyEmbeddings,
      TacamtBlock block,
      BlockCache bc,
      bool isTraining = false,
      Random dropoutRng = null,
      int globalBypassCount = 0)
        {
            int priceEmbeddingDim = _config.Price.EmbeddingDim;
            int priceNumHeads = _config.Price.NumHeads;

            return _accel.ContentAwareCrossAttentionWithCache(
                Q,
                K,
                V,
                timeDiffs,
                keyTimesFromRef,
                queryEmbeddings,
                keyEmbeddings,
                block,
                bc,
                priceEmbeddingDim,
                priceNumHeads,
                enableDecayBias: _config.DecayNetwork.Enabled,
                isTraining: isTraining,
                dropoutRng: dropoutRng);
        }
        internal (float[,], float[,]) ProjectToOutput(float[,] hidden)
        {
            int sl = hidden.GetLength(0);

            var pred = _accel.ProjectOutputBatch(hidden, OutputProjection, OutputBias, sl, _config.Output.OutputDim);

            float[,] conf = null;

            if (_config.Output.UseConfidenceHead)
            {
                conf = new float[sl, 1];

                for (int i = 0; i < sl; i++)
                {
                    float l = ConfidenceBias[0];
                    for (int k = 0; k < _config.Price.EmbeddingDim; k++)
                    {
                        l += ConfidenceProjection[0, k] * hidden[i, k];
                    }
                    conf[i, 0] = Sigmoid(l);
                }
            }

            return (pred, conf);
        }

        private float[,] ForwardTextEncoder(int[] tokenIds)
        {
            int sl = tokenIds.Length; var emb = EmbedTextTokens(tokenIds, sl);
            bool[,] mask = _config.Text.UseDecoderOnly ? CreateCausalMask(sl) : null;
            var x = emb;

            foreach (var b in TextBlocks)
            {
                x = b.Forward(x, mask);
            }
            return x;
        }
        private float[,] ForwardTextEncoderWithCache(int[] tokenIds, MultimodalForwardCache cache)
        {
            int sl = tokenIds.Length;
            int ed = _config.Text.EmbeddingDim;

            var emb = EmbedTextTokens(tokenIds, sl);

            cache.TextEmbedded = emb;
            cache.TextTokenIds = tokenIds;

            bool[,] mask = _config.Text.UseDecoderOnly ? CreateCausalMask(sl) : null;

            var x = emb;

            for (int layer = 0; layer < _config.Text.NumLayers; layer++)
            {
                cache.TextLayerInputs.Add(x);

                var b = TextBlocks[layer];

                var ac = cache.TextAttentionCaches[layer];
                ac.Input = x;

                // ---- Self Attention ----
                var ao = AttentionForwardWithCache(
                    b.Attention,        // FIX 1: was b.SelfAttention
                    x,
                    x,
                    x,
                    mask,
                    ac
                );

                var ar = _accel.MatrixAdd(x, ao);

                // ---- LN1 ----
                var l1c = cache.TextLN1Caches[layer];

                var (n1, m1, v1, nr1) =
                    _accel.LayerNormForward(ar, b.LN1Gamma, b.LN1Beta);  // FIX 2: was b.LNSelfGamma, b.LNSelfBeta

                l1c.Input = ar;
                l1c.Mean = m1;
                l1c.Variance = v1;
                l1c.Normalized = nr1;

                // ---- FFN ----
                var fo = _accel.FFNForwardBatch(
                    n1,
                    sl,
                    ed,
                    b.FeedForwardNetwork.ForwardPassOnly
                );

                cache.TextFFNInputs.Add(n1);
                cache.TextFFNOutputs.Add(fo);

                var fr = _accel.MatrixAdd(n1, fo);

                // ---- LN2 ----
                var l2c = cache.TextLN2Caches[layer];

                var (n2, m2, v2, nr2) = _accel.LayerNormForward(fr, b.LN2Gamma, b.LN2Beta);  // FIX 3: was b.LNFFNGamma, b.LNFFNBeta

                l2c.Input = fr;
                l2c.Mean = m2;
                l2c.Variance = v2;
                l2c.Normalized = nr2;

                x = n2;
            }

            cache.TextFinalHidden = x;

            return x;
        }
        private void InitTextEncoder()
        {
            TextTokenEmbedding = new float[_config.Text.VocabSize, _config.Text.EmbeddingDim];

            float std = MathF.Sqrt(1.0f / _config.Text.EmbeddingDim);

            for (int i = 0; i < _config.Text.VocabSize; i++)
            {
                for (int j = 0; j < _config.Text.EmbeddingDim; j++)
                {
                    TextTokenEmbedding[i, j] = SampleGaussian() * std;
                }
            }

            TextBlocks = new TransformerBlock[_config.Text.NumLayers];

            for (int i = 0; i < _config.Text.NumLayers; i++)
            {
                TextBlocks[i] = new TransformerBlock(
                    embeddingDim: _config.Text.EmbeddingDim,
                    numHeads: _config.Text.NumHeads,
                    feedForwardDim: _config.Text.FeedForwardDim,
                    ffnActivationType: _config.Runtime.FFNActivationType,
                    costFunction: CostFunctionType.mse,
                    activationDistribution: ActivationDistribution.Normal,
                    l2RegulationLamda: _config.Regularization.L2RegulationLamda,
                    gradientClippingThreshold: _config.Regularization.GradientClippingThreshold,
                    accelerationType: _config.Runtime.AccelerationType,
                    accelerationDeviceId: _config.Runtime.AccelerationDeviceId,
                    accel: _accel,
                    random: _random);

                DeterministicallyInitializeFeedForwardNetwork(TextBlocks[i].FeedForwardNetwork, _random);
            }
        }
        private static void DeterministicallyInitializeFeedForwardNetwork(NeuralNetwork network, Random random)
        {
            if (network == null)
                throw new ArgumentNullException(nameof(network));

            random ??= new Random();

            CallaghanDev.ML.Data state = network.GetInternalData();

            if (state == null)
                return;

            var seen = new HashSet<object>();
            ReinitializeFloatStorage(state, random, seen);

            network.RestoreState(state);
        }

        private static void ReinitializeFloatStorage(object value, Random random, HashSet<object> seen)
        {
            if (value == null)
                return;

            if (value is string)
                return;

            if (value is float[,] matrix)
            {
                FillDeterministicMatrix(matrix, random);
                return;
            }

            if (value is float[] vector)
            {
                Array.Clear(vector, 0, vector.Length);
                return;
            }

            Type type = value.GetType();

            if (type.IsPrimitive || type.IsEnum)
                return;

            if (!type.IsValueType && !seen.Add(value))
                return;

            if (value is System.Collections.IEnumerable enumerable && value is not string)
            {
                foreach (object child in enumerable)
                    ReinitializeFloatStorage(child, random, seen);

                return;
            }

            var fields = type.GetFields(
                System.Reflection.BindingFlags.Instance |
                System.Reflection.BindingFlags.Public |
                System.Reflection.BindingFlags.NonPublic);

            Array.Sort(fields, (a, b) => string.CompareOrdinal(a.Name, b.Name));

            foreach (var field in fields)
            {
                if (field.FieldType.IsPointer)
                    continue;

                object child;

                try
                {
                    child = field.GetValue(value);
                }
                catch
                {
                    continue;
                }

                ReinitializeFloatStorage(child, random, seen);
            }

            var properties = type.GetProperties(
                System.Reflection.BindingFlags.Instance |
                System.Reflection.BindingFlags.Public |
                System.Reflection.BindingFlags.NonPublic);

            Array.Sort(properties, (a, b) => string.CompareOrdinal(a.Name, b.Name));

            foreach (var property in properties)
            {
                if (!property.CanRead)
                    continue;

                if (property.GetIndexParameters().Length != 0)
                    continue;

                object child;

                try
                {
                    child = property.GetValue(value);
                }
                catch
                {
                    continue;
                }

                ReinitializeFloatStorage(child, random, seen);
            }
        }

        private static void FillDeterministicMatrix(float[,] matrix, Random random)
        {
            int rows = matrix.GetLength(0);
            int cols = matrix.GetLength(1);

            float std = MathF.Sqrt(2.0f / Math.Max(1, rows + cols));

            for (int r = 0; r < rows; r++)
            {
                for (int c = 0; c < cols; c++)
                {
                    matrix[r, c] = SampleGaussian(random) * std;
                }
            }
        }

        private static float SampleGaussian(Random random)
        {
            double u1 = 1.0 - random.NextDouble();
            double u2 = 1.0 - random.NextDouble();

            return (float)(Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2));
        }
        private void InitPriceDecoder()
        {
            PriceInputProjection = new float[_config.Price.EmbeddingDim, _config.Price.InputFeatureDim];
            PriceInputProjectionBias = new float[_config.Price.EmbeddingDim];

            float std = MathF.Sqrt(2.0f / (_config.Price.InputFeatureDim + _config.Price.EmbeddingDim));

            for (int i = 0; i < _config.Price.EmbeddingDim; i++)
            {
                for (int j = 0; j < _config.Price.InputFeatureDim; j++)
                {
                    PriceInputProjection[i, j] = SampleGaussian() * std;
                }
            }

            PriceBlocks = new TacamtBlock[_config.Price.NumLayers];

            for (int i = 0; i < _config.Price.NumLayers; i++)
            {
                PriceBlocks[i] = new TacamtBlock(
                    _config.Price.EmbeddingDim,
                    _config.Price.NumHeads,
                    _config.Price.FeedForwardDim,
                    _config.Runtime.FFNActivationType,
                    _accel,
                    _random,
                    _config.Regularization.L2RegulationLamda,
                    _config.DecayNetwork.ProjectionDim,
                    _config.DecayNetwork.HiddenDim,
                    _config.DecayNetwork.MemAttentionDropout,
                    _config.DecayNetwork.MlpDropout,
                    _config.DecayNetwork.WeightDecay,
                    _config.DecayNetwork.TimeEncodingBases);

                DeterministicallyInitializeFeedForwardNetwork(PriceBlocks[i].FeedForwardNetwork, _random);
            }
        }
        private void InitOutputHead()
        {
            int ed = _config.Price.EmbeddingDim;
            OutputProjection = new float[_config.Output.OutputDim, ed];
            OutputBias = new float[_config.Output.OutputDim];

            float std = MathF.Sqrt(2.0f / (ed + _config.Output.OutputDim));

            for (int i = 0; i < _config.Output.OutputDim; i++)
            {
                for (int j = 0; j < ed; j++)
                {
                    OutputProjection[i, j] = SampleGaussian() * std;
                }
            }
            if (_config.Output.UseConfidenceHead)
            {
                ConfidenceProjection = new float[1, ed];
                ConfidenceBias = new float[1];
                float cs = MathF.Sqrt(2.0f / (ed + 1));

                for (int j = 0; j < ed; j++)
                {
                    ConfidenceProjection[0, j] = SampleGaussian() * cs;
                }
            }
        }
        private float[,] EmbedTextTokens(int[] ids, int sl)
        {
            if (ids == null)
                throw new ArgumentNullException(nameof(ids));

            if (sl <= 0)
                throw new ArgumentOutOfRangeException(nameof(sl), "Text sequence must contain at least one token.");

            if (sl > ids.Length)
                throw new ArgumentException("sl cannot exceed ids.Length.", nameof(sl));

            int vocabSize = TextTokenEmbedding.GetLength(0);
            int embeddingDim = _config.Text.EmbeddingDim;

            var embedded = new float[sl, embeddingDim];

            for (int i = 0; i < sl; i++)
            {
                int tokenId = ids[i];

                if (tokenId < 0 || tokenId >= vocabSize)
                {
                    throw new InvalidOperationException(
                        $"Token ID {tokenId} at position {i} is out of bounds for embedding table size {vocabSize}. " +
                        "The tokenizer vocab size exceeds TextVocabSize in the config, or the token stream is invalid.");
                }

                for (int j = 0; j < embeddingDim; j++)
                    embedded[i, j] = TextTokenEmbedding[tokenId, j];
            }

            return embedded;
        }

        // Existing method kept for compatibility
        private float[,] EmbedPriceSequence(float[,] ps, int sl)
        {
            // Old behavior: assumes rows start at 0 and length == sl
            // Delegates to new overload
            return EmbedPriceSequence(ps, rowStart: 0, rowCount: sl);
        }

        //  offset-aware
        private float[,] EmbedPriceSequence(float[,] ps, int rowStart, int rowCount)
        {
            if (ps == null)
            {
                throw new ArgumentNullException(nameof(ps));
            }
            if (rowStart < 0)
            {
                throw new ArgumentOutOfRangeException(nameof(rowStart));
            }
            if (rowCount < 0)
            {
                throw new ArgumentOutOfRangeException(nameof(rowCount));
            }
            if (rowStart + rowCount > ps.GetLength(0))
            {
                throw new ArgumentException("rowStart + rowCount exceeds ps row count.");
            }

            var projected = _accel.BatchDotProduct(PriceInputProjection, ps, rowStart, rowCount);

            return _accel.MatrixAddBias(projected, PriceInputProjectionBias);
        }
        private bool[,] CreateCausalMask(int sl)
        {
            var m = new bool[sl, sl];

            for (int i = 0; i < sl; i++)
            {
                for (int j = 0; j <= i; j++)
                {
                    m[i, j] = true;
                }
            }
            return m;
        }

        private float[,] ComputeProjection(float[,] input, float[,] w, float[] b)
        {
            var p = _accel.BatchDotProduct(w, input);
            int r = p.GetLength(0), c = p.GetLength(1);
            var res = new float[r, c];

            for (int i = 0; i < r; i++)
            {
                for (int j = 0; j < c; j++)
                {
                    res[i, j] = p[i, j] + b[j];
                }
            }
            return res;
        }

        private float[,] AttentionForwardWithCache(MultiHeadAttention attn, float[,] qs, float[,] ks, float[,] vs, bool[,] mask, AttentionCache cache)
        {
            int ed = _config.Text.EmbeddingDim;
            int nh = _config.Text.NumHeads;
            int hd = ed / nh;

            var Q = ComputeProjection(qs, attn.WQ, attn.BiasQ);
            var K = ComputeProjection(ks, attn.WK, attn.BiasK);
            var V = ComputeProjection(vs, attn.WV, attn.BiasV);

            RotaryPositionEmbedding.ApplyInPlace(Q, K, nh);

            cache.Q = Q;
            cache.K = K;
            cache.V = V;

            float s = 1.0f / MathF.Sqrt(hd);

            var c = _accel.MultiHeadAttentionForward(Q, K, V, nh, s, mask);
            cache.AttentionOutput = c;

            return ComputeProjection(c, attn.WO, attn.BiasO);
        }
        protected float SampleGaussian()
        {
            float u1 = 1f - _random.NextSingle(), u2 = 1f - _random.NextSingle();

            return MathF.Sqrt(-2f * MathF.Log(u1)) * MathF.Cos(2f * MathF.PI * u2);

        }
        private static float Sigmoid(float x)
        {
            if (x >= 0)
            {
                float ex = MathF.Exp(-x);
                return 1f / (1f + ex);
            }
            else
            {
                float ex = MathF.Exp(x);
                return ex / (1f + ex);
            }
        }

        public float[,] EncodePriceHistory(float[,] histPrices)
        {
            int sl = histPrices.GetLength(0);

            var emb = EmbedPriceSequence(histPrices, sl);

            bool[,] mask = _config.Price.UseDecoderOnly ? CreateCausalMask(sl) : null;

            var x = emb;

            foreach (var block in PriceBlocks)
            {
                // No context for pure price encoding
                block.SetContext(null);
                block.SetTimeData(null, null);
                block.SetTraining(false);

                x = block.Forward(x, mask);
            }

            return x;
        }

        public float[,] EncodePriceHistoryWithCache(
            float[,] histPrices,
            MultimodalForwardCache cache,
            bool isTraining = false,
            Random dropoutRng = null)
        {
            if (histPrices == null)
                throw new ArgumentNullException(nameof(histPrices));

            if (cache == null)
                throw new ArgumentNullException(nameof(cache));

            if (histPrices.GetLength(0) <= 0)
                throw new ArgumentException("histPrices must contain at least one row.", nameof(histPrices));

            var hidden = ForwardPriceDecoderWithCache(
                histPrices,
                storyHidden: null,
                storyTimes: null,
                cache: cache,
                isTraining: isTraining,
                dropoutRng: dropoutRng);

            cache.PriceFinalHidden = hidden;

            return hidden;
        }

        public (float[,] predictions, float[,] confidence) ForwardWithPriceHistoryContextAndCache(
            NewsStory[] stories,
            float[,] currentPriceInput,
            float[,] historyPrices,
            float[] priceCtxTimes,
            MultimodalForwardCache cache,
            bool isTraining = true,
            Random dropoutRng = null)
        {
            if (historyPrices == null)
                throw new ArgumentNullException(nameof(historyPrices));

            if (cache == null)
                throw new ArgumentNullException(nameof(cache));

            cache.PriceContextEncoderCache = new MultimodalForwardCache(
                _config.Text.NumLayers,
                _config.Price.NumLayers);

            float[,] priceCtxHidden = EncodePriceHistoryWithCache(
                historyPrices,
                cache.PriceContextEncoderCache,
                isTraining,
                dropoutRng);

            return ForwardWithPriceContextAndCache(
                stories,
                currentPriceInput,
                priceCtxHidden,
                priceCtxTimes,
                cache,
                isTraining,
                dropoutRng);
        }

        public (float[,] predictions, float[,] confidence) ForwardWithPriceContextAndCache(
        NewsStory[] stories,
        float[,] currentPriceInput,
        float[,] priceCtxHidden,
        float[] priceCtxTimes,
        MultimodalForwardCache cache,
        bool isTraining = true,
        Random dropoutRng = null)
        {
            if (currentPriceInput == null)
                throw new ArgumentNullException(nameof(currentPriceInput));

            if (cache == null)
                throw new ArgumentNullException(nameof(cache));

            int embeddingDim = _config.Price.EmbeddingDim;

            float[,] newsHidden = null;
            float[] newsTimes = null;
            int numNews = 0;

            if (stories != null && stories.Length > 0)
            {
                (newsHidden, newsTimes) = EncodeStoriesWithCache(stories, cache);
                numNews = newsHidden.GetLength(0);

                if (newsHidden.GetLength(1) != embeddingDim)
                    throw new InvalidOperationException("Text embedding dim must match price embedding dim for cross-attention.");
            }
            else
            {
                cache.TextFinalHidden = null;
                cache.TextTokenIds = null;
                cache.StoryCaches = new List<MultimodalForwardCache>();
                cache.StoryTokenCounts = Array.Empty<int>();
            }

            int numPriceCtx = priceCtxHidden != null ? priceCtxHidden.GetLength(0) : 0;

            if (numPriceCtx > 0)
            {
                if (priceCtxHidden.GetLength(1) != embeddingDim)
                    throw new ArgumentException($"priceCtxHidden embedding dim must be {embeddingDim}.", nameof(priceCtxHidden));

                if (priceCtxTimes == null || priceCtxTimes.Length != numPriceCtx)
                    throw new ArgumentException("priceCtxTimes length must match priceCtxHidden row count.", nameof(priceCtxTimes));
            }

            int totalCtx = numNews + numPriceCtx;

            float[,] combinedHidden = null;
            float[] combinedTimes = null;

            if (totalCtx > 0)
            {
                combinedHidden = new float[totalCtx, embeddingDim];
                combinedTimes = new float[totalCtx];
                var typeIndices = new int[totalCtx];

                for (int i = 0; i < numNews; i++)
                {
                    combinedTimes[i] = newsTimes[i];
                    typeIndices[i] = 0;

                    for (int d = 0; d < embeddingDim; d++)
                        combinedHidden[i, d] = newsHidden[i, d];
                }

                for (int i = 0; i < numPriceCtx; i++)
                {
                    int dst = numNews + i;

                    combinedTimes[dst] = priceCtxTimes[i];
                    typeIndices[dst] = 1;

                    for (int d = 0; d < embeddingDim; d++)
                        combinedHidden[dst, d] = priceCtxHidden[i, d];
                }

                _accel.ApplyContextTypeEmbedding(combinedHidden, ContextTypeEmbedding, typeIndices);
            }

            cache.NumNewsContext = numNews;
            cache.NumPriceContext = numPriceCtx;
            cache.PriceContextHidden = priceCtxHidden;
            cache.TextFinalHidden = combinedHidden;
            cache.StoryArrivalTimes = combinedTimes;

            var priceHidden = ForwardPriceDecoderWithCache(
                currentPriceInput,
                combinedHidden,
                combinedTimes,
                cache,
                isTraining,
                dropoutRng);

            cache.PriceFinalHidden = priceHidden;

            return ProjectToOutput(priceHidden);
        }


        #region Save/Load
        private const string FeedForwardNetworkFileName = "network.json";

        private static string GetFeedForwardNetworkDirectory(string rootDir, string folderName)
        {
            return System.IO.Path.Combine(rootDir, folderName);
        }

        private static string GetFeedForwardNetworkFilePath(string rootDir, string folderName)
        {
            return System.IO.Path.Combine(
                GetFeedForwardNetworkDirectory(rootDir, folderName),
                FeedForwardNetworkFileName);
        }

        private static void EnsureFeedForwardDirectory(string ffnDirectory)
        {
            // Handles previous broken saves where "text_ffn_0" may have been created as a file.
            if (System.IO.File.Exists(ffnDirectory))
                System.IO.File.Delete(ffnDirectory);

            System.IO.Directory.CreateDirectory(ffnDirectory);
        }

        private static string ResolveFeedForwardNetworkFilePath(string rootDir, string folderName)
        {
            string ffnDirectory = GetFeedForwardNetworkDirectory(rootDir, folderName);
            string expectedFile = System.IO.Path.Combine(ffnDirectory, FeedForwardNetworkFileName);

            if (System.IO.File.Exists(expectedFile))
                return expectedFile;

            // Backward compatibility for old saves where the FFN was saved directly as:
            // "text_ffn_0" / "price_ffn_0" instead of inside that directory.
            if (System.IO.File.Exists(ffnDirectory))
                return ffnDirectory;

            if (!System.IO.Directory.Exists(ffnDirectory))
            {
                throw new System.IO.DirectoryNotFoundException(
                    $"Feed-forward network directory not found: {ffnDirectory}");
            }

            throw new System.IO.FileNotFoundException(
                $"Feed-forward network file not found. Expected: {expectedFile}",
                expectedFile);
        }

        private static void RestoreFeedForwardNetworkState(NeuralNetwork target, NeuralNetwork loaded)
        {
            if (target == null)
                throw new ArgumentNullException(nameof(target));

            if (loaded == null)
                throw new ArgumentNullException(nameof(loaded));

            target.RestoreState(loaded.GetInternalData());
        }
        private static MultimodalTransformerConfig BuildConfig(Dictionary<string, JsonElement> d)
        {
            return new MultimodalTransformerConfig
            {
                Text = new TextEncoderConfig
                {
                    VocabSize = d["TextVocabSize"].GetInt32(),
                    MaxSequenceLength = d["TextMaxSequenceLength"].GetInt32(),
                    EmbeddingDim = d["TextEmbeddingDim"].GetInt32(),
                    NumHeads = d["TextNumHeads"].GetInt32(),
                    NumLayers = d["TextNumLayers"].GetInt32(),
                    FeedForwardDim = d["TextFeedForwardDim"].GetInt32(),
                    UseDecoderOnly = d["TextUseDecoderOnly"].GetBoolean(),
                    Freeze = d["FreezeTextEncoder"].GetBoolean()
                },

                Price = new PriceDecoderConfig
                {
                    InputFeatureDim = d["PriceInputFeatureDim"].GetInt32(),
                    MaxSequenceLength = d["PriceMaxSequenceLength"].GetInt32(),
                    EmbeddingDim = d["PriceEmbeddingDim"].GetInt32(),
                    NumHeads = d["PriceNumHeads"].GetInt32(),
                    NumLayers = d["PriceNumLayers"].GetInt32(),
                    FeedForwardDim = d["PriceFeedForwardDim"].GetInt32(),
                    UseDecoderOnly = d["PriceUseDecoderOnly"].GetBoolean()
                },

                Output = new OutputHeadConfig
                {
                    OutputDim = d["OutputDim"].GetInt32(),
                    UseConfidenceHead = d["UseConfidenceHead"].GetBoolean()
                },

                Runtime = new RuntimeConfig
                {
                    FFNActivationType = (ActivationType)d["FFNActivationType"].GetInt32(),
                    AccelerationType = (AccelerationType)d["AccelerationType"].GetInt32(),
                    AccelerationDeviceId = d["AccelerationDeviceId"].GetInt32()
                },

                Regularization = new RegularizationConfig
                {
                    L2RegulationLamda = d["L2RegulationLamda"].GetSingle(),
                    GradientClippingThreshold = d["GradientClippingThreshold"].GetSingle()
                },

                DecayNetwork = new DecayNetworkConfig
                {
                    Enabled = d.ContainsKey("DecayEnabled")
                        ? d["DecayEnabled"].GetBoolean()
                        : d.ContainsKey("DecayProjectionDim"),

                    ProjectionDim = d.ContainsKey("DecayProjectionDim")
                        ? d["DecayProjectionDim"].GetInt32()
                        : 16,

                    HiddenDim = d.ContainsKey("DecayHiddenDim")
                        ? d["DecayHiddenDim"].GetInt32()
                        : 32,

                    MemAttentionDropout = d.ContainsKey("DecayMemAttnDropout")
                        ? d["DecayMemAttnDropout"].GetSingle()
                        : 0.2f,

                    MlpDropout = d.ContainsKey("DecayMLPDropout")
                        ? d["DecayMLPDropout"].GetSingle()
                        : 0.2f,

                    WeightDecay = d.ContainsKey("DecayWeightDecay")
                        ? d["DecayWeightDecay"].GetSingle()
                        : 0f,

                    TimeEncodingBases = d.ContainsKey("DecayTimeEncodingBases")
                        ? d["DecayTimeEncodingBases"].GetInt32()
                        : 16
                },

                PriceContext = new PriceContextConfig
                {
                    Enabled = d.ContainsKey("PriceContextEnabled")
                        ? d["PriceContextEnabled"].GetBoolean()
                        : d.ContainsKey("PriceContextMinHistoryLength"),

                    MinHistoryLength = d.ContainsKey("PriceContextMinHistoryLength")
                        ? d["PriceContextMinHistoryLength"].GetInt32()
                        : 5,

                    MinCurrentLength = d.ContainsKey("PriceContextMinCurrentLength")
                        ? d["PriceContextMinCurrentLength"].GetInt32()
                        : 5
                },

                RequireSharedCrossAttentionEmbeddingDim =
                    d.ContainsKey("RequireSharedCrossAttentionEmbeddingDim")
                        ? d["RequireSharedCrossAttentionEmbeddingDim"].GetBoolean()
                        : true
            };
        }
        public void Save(string dir)
        {
            if (string.IsNullOrWhiteSpace(dir))
                throw new ArgumentException("Save directory cannot be null or empty.", nameof(dir));

            System.IO.Directory.CreateDirectory(dir);

            var pruning = PruningConfig ?? new MemoryPruningConfig();

            // ===============================
            // CONFIG
            // ===============================
            var cd = new Dictionary<string, object>
            {
                ["TextVocabSize"] = _config.Text.VocabSize,
                ["TextMaxSequenceLength"] = _config.Text.MaxSequenceLength,
                ["TextEmbeddingDim"] = _config.Text.EmbeddingDim,
                ["TextNumHeads"] = _config.Text.NumHeads,
                ["TextNumLayers"] = _config.Text.NumLayers,
                ["TextFeedForwardDim"] = _config.Text.FeedForwardDim,
                ["TextUseDecoderOnly"] = _config.Text.UseDecoderOnly,
                ["FreezeTextEncoder"] = _config.Text.Freeze,

                ["PriceInputFeatureDim"] = _config.Price.InputFeatureDim,
                ["PriceMaxSequenceLength"] = _config.Price.MaxSequenceLength,
                ["PriceEmbeddingDim"] = _config.Price.EmbeddingDim,
                ["PriceNumHeads"] = _config.Price.NumHeads,
                ["PriceNumLayers"] = _config.Price.NumLayers,
                ["PriceFeedForwardDim"] = _config.Price.FeedForwardDim,
                ["PriceUseDecoderOnly"] = _config.Price.UseDecoderOnly,

                ["OutputDim"] = _config.Output.OutputDim,
                ["UseConfidenceHead"] = _config.Output.UseConfidenceHead,

                ["FFNActivationType"] = (int)_config.Runtime.FFNActivationType,
                ["AccelerationType"] = (int)_config.Runtime.AccelerationType,
                ["AccelerationDeviceId"] = _config.Runtime.AccelerationDeviceId,

                ["L2RegulationLamda"] = _config.Regularization.L2RegulationLamda,
                ["GradientClippingThreshold"] = _config.Regularization.GradientClippingThreshold,

                ["DecayEnabled"] = _config.DecayNetwork.Enabled,
                ["DecayProjectionDim"] = _config.DecayNetwork.ProjectionDim,
                ["DecayHiddenDim"] = _config.DecayNetwork.HiddenDim,
                ["DecayMemAttnDropout"] = _config.DecayNetwork.MemAttentionDropout,
                ["DecayMLPDropout"] = _config.DecayNetwork.MlpDropout,
                ["DecayWeightDecay"] = _config.DecayNetwork.WeightDecay,
                ["DecayTimeEncodingBases"] = _config.DecayNetwork.TimeEncodingBases,

                ["PriceContextEnabled"] = _config.PriceContext.Enabled,
                ["PriceContextMinHistoryLength"] = _config.PriceContext.MinHistoryLength,
                ["PriceContextMinCurrentLength"] = _config.PriceContext.MinCurrentLength,

                ["RequireSharedCrossAttentionEmbeddingDim"] = _config.RequireSharedCrossAttentionEmbeddingDim,

                ["PruningAttentionScoreAlpha"] = pruning.AttentionScoreAlpha,
                ["PruningMinQueryCountForPruning"] = pruning.MinQueryCountForPruning,
                ["PruningNewEntryReserveFraction"] = pruning.NewEntryReserveFraction,
                ["PruningUseAttentionBasedPruning"] = pruning.UseAttentionBasedPruning
            };

            string configPath = System.IO.Path.Combine(dir, "config.json");

            System.IO.File.WriteAllText(
                configPath,
                JsonSerializer.Serialize(cd, new JsonSerializerOptions { WriteIndented = true }));

            // ===============================
            // WEIGHTS
            // ===============================
            string weightsPath = System.IO.Path.Combine(dir, "weights.bin");

            using (var s = new System.IO.FileStream(weightsPath, System.IO.FileMode.Create, System.IO.FileAccess.Write))
            using (var w = new System.IO.BinaryWriter(s))
            {
                w.Write(1); // weights format version

                WM(w, TextTokenEmbedding);

                for (int i = 0; i < _config.Text.NumLayers; i++)
                {
                    var b = TextBlocks[i];

                    WA(w, b.Attention);
                    WV(w, b.LN1Gamma);
                    WV(w, b.LN1Beta);
                    WV(w, b.LN2Gamma);
                    WV(w, b.LN2Beta);
                }

                WM(w, PriceInputProjection);
                WV(w, PriceInputProjectionBias);

                for (int i = 0; i < _config.Price.NumLayers; i++)
                {
                    var b = PriceBlocks[i];

                    WA(w, b.SelfAttention);
                    WV(w, b.LNSelfGamma);
                    WV(w, b.LNSelfBeta);

                    WA(w, b.CrossAttention);
                    WV(w, b.LnCrossGamma);
                    WV(w, b.LnCrossBeta);

                    WV(w, b.LNFFNGamma);
                    WV(w, b.LNFFNBeta);

                    b.DecayNetwork.WriteTo(w);
                }

                WM(w, OutputProjection);
                WV(w, OutputBias);

                if (_config.Output.UseConfidenceHead)
                {
                    WM(w, ConfidenceProjection);
                    WV(w, ConfidenceBias);
                }

                WM(w, ContextTypeEmbedding);
            }

            // ===============================
            // FFNs
            // ===============================
            for (int i = 0; i < _config.Text.NumLayers; i++)
            {
                string folderName = $"text_ffn_{i}";
                string ffnDirectory = GetFeedForwardNetworkDirectory(dir, folderName);
                string ffnFilePath = GetFeedForwardNetworkFilePath(dir, folderName);

                EnsureFeedForwardDirectory(ffnDirectory);
                TextBlocks[i].FeedForwardNetwork.Save(ffnFilePath);
            }

            for (int i = 0; i < _config.Price.NumLayers; i++)
            {
                string folderName = $"price_ffn_{i}";
                string ffnDirectory = GetFeedForwardNetworkDirectory(dir, folderName);
                string ffnFilePath = GetFeedForwardNetworkFilePath(dir, folderName);

                EnsureFeedForwardDirectory(ffnDirectory);
                PriceBlocks[i].FeedForwardNetwork.Save(ffnFilePath);
            }

            // ===============================
            // MEMORY
            // ===============================
            var newsMemory = (NewsMemory ?? new List<NewsMemoryEntry>())
                .Where(e => e != null)
                .ToList();

            var priceMemory = (PriceMemory ?? new List<PriceMemoryEntry>())
                .Where(e => e != null)
                .ToList();

            using (var s = new System.IO.FileStream(System.IO.Path.Combine(dir, "news_memory.bin"), System.IO.FileMode.Create, System.IO.FileAccess.Write))
            using (var w = new System.IO.BinaryWriter(s))
            {
                w.Write(LastPriceTimestamp);
                w.Write(NewsMemoryVersionMarkerWithTokenIds);
                w.Write(newsMemory.Count);

                int ed = _config.Price.EmbeddingDim;

                foreach (var e in newsMemory)
                {
                    w.Write(e.AbsoluteTimestamp);
                    w.Write(e.AttentionScore);
                    w.Write(e.QueryCount);

                    int tokenCount = e.TokenIds?.Length ?? 0;
                    w.Write(tokenCount);

                    for (int t = 0; t < tokenCount; t++)
                        w.Write(e.TokenIds[t]);

                    for (int d = 0; d < ed; d++)
                    {
                        float value =
                            e.HiddenState != null && d < e.HiddenState.Length
                                ? e.HiddenState[d]
                                : 0f;

                        w.Write(value);
                    }
                }
            }

            using (var s = new System.IO.FileStream(System.IO.Path.Combine(dir, "price_memory.bin"), System.IO.FileMode.Create, System.IO.FileAccess.Write))
            using (var w = new System.IO.BinaryWriter(s))
            {
                w.Write(priceMemory.Count);

                int ed = _config.Price.EmbeddingDim;

                foreach (var e in priceMemory)
                {
                    w.Write(e.AbsoluteTimestamp);
                    w.Write(e.AttentionScore);
                    w.Write(e.QueryCount);

                    for (int d = 0; d < ed; d++)
                    {
                        float value =
                            e.HiddenState != null && d < e.HiddenState.Length
                                ? e.HiddenState[d]
                                : 0f;

                        w.Write(value);
                    }
                }
            }

            // ===============================
            // TOKENIZER
            // ===============================
            if (Tokenizer != null)
            {
                string tokenizerDir = System.IO.Path.Combine(dir, "tokenizer");

                if (System.IO.File.Exists(tokenizerDir))
                    System.IO.File.Delete(tokenizerDir);

                Tokenizer.Save(tokenizerDir);
            }
        }
        public static Model Load(string dir)
        {
            if (string.IsNullOrWhiteSpace(dir))
                throw new ArgumentException("Load directory cannot be null or empty.", nameof(dir));

            string configPath = System.IO.Path.Combine(dir, "config.json");

            if (!System.IO.File.Exists(configPath))
                throw new System.IO.FileNotFoundException("Model config file not found.", configPath);

            var d = JsonSerializer.Deserialize<Dictionary<string, JsonElement>>(
                System.IO.File.ReadAllText(configPath));

            if (d == null)
                throw new System.IO.InvalidDataException("config.json could not be deserialized.");

            var cfg = BuildConfig(d);
            cfg.Validate();

            var m = new Model(cfg);

            // ===============================
            // PRUNING CONFIG
            // ===============================
            m.PruningConfig ??= new MemoryPruningConfig();

            if (d.ContainsKey("PruningAttentionScoreAlpha"))
                m.PruningConfig.AttentionScoreAlpha = d["PruningAttentionScoreAlpha"].GetSingle();

            if (d.ContainsKey("PruningMinQueryCountForPruning"))
                m.PruningConfig.MinQueryCountForPruning = d["PruningMinQueryCountForPruning"].GetInt32();

            if (d.ContainsKey("PruningNewEntryReserveFraction"))
                m.PruningConfig.NewEntryReserveFraction = d["PruningNewEntryReserveFraction"].GetSingle();

            if (d.ContainsKey("PruningUseAttentionBasedPruning"))
                m.PruningConfig.UseAttentionBasedPruning = d["PruningUseAttentionBasedPruning"].GetBoolean();

            // ===============================
            // WEIGHTS
            // ===============================
            string weightsPath = System.IO.Path.Combine(dir, "weights.bin");

            if (!System.IO.File.Exists(weightsPath))
                throw new System.IO.FileNotFoundException("Model weights file not found.", weightsPath);

            using (var s = new System.IO.FileStream(weightsPath, System.IO.FileMode.Open, System.IO.FileAccess.Read))
            using (var r = new System.IO.BinaryReader(s))
            {
                int version = r.ReadInt32();

                if (version != 1)
                    throw new System.IO.InvalidDataException($"Unsupported TACAMT weights version: {version}");

                RM(r, m.TextTokenEmbedding);

                for (int i = 0; i < cfg.Text.NumLayers; i++)
                {
                    var b = m.TextBlocks[i];

                    RA(r, b.Attention);
                    RV(r, b.LN1Gamma);
                    RV(r, b.LN1Beta);
                    RV(r, b.LN2Gamma);
                    RV(r, b.LN2Beta);
                }

                RM(r, m.PriceInputProjection);
                RV(r, m.PriceInputProjectionBias);

                for (int i = 0; i < cfg.Price.NumLayers; i++)
                {
                    var b = m.PriceBlocks[i];

                    RA(r, b.SelfAttention);
                    RV(r, b.LNSelfGamma);
                    RV(r, b.LNSelfBeta);

                    RA(r, b.CrossAttention);
                    RV(r, b.LnCrossGamma);
                    RV(r, b.LnCrossBeta);

                    RV(r, b.LNFFNGamma);
                    RV(r, b.LNFFNBeta);

                    b.DecayNetwork = ContentAwareDecayNetwork.ReadFrom(r);
                }

                RM(r, m.OutputProjection);
                RV(r, m.OutputBias);

                if (cfg.Output.UseConfidenceHead)
                {
                    RM(r, m.ConfidenceProjection);
                    RV(r, m.ConfidenceBias);
                }

                RM(r, m.ContextTypeEmbedding);

                if (s.Position != s.Length)
                    throw new System.IO.InvalidDataException("Weights file was not fully consumed.");
            }

            // ===============================
            // FFNs
            // ===============================
            for (int i = 0; i < cfg.Text.NumLayers; i++)
            {
                string ffnPath = ResolveFeedForwardNetworkFilePath(dir, $"text_ffn_{i}");

                var loaded = NeuralNetwork.Load(
                    ffnPath,
                    cfg.Runtime.AccelerationType);

                RestoreFeedForwardNetworkState(
                    m.TextBlocks[i].FeedForwardNetwork,
                    loaded);
            }

            for (int i = 0; i < cfg.Price.NumLayers; i++)
            {
                string ffnPath = ResolveFeedForwardNetworkFilePath(dir, $"price_ffn_{i}");

                var loaded = NeuralNetwork.Load(
                    ffnPath,
                    cfg.Runtime.AccelerationType);

                RestoreFeedForwardNetworkState(
                    m.PriceBlocks[i].FeedForwardNetwork,
                    loaded);
            }

            // ===============================
            // NEWS MEMORY
            // ===============================
            string newsMemoryPath = System.IO.Path.Combine(dir, "news_memory.bin");

            if (System.IO.File.Exists(newsMemoryPath))
            {
                using (var s = new System.IO.FileStream(newsMemoryPath, System.IO.FileMode.Open, System.IO.FileAccess.Read))
                using (var r = new System.IO.BinaryReader(s))
                {
                    m.LastPriceTimestamp = r.ReadDouble();

                    int countOrVersionMarker = r.ReadInt32();
                    bool hasTokenIds = false;
                    int count;

                    if (countOrVersionMarker < 0)
                    {
                        int memoryVersion = -countOrVersionMarker;
                        hasTokenIds = memoryVersion >= NewsMemoryFormatVersionWithTokenIds;
                        count = r.ReadInt32();
                    }
                    else
                    {
                        // Backward compatibility:
                        // [LastPriceTimestamp][Count][Entry...]
                        count = countOrVersionMarker;
                    }

                    if (count < 0)
                        throw new System.IO.InvalidDataException($"Invalid news memory count: {count}");

                    int ed = cfg.Price.EmbeddingDim;

                    m.NewsMemory = new List<NewsMemoryEntry>(count);

                    for (int i = 0; i < count; i++)
                    {
                        var e = new NewsMemoryEntry
                        {
                            AbsoluteTimestamp = r.ReadDouble(),
                            AttentionScore = r.ReadSingle(),
                            QueryCount = r.ReadInt32(),
                            HiddenState = new float[ed]
                        };

                        if (hasTokenIds)
                        {
                            int tokenCount = r.ReadInt32();

                            if (tokenCount < 0)
                                throw new System.IO.InvalidDataException($"Invalid token count in news memory: {tokenCount}");

                            if (tokenCount > 0)
                            {
                                e.TokenIds = new int[tokenCount];

                                for (int t = 0; t < tokenCount; t++)
                                    e.TokenIds[t] = r.ReadInt32();
                            }
                            else
                            {
                                e.TokenIds = Array.Empty<int>();
                            }
                        }

                        for (int ddd = 0; ddd < ed; ddd++)
                            e.HiddenState[ddd] = r.ReadSingle();

                        m.NewsMemory.Add(e);
                    }
                }
            }

            // ===============================
            // PRICE MEMORY
            // ===============================
            string priceMemoryPath = System.IO.Path.Combine(dir, "price_memory.bin");

            if (System.IO.File.Exists(priceMemoryPath))
            {
                using (var s = new System.IO.FileStream(priceMemoryPath, System.IO.FileMode.Open, System.IO.FileAccess.Read))
                using (var r = new System.IO.BinaryReader(s))
                {
                    int count = r.ReadInt32();

                    if (count < 0)
                        throw new System.IO.InvalidDataException($"Invalid price memory count: {count}");

                    int ed = cfg.Price.EmbeddingDim;

                    m.PriceMemory = new List<PriceMemoryEntry>(count);

                    for (int i = 0; i < count; i++)
                    {
                        var e = new PriceMemoryEntry
                        {
                            AbsoluteTimestamp = r.ReadDouble(),
                            AttentionScore = r.ReadSingle(),
                            QueryCount = r.ReadInt32(),
                            HiddenState = new float[ed]
                        };

                        for (int ddd = 0; ddd < ed; ddd++)
                            e.HiddenState[ddd] = r.ReadSingle();

                        m.PriceMemory.Add(e);
                    }
                }
            }

            // ===============================
            // TOKENIZER
            // ===============================
            string tokenizerDir = System.IO.Path.Combine(dir, "tokenizer");

            if (System.IO.Directory.Exists(tokenizerDir))
            {
                try
                {
                    var tokenizer = BPETokenizer.Load(tokenizerDir);
                    m.SetTokenizer(tokenizer);
                }
                catch
                {
                    // Preserve current behavior: tokenizer loading is optional.
                }
            }

            return m;
        }
        static void WM(System.IO.BinaryWriter w, float[,] m)
        {
            int r = m.GetLength(0), c = m.GetLength(1);
            w.Write(r); w.Write(c);

            for (int i = 0; i < r; i++)
            {
                for (int j = 0; j < c; j++)
                {
                    w.Write(m[i, j]);
                }
            }
        }
        static void WV(System.IO.BinaryWriter w, float[] v)
        {
            w.Write(v.Length);

            for (int i = 0; i < v.Length; i++)
            {
                w.Write(v[i]);
            }
        }
        static void WA(System.IO.BinaryWriter w, MultiHeadAttention a)
        {
            WM(w, a.WQ);
            WM(w, a.WK);
            WM(w, a.WV);
            WM(w, a.WO);
            WV(w, a.BiasQ);
            WV(w, a.BiasK);
            WV(w, a.BiasV);
            WV(w, a.BiasO);
        }
        static void RM(System.IO.BinaryReader r, float[,] m)
        {
            int rows = r.ReadInt32();
            int cols = r.ReadInt32();

            if (rows != m.GetLength(0) || cols != m.GetLength(1))
            {
                throw new InvalidOperationException($"Matrix shape mismatch. File has [{rows},{cols}], target is [{m.GetLength(0)},{m.GetLength(1)}].");
            }

            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < cols; j++)
                {
                    m[i, j] = r.ReadSingle();
                }
            }
        }

        static void RV(System.IO.BinaryReader r, float[] v)
        {
            int l = r.ReadInt32();

            if (l != v.Length)
            {
                throw new InvalidOperationException($"Vector length mismatch. File has {l}, target is {v.Length}.");
            }

            for (int i = 0; i < l; i++)
            {
                v[i] = r.ReadSingle();
            }
        }
        static void RA(System.IO.BinaryReader r, MultiHeadAttention a)
        {
            RM(r, a.WQ);
            RM(r, a.WK);
            RM(r, a.WV);
            RM(r, a.WO);
            RV(r, a.BiasQ);
            RV(r, a.BiasK);
            RV(r, a.BiasV);
            RV(r, a.BiasO);
        }
        #endregion
    }
}