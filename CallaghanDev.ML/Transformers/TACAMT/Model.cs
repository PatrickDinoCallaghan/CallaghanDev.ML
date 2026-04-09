using CallaghanDev.ML.AccelerationManagers;
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

        public (float[,] predictions, float[,] confidence) ForwardWithCache(NewsStory[] stories, float[,] priceSequence, MultimodalForwardCache cache, bool isTraining = false, Random dropoutRng = null)
        {
            int sl = priceSequence.GetLength(0);

            return ForwardWithCache(stories, priceSequence, rowStart: 0, rowCount: sl,  cache, isTraining, dropoutRng);
        }


        public (float[,] predictions, float[,] confidence) ForwardWithCache(NewsStory[] stories, float[,] priceSequence, int rowStart, int rowCount,  MultimodalForwardCache cache, bool isTraining = false, Random dropoutRng = null)
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
            int embDim = _config.Price.EmbeddingDim;

            var ctxH = new List<float[]>();
            var ctxT = new List<float>();
            var ctxTypes = new List<int>(); // 0 = news, 1 = price

            // Existing news memory: positive age from query time
            if (NewsMemory != null)
            {
                foreach (var e in NewsMemory)
                {
                    ctxH.Add(e.HiddenState);
                    ctxT.Add((float)((currentAbsoluteTimestamp - e.AbsoluteTimestamp) / timeUnitsPerPosition));
                    ctxTypes.Add(0);
                }
            }

            float[,] newSH = null;

            // New stories: convert arrival offsets into positive age
            // Convention:
            //   ArrivalTime = 0   => current story, age 0
            //   ArrivalTime = -3  => story happened 3 units before query, age 3
            if (newStories != null && newStories.Length > 0)
            {
                float[] nst;
                (newSH, nst) = EncodeStories(newStories);

                for (int i = 0; i < nst.Length; i++)
                {
                    var v = new float[embDim];
                    for (int d = 0; d < embDim; d++)
                    {
                        v[d] = newSH[i, d];
                    }

                    ctxH.Add(v);
                    ctxT.Add(-nst[i]);
                    ctxTypes.Add(0);
                }
            }

            // Existing price memory: positive age from query time
            if (PriceMemory != null)
            {
                foreach (var e in PriceMemory)
                {
                    ctxH.Add(e.HiddenState);
                    ctxT.Add((float)((currentAbsoluteTimestamp - e.AbsoluteTimestamp) / timeUnitsPerPosition));
                    ctxTypes.Add(1);
                }
            }

            float[,] cH = null;
            float[] cT = null;

            if (ctxH.Count > 0)
            {
                cH = new float[ctxH.Count, embDim];
                cT = new float[ctxH.Count];

                for (int i = 0; i < ctxH.Count; i++)
                {
                    int cd = Math.Min(embDim, ctxH[i].Length);
                    for (int d = 0; d < cd; d++)
                    {
                        cH[i, d] = ctxH[i][d];
                    }

                    cT[i] = ctxT[i];
                }

                _accel.ApplyContextTypeEmbedding(cH, ContextTypeEmbedding, ctxTypes.ToArray());
            }

            var priceHidden = ForwardPriceDecoder(priceSequence, cH, cT);
            var (predictions, confidenceMatrix) = ProjectToOutput(priceHidden);

            int lastPos = predictions.GetLength(0) - 1;
            var prediction = new float[_config.Output.OutputDim];

            for (int j = 0; j < _config.Output.OutputDim; j++)
            {
                prediction[j] = predictions[lastPos, j];
            }

            float conf = _config.Output.UseConfidenceHead
                ? confidenceMatrix[lastPos, 0]
                : 1.0f;

            UpdateMemoryAttentionScores(priceHidden, cH, ctxH.Count);

            // Persist newly seen stories using absolute timestamps
            if (newSH != null)
            {
                for (int i = 0; i < newStories.Length; i++)
                {
                    var hv = new float[embDim];
                    for (int d = 0; d < embDim; d++)
                    {
                        hv[d] = newSH[i, d];
                    }

                    NewsMemory.Add(new NewsMemoryEntry
                    {
                        HiddenState = hv,
                        AbsoluteTimestamp = currentAbsoluteTimestamp + newStories[i].ArrivalTime * timeUnitsPerPosition
                    });
                }
            }

            int priceSeqLen = priceSequence.GetLength(0);

            for (int t = 0; t < priceSeqLen; t++)
            {
                var pv = new float[embDim];
                for (int d = 0; d < embDim; d++)
                {
                    pv[d] = priceHidden[t, d];
                }

                PriceMemory.Add(new PriceMemoryEntry
                {
                    HiddenState = pv,
                    AbsoluteTimestamp = currentAbsoluteTimestamp + t * timeUnitsPerPosition
                });
            }

            LastPriceTimestamp = currentAbsoluteTimestamp;
            PruneNewsMemory(maxNewsMemorySize);
            PricePruneMemory(maxPriceMemorySize);

            return (prediction, conf);
        }
        /// <summary>
        /// Updates the running attention scores for all memory entries by computing
        /// approximate attention weights from the last price position to each context entry.
        /// This uses a simplified dot-product similarity (without full multi-head attention)
        /// to avoid the cost of a full forward pass, but captures the essential signal:
        /// entries that the model's queries are similar to get high scores.
        /// </summary>
        private void UpdateMemoryAttentionScores(float[,] priceHidden, float[,] contextHidden, int totalCtx)
        {
            if (contextHidden == null || totalCtx == 0) return;

            int embDim = _config.Price.EmbeddingDim;
            int lastPos = priceHidden.GetLength(0) - 1;
            float alpha = PruningConfig.AttentionScoreAlpha;


            float scale = 1.0f / MathF.Sqrt(embDim);
            var scores = _accel.ComputeMemoryAttentionScores(priceHidden, lastPos, contextHidden, totalCtx, scale);
            float maxScore = scores.Length > 0 ? scores.Max() : float.MinValue;

            // Softmax
            float sumExp = 0;
            for (int s = 0; s < totalCtx; s++)
            {
                scores[s] = MathF.Exp(scores[s] - maxScore);
                sumExp += scores[s];
            }
            if (sumExp > 0)
            {
                for (int s = 0; s < totalCtx; s++)
                {
                    scores[s] /= sumExp;
                }
            }

            // Update EMA scores for news memory entries
            int newsCount = NewsMemory?.Count ?? 0;
            for (int i = 0; i < newsCount && i < totalCtx; i++)
            {
                var entry = NewsMemory[i];
                entry.AttentionScore = alpha * scores[i] + (1 - alpha) * entry.AttentionScore;
                entry.QueryCount++;
            }

            // Update EMA scores for price memory entries
            int priceCount = PriceMemory?.Count ?? 0;
            int priceOffset = totalCtx - priceCount;

            for (int i = 0; i < priceCount && (priceOffset + i) < totalCtx; i++)
            {
                var entry = PriceMemory[i];

                int ctxIdx = priceOffset + i;

                if (ctxIdx < totalCtx)
                {
                    entry.AttentionScore = alpha * scores[ctxIdx] + (1 - alpha) * entry.AttentionScore;
                    entry.QueryCount++;
                }
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
            if (NewsMemory.Count <= maxSize)
            {
                return;
            }
            if (!PruningConfig.UseAttentionBasedPruning)
            {
                // Fallback: oldest-first
                NewsMemory = NewsMemory.OrderByDescending(e => e.AbsoluteTimestamp).Take(maxSize).ToList();
                return;
            }

            int reserveCount = Math.Max(1, (int)(maxSize * PruningConfig.NewEntryReserveFraction));
            int scorePruneCount = maxSize - reserveCount;

            // Sort by timestamp to identify newest entries
            var byTime = NewsMemory.OrderByDescending(e => e.AbsoluteTimestamp).ToList();

            // Newest entries get reserved slots regardless of score
            var reserved = byTime.Take(reserveCount).ToHashSet();
            var candidates = byTime.Skip(reserveCount).ToList();

            // Among candidates, keep the ones with highest attention scores
            // Entries not yet queried enough get maximum score (cold-start protection)
            var kept = candidates.OrderByDescending(e => e.QueryCount >= PruningConfig.MinQueryCountForPruning ? e.AttentionScore : float.MaxValue).Take(scorePruneCount).ToList();

            NewsMemory = reserved.Concat(kept).ToList();
        }

        /// <summary>
        /// Prune price memory using attention-based strategy. Same logic as news pruning.
        /// </summary>
        internal void PricePruneMemory(int maxSize)
        {
            if (PriceMemory.Count <= maxSize)
            {
                return;
            }

            if (!PruningConfig.UseAttentionBasedPruning)
            {
                PriceMemory = PriceMemory.OrderByDescending(e => e.AbsoluteTimestamp).Take(maxSize).ToList();
                return;
            }

            int reserveCount = Math.Max(1, (int)(maxSize * PruningConfig.NewEntryReserveFraction));
            int scorePruneCount = maxSize - reserveCount;

            var byTime = PriceMemory.OrderByDescending(e => e.AbsoluteTimestamp).ToList();
            var reserved = byTime.Take(reserveCount).ToHashSet();
            var candidates = byTime.Skip(reserveCount).ToList();

            var kept = candidates.OrderByDescending(e => e.QueryCount >= PruningConfig.MinQueryCountForPruning ? e.AttentionScore : float.MaxValue)
                .Take(scorePruneCount)
                .ToList();

            PriceMemory = reserved.Concat(kept).ToList();
        }

        public void UpdateNewsMemory(NewsStory[] stories, double ts, double tpp = 1.0, int max = 100)
        {
            if (stories == null || stories.Length == 0)
            {
                return;
            }
            var (sh, _) = EncodeStories(stories); int ed = _config.Price.EmbeddingDim;
            for (int i = 0; i < stories.Length; i++)
            {
                var hv = new float[ed];
                for (int d = 0; d < ed; d++)
                {
                    hv[d] = sh[i, d];
                }
                NewsMemory.Add(new NewsMemoryEntry
                {
                    HiddenState = hv,
                    AbsoluteTimestamp = ts + stories[i].ArrivalTime * tpp
                });
            }
            LastPriceTimestamp = ts;
            PruneNewsMemory(max);
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
            int n = stories.Length, ed = _config.Text.EmbeddingDim;
            var sh = new float[n, ed]; var at = new float[n];
            for (int s = 0; s < n; s++)
            {
                var th = ForwardTextEncoder(stories[s].TokenIds); int sl = th.GetLength(0);
                for (int d = 0; d < ed; d++)
                {
                    float sum = 0;
                    for (int t = 0; t < sl; t++)
                        sum += th[t, d];
                    sh[s, d] = sum / sl;  // mean pool - outside the t loop
                }
                at[s] = stories[s].ArrivalTime;
            }
            return (sh, at);
        }

        internal (float[,], float[]) EncodeStoriesWithCache(NewsStory[] stories, MultimodalForwardCache cache)
        {
            int n = stories.Length, ed = _config.Text.EmbeddingDim;
            var sh = new float[n, ed]; var at = new float[n];
            cache.StoryCaches = new List<MultimodalForwardCache>(); cache.StoryTokenCounts = new int[n];
            for (int s = 0; s < n; s++)
            {
                var sc = new MultimodalForwardCache(_config.Text.NumLayers, 0);
                var th = ForwardTextEncoderWithCache(stories[s].TokenIds, sc);
                cache.StoryCaches.Add(sc); cache.StoryTokenCounts[s] = stories[s].TokenIds.Length;
                int sl = th.GetLength(0);
                for (int d = 0; d < ed; d++)
                {
                    float sum = 0;
                    for (int t = 0; t < sl; t++)
                    {
                        sum += th[t, d]; sh[s, d] = sum / sl;
                    }
                }
                at[s] = stories[s].ArrivalTime;
            }
            cache.TextFinalHidden = sh; cache.TextTokenIds = stories.SelectMany(s => s.TokenIds).ToArray();
            return (sh, at);
        }

        private float[,] ForwardPriceDecoder(float[,] priceSequence, float[,] ctx, float[] ctxTimes)
        {
            int sl = priceSequence.GetLength(0);

            var emb = EmbedPriceSequence(priceSequence, sl);

            bool[,] mask = _config.Price.UseDecoderOnly  ? CreateCausalMask(sl)  : null;

            var x = emb;

            // Precompute once (not inside loop)
            float[,] td = null;
            float[] ktr = null;

            if (ctx != null && ctxTimes != null)
            {
                td = _accel.ComputeTimeDiffMatrix(sl, ctxTimes);
                ktr = ctxTimes;
            }

            foreach (var block in PriceBlocks)
            {
                block.SetContext(ctx);
                block.SetTimeData(td, ktr);
                block.SetTraining(false);

                x = block.Forward(x, mask);
            }

            return x;
        }

        /// <summary>
        /// Main forward pass (recommended clean usage)
        /// </summary>
        public float[,] Forward(float[,] x, float[,] context = null, bool[,] selfMask = null, float[,] timeDiffs = null, float[] keyTimes = null, bool isTraining = false, Random rng = null)
        {
            foreach (var block in PriceBlocks)
            {
                block.SetContext(context);
                block.SetTimeData(timeDiffs, keyTimes);
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

        /// <summary>
        /// Helper: precompute time-difference matrix using accel
        /// </summary>
        public float[,] ComputeTimeDiffs(int seqLen, float[] keyTimes)
        {
            return _accel.ComputeTimeDiffMatrix(seqLen, keyTimes);
        }

        // Existing signature remains for compatibility.
        // It now delegates to the new overload.
        internal float[,] ForwardPriceDecoderWithCache(float[,] priceSequence, float[,] storyHidden, float[] storyTimes,  MultimodalForwardCache cache,  bool isTraining = true,  Random dropoutRng = null)
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

        // accepts rowStart/rowCount so caller can avoid SliceRows.
        internal float[,] ForwardPriceDecoderWithCache(float[,] priceSequence, int rowStart, int rowCount, float[,] storyHidden, float[] storyTimes,  MultimodalForwardCache cache, bool isTraining = true, Random dropoutRng = null)
        {
            if (priceSequence == null)
            {
                throw new ArgumentNullException(nameof(priceSequence));
            }
            if (rowStart < 0)
            {
                throw new ArgumentOutOfRangeException(nameof(rowStart));
            }

            if (rowCount < 0)
            {
                throw new ArgumentOutOfRangeException(nameof(rowCount));
            }
            if (rowStart + rowCount > priceSequence.GetLength(0))
            {
                throw new ArgumentException("rowStart + rowCount exceeds priceSequence row count.");
            }

            int sl = rowCount;
            int ed = _config.Price.EmbeddingDim;
            int nh = _config.Price.NumHeads;
            int hd = ed / nh;

            float scale = 1.0f / MathF.Sqrt(hd);

            // IMPORTANT: this requires an EmbedPriceSequence overload that supports offsets.
            // See "Methods you must change" below.
            //var emb = EmbedPriceSequence(priceSequence, rowStart, sl);
            var emb = EmbedPriceSequence(priceSequence, sl);
            cache.PriceEmbedded = emb;
            cache.PriceContinuousInput = priceSequence; // same as before (reference)

            bool[,] selfMask = _config.Price.UseDecoderOnly ? CreateCausalMask(sl) : null;
            var x = emb;

            for (int layer = 0; layer < _config.Price.NumLayers; layer++)
            {
                var block = PriceBlocks[layer];
                var bc = cache.PriceBlockCaches[layer];
                bc.BlockInput = x;

                var sQ = ComputeProjection(x, block.SelfAttention.WQ, block.SelfAttention.BiasQ);
                var sK = ComputeProjection(x, block.SelfAttention.WK, block.SelfAttention.BiasK);
                var sV = ComputeProjection(x, block.SelfAttention.WV, block.SelfAttention.BiasV);

                RotaryPositionEmbedding.ApplyInPlace(sQ, sK, nh);

                bc.SelfQ = sQ;
                bc.SelfK = sK;
                bc.SelfV = sV;

                var sao = _accel.MultiHeadAttentionForward(sQ, sK, sV, nh, scale, selfMask);
                bc.SelfAttnOutput = sao;

                var sp = ComputeProjection(sao, block.SelfAttention.WO, block.SelfAttention.BiasO);
                var sr = _accel.MatrixAdd(x, sp);
                bc.SelfResidualInput = sr;

                var (ns, sm, sv, sn) = _accel.LayerNormForward(sr, block.LNSelfGamma, block.LNSelfBeta);
                bc.LNSelfCache.Input = sr;
                bc.LNSelfCache.Mean = sm;
                bc.LNSelfCache.Variance = sv;
                bc.LNSelfCache.Normalized = sn;
                bc.NormedSelf = ns;

                float[,] nc;

                if (storyHidden != null)
                {
                    float[,] td = null;
                    float[] ktr = null;

                    if (storyTimes != null)
                    {
                        // sl is already the logical sequence length, so time diff matrix stays correct.
                        td = _accel.ComputeTimeDiffMatrix(sl, storyTimes);
                        ktr = storyTimes;
                        bc.TimeDiffs = td;
                        bc.KeyTimesFromRef = ktr;
                    }

                    var cQ = ComputeProjection(ns, block.CrossAttention.WQ, block.CrossAttention.BiasQ);
                    var cK = ComputeProjection(storyHidden, block.CrossAttention.WK, block.CrossAttention.BiasK);
                    var cV = ComputeProjection(storyHidden, block.CrossAttention.WV, block.CrossAttention.BiasV);

                    RotaryPositionEmbedding.ApplyInPlace(cQ, cK, nh);

                    bc.CrossQ = cQ;
                    bc.CrossK = cK;
                    bc.CrossV = cV;

                    var cao = ContentAwareCrossAttentionWithCache(
                        cQ, cK, cV,
                        td, ktr,
                        ns, storyHidden,
                        block, bc,
                        isTraining, dropoutRng);

                    bc.CrossAttnOutput = cao;

                    var cp = ComputeProjection(cao, block.CrossAttention.WO, block.CrossAttention.BiasO);
                    var cr = _accel.MatrixAdd(ns, cp);
                    bc.CrossResidualInput = cr;

                    var (ncr, cm, cvr, cn) = _accel.LayerNormForward(cr, block.LnCrossGamma, block.LnCrossBeta);
                    bc.LNCrossCache.Input = cr;
                    bc.LNCrossCache.Mean = cm;
                    bc.LNCrossCache.Variance = cvr;
                    bc.LNCrossCache.Normalized = cn;
                    bc.NormedCross = ncr;

                    nc = ncr;
                }
                else
                {
                    bc.CrossQ = null;
                    bc.CrossK = null;
                    bc.CrossV = null;
                    bc.CrossAttnOutput = null;
                    bc.TimeDiffs = null;
                    bc.DecayCache = null;

                    var (ncr, cm, cvr, cn) = _accel.LayerNormForward(ns, block.LnCrossGamma, block.LnCrossBeta);
                    bc.LNCrossCache.Input = ns;
                    bc.LNCrossCache.Mean = cm;
                    bc.LNCrossCache.Variance = cvr;
                    bc.LNCrossCache.Normalized = cn;
                    bc.NormedCross = ncr;

                    nc = ncr;
                }

                // NOTE: this still allocates ffnIn rows; speeding that up requires changes elsewhere.
                var ffnIn = new float[sl][];

                for (int i = 0; i < sl; i++)
                {
                    var ir = new float[ed];

                    for (int j = 0; j < ed; j++)
                    {
                        ir[j] = nc[i, j];
                    }

                    ffnIn[i] = ir;
                }

                var ffOut = _accel.FFNForwardBatch(nc, sl, ed, block.FeedForwardNetwork.ForwardPassOnly);

                bc.FFNInputRows = ffnIn;
                bc.FFNOutput = ffOut;

                var fr = _accel.MatrixAdd(nc, ffOut);
                bc.FFNResidualInput = fr;

                var (nf, fm, fv, fn) = _accel.LayerNormForward(fr, block.LNFFNGamma, block.LNFFNBeta);

                bc.LNFFNCache.Input = fr;
                bc.LNFFNCache.Mean = fm;
                bc.LNFFNCache.Variance = fv;
                bc.LNFFNCache.Normalized = fn;

                x = nf;
            }

            return x;
        }

        private float[,] ContentAwareCrossAttentionWithCache(float[,] Q, float[,] K, float[,] V, float[,] timeDiffs, float[] keyTimesFromRef, float[,] queryEmbeddings, float[,] keyEmbeddings, TacamtBlock block, BlockCache bc, bool isTraining = false, Random dropoutRng = null)
        {
            int psl = Q.GetLength(0);
            int tsl = K.GetLength(0);

            int PriceEmbeddingDim = _config.Price.EmbeddingDim;
            int PriceNumHeads = _config.Price.NumHeads;

            return _accel.ContentAwareCrossAttentionWithCache(Q, K, V, timeDiffs, keyTimesFromRef, queryEmbeddings, keyEmbeddings, block, bc, PriceEmbeddingDim, PriceNumHeads, isTraining, dropoutRng);
         
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
                for (int j = 0; j < _config.Text.EmbeddingDim; j++)
                    TextTokenEmbedding[i, j] = SampleGaussian() * std;


            // No throwaway TransformerConfig needed — pass parameters directly.
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
            }
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
                    _config.DecayNetwork.TimeEncodingBases
                );
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
            int maxTokenId = TextTokenEmbedding.GetLength(0) - 1;

            for (int i = 0; i < sl; i++)
            {
                if (ids[i] < 0 || ids[i] > maxTokenId)
                {
                    throw new InvalidOperationException(
                        $"Token ID {ids[i]} at position {i} is out of bounds for embedding table (size {maxTokenId + 1}). " +
                        "The tokenizer vocab size exceeds TextVocabSize in the config. Increase TextVocabSize or retrain the tokenizer.");
                }
            }

            if (sl <= _config.Text.MaxSequenceLength)
            {
                var e = new float[sl, _config.Text.EmbeddingDim];

                for (int i = 0; i < sl; i++)
                    for (int j = 0; j < _config.Text.EmbeddingDim; j++)
                        e[i, j] = TextTokenEmbedding[ids[i], j];

                return e;
            }

            int chunkSize = _config.Text.MaxSequenceLength;
            int numChunks = (sl + chunkSize - 1) / chunkSize;
            var pooled = new float[sl, _config.Text.EmbeddingDim];

            int offset = 0;
            for (int chunk = 0; chunk < numChunks; chunk++)
            {
                int len = Math.Min(chunkSize, sl - offset);
                var chunkEmb = new float[len, _config.Text.EmbeddingDim];

                for (int i = 0; i < len; i++)
                    for (int j = 0; j < _config.Text.EmbeddingDim; j++)
                        chunkEmb[i, j] = TextTokenEmbedding[ids[offset + i], j];

                bool[,] mask = _config.Text.UseDecoderOnly ? CreateCausalMask(len) : null;
                var hidden = chunkEmb;

                foreach (var b in TextBlocks)
                    hidden = b.Forward(hidden, mask);

                for (int i = 0; i < len; i++)
                    for (int j = 0; j < _config.Text.EmbeddingDim; j++)
                        pooled[offset + i, j] = hidden[i, j];

                offset += len;
            }

            return pooled;
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
                throw new ArgumentNullException(nameof(ps));
            if (rowStart < 0)
                throw new ArgumentOutOfRangeException(nameof(rowStart));
            if (rowCount < 0)
                throw new ArgumentOutOfRangeException(nameof(rowCount));
            if (rowStart + rowCount > ps.GetLength(0))
                throw new ArgumentException("rowStart + rowCount exceeds ps row count.");

            var projected = _accel.BatchDotProduct(
                PriceInputProjection,
                ps,
                rowStart,
                rowCount);

            return _accel.MatrixAddBias(projected, PriceInputProjectionBias);
        }
        private float[,] CreatePositionalEncoding(int ml, int d) 
        {
            var pe = new float[ml, d];

            for (int p = 0; p < ml; p++)
            {
                for (int i = 0; i < d; i++)
                {
                    float a = p / MathF.Pow(10000, 2.0f * (i / 2) / d); pe[p, i] = (i % 2 == 0) ? MathF.Sin(a) : MathF.Cos(a);
                }
            }
            return pe; 
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

            bool[,] mask = _config.Price.UseDecoderOnly  ? CreateCausalMask(sl)  : null;

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

        public (float[,] predictions, float[,] confidence) ForwardWithPriceContextAndCache(NewsStory[] stories, float[,] currentPriceInput, float[,] priceCtxHidden, float[] priceCtxTimes, MultimodalForwardCache cache, bool isTraining = true, Random dropoutRng = null)
        {
            int ed = _config.Price.EmbeddingDim;

            float[,] newsHidden = null; 
            float[] newsTimes = null;
            int numNews = 0;

            if (stories != null && stories.Length > 0)
            {
                (newsHidden, newsTimes) = EncodeStoriesWithCache(stories, cache); numNews = newsHidden.GetLength(0);
            }
            else
            {
                cache.TextFinalHidden = null;
                cache.TextTokenIds = null;
            }

            int numPriceCtx = priceCtxHidden != null ? priceCtxHidden.GetLength(0) : 0;
            int totalCtx = numNews + numPriceCtx;

            float[,] combinedHidden = null;
            float[] combinedTimes = null;

            if (totalCtx > 0)
            {
                combinedHidden = new float[totalCtx, ed];
                combinedTimes = new float[totalCtx];

                for (int i = 0; i < numNews; i++)
                {
                    for (int d = 0; d < ed; d++)
                    {
                        combinedHidden[i, d] = newsHidden[i, d];
                        combinedTimes[i] = newsTimes[i];
                    }
                }
                for (int i = 0; i < numPriceCtx; i++)
                {
                    int ci = numNews + i;
                    for (int d = 0; d < ed; d++)
                    {
                        combinedHidden[ci, d] = priceCtxHidden[i, d];
                        combinedTimes[ci] = priceCtxTimes[i];
                    }
                }

                // Apply context type embeddings
                for (int i = 0; i < numNews; i++)
                {
                    for (int d = 0; d < ed; d++)
                    {
                        combinedHidden[i, d] += ContextTypeEmbedding[0, d];  // news
                    }
                }

                for (int i = 0; i < numPriceCtx; i++)
                {
                    for (int d = 0; d < ed; d++)
                    {
                        combinedHidden[numNews + i, d] += ContextTypeEmbedding[1, d];  // price memory
                    }
                }
            }

            cache.NumNewsContext = numNews; cache.NumPriceContext = numPriceCtx; cache.PriceContextHidden = priceCtxHidden;

            cache.TextFinalHidden = combinedHidden; cache.StoryArrivalTimes = combinedTimes;

            var priceHidden = ForwardPriceDecoderWithCache(currentPriceInput, combinedHidden, combinedTimes, cache, isTraining, dropoutRng);

            cache.PriceFinalHidden = priceHidden;

            return ProjectToOutput(priceHidden);
        }

        #region Save/Load
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
                    Enabled = d.ContainsKey("DecayProjectionDim"),
                    ProjectionDim = d.ContainsKey("DecayProjectionDim") ? d["DecayProjectionDim"].GetInt32() : 16,
                    HiddenDim = d.ContainsKey("DecayHiddenDim") ? d["DecayHiddenDim"].GetInt32() : 32,
                    MemAttentionDropout = d.ContainsKey("DecayMemAttnDropout") ? d["DecayMemAttnDropout"].GetSingle() : 0.2f,
                    MlpDropout = d.ContainsKey("DecayMLPDropout") ? d["DecayMLPDropout"].GetSingle() : 0.2f,
                    WeightDecay = d.ContainsKey("DecayWeightDecay") ? d["DecayWeightDecay"].GetSingle() : 0f,
                    TimeEncodingBases = d.ContainsKey("DecayTimeEncodingBases") ? d["DecayTimeEncodingBases"].GetInt32() : 16
                },

                PriceContext = new PriceContextConfig
                {
                    Enabled = d.ContainsKey("PriceContextMinHistoryLength"),
                    MinHistoryLength = d.ContainsKey("PriceContextMinHistoryLength") ? d["PriceContextMinHistoryLength"].GetInt32() : 5,
                    MinCurrentLength = d.ContainsKey("PriceContextMinCurrentLength") ? d["PriceContextMinCurrentLength"].GetInt32() : 5
                },

                RequireSharedCrossAttentionEmbeddingDim = true
            };
        }
        public void Save(string dir)
        {
            if (!Directory.Exists(dir)) Directory.CreateDirectory(dir);

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

                ["DecayProjectionDim"] = _config.DecayNetwork.ProjectionDim,
                ["DecayHiddenDim"] = _config.DecayNetwork.HiddenDim,
                ["DecayMemAttnDropout"] = _config.DecayNetwork.MemAttentionDropout,
                ["DecayMLPDropout"] = _config.DecayNetwork.MlpDropout,
                ["DecayWeightDecay"] = _config.DecayNetwork.WeightDecay,
                ["DecayTimeEncodingBases"] = _config.DecayNetwork.TimeEncodingBases,

                ["PriceContextMinHistoryLength"] = _config.PriceContext.MinHistoryLength,
                ["PriceContextMinCurrentLength"] = _config.PriceContext.MinCurrentLength,

                ["PruningAttentionScoreAlpha"] = PruningConfig.AttentionScoreAlpha,
                ["PruningMinQueryCountForPruning"] = PruningConfig.MinQueryCountForPruning,
                ["PruningNewEntryReserveFraction"] = PruningConfig.NewEntryReserveFraction,
                ["PruningUseAttentionBasedPruning"] = PruningConfig.UseAttentionBasedPruning
            };

            File.WriteAllText(
                Path.Combine(dir, "config.json"),
                JsonSerializer.Serialize(cd, new JsonSerializerOptions { WriteIndented = true })
            );

            // ===============================
            // WEIGHTS
            // ===============================
            using (var s = new FileStream(Path.Combine(dir, "weights.bin"), FileMode.Create))
            using (var w = new BinaryWriter(s))
            {
                w.Write(1); // 🔥 VERSION

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

                WM(w, ContextTypeEmbedding); // 🔥 ALWAYS
            }

            // ===============================
            // FFNs (CRITICAL)
            // ===============================
            for (int i = 0; i < _config.Text.NumLayers; i++)
                TextBlocks[i].FeedForwardNetwork.Save(Path.Combine(dir, $"text_ffn_{i}"));

            for (int i = 0; i < _config.Price.NumLayers; i++)
                PriceBlocks[i].FeedForwardNetwork.Save(Path.Combine(dir, $"price_ffn_{i}"));

            // ===============================
            // MEMORY
            // ===============================
            using (var s = new FileStream(Path.Combine(dir, "news_memory.bin"), FileMode.Create))
            using (var w = new BinaryWriter(s))
            {
                w.Write(LastPriceTimestamp);
                w.Write(NewsMemory.Count);

                int ed = _config.Price.EmbeddingDim;

                foreach (var e in NewsMemory)
                {
                    w.Write(e.AbsoluteTimestamp);
                    w.Write(e.AttentionScore);
                    w.Write(e.QueryCount);

                    for (int d = 0; d < ed; d++)
                        w.Write(e.HiddenState[d]);
                }
            }

            using (var s = new FileStream(Path.Combine(dir, "price_memory.bin"), FileMode.Create))
            using (var w = new BinaryWriter(s))
            {
                w.Write(PriceMemory.Count);

                int ed = _config.Price.EmbeddingDim;

                foreach (var e in PriceMemory)
                {
                    w.Write(e.AbsoluteTimestamp);
                    w.Write(e.AttentionScore);
                    w.Write(e.QueryCount);

                    for (int d = 0; d < ed; d++)
                        w.Write(e.HiddenState[d]);
                }
            }

            if (Tokenizer != null)
                Tokenizer.Save(Path.Combine(dir, "tokenizer"));
        }
        public static Model Load(string dir)
        {
            var d = JsonSerializer.Deserialize<Dictionary<string, JsonElement>>(
                File.ReadAllText(Path.Combine(dir, "config.json")));

            var cfg = BuildConfig(d);
            cfg.Validate();

            var m = new Model(cfg);

            // ===============================
            // PRUNING CONFIG
            // ===============================
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
            using (var s = new FileStream(Path.Combine(dir, "weights.bin"), FileMode.Open))
            using (var r = new BinaryReader(s))
            {
                int version = r.ReadInt32(); // 🔥

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

                RM(r, m.ContextTypeEmbedding); // 🔥 ALWAYS LOAD

                if (s.Position != s.Length)
                    throw new InvalidOperationException("Weights not fully consumed");
            }

            // ===============================
            // FFNs (CRITICAL FIX)
            // ===============================
            for (int i = 0; i < cfg.Text.NumLayers; i++)
            {
                var loaded = NeuralNetwork.Load(Path.Combine(dir, $"text_ffn_{i}"), cfg.Runtime.AccelerationType);
                m.TextBlocks[i].FeedForwardNetwork.RestoreState(loaded.GetInternalData());
            }

            for (int i = 0; i < cfg.Price.NumLayers; i++)
            {
                var loaded = NeuralNetwork.Load(Path.Combine(dir, $"price_ffn_{i}"), cfg.Runtime.AccelerationType);
                m.PriceBlocks[i].FeedForwardNetwork.RestoreState(loaded.GetInternalData());
            }

            // ===============================
            // MEMORY
            // ===============================
            var nmp = Path.Combine(dir, "news_memory.bin");
            if (File.Exists(nmp))
                using (var s = new FileStream(nmp, FileMode.Open))
                using (var r = new BinaryReader(s))
                {
                    m.LastPriceTimestamp = r.ReadDouble();
                    int c = r.ReadInt32();
                    int ed = cfg.Price.EmbeddingDim;

                    m.NewsMemory = new List<NewsMemoryEntry>(c);

                    for (int i = 0; i < c; i++)
                    {
                        var e = new NewsMemoryEntry
                        {
                            AbsoluteTimestamp = r.ReadDouble(),
                            AttentionScore = r.ReadSingle(),
                            QueryCount = r.ReadInt32(),
                            HiddenState = new float[ed]
                        };

                        for (int ddd = 0; ddd < ed; ddd++)
                            e.HiddenState[ddd] = r.ReadSingle();

                        m.NewsMemory.Add(e);
                    }
                }

            var pmp = Path.Combine(dir, "price_memory.bin");
            if (File.Exists(pmp))
                using (var s = new FileStream(pmp, FileMode.Open))
                using (var r = new BinaryReader(s))
                {
                    int c = r.ReadInt32();
                    int ed = cfg.Price.EmbeddingDim;

                    m.PriceMemory = new List<PriceMemoryEntry>(c);

                    for (int i = 0; i < c; i++)
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

            var tokDir = Path.Combine(dir, "tokenizer");
            if (Directory.Exists(tokDir))
            {
                try { m.Tokenizer = BPETokenizer.Load(tokDir); } catch { }
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
            int rows = r.ReadInt32(), cols = r.ReadInt32();

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