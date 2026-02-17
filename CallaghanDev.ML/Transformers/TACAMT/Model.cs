using CallaghanDev.ML.AccelerationManagers;
using CallaghanDev.ML.Enums;
using CallaghanDev.ML.Transformers.Cache;
using System;
using System.Collections.Generic;
using System.Linq;

namespace CallaghanDev.ML.Transformers.TACAMT
{
    public partial class Model
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


        private readonly Config _config;
        private readonly Random _random;
        private readonly IAccelerationManager _accel;

        public Config Config => _config;
        public IAccelerationManager AccelerationManager => _accel;

        public float[,] TextTokenEmbedding { get; set; }
        public float[,] TextPositionalEncoding { get; set; }
        public Transformers.TransformerBlock[] TextBlocks { get; set; }

        public float[,] PriceInputProjection { get; set; }
        public float[] PriceInputProjectionBias { get; set; }
        public float[,] PricePositionalEncoding { get; set; }
        public TransformerBlock[] PriceBlocks { get; set; }

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

        public Model(Config config, Random random = null)
        {
            config.Validate();
            _config = config;
            _random = random ?? new Random();
            if (_config.AccelerationType == AccelerationType.GPU || _config.AccelerationType == AccelerationType.CUDA)
            {
                _accel = new AccelerationGPU(_config.AccelerationType, _config.AccelerationDeviceId);
            }
            else if (_config.AccelerationType == AccelerationType.CPU)
            {
                _accel = new AccelerationCPU();
            }
            else if (_config.AccelerationType == AccelerationType.MultiThreadCPU)
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

            int ed = _config.PriceEmbeddingDim;
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
        /// Attach a trained BPE tokenizer. Validates that vocab size matches config.
        /// </summary>
        public void SetTokenizer(BPETokenizer tokenizer)
        {
            if (tokenizer.VocabSize != _config.TextVocabSize)
            {
                throw new ArgumentException($"Tokenizer vocab size ({tokenizer.VocabSize}) does not match config TextVocabSize ({_config.TextVocabSize}). Either retrain the tokenizer with vocabSize={_config.TextVocabSize} or update the config.");
            }
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
                int ed = _config.PriceEmbeddingDim;
                for (int i = 0; i < numNews; i++)
                {
                    for (int d = 0; d < ed; d++)
                    {
                        sh[i, d] += ContextTypeEmbedding[0, d];
                    }
                }
            }

            return ProjectToOutput(ForwardPriceDecoder(priceSequence, sh, st));
        }
        public (float[,] predictions, float[,] confidence) ForwardWithCache(NewsStory[] stories, float[,] priceSequence, MultimodalForwardCache cache, bool isTraining = false, Random dropoutRng = null)
        {
            float[,] sh = null; float[] st = null;
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

            // Apply context type embedding: type 0 = news
            if (sh != null)
            {
                int numNews = sh.GetLength(0);
                int ed = _config.PriceEmbeddingDim;
                for (int i = 0; i < numNews; i++)
                {
                    for (int d = 0; d < ed; d++)
                    {
                        sh[i, d] += ContextTypeEmbedding[0, d];
                    }
                }
            }

            var ph = ForwardPriceDecoderWithCache(priceSequence, sh, st, cache, isTraining, dropoutRng);
            cache.PriceFinalHidden = ph;
            return ProjectToOutput(ph);
        }

        public (float[,], float[,]) Forward(int[] t, float[,] p)
        {
            return t != null && t.Length > 0 ? Forward(new[] { new NewsStory(t, 0f) }, p) : Forward((NewsStory[])null, p);
        }
        public (float[,], float[,]) ForwardWithCache(int[] t, float[,] p, MultimodalForwardCache c, bool isTraining = false, Random dropoutRng = null)
        {
            return t != null && t.Length > 0 ? ForwardWithCache(new[] { new NewsStory(t, 0f) }, p, c, isTraining, dropoutRng) : ForwardWithCache((NewsStory[])null, p, c, isTraining, dropoutRng);
        }
            

        public (float[] prediction, float confidence) PredictNext(NewsStory[] stories, float[,] priceSequence)
        {
            var (pred, conf) = Forward(stories, priceSequence);
            int last = pred.GetLength(0) - 1;
            var p = new float[_config.OutputDim];

            for (int j = 0; j < _config.OutputDim; j++)
            {
                p[j] = pred[last, j];
            }
            return (p, _config.UseConfidenceHead ? conf[last, 0] : 1.0f);
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

        public (float[] prediction, float confidence) PredictWithMemory(NewsStory[] newStories, float[,] priceSequence, double currentAbsoluteTimestamp, double timeUnitsPerPosition = 1.0, int maxNewsMemorySize = 100, int maxPriceMemorySize = 200)
        {
            int embDim = _config.PriceEmbeddingDim;
            var ctxH = new List<float[]>(); var ctxT = new List<float>();
            var ctxTypes = new List<int>();  // Track type per entry

            if (NewsMemory != null) foreach (var e in NewsMemory)
            { 
                ctxH.Add(e.HiddenState); 
                ctxT.Add(-(float)((currentAbsoluteTimestamp - e.AbsoluteTimestamp) / timeUnitsPerPosition)); 
                ctxTypes.Add(0); 
            }

            float[,] newSH = null;
            if (newStories != null && newStories.Length > 0)
            {
                float[] nst; (newSH, nst) = EncodeStories(newStories);
                for (int i = 0; i < nst.Length; i++)
                {
                    var v = new float[embDim];
                    for (int d = 0; d < embDim; d++)
                    {
                        v[d] = newSH[i, d];
                    }
                    ctxH.Add(v);
                    ctxT.Add(nst[i]);
                    ctxTypes.Add(0);
                }
            }

            if (PriceMemory != null)
            {
                foreach (var e in PriceMemory)
                { 
                    ctxH.Add(e.HiddenState); 
                    ctxT.Add(-(float)((currentAbsoluteTimestamp - e.AbsoluteTimestamp) / timeUnitsPerPosition)); 
                    ctxTypes.Add(1); 
                }
            }

            float[,] cH = null; float[] cT = null;
            if (ctxH.Count > 0)
            {
                cH = new float[ctxH.Count, embDim]; cT = new float[ctxH.Count];
                for (int i = 0; i < ctxH.Count; i++)
                {
                    int cd = Math.Min(embDim, ctxH[i].Length);
                    for (int d = 0; d < cd; d++)
                    {
                        cH[i, d] = ctxH[i][d];
                    }
                    cT[i] = ctxT[i];
                }

                // Apply context type embeddings
                for (int i = 0; i < ctxH.Count; i++)
                {
                    for (int d = 0; d < embDim; d++)
                    {
                        cH[i, d] += ContextTypeEmbedding[ctxTypes[i], d];
                    }
                }
            }

            var priceHidden = ForwardPriceDecoder(priceSequence, cH, cT);
            var (predictions, confidenceMatrix) = ProjectToOutput(priceHidden);
            int lastPos = predictions.GetLength(0) - 1;
            var prediction = new float[_config.OutputDim];

            for (int j = 0; j < _config.OutputDim; j++)
            {
                prediction[j] = predictions[lastPos, j];
            }

            float conf = _config.UseConfidenceHead ? confidenceMatrix[lastPos, 0] : 1.0f;

            UpdateMemoryAttentionScores(priceHidden, cH, ctxH.Count);

            if (newSH != null)
            {
                for (int i = 0; i < newStories.Length; i++)
                {
                    var hv = new float[embDim]; for (int d = 0; d < embDim; d++) hv[d] = newSH[i, d];
                    NewsMemory.Add(new NewsMemoryEntry { HiddenState = hv, AbsoluteTimestamp = currentAbsoluteTimestamp + newStories[i].ArrivalTime * timeUnitsPerPosition });
                }
            }

            int priceSeqLen = priceSequence.GetLength(0);

            for (int t = 0; t < priceSeqLen; t++)
            {
                var pv = new float[embDim]; for (int d = 0; d < embDim; d++) pv[d] = priceHidden[t, d];
                PriceMemory.Add(new PriceMemoryEntry { HiddenState = pv, AbsoluteTimestamp = currentAbsoluteTimestamp + t * timeUnitsPerPosition });
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

            int embDim = _config.PriceEmbeddingDim;
            int lastPos = priceHidden.GetLength(0) - 1;
            float alpha = PruningConfig.AttentionScoreAlpha;

            // Compute approximate attention weights from last price position to each context entry
            var scores = new float[totalCtx];
            float maxScore = float.MinValue;
            float scale = 1.0f / MathF.Sqrt(embDim);

            for (int s = 0; s < totalCtx; s++)
            {
                float dot = 0;
                for (int d = 0; d < embDim; d++)
                {
                    dot += priceHidden[lastPos, d] * contextHidden[s, d];
                }
                scores[s] = dot * scale;
                if (scores[s] > maxScore)
                { 
                    maxScore = scores[s];
                }
            }

            // Softmax
            float sumExp = 0;
            for (int s = 0; s < totalCtx; s++)
            {
                scores[s] = MathF.Exp(scores[s] - maxScore);
                sumExp += scores[s];
            }
            if (sumExp > 0)
                for (int s = 0; s < totalCtx; s++)
                    scores[s] /= sumExp;

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
            if (NewsMemory.Count <= maxSize) return;

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
            var kept = candidates.OrderByDescending(e => e.QueryCount >= PruningConfig.MinQueryCountForPruning  ? e.AttentionScore : float.MaxValue)
                .Take(scorePruneCount)
                .ToList();

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
            var (sh, _) = EncodeStories(stories); int ed = _config.PriceEmbeddingDim;
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

        internal (float[,], float[]) EncodeStories(NewsStory[] stories)
        {
            int n = stories.Length, ed = _config.TextEmbeddingDim;
            var sh = new float[n, ed]; var at = new float[n];
            for (int s = 0; s < n; s++)
            {
                var th = ForwardTextEncoder(stories[s].TokenIds); int sl = th.GetLength(0);
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
            return (sh, at);
        }

        internal (float[,], float[]) EncodeStoriesWithCache(NewsStory[] stories, MultimodalForwardCache cache)
        {
            int n = stories.Length, ed = _config.TextEmbeddingDim;
            var sh = new float[n, ed]; var at = new float[n];
            cache.StoryCaches = new List<MultimodalForwardCache>(); cache.StoryTokenCounts = new int[n];
            for (int s = 0; s < n; s++)
            {
                var sc = new MultimodalForwardCache(_config.TextNumLayers, 0);
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
            int sl = priceSequence.GetLength(0); var emb = EmbedPriceSequence(priceSequence, sl);
            bool[,] mask = _config.PriceUseDecoderOnly ? CreateCausalMask(sl) : null;
            var x = emb;
            foreach (var block in PriceBlocks)
            {
                float[,] td = null; float[] ktr = null;
                if (ctx != null && ctxTimes != null) 
                { 
                    td = block.ComputeTimeDiffMatrix(sl, ctxTimes); 
                    ktr = ctxTimes;
                }
                x = block.Forward(x, ctx, mask, _accel, td, ktr, isTraining: false);
            }
            return x;
        }

        internal float[,] ForwardPriceDecoderWithCache(float[,] priceSequence, float[,] storyHidden, float[] storyTimes, MultimodalForwardCache cache, bool isTraining = true, Random dropoutRng = null)
        {
            int sl = priceSequence.GetLength(0), ed = _config.PriceEmbeddingDim, nh = _config.PriceNumHeads, hd = ed / nh;
            float scale = 1.0f / MathF.Sqrt(hd);
            var emb = EmbedPriceSequence(priceSequence, sl); cache.PriceEmbedded = emb; cache.PriceContinuousInput = priceSequence;
            bool[,] selfMask = _config.PriceUseDecoderOnly ? CreateCausalMask(sl) : null;
            var x = emb;

            for (int layer = 0; layer < _config.PriceNumLayers; layer++)
            {
                var block = PriceBlocks[layer]; var bc = cache.PriceBlockCaches[layer]; bc.BlockInput = x;
                var sQ = ComputeProjection(x, block.SelfAttention.WQ, block.SelfAttention.BiasQ);
                var sK = ComputeProjection(x, block.SelfAttention.WK, block.SelfAttention.BiasK);
                var sV = ComputeProjection(x, block.SelfAttention.WV, block.SelfAttention.BiasV);
                bc.SelfQ = sQ; bc.SelfK = sK; bc.SelfV = sV;
                var sao = _accel.MultiHeadAttentionForward(sQ, sK, sV, nh, scale, selfMask); bc.SelfAttnOutput = sao;
                var sp = ComputeProjection(sao, block.SelfAttention.WO, block.SelfAttention.BiasO);
                var sr = _accel.MatrixAdd(x, sp); bc.SelfResidualInput = sr;
                var (ns, sm, sv, sn) = _accel.LayerNormForward(sr, block.LNSelfGamma, block.LNSelfBeta);
                bc.LNSelfCache.Input = sr; bc.LNSelfCache.Mean = sm; bc.LNSelfCache.Variance = sv; bc.LNSelfCache.Normalized = sn; bc.NormedSelf = ns;

                float[,] nc;
                if (storyHidden != null)
                {
                    float[,] td = null; float[] ktr = null;
                    if (storyTimes != null)
                    { 
                        td = block.ComputeTimeDiffMatrix(sl, storyTimes); 
                        ktr = storyTimes; 
                        bc.TimeDiffs = td; 
                        bc.KeyTimesFromRef = ktr;
                    }

                    var cQ = ComputeProjection(ns, block.CrossAttention.WQ, block.CrossAttention.BiasQ);
                    var cK = ComputeProjection(storyHidden, block.CrossAttention.WK, block.CrossAttention.BiasK);
                    var cV = ComputeProjection(storyHidden, block.CrossAttention.WV, block.CrossAttention.BiasV);
                    bc.CrossQ = cQ; bc.CrossK = cK; bc.CrossV = cV;
                    var cao = ContentAwareCrossAttentionWithCache(cQ, cK, cV, td, ktr, ns, storyHidden, block, bc, isTraining, dropoutRng);
                    bc.CrossAttnOutput = cao;
                    var cp = ComputeProjection(cao, block.CrossAttention.WO, block.CrossAttention.BiasO);
                    var cr = _accel.MatrixAdd(ns, cp); bc.CrossResidualInput = cr;
                    var (ncr, cm, cvr, cn) = _accel.LayerNormForward(cr, block.LNCrossGamma, block.LNCrossBeta);
                    bc.LNCrossCache.Input = cr; bc.LNCrossCache.Mean = cm; bc.LNCrossCache.Variance = cvr; bc.LNCrossCache.Normalized = cn; bc.NormedCross = ncr; nc = ncr;
                }
                else
                {
                    bc.CrossQ = null; bc.CrossK = null; bc.CrossV = null; bc.CrossAttnOutput = null; bc.TimeDiffs = null; bc.DecayCache = null;
                    var (ncr, cm, cvr, cn) = _accel.LayerNormForward(ns, block.LNCrossGamma, block.LNCrossBeta);
                    bc.LNCrossCache.Input = ns; bc.LNCrossCache.Mean = cm; bc.LNCrossCache.Variance = cvr; bc.LNCrossCache.Normalized = cn; bc.NormedCross = ncr; nc = ncr;
                }

                var ffnIn = new float[sl][]; var ffOut = new float[sl, ed];
                for (int i = 0; i < sl; i++)
                {
                    var ir = new float[ed]; for (int j = 0; j < ed; j++) ir[j] = nc[i, j]; ffnIn[i] = ir;
                    var or2 = block.FeedForwardNetwork.ForwardPassOnly(ir); for (int j = 0; j < ed; j++) ffOut[i, j] = or2[j];
                }
                bc.FFNInputRows = ffnIn; bc.FFNOutput = ffOut;
                var fr = _accel.MatrixAdd(nc, ffOut); bc.FFNResidualInput = fr;
                var (nf, fm, fv, fn) = _accel.LayerNormForward(fr, block.LNFFNGamma, block.LNFFNBeta);
                bc.LNFFNCache.Input = fr; bc.LNFFNCache.Mean = fm; bc.LNFFNCache.Variance = fv; bc.LNFFNCache.Normalized = fn;
                x = nf;
            }
            return x;
        }

        private float[,] ContentAwareCrossAttentionWithCache(float[,] Q, float[,] K, float[,] V, float[,] timeDiffs, float[] keyTimesFromRef, float[,] queryEmbeddings, float[,] keyEmbeddings, TransformerBlock block, BlockCache bc, bool isTraining = false, Random dropoutRng = null)
        {
            int psl = Q.GetLength(0), tsl = K.GetLength(0), ed = _config.PriceEmbeddingDim, nh = _config.PriceNumHeads, hd = ed / nh;
            float scale = 1.0f / MathF.Sqrt(hd);
            var output = new float[psl, ed];

            float[,,] decayBias = null;
            if (timeDiffs != null)
            {
                var (bias, decayCache) = block.DecayNetwork.Forward(queryEmbeddings, keyEmbeddings, timeDiffs, keyTimesFromRef, isTraining, dropoutRng);

                decayBias = bias; bc.DecayCache = decayCache;
            }

            bc.CrossAttentionWeights = new float[nh][,]; bc.CrossScoresPreSoftmax = new float[nh][,];
            for (int h = 0; h < nh; h++)
            {
                int si = h * hd;
                var scores = new float[psl, tsl]; var weights = new float[psl, tsl];
                for (int p = 0; p < psl; p++)
                {
                    float mx = float.MinValue;
                    for (int s = 0; s < tsl; s++)
                    {
                        float dot = 0; for (int d = 0; d < hd; d++) dot += Q[p, si + d] * K[s, si + d];
                        float sc = dot * scale; if (decayBias != null) sc += decayBias[p, s, h];
                        scores[p, s] = sc; if (sc > mx) mx = sc;
                    }
                    float se = 0;
                    for (int s = 0; s < tsl; s++) 
                    { 
                        weights[p, s] = MathF.Exp(scores[p, s] - mx);
                        se += weights[p, s]; 
                    }
                    if (se > 0) 
                    { 
                        for (int s = 0; s < tsl; s++) 
                        { weights[p, s] /= se; 
                        }
                    }
                    for (int d = 0; d < hd; d++)
                    { 
                        float v = 0; 
                        for (int s = 0; s < tsl; s++)
                        {
                            v += weights[p, s] * V[s, si + d];
                            output[p, si + d] = v;
                        }
                    }
                }
                bc.CrossAttentionWeights[h] = weights; bc.CrossScoresPreSoftmax[h] = scores;
            }
            return output;
        }

        internal (float[,], float[,]) ProjectToOutput(float[,] hidden)
        {
            int sl = hidden.GetLength(0), ed = _config.PriceEmbeddingDim;
            var pred = new float[sl, _config.OutputDim];
            for (int i = 0; i < sl; i++)
            {
                var row = new float[ed]; for (int k = 0; k < ed; k++) row[k] = hidden[i, k];
                var o = _accel.CalculateDotProduct(OutputProjection, row);
                for (int j = 0; j < _config.OutputDim; j++)
                {
                    pred[i, j] = o[j] + OutputBias[j];
                }

            }
            float[,] conf = null;
            if (_config.UseConfidenceHead)
            {
                conf = new float[sl, 1];
                for (int i = 0; i < sl; i++)
                {
                    float l = ConfidenceBias[0];
                    for (int k = 0; k < ed; k++)
                    {
                        l += ConfidenceProjection[0, k] * hidden[i, k];
                        conf[i, 0] = Sigmoid(l);
                    }
                }
            }
            return (pred, conf);
        }

        private float[,] ForwardTextEncoder(int[] tokenIds)
        {
            int sl = tokenIds.Length; var emb = EmbedTextTokens(tokenIds, sl);
            bool[,] mask = _config.TextUseDecoderOnly ? CreateCausalMask(sl) : null;
            var x = emb; foreach (var b in TextBlocks) x = b.Forward(x, mask); 
            return x;
        }

        private float[,] ForwardTextEncoderWithCache(int[] tokenIds, MultimodalForwardCache cache)
        {
            int sl = tokenIds.Length, ed = _config.TextEmbeddingDim;
            var emb = EmbedTextTokens(tokenIds, sl); cache.TextEmbedded = emb; cache.TextTokenIds = tokenIds;
            bool[,] mask = _config.TextUseDecoderOnly ? CreateCausalMask(sl) : null;
            var x = emb;


            for (int layer = 0; layer < _config.TextNumLayers; layer++)
            {
                cache.TextLayerInputs.Add(x); 
                var b = TextBlocks[layer];
                var ac = cache.TextAttentionCaches[layer]; ac.Input = x;

                var ao = AttentionForwardWithCache(b.Attention, x, x, x, mask, ac);
                var ar = _accel.MatrixAdd(x, ao);

                var l1c = cache.TextLN1Caches[layer];

                var (n1, m1, v1, nr1) = _accel.LayerNormForward(ar, b.LN1Gamma, b.LN1Beta);

                l1c.Input = ar; l1c.Mean = m1; l1c.Variance = v1; l1c.Normalized = nr1;

                var fi = new float[sl][]; var fo = new float[sl, ed];

                for (int i = 0; i < sl; i++)
                {
                    var ir = new float[ed];
                    for (int j = 0; j < ed; j++)
                    { 
                        ir[j] = n1[i, j]; 
                        fi[i] = ir;
                    }
                    var or2 = b.FeedForwardNetwork.ForwardPassOnly(ir);
                    for (int j = 0; j < ed; j++) 
                    {
                        fo[i, j] = or2[j]; 
                    }
                }

                cache.TextFFNInputs.Add(fi); cache.TextFFNOutputs.Add(fo);

                var fr = _accel.MatrixAdd(n1, fo);

                var l2c = cache.TextLN2Caches[layer];

                var (n2, m2, v2, nr2) = _accel.LayerNormForward(fr, b.LN2Gamma, b.LN2Beta);

                l2c.Input = fr; 
                l2c.Mean = m2;
                l2c.Variance = v2;
                l2c.Normalized = nr2;

                x = n2;
            }
            cache.TextFinalHidden = x; return x;
        }

        private void InitTextEncoder()
        {
            TextTokenEmbedding = new float[_config.TextVocabSize, _config.TextEmbeddingDim];
            float std = MathF.Sqrt(1.0f / _config.TextEmbeddingDim);

            for (int i = 0; i < _config.TextVocabSize; i++)
            {
                for (int j = 0; j < _config.TextEmbeddingDim; j++)
                {
                    TextTokenEmbedding[i, j] = SampleGaussian() * std;
                }
            }


            TextPositionalEncoding = CreatePositionalEncoding(_config.TextMaxSequenceLength, _config.TextEmbeddingDim);
            var tc = new TransformerConfig 
            { 
                VocabSize = _config.TextVocabSize, 
                MaxSequenceLength = _config.TextMaxSequenceLength, 
                EmbeddingDim = _config.TextEmbeddingDim, 
                NumHeads = _config.TextNumHeads,
                NumLayers = _config.TextNumLayers, 
                FeedForwardDim = _config.TextFeedForwardDim,
                FFNActivationType = _config.FFNActivationType,
                UseDecoderOnly = _config.TextUseDecoderOnly,
                AccelerationType = _config.AccelerationType,
                AccelerationDeviceId = _config.AccelerationDeviceId,
                L2RegulationLamda = _config.L2RegulationLamda 
            };

            TextBlocks = new Transformers.TransformerBlock[_config.TextNumLayers];

            for (int i = 0; i < _config.TextNumLayers; i++)
            {
                TextBlocks[i] = new Transformers.TransformerBlock(tc, _accel, _random);
            }
        }

        private void InitPriceDecoder()
        {
            PriceInputProjection = new float[_config.PriceEmbeddingDim, _config.PriceInputFeatureDim];
            PriceInputProjectionBias = new float[_config.PriceEmbeddingDim];
            float std = MathF.Sqrt(2.0f / (_config.PriceInputFeatureDim + _config.PriceEmbeddingDim));
            for (int i = 0; i < _config.PriceEmbeddingDim; i++) for (int j = 0; j < _config.PriceInputFeatureDim; j++) PriceInputProjection[i, j] = SampleGaussian() * std;
            PricePositionalEncoding = CreatePositionalEncoding(_config.PriceMaxSequenceLength, _config.PriceEmbeddingDim);
            PriceBlocks = new TransformerBlock[_config.PriceNumLayers];
            for (int i = 0; i < _config.PriceNumLayers; i++)
            {
                PriceBlocks[i] = new TransformerBlock(
                    _config.PriceEmbeddingDim,
                    _config.PriceNumHeads, 
                    _config.PriceFeedForwardDim,
                    _config.FFNActivationType, 
                    _accel, 
                    _random, 
                    _config.AccelerationType,
                    _config.AccelerationDeviceId,
                    _config.L2RegulationLamda, 
                    _config.DecayProjectionDim, 
                    _config.DecayHiddenDim,
                    _config.DecayMemAttnDropout,
                    _config.DecayMLPDropout,
                    _config.DecayWeightDecay,
                    _config.DecayTimeEncodingBases);
            }
        }

        private void InitOutputHead()
        {
            int ed = _config.PriceEmbeddingDim;
            OutputProjection = new float[_config.OutputDim, ed]; 
            OutputBias = new float[_config.OutputDim];

            float std = MathF.Sqrt(2.0f / (ed + _config.OutputDim));

            for (int i = 0; i < _config.OutputDim; i++)
            {
                for (int j = 0; j < ed; j++)
                {
                    OutputProjection[i, j] = SampleGaussian() * std;
                }
            }
            if (_config.UseConfidenceHead)
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

        private float[,] EmbedTextTokens(int[] ids, int sl) { var e = new float[sl, _config.TextEmbeddingDim]; for (int i = 0; i < sl; i++) for (int j = 0; j < _config.TextEmbeddingDim; j++) e[i, j] = TextTokenEmbedding[ids[i], j] + TextPositionalEncoding[i, j]; return e; }
        private float[,] EmbedPriceSequence(float[,] ps, int sl) { var p = _accel.BatchDotProduct(PriceInputProjection, ps); var e = new float[sl, _config.PriceEmbeddingDim]; for (int i = 0; i < sl; i++) for (int j = 0; j < _config.PriceEmbeddingDim; j++) e[i, j] = p[i, j] + PriceInputProjectionBias[j] + PricePositionalEncoding[i, j]; return e; }
        private float[,] CreatePositionalEncoding(int ml, int d) { var pe = new float[ml, d]; for (int p = 0; p < ml; p++) for (int i = 0; i < d; i++) { float a = p / MathF.Pow(10000, 2.0f * (i / 2) / d); pe[p, i] = (i % 2 == 0) ? MathF.Sin(a) : MathF.Cos(a); } return pe; }
        private bool[,] CreateCausalMask(int sl) { var m = new bool[sl, sl]; for (int i = 0; i < sl; i++) for (int j = 0; j <= i; j++) m[i, j] = true; return m; }
        private float[,] ComputeProjection(float[,] input, float[,] w, float[] b) { var p = _accel.BatchDotProduct(w, input); int r = p.GetLength(0), c = p.GetLength(1); var res = new float[r, c]; for (int i = 0; i < r; i++) for (int j = 0; j < c; j++) res[i, j] = p[i, j] + b[j]; return res; }

        private float[,] AttentionForwardWithCache(MultiHeadAttention attn, float[,] qs, float[,] ks, float[,] vs, bool[,] mask, AttentionCache cache)
        {
            int ed = _config.TextEmbeddingDim, hd = ed / _config.TextNumHeads;
            var Q = ComputeProjection(qs, attn.WQ, attn.BiasQ); var K = ComputeProjection(ks, attn.WK, attn.BiasK); var V = ComputeProjection(vs, attn.WV, attn.BiasV);
            cache.Q = Q; cache.K = K; cache.V = V; float s = 1.0f / MathF.Sqrt(hd);
            var c = _accel.MultiHeadAttentionForward(Q, K, V, _config.TextNumHeads, s, mask); cache.AttentionOutput = c;
            return ComputeProjection(c, attn.WO, attn.BiasO);
        }

        private float SampleGaussian() { float u1 = 1f - _random.NextSingle(), u2 = 1f - _random.NextSingle(); return MathF.Sqrt(-2f * MathF.Log(u1)) * MathF.Cos(2f * MathF.PI * u2); }
        private static float Sigmoid(float x) { if (x >= 0) { float ex = MathF.Exp(-x); return 1f / (1f + ex); } else { float ex = MathF.Exp(x); return ex / (1f + ex); } }

        public float[,] EncodePriceHistory(float[,] histPrices)
        {
            int sl = histPrices.GetLength(0);
            var emb = EmbedPriceSequence(histPrices, sl);
            bool[,] mask = _config.PriceUseDecoderOnly ? CreateCausalMask(sl) : null;
            var x = emb;

            foreach (var block in PriceBlocks)
            {
                x = block.Forward(x, null, mask, _accel, isTraining: false);
            }
            return x;
        }

        public (float[,] predictions, float[,] confidence) ForwardWithPriceContextAndCache(NewsStory[] stories, float[,] currentPriceInput, float[,] priceCtxHidden, float[] priceCtxTimes, MultimodalForwardCache cache, bool isTraining = true, Random dropoutRng = null)
        {
            int ed = _config.PriceEmbeddingDim;
            float[,] newsHidden = null; float[] newsTimes = null; int numNews = 0;
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
        public void Save(string dir)
        {
            if (!System.IO.Directory.Exists(dir)) System.IO.Directory.CreateDirectory(dir);
            var cd = new Dictionary<string, object>
            {
                ["TextVocabSize"] = _config.TextVocabSize,
                ["TextMaxSequenceLength"] = _config.TextMaxSequenceLength,
                ["TextEmbeddingDim"] = _config.TextEmbeddingDim,
                ["TextNumHeads"] = _config.TextNumHeads,
                ["TextNumLayers"] = _config.TextNumLayers,
                ["TextFeedForwardDim"] = _config.TextFeedForwardDim,
                ["TextUseDecoderOnly"] = _config.TextUseDecoderOnly,
                ["FreezeTextEncoder"] = _config.FreezeTextEncoder,
                ["PriceInputFeatureDim"] = _config.PriceInputFeatureDim,
                ["PriceMaxSequenceLength"] = _config.PriceMaxSequenceLength,
                ["PriceEmbeddingDim"] = _config.PriceEmbeddingDim,
                ["PriceNumHeads"] = _config.PriceNumHeads,
                ["PriceNumLayers"] = _config.PriceNumLayers,
                ["PriceFeedForwardDim"] = _config.PriceFeedForwardDim,
                ["PriceUseDecoderOnly"] = _config.PriceUseDecoderOnly,
                ["OutputDim"] = _config.OutputDim,
                ["UseConfidenceHead"] = _config.UseConfidenceHead,
                ["FFNActivationType"] = (int)_config.FFNActivationType,
                ["AccelerationType"] = (int)_config.AccelerationType,
                ["AccelerationDeviceId"] = _config.AccelerationDeviceId,
                ["L2RegulationLamda"] = _config.L2RegulationLamda,
                ["GradientClippingThreshold"] = _config.GradientClippingThreshold,
                ["DecayProjectionDim"] = _config.DecayProjectionDim,
                ["DecayHiddenDim"] = _config.DecayHiddenDim,
                ["DecayMemAttnDropout"] = _config.DecayMemAttnDropout,
                ["DecayMLPDropout"] = _config.DecayMLPDropout,
                ["DecayWeightDecay"] = _config.DecayWeightDecay,
                ["DecayTimeEncodingBases"] = _config.DecayTimeEncodingBases,
                // Price context training config
                ["PriceContextMinHistoryLength"] = _config.PriceContextMinHistoryLength,
                ["PriceContextMinCurrentLength"] = _config.PriceContextMinCurrentLength,
                // Memory pruning config
                ["PruningAttentionScoreAlpha"] = PruningConfig.AttentionScoreAlpha,
                ["PruningMinQueryCountForPruning"] = PruningConfig.MinQueryCountForPruning,
                ["PruningNewEntryReserveFraction"] = PruningConfig.NewEntryReserveFraction,
                ["PruningUseAttentionBasedPruning"] = PruningConfig.UseAttentionBasedPruning
            };
            System.IO.File.WriteAllText(System.IO.Path.Combine(dir, "config.json"), System.Text.Json.JsonSerializer.Serialize(cd, new System.Text.Json.JsonSerializerOptions { WriteIndented = true }));

            using (var s = new System.IO.FileStream(System.IO.Path.Combine(dir, "weights.bin"), System.IO.FileMode.Create))
            using (var w = new System.IO.BinaryWriter(s))
            {
                WM(w, TextTokenEmbedding);
                for (int i = 0; i < _config.TextNumLayers; i++) { var b = TextBlocks[i]; WA(w, b.Attention); WV(w, b.LN1Gamma); WV(w, b.LN1Beta); WV(w, b.LN2Gamma); WV(w, b.LN2Beta); }
                WM(w, PriceInputProjection); WV(w, PriceInputProjectionBias);
                for (int i = 0; i < _config.PriceNumLayers; i++) { var b = PriceBlocks[i]; WA(w, b.SelfAttention); WV(w, b.LNSelfGamma); WV(w, b.LNSelfBeta); WA(w, b.CrossAttention); WV(w, b.LNCrossGamma); WV(w, b.LNCrossBeta); WV(w, b.LNFFNGamma); WV(w, b.LNFFNBeta); b.DecayNetwork.WriteTo(w); }
                WM(w, OutputProjection); WV(w, OutputBias);
                if (_config.UseConfidenceHead) 
                {
                    WM(w, ConfidenceProjection); WV(w, ConfidenceBias);
                }
                WM(w, ContextTypeEmbedding);
            }
            for (int i = 0; i < _config.TextNumLayers; i++) TextBlocks[i].FeedForwardNetwork.Save(System.IO.Path.Combine(dir, $"text_ffn_{i}"));
            for (int i = 0; i < _config.PriceNumLayers; i++) PriceBlocks[i].FeedForwardNetwork.Save(System.IO.Path.Combine(dir, $"price_ffn_{i}"));

            // News memory (with attention scores)
            using (var s = new System.IO.FileStream(System.IO.Path.Combine(dir, "news_memory.bin"), System.IO.FileMode.Create))
            using (var w = new System.IO.BinaryWriter(s))
            {
                w.Write(LastPriceTimestamp); w.Write(NewsMemory.Count); int ed = _config.PriceEmbeddingDim;
                foreach (var e in NewsMemory) { w.Write(e.AbsoluteTimestamp); w.Write(e.AttentionScore); w.Write(e.QueryCount); for (int d = 0; d < ed; d++) w.Write(e.HiddenState[d]); }
            }
            // Price memory (with attention scores)
            using (var s = new System.IO.FileStream(System.IO.Path.Combine(dir, "price_memory.bin"), System.IO.FileMode.Create))
            using (var w = new System.IO.BinaryWriter(s))
            {
                w.Write(PriceMemory.Count); int ed = _config.PriceEmbeddingDim;
                foreach (var e in PriceMemory) { w.Write(e.AbsoluteTimestamp); w.Write(e.AttentionScore); w.Write(e.QueryCount); for (int d = 0; d < ed; d++) w.Write(e.HiddenState[d]); }
            }

            // Save tokenizer if attached
            if (Tokenizer != null)
            {
                Tokenizer.Save(System.IO.Path.Combine(dir, "tokenizer"));
            }
        }

        public static Model Load(string dir)
        {
            var d = System.Text.Json.JsonSerializer.Deserialize<Dictionary<string, System.Text.Json.JsonElement>>(System.IO.File.ReadAllText(System.IO.Path.Combine(dir, "config.json")));
            
            var cfg = new Config
            {
                TextVocabSize = d["TextVocabSize"].GetInt32(),
                TextMaxSequenceLength = d["TextMaxSequenceLength"].GetInt32(),
                TextEmbeddingDim = d["TextEmbeddingDim"].GetInt32(),
                TextNumHeads = d["TextNumHeads"].GetInt32(),
                TextNumLayers = d["TextNumLayers"].GetInt32(),
                TextFeedForwardDim = d["TextFeedForwardDim"].GetInt32(),
                TextUseDecoderOnly = d["TextUseDecoderOnly"].GetBoolean(),
                FreezeTextEncoder = d["FreezeTextEncoder"].GetBoolean(),
                PriceInputFeatureDim = d["PriceInputFeatureDim"].GetInt32(),
                PriceMaxSequenceLength = d["PriceMaxSequenceLength"].GetInt32(),
                PriceEmbeddingDim = d["PriceEmbeddingDim"].GetInt32(),
                PriceNumHeads = d["PriceNumHeads"].GetInt32(),
                PriceNumLayers = d["PriceNumLayers"].GetInt32(),
                PriceFeedForwardDim = d["PriceFeedForwardDim"].GetInt32(),
                PriceUseDecoderOnly = d["PriceUseDecoderOnly"].GetBoolean(),
                OutputDim = d["OutputDim"].GetInt32(),
                UseConfidenceHead = d["UseConfidenceHead"].GetBoolean(),
                FFNActivationType = (ActivationType)d["FFNActivationType"].GetInt32(),
                AccelerationType = (AccelerationType)d["AccelerationType"].GetInt32(),
                AccelerationDeviceId = d["AccelerationDeviceId"].GetInt32(),
                L2RegulationLamda = d["L2RegulationLamda"].GetSingle(),
                GradientClippingThreshold = d["GradientClippingThreshold"].GetSingle()
            };
            if (d.ContainsKey("DecayProjectionDim")) cfg.DecayProjectionDim = d["DecayProjectionDim"].GetInt32();
            if (d.ContainsKey("DecayHiddenDim")) cfg.DecayHiddenDim = d["DecayHiddenDim"].GetInt32();
            if (d.ContainsKey("DecayMemAttnDropout")) cfg.DecayMemAttnDropout = d["DecayMemAttnDropout"].GetSingle();
            if (d.ContainsKey("DecayMLPDropout")) cfg.DecayMLPDropout = d["DecayMLPDropout"].GetSingle();
            if (d.ContainsKey("DecayWeightDecay")) cfg.DecayWeightDecay = d["DecayWeightDecay"].GetSingle();
            if (d.ContainsKey("DecayTimeEncodingBases")) cfg.DecayTimeEncodingBases = d["DecayTimeEncodingBases"].GetInt32();
            // Price context training config
            if (d.ContainsKey("PriceContextMinHistoryLength")) cfg.PriceContextMinHistoryLength = d["PriceContextMinHistoryLength"].GetInt32();
            if (d.ContainsKey("PriceContextMinCurrentLength")) cfg.PriceContextMinCurrentLength = d["PriceContextMinCurrentLength"].GetInt32();

            var m = new Model(cfg);

            // Memory pruning config
            if (d.ContainsKey("PruningAttentionScoreAlpha")) m.PruningConfig.AttentionScoreAlpha = d["PruningAttentionScoreAlpha"].GetSingle();
            if (d.ContainsKey("PruningMinQueryCountForPruning")) m.PruningConfig.MinQueryCountForPruning = d["PruningMinQueryCountForPruning"].GetInt32();
            if (d.ContainsKey("PruningNewEntryReserveFraction")) m.PruningConfig.NewEntryReserveFraction = d["PruningNewEntryReserveFraction"].GetSingle();
            if (d.ContainsKey("PruningUseAttentionBasedPruning")) m.PruningConfig.UseAttentionBasedPruning = d["PruningUseAttentionBasedPruning"].GetBoolean();
            using (var s = new System.IO.FileStream(System.IO.Path.Combine(dir, "weights.bin"), System.IO.FileMode.Open))
            using (var r = new System.IO.BinaryReader(s))
            {
                RM(r, m.TextTokenEmbedding);
                for (int i = 0; i < cfg.TextNumLayers; i++) { var b = m.TextBlocks[i]; RA(r, b.Attention); RV(r, b.LN1Gamma); RV(r, b.LN1Beta); RV(r, b.LN2Gamma); RV(r, b.LN2Beta); }
                RM(r, m.PriceInputProjection); RV(r, m.PriceInputProjectionBias);
                for (int i = 0; i < cfg.PriceNumLayers; i++) { var b = m.PriceBlocks[i]; RA(r, b.SelfAttention); RV(r, b.LNSelfGamma); RV(r, b.LNSelfBeta); RA(r, b.CrossAttention); RV(r, b.LNCrossGamma); RV(r, b.LNCrossBeta); RV(r, b.LNFFNGamma); RV(r, b.LNFFNBeta); b.DecayNetwork = ContentAwareDecayNetwork.ReadFrom(r); }
                RM(r, m.OutputProjection); RV(r, m.OutputBias);
                if (cfg.UseConfidenceHead) { RM(r, m.ConfidenceProjection); RV(r, m.ConfidenceBias); }
                if (s.Position < s.Length)
                {
                    RM(r, m.ContextTypeEmbedding);
                }
            }
            for (int i = 0; i < cfg.TextNumLayers; i++) m.TextBlocks[i].FeedForwardNetwork = NeuralNetwork.Load(System.IO.Path.Combine(dir, $"text_ffn_{i}"), cfg.AccelerationType);
            for (int i = 0; i < cfg.PriceNumLayers; i++) m.PriceBlocks[i].FeedForwardNetwork = NeuralNetwork.Load(System.IO.Path.Combine(dir, $"price_ffn_{i}"), cfg.AccelerationType);

            // Load news memory (with attention scores)
            var nmp = System.IO.Path.Combine(dir, "news_memory.bin");
            if (System.IO.File.Exists(nmp)) using (var s = new System.IO.FileStream(nmp, System.IO.FileMode.Open)) using (var r = new System.IO.BinaryReader(s))
            {
                m.LastPriceTimestamp = r.ReadDouble(); int c = r.ReadInt32(); int ed = cfg.PriceEmbeddingDim; m.NewsMemory = new List<NewsMemoryEntry>(c);
                for (int i = 0; i < c; i++) { var e = new NewsMemoryEntry { AbsoluteTimestamp = r.ReadDouble(), AttentionScore = r.ReadSingle(), QueryCount = r.ReadInt32(), HiddenState = new float[ed] }; for (int dd = 0; dd < ed; dd++) e.HiddenState[dd] = r.ReadSingle(); m.NewsMemory.Add(e); }
            }

            // Load price memory (with attention scores)
            var pmp = System.IO.Path.Combine(dir, "price_memory.bin");
            if (System.IO.File.Exists(pmp)) using (var s = new System.IO.FileStream(pmp, System.IO.FileMode.Open)) using (var r = new System.IO.BinaryReader(s))
            {
                int c = r.ReadInt32(); int ed = cfg.PriceEmbeddingDim; m.PriceMemory = new List<PriceMemoryEntry>(c);
                for (int i = 0; i < c; i++) { var e = new PriceMemoryEntry { AbsoluteTimestamp = r.ReadDouble(), AttentionScore = r.ReadSingle(), QueryCount = r.ReadInt32(), HiddenState = new float[ed] }; for (int dd = 0; dd < ed; dd++) e.HiddenState[dd] = r.ReadSingle(); m.PriceMemory.Add(e); }
            }

            // Load tokenizer if saved
            var tokDir = System.IO.Path.Combine(dir, "tokenizer");
            if (System.IO.Directory.Exists(tokDir))
            {
                try { m.Tokenizer = BPETokenizer.Load(tokDir); }
                catch { /* tokenizer optional */ }
            }

            return m;
        }

        static void WM(System.IO.BinaryWriter w, float[,] m) { int r = m.GetLength(0), c = m.GetLength(1); w.Write(r); w.Write(c); for (int i = 0; i < r; i++) for (int j = 0; j < c; j++) w.Write(m[i, j]); }
        static void WV(System.IO.BinaryWriter w, float[] v) { w.Write(v.Length); for (int i = 0; i < v.Length; i++) w.Write(v[i]); }
        static void WA(System.IO.BinaryWriter w, MultiHeadAttention a) { WM(w, a.WQ); WM(w, a.WK); WM(w, a.WV); WM(w, a.WO); WV(w, a.BiasQ); WV(w, a.BiasK); WV(w, a.BiasV); WV(w, a.BiasO); }
        static void RM(System.IO.BinaryReader r, float[,] m) { int rows = r.ReadInt32(), cols = r.ReadInt32(); for (int i = 0; i < rows; i++) for (int j = 0; j < cols; j++) m[i, j] = r.ReadSingle(); }
        static void RV(System.IO.BinaryReader r, float[] v) { int l = r.ReadInt32(); for (int i = 0; i < l; i++) v[i] = r.ReadSingle(); }
        static void RA(System.IO.BinaryReader r, MultiHeadAttention a) { RM(r, a.WQ); RM(r, a.WK); RM(r, a.WV); RM(r, a.WO); RV(r, a.BiasQ); RV(r, a.BiasK); RV(r, a.BiasV); RV(r, a.BiasO); }
        #endregion
    }
}
