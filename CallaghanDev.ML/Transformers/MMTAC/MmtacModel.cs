using CallaghanDev.ML.AccelerationManagers;
using CallaghanDev.ML.AccelerationManagers.GPU;
using CallaghanDev.ML.Enums;
using CallaghanDev.ML.Transformers.Configuration;
using CallaghanDev.ML.Transformers.MultiTypeTransformer;
using CallaghanDev.ML.Transformers.TACAMT;
using System.Text.Json;
using static CallaghanDev.ML.Transformers.MMTAC.MmtacTrainer;

namespace CallaghanDev.ML.Transformers.MMTAC
{
    /// <summary>
    /// Multimodal Market Transformer with Additional Context (MMTAC).
    ///
    /// Output heads (per timestep):
    ///   Regression  : High, Low, Close          - linear,   MSE
    ///   Range       : High - Low                - softplus, MSE
    ///   Quality     : reliability score [0,1]   - sigmoid,  MSE
    ///   Direction   : P(close_t+1 > close_t)    - sigmoid,  BCE
    ///   MidDir      : P(mid-window up-move)      - sigmoid,  BCE
    ///   Confidence  : optional reliability       - sigmoid,  BCE
    /// </summary>
    public class MmtacModel
    {
        //  Config & infrastructure 
        private readonly MmtacConfig _config;
        private readonly Random _random;
        private readonly IAccelerationManager _accel;

        public MmtacConfig Config => _config;
        public IAccelerationManager AccelerationManager => _accel;

        //  Text encoder weights 
        public float[,] TextTokenEmbedding { get; set; }
        public TransformerBlock[] TextBlocks { get; set; }

        //  Price decoder weights 
        public float[,] PriceInputProjection { get; set; }
        public float[] PriceInputProjectionBias { get; set; }
        public TacamtBlock[] PriceBlocks { get; set; }

        //  Global feature projection 
        public float[,] GlobalFeatureProjection { get; set; }
        public float[] GlobalFeatureBias { get; set; }

        //  Context-type embedding 
        // Shape: [3, PriceEmbeddingDim]  Row 0=news, 1=price-memory, 2=global
        public float[,] ContextTypeEmbedding { get; set; }

        //  Output head weights 

        /// <summary>Regression head: High, Low, Close.  Shape [3, embDim].  Linear, no activation.</summary>
        public float[,] RegressionProjection { get; set; }
        public float[] RegressionBias { get; set; }

        /// <summary>Range head: High-Low >= 0.  Shape [1, embDim].  Softplus activation.</summary>
        public float[,] RangeProjection { get; set; }
        public float[] RangeBias { get; set; }

        /// <summary>Quality head: reliability in [0,1].  Shape [1, embDim].  Sigmoid activation.</summary>
        public float[,] QualityProjection { get; set; }
        public float[] QualityBias { get; set; }

        /// <summary>Direction classification head.  Shape [1, embDim].  Sigmoid.</summary>
        public float[,] DirectionProjection { get; set; }
        public float[] DirectionBias { get; set; }

        /// <summary>MidWindowDirection classification head.  Shape [1, embDim].  Sigmoid.</summary>
        public float[,] MidDirectionProjection { get; set; }
        public float[] MidDirectionBias { get; set; }

        /// <summary>Optional confidence head.  Shape [1, embDim].  Null when disabled.</summary>
        public float[,] ConfidenceProjection { get; set; }
        public float[] ConfidenceBias { get; set; }

        //  Persistent memory 
        public List<NewsMemoryEntry> NewsMemory { get; set; } = new List<NewsMemoryEntry>();
        public List<PriceMemoryEntry> PriceMemory { get; set; } = new List<PriceMemoryEntry>();
        public double LastPriceTimestamp { get; set; } = 0;
        public MemoryPruningConfig PruningConfig { get; set; } = new MemoryPruningConfig();

        //  Tokenizer 
        public BPETokenizer Tokenizer { get; private set; }

        // 
        // Constructor
        // 
        public MmtacModel(MmtacConfig config, Random random = null)
        {
            if (config == null)
                throw new ArgumentNullException(nameof(config));

            config.Validate();

            _config = config;
            _random = random ?? new Random();
            _accel = CreateAccel(config.Runtime);

            // Copy runtime pruning settings from config. Do not keep the default
            // field initializer, otherwise config.Pruning is silently ignored.
            PruningConfig = new MemoryPruningConfig
            {
                AttentionScoreAlpha = config.Pruning.AttentionScoreAlpha,
                MinQueryCountForPruning = config.Pruning.MinQueryCountForPruning,
                NewEntryReserveFraction = config.Pruning.NewEntryReserveFraction,
                UseAttentionBasedPruning = config.Pruning.UseAttentionBasedPruning
            };

            InitTextEncoder();
            InitPriceDecoder();
            InitGlobalProjection();
            InitContextTypeEmbedding();
            InitOutputHeads();
        }
        // 
        // Tokeniser helpers
        // 

        public void SetTokenizer(BPETokenizer tokenizer)
        {
            if (tokenizer == null)
                throw new ArgumentNullException(nameof(tokenizer));

            if (tokenizer.VocabSize > _config.Text.VocabSize)
            {
                throw new ArgumentException(
                    $"Tokenizer vocab size ({tokenizer.VocabSize}) exceeds config TextVocabSize ({_config.Text.VocabSize}).",
                    nameof(tokenizer));
            }

            Tokenizer = tokenizer;
        }

        public void TrainTokenizer(string[] texts, int minFrequency = 10)
        {
            if (texts == null || texts.Length == 0)
                throw new ArgumentException("Cannot train tokenizer on empty corpus.");
            var tok = new BPETokenizer();
            tok.Train(texts, _config.Text.VocabSize, minFrequency);
            Tokenizer = tok;
        }

        public NewsStory[] TokenizeStories(string[] texts, float[] arrivalTimes)
        {
            if (Tokenizer == null)
                throw new InvalidOperationException("No tokenizer attached. Call SetTokenizer() first.");
            if (texts.Length != arrivalTimes.Length)
                throw new ArgumentException("texts and arrivalTimes must have the same length.");

            var out_ = new NewsStory[texts.Length];
            for (int i = 0; i < texts.Length; i++)
                out_[i] = new NewsStory(Tokenizer.Encode(texts[i]), arrivalTimes[i]);
            return out_;
        }

        // 
        // Forward – primary entry point
        // 

        /// <summary>
        /// Run a full forward pass.  Returns per-timestep outputs for all heads.
        /// regression[t,0..2] = High, Low, Close  (linear)
        /// range[t,0]         = Range             (softplus)
        /// quality[t,0]       = Quality           (sigmoid)
        /// direction[t,0]     = Direction prob    (sigmoid)
        /// midDirection[t,0]  = MidDir prob       (sigmoid)
        /// confidence[t,0]    = Confidence prob   (sigmoid, null if disabled)
        /// </summary>
        public (float[,] regression, float[,] range, float[,] quality, float[,] direction, float[,] midDirection, float[,] confidence) Forward(MultimodalInput input)
        {
            if (input == null)
                throw new ArgumentNullException(nameof(input));
            if (input.PriceSequence == null)
                throw new ArgumentNullException(nameof(input.PriceSequence));
            if (input.PriceSequence.GetLength(0) <= 0)
                throw new ArgumentException("PriceSequence must contain at least one row.", nameof(input));

            float[,] storyHidden = null;
            float[] storyTimes = null;

            if (input.NewsStories != null && input.NewsStories.Length > 0)
                (storyHidden, storyTimes) = EncodeStories(input.NewsStories);

            float[,] contextHidden;
            float[] contextTimes;
            int numGlobalContext = (_config.Global.GlobalFeatureDim > 0 && input.GlobalFeatures != null) ? 1 : 0;

            BuildContext(storyHidden, storyTimes, input.GlobalFeatures, null, null, out contextHidden, out contextTimes);

            var priceHidden = ForwardPriceDecoder(input.PriceSequence, contextHidden, contextTimes, numGlobalContext);
            return ProjectToOutputs(priceHidden);
        }

        /// <summary>Convenience overload - returns only the last-timestep prediction.</summary>
        public ModelPrediction PredictNext(MultimodalInput input)
        {
            var (reg, range, quality, dir, midDir, conf) = Forward(input);
            int last = reg.GetLength(0) - 1;
            return new ModelPrediction
            {
                High = reg[last, 0],
                Low = reg[last, 1],
                Close = reg[last, 2],
                Range = range[last, 0],
                Quality = quality[last, 0],
                DirectionProb = dir[last, 0],
                MidWindowDirectionProb = midDir[last, 0],
                Confidence = conf != null ? conf[last, 0] : 1f
            };
        }

        // 
        // Forward with cache (used by trainer for backprop)
        // 

        internal (float[,] regression, float[,] range, float[,] quality, float[,] direction, float[,] midDirection, float[,] confidence) ForwardWithCache(MultimodalInput input, MmtacForwardCache cache, bool isTraining = false, Random dropoutRng = null)
        {
            if (input == null)
                throw new ArgumentNullException(nameof(input));
            if (input.PriceSequence == null)
                throw new ArgumentNullException(nameof(input.PriceSequence));

            int seqLen = input.PriceSequence.GetLength(0);
            if (seqLen <= 0)
                throw new ArgumentException("PriceSequence must contain at least one row.", nameof(input));

            return ForwardWithCache(input, 0, seqLen, cache, isTraining, dropoutRng);
        }

        internal (float[,] regression, float[,] range, float[,] quality, float[,] direction, float[,] midDirection, float[,] confidence) ForwardWithCache(MultimodalInput input, int rowStart, int rowCount, MmtacForwardCache cache, bool isTraining = false, Random dropoutRng = null)
        {
            if (input == null)
                throw new ArgumentNullException(nameof(input));
            if (input.PriceSequence == null)
                throw new ArgumentNullException(nameof(input.PriceSequence));
            if (cache == null)
                throw new ArgumentNullException(nameof(cache));
            if (rowStart < 0 || rowCount <= 0)
                throw new ArgumentOutOfRangeException($"{nameof(rowStart)} and {nameof(rowCount)} must describe a non-empty slice.");
            if (rowStart + rowCount > input.PriceSequence.GetLength(0))
                throw new ArgumentException("rowStart + rowCount exceeds PriceSequence row count.");

            cache.Reset();

            float[,] storyHidden = null;
            float[] storyTimes = null;

            if (input.NewsStories != null && input.NewsStories.Length > 0)
            {
                (storyHidden, storyTimes) = EncodeStoriesWithCache(input.NewsStories, cache);
            }
            else
            {
                cache.TextFinalHidden = null;
                cache.TextTokenIds = null;
                cache.StoryCaches = null;
                cache.StoryTokenCounts = null;
            }

            float[] globalToken = null;
            if (_config.Global.GlobalFeatureDim > 0 && input.GlobalFeatures != null)
            {
                globalToken = EmbedGlobalFeatures(input.GlobalFeatures);
                cache.GlobalTokenEmbedded = globalToken;
                cache.GlobalRawInput = input.GlobalFeatures;
            }

            float[,] contextHidden;
            float[] contextTimes;
            BuildContext(storyHidden, storyTimes, null, globalToken, cache, out contextHidden, out contextTimes);

            int numGlobal = globalToken != null ? 1 : 0;
            int numLiveNews = storyHidden != null ? storyHidden.GetLength(0) : 0;

            cache.NumGlobalContext = numGlobal;
            cache.NumStoredNewsContext = 0;
            cache.NumLiveNewsContext = numLiveNews;
            cache.NumNewsContext = numLiveNews;
            cache.NumPriceContext = 0;

            // In price-decoder backprop this field represents the combined context matrix.
            cache.TextFinalHidden = contextHidden;
            cache.StoryArrivalTimes = contextTimes;

            var priceHidden = ForwardPriceDecoderWithCache(
                input.PriceSequence,
                rowStart,
                rowCount,
                contextHidden,
                contextTimes,
                cache,
                isTraining,
                dropoutRng);

            cache.PriceFinalHidden = priceHidden;
            return ProjectToOutputs(priceHidden, cache);
        }
        // 
        // Persistent memory inference
        // 
        public ModelPrediction PredictWithMemory(
     MultimodalInput input,
     double currentAbsoluteTimestamp,
     double timeUnitsPerPosition = 1.0,
     int maxNewsMemorySize = 100,
     int maxPriceMemorySize = 200)
        {
            if (input == null)
                throw new ArgumentNullException(nameof(input));

            if (input.PriceSequence == null)
                throw new ArgumentNullException(nameof(input.PriceSequence));

            if (input.PriceSequence.GetLength(0) <= 0)
                throw new ArgumentException("PriceSequence must contain at least one row.", nameof(input));

            if (input.PriceSequence.GetLength(1) != _config.Price.InputFeatureDim)
            {
                throw new ArgumentException(
                    $"PriceSequence feature dimension ({input.PriceSequence.GetLength(1)}) must equal Price.InputFeatureDim ({_config.Price.InputFeatureDim}).",
                    nameof(input));
            }

            if (timeUnitsPerPosition == 0.0)
                throw new ArgumentOutOfRangeException(nameof(timeUnitsPerPosition), "Must be non-zero.");

            int embDim = _config.Price.EmbeddingDim;

            var ctxH = new List<float[]>();
            var ctxT = new List<float>();
            var ctxTypes = new List<int>();

            float[] globalToken = null;

            if (_config.Global.GlobalFeatureDim > 0 && input.GlobalFeatures != null)
            {
                globalToken = EmbedGlobalFeatures(input.GlobalFeatures);

                ctxH.Add(globalToken);
                ctxT.Add(0f);
                ctxTypes.Add(2);
            }

            int globalOffset = globalToken != null ? 1 : 0;

            int storedNewsMemoryCount = NewsMemory?.Count ?? 0;
            int storedPriceMemoryCount = PriceMemory?.Count ?? 0;

            if (NewsMemory != null)
            {
                foreach (var e in NewsMemory)
                {
                    if (e?.HiddenState == null)
                        continue;

                    float relTime = -(float)((currentAbsoluteTimestamp - e.AbsoluteTimestamp) / timeUnitsPerPosition);

                    var v = new float[embDim];
                    int copyDim = Math.Min(embDim, e.HiddenState.Length);

                    for (int d = 0; d < copyDim; d++)
                        v[d] = e.HiddenState[d];

                    ctxH.Add(v);
                    ctxT.Add(relTime);
                    ctxTypes.Add(0);
                }
            }

            float[,] newStoryHidden = null;
            float[] newStoryTimes = null;
            int liveNewsContextCount = 0;

            if (input.NewsStories != null && input.NewsStories.Length > 0)
            {
                (newStoryHidden, newStoryTimes) = EncodeStories(input.NewsStories);

                int storyRows = newStoryHidden.GetLength(0);

                for (int i = 0; i < storyRows; i++)
                {
                    var story = input.NewsStories[i];

                    if (story == null || story.TokenIds == null || story.TokenIds.Length == 0)
                        continue;

                    var v = new float[embDim];

                    for (int d = 0; d < embDim; d++)
                        v[d] = newStoryHidden[i, d];

                    ctxH.Add(v);
                    ctxT.Add(newStoryTimes != null ? newStoryTimes[i] : story.ArrivalTime);
                    ctxTypes.Add(0);
                    liveNewsContextCount++;
                }
            }

            if (PriceMemory != null)
            {
                foreach (var e in PriceMemory)
                {
                    if (e?.HiddenState == null)
                        continue;

                    float relTime = -(float)((currentAbsoluteTimestamp - e.AbsoluteTimestamp) / timeUnitsPerPosition);

                    var v = new float[embDim];
                    int copyDim = Math.Min(embDim, e.HiddenState.Length);

                    for (int d = 0; d < copyDim; d++)
                        v[d] = e.HiddenState[d];

                    ctxH.Add(v);
                    ctxT.Add(relTime);
                    ctxTypes.Add(1);
                }
            }

            float[,] contextHidden = null;
            float[] contextTimes = null;
            int totalCtx = ctxH.Count;

            if (totalCtx > 0)
            {
                contextHidden = new float[totalCtx, embDim];
                contextTimes = new float[totalCtx];

                for (int i = 0; i < totalCtx; i++)
                {
                    int copyDim = Math.Min(embDim, ctxH[i].Length);

                    for (int d = 0; d < copyDim; d++)
                        contextHidden[i, d] = ctxH[i][d];

                    contextTimes[i] = ctxT[i];
                }

                ApplyContextTypeEmbeddings(contextHidden, ctxTypes.ToArray());
            }

            var priceHidden = ForwardPriceDecoder(
                input.PriceSequence,
                contextHidden,
                contextTimes,
                globalOffset);

            var (reg, range, quality, dir, midDir, conf) = ProjectToOutputs(priceHidden);

            int last = reg.GetLength(0) - 1;

            UpdateMemoryAttentionScores(
                priceHidden,
                contextHidden,
                totalCtx,
                globalOffset,
                storedNewsMemoryCount,
                liveNewsContextCount,
                storedPriceMemoryCount);

            if (newStoryHidden != null && input.NewsStories != null)
            {
                int storyRows = newStoryHidden.GetLength(0);

                for (int i = 0; i < storyRows; i++)
                {
                    var story = input.NewsStories[i];

                    if (story == null || story.TokenIds == null || story.TokenIds.Length == 0)
                        continue;

                    var hv = new float[embDim];

                    for (int d = 0; d < embDim; d++)
                        hv[d] = newStoryHidden[i, d];

                    float arrivalTime = newStoryTimes != null ? newStoryTimes[i] : story.ArrivalTime;

                    NewsMemory.Add(new NewsMemoryEntry
                    {
                        HiddenState = hv,
                        AbsoluteTimestamp = currentAbsoluteTimestamp + arrivalTime * timeUnitsPerPosition
                    });
                }
            }

            int priceSeqLen = input.PriceSequence.GetLength(0);

            for (int t = 0; t < priceSeqLen; t++)
            {
                var pv = new float[embDim];

                for (int d = 0; d < embDim; d++)
                    pv[d] = priceHidden[t, d];

                PriceMemory.Add(new PriceMemoryEntry
                {
                    HiddenState = pv,
                    AbsoluteTimestamp = currentAbsoluteTimestamp + t * timeUnitsPerPosition
                });
            }

            LastPriceTimestamp =
                currentAbsoluteTimestamp + Math.Max(0, priceSeqLen - 1) * timeUnitsPerPosition;

            PruneNewsMemory(maxNewsMemorySize);

            // Critical fix:
            // The just-written price sequence must survive pruning. In the carry task,
            // maxPriceMemory == seqLen, so the next sample must see this whole sequence.
            PricePruneMemoryAfterAppend(maxPriceMemorySize, priceSeqLen);
            return new ModelPrediction
            {
                High = reg[last, 0],
                Low = reg[last, 1],
                Close = reg[last, 2],
                Range = range[last, 0],
                Quality = quality[last, 0],
                DirectionProb = dir[last, 0],
                MidWindowDirectionProb = midDir[last, 0],
                Confidence = conf != null ? conf[last, 0] : 1f
            };
        }
        public void ClearNewsMemory()
        {
            NewsMemory.Clear();
        }
        public void ClearPriceMemory()
        {
            PriceMemory.Clear();
        }
        public void ClearAllMemory()
        {
            ClearNewsMemory();
            ClearPriceMemory();
        }

        // 
        // Global feature embedding
        // 

        internal float[] EmbedGlobalFeatures(float[] globalFeatures)
        {
            if (globalFeatures == null)
                throw new ArgumentNullException(nameof(globalFeatures));

            int gd = _config.Global.GlobalFeatureDim;
            int ed = _config.Price.EmbeddingDim;

            if (gd <= 0)
                throw new InvalidOperationException("GlobalFeatureDim is zero, but EmbedGlobalFeatures was called.");

            if (globalFeatures.Length != gd)
            {
                throw new ArgumentException(
                    $"globalFeatures.Length ({globalFeatures.Length}) must equal GlobalFeatureDim ({gd}).",
                    nameof(globalFeatures));
            }

            var output = new float[ed];

            for (int d = 0; d < ed; d++)
            {
                float sum = GlobalFeatureBias[d];

                for (int g = 0; g < gd; g++)
                    sum += GlobalFeatureProjection[d, g] * globalFeatures[g];

                output[d] = sum;
            }

            return output;
        }
        // 
        // Context assembly
        // 

        internal void BuildContext(float[,] newsHidden, float[] newsTimes, float[] globalFeatures, float[] preEmbeddedGlobal, MmtacForwardCache cache, out float[,] contextHidden, out float[] contextTimes)
        {
            int ed = _config.Price.EmbeddingDim;

            float[] globalToken = preEmbeddedGlobal;
            if (globalToken == null && _config.Global.GlobalFeatureDim > 0 && globalFeatures != null)
                globalToken = EmbedGlobalFeatures(globalFeatures);

            int numGlobal = globalToken != null ? 1 : 0;
            int numNews = newsHidden != null ? newsHidden.GetLength(0) : 0;
            int total = numGlobal + numNews;

            if (total == 0) { contextHidden = null; contextTimes = null; return; }

            contextHidden = new float[total, ed];
            contextTimes = new float[total];
            int row = 0;

            if (globalToken != null)
            {
                for (int d = 0; d < ed; d++)
                    contextHidden[row, d] = globalToken[d] + ContextTypeEmbedding[2, d];
                contextTimes[row] = 0f;
                row++;
            }

            for (int i = 0; i < numNews; i++)
            {
                for (int d = 0; d < ed; d++)
                    contextHidden[row, d] = newsHidden[i, d] + ContextTypeEmbedding[0, d];
                contextTimes[row] = newsTimes != null ? newsTimes[i] : 0f;
                row++;
            }

            if (cache != null)
            {
                cache.NumGlobalContext = numGlobal;
                cache.NumNewsContext = numNews;
            }
        }

        // 
        // Price decoder forward (no cache)
        // 

        private float[,] ForwardPriceDecoder(float[,] priceSequence, float[,] contextHidden, float[] contextTimes, int globalBypassCount = 0)
        {
            if (priceSequence == null)
                throw new ArgumentNullException(nameof(priceSequence));

            int seqLen = priceSequence.GetLength(0);
            if (seqLen <= 0)
                throw new ArgumentException("priceSequence must contain at least one row.", nameof(priceSequence));

            if (contextHidden != null)
            {
                if (contextHidden.GetLength(1) != _config.Price.EmbeddingDim)
                    throw new ArgumentException("contextHidden embedding dimension does not match Price.EmbeddingDim.", nameof(contextHidden));

                if (contextTimes != null && contextTimes.Length != contextHidden.GetLength(0))
                    throw new ArgumentException("contextTimes length must match contextHidden row count.", nameof(contextTimes));
            }

            var x = EmbedPriceSequence(priceSequence, 0, seqLen);
            bool[,] mask = _config.Price.UseDecoderOnly ? CreateCausalMask(seqLen) : null;

            float[,] timeDiffs = null;
            float[] keyTimes = null;

            if (_config.Decay.Enabled && contextHidden != null && contextTimes != null)
            {
                timeDiffs = _accel.ComputeTimeDiffMatrix(seqLen, contextTimes);
                keyTimes = contextTimes;

                if (_config.Global.BypassDecay && globalBypassCount > 0)
                {
                    int bypass = Math.Min(globalBypassCount, contextTimes.Length);
                    for (int qi = 0; qi < seqLen; qi++)
                        for (int gi = 0; gi < bypass; gi++)
                            timeDiffs[qi, gi] = 0f;
                }
            }

            foreach (var block in PriceBlocks)
            {
                block.SetContext(contextHidden);
                block.SetTimeData(timeDiffs, keyTimes);
                block.SetTraining(false);
                x = block.Forward(x, mask);
            }

            return x;
        }
        // 
        // Price decoder forward (with cache)
        // 

        public float[,] ForwardPriceDecoderWithCache(float[,] priceSequence, int rowStart, int rowCount, float[,] contextHidden, float[] contextTimes, MmtacForwardCache cache, bool isTraining = true, Random dropoutRng = null)
        {
            if (priceSequence == null)
                throw new ArgumentNullException(nameof(priceSequence));
            if (cache == null)
                throw new ArgumentNullException(nameof(cache));
            if (rowStart < 0 || rowCount <= 0 || rowStart + rowCount > priceSequence.GetLength(0))
                throw new ArgumentOutOfRangeException($"Invalid price row slice: start={rowStart}, count={rowCount}.");

            int seqLen = rowCount;
            int featureDim = priceSequence.GetLength(1);
            int ed = _config.Price.EmbeddingDim;
            int nh = _config.Price.NumHeads;
            int hd = ed / nh;
            float scale = 1.0f / MathF.Sqrt(hd);

            if (contextHidden != null)
            {
                if (contextHidden.GetLength(1) != ed)
                    throw new ArgumentException("contextHidden embedding dimension does not match Price.EmbeddingDim.", nameof(contextHidden));
                if (contextTimes != null && contextTimes.Length != contextHidden.GetLength(0))
                    throw new ArgumentException("contextTimes length must match contextHidden row count.", nameof(contextTimes));
            }

            // Store the exact slice used for the projection. BackpropInputProjection indexes from zero.
            var priceInputSlice = new float[seqLen, featureDim];
            for (int i = 0; i < seqLen; i++)
                for (int f = 0; f < featureDim; f++)
                    priceInputSlice[i, f] = priceSequence[rowStart + i, f];

            var emb = EmbedPriceSequence(priceInputSlice, 0, seqLen);
            cache.PriceEmbedded = emb;
            cache.PriceContinuousInput = priceInputSlice;

            bool[,] selfMask = _config.Price.UseDecoderOnly ? CreateCausalMask(seqLen) : null;
            var x = emb;

            for (int layer = 0; layer < _config.Price.NumLayers; layer++)
            {
                var block = PriceBlocks[layer];
                var bc = cache.PriceBlockCaches[layer];
                bc.BlockInput = x;

                var selfQ = ComputeProjection(x, block.SelfAttention.WQ, block.SelfAttention.BiasQ);
                var selfK = ComputeProjection(x, block.SelfAttention.WK, block.SelfAttention.BiasK);
                var selfV = ComputeProjection(x, block.SelfAttention.WV, block.SelfAttention.BiasV);
                RotaryPositionEmbedding.ApplyInPlace(selfQ, selfK, nh);
                bc.SelfQ = selfQ;
                bc.SelfK = selfK;
                bc.SelfV = selfV;

                var selfAttnOutput = _accel.MultiHeadAttentionForward(selfQ, selfK, selfV, nh, scale, selfMask);
                bc.SelfAttnOutput = selfAttnOutput;

                var selfProjected = ComputeProjection(selfAttnOutput, block.SelfAttention.WO, block.SelfAttention.BiasO);
                var selfResidual = _accel.MatrixAdd(x, selfProjected);
                bc.SelfResidualInput = selfResidual;

                var (normedSelf, selfMean, selfVar, selfNorm) = _accel.LayerNormForward(selfResidual, block.LNSelfGamma, block.LNSelfBeta);
                bc.LNSelfCache.Input = selfResidual;
                bc.LNSelfCache.Mean = selfMean;
                bc.LNSelfCache.Variance = selfVar;
                bc.LNSelfCache.Normalized = selfNorm;
                bc.NormedSelf = normedSelf;

                float[,] normedCross;
                bool hasContext = contextHidden != null && contextHidden.GetLength(0) > 0;

                if (hasContext)
                {
                    float[,] timeDiffs = null;
                    float[] keyTimes = null;
                    bc.TimeDiffs = null;
                    bc.KeyTimesFromRef = null;
                    bc.DecayCache = null;

                    if (_config.Decay.Enabled && contextTimes != null)
                    {
                        timeDiffs = _accel.ComputeTimeDiffMatrix(seqLen, contextTimes);
                        keyTimes = contextTimes;

                        if (_config.Global.BypassDecay && cache.NumGlobalContext > 0)
                        {
                            int bypass = Math.Min(cache.NumGlobalContext, contextTimes.Length);

                            for (int qi = 0; qi < seqLen; qi++)
                            {
                                for (int gi = 0; gi < bypass; gi++)
                                    timeDiffs[qi, gi] = 0f;
                            }
                        }

                        bc.TimeDiffs = timeDiffs;
                        bc.KeyTimesFromRef = keyTimes;
                    }

                    var crossQ = ComputeProjection(normedSelf, block.CrossAttention.WQ, block.CrossAttention.BiasQ);
                    var crossK = ComputeProjection(contextHidden, block.CrossAttention.WK, block.CrossAttention.BiasK);
                    var crossV = ComputeProjection(contextHidden, block.CrossAttention.WV, block.CrossAttention.BiasV);
                    RotaryPositionEmbedding.ApplyInPlace(crossQ, crossK, nh);
                    bc.CrossQ = crossQ;
                    bc.CrossK = crossK;
                    bc.CrossV = crossV;

                    var crossAttnOutput = ContentAwareCrossAttentionWithCache(
                        crossQ,
                        crossK,
                        crossV,
                        timeDiffs,
                        keyTimes,
                        normedSelf,
                        contextHidden,
                        block,
                        bc,
                        isTraining,
                        dropoutRng,
                        cache.NumGlobalContext);

                    bc.CrossAttnOutput = crossAttnOutput;

                    var crossProjected = ComputeProjection(crossAttnOutput, block.CrossAttention.WO, block.CrossAttention.BiasO);
                    var crossResidual = _accel.MatrixAdd(normedSelf, crossProjected);
                    bc.CrossResidualInput = crossResidual;

                    var (ncr, crossMean, crossVar, crossNorm) = _accel.LayerNormForward(crossResidual, block.LnCrossGamma, block.LnCrossBeta);
                    bc.LNCrossCache.Input = crossResidual;
                    bc.LNCrossCache.Mean = crossMean;
                    bc.LNCrossCache.Variance = crossVar;
                    bc.LNCrossCache.Normalized = crossNorm;
                    bc.NormedCross = ncr;
                    normedCross = ncr;
                }
                else
                {
                    bc.CrossQ = null;
                    bc.CrossK = null;
                    bc.CrossV = null;
                    bc.CrossAttnOutput = null;
                    bc.CrossAttentionWeights = null;
                    bc.CrossScoresPreSoftmax = null;
                    bc.TimeDiffs = null;
                    bc.KeyTimesFromRef = null;
                    bc.DecayCache = null;
                    bc.CrossResidualInput = normedSelf;

                    var (ncr, crossMean, crossVar, crossNorm) = _accel.LayerNormForward(normedSelf, block.LnCrossGamma, block.LnCrossBeta);
                    bc.LNCrossCache.Input = normedSelf;
                    bc.LNCrossCache.Mean = crossMean;
                    bc.LNCrossCache.Variance = crossVar;
                    bc.LNCrossCache.Normalized = crossNorm;
                    bc.NormedCross = ncr;
                    normedCross = ncr;
                }

                var ffnInputRows = new float[seqLen][];
                for (int i = 0; i < seqLen; i++)
                {
                    var row = new float[ed];
                    for (int j = 0; j < ed; j++)
                        row[j] = normedCross[i, j];
                    ffnInputRows[i] = row;
                }

                var ffnOutput = _accel.FFNForwardBatch(normedCross, seqLen, ed, block.FeedForwardNetwork.ForwardPassOnly);
                bc.FFNInputRows = ffnInputRows;
                bc.FFNOutput = ffnOutput;

                var ffnResidual = _accel.MatrixAdd(normedCross, ffnOutput);
                bc.FFNResidualInput = ffnResidual;

                var (normedFfn, ffnMean, ffnVar, ffnNorm) = _accel.LayerNormForward(ffnResidual, block.LNFFNGamma, block.LNFFNBeta);
                bc.LNFFNCache.Input = ffnResidual;
                bc.LNFFNCache.Mean = ffnMean;
                bc.LNFFNCache.Variance = ffnVar;
                bc.LNFFNCache.Normalized = ffnNorm;
                x = normedFfn;
            }

            return x;
        }

        // 
        // Price-context training forward
        // 

        internal (float[,] reg, float[,] range, float[,] quality, float[,] dir, float[,] midDir, float[,] conf) ForwardWithPriceContextAndCache(MultimodalInput input, float[,] priceCtxHidden, float[] priceCtxTimes, MmtacForwardCache cache, bool isTraining = true, Random dropoutRng = null)
        {
            if (input == null)
                throw new ArgumentNullException(nameof(input));
            if (input.PriceSequence == null)
                throw new ArgumentNullException(nameof(input.PriceSequence));
            if (cache == null)
                throw new ArgumentNullException(nameof(cache));

            cache.Reset();

            int ed = _config.Price.EmbeddingDim;

            float[,] newsHidden = null;
            float[] newsTimes = null;
            int numNews = 0;

            if (input.NewsStories != null && input.NewsStories.Length > 0)
            {
                (newsHidden, newsTimes) = EncodeStoriesWithCache(input.NewsStories, cache);
                numNews = newsHidden.GetLength(0);
            }
            else
            {
                cache.TextFinalHidden = null;
                cache.TextTokenIds = null;
                cache.StoryCaches = null;
                cache.StoryTokenCounts = null;
            }

            float[] globalToken = null;
            if (_config.Global.GlobalFeatureDim > 0 && input.GlobalFeatures != null)
            {
                globalToken = EmbedGlobalFeatures(input.GlobalFeatures);
                cache.GlobalTokenEmbedded = globalToken;
                cache.GlobalRawInput = input.GlobalFeatures;
            }

            int numGlobal = globalToken != null ? 1 : 0;
            int numPriceCtx = priceCtxHidden != null ? priceCtxHidden.GetLength(0) : 0;

            if (priceCtxHidden != null && priceCtxHidden.GetLength(1) != ed)
                throw new ArgumentException("priceCtxHidden embedding dimension does not match Price.EmbeddingDim.", nameof(priceCtxHidden));
            if (priceCtxTimes != null && priceCtxTimes.Length != numPriceCtx)
                throw new ArgumentException("priceCtxTimes length must match priceCtxHidden row count.", nameof(priceCtxTimes));

            int totalCtx = numGlobal + numNews + numPriceCtx;
            float[,] combinedHidden = null;
            float[] combinedTimes = null;

            if (totalCtx > 0)
            {
                combinedHidden = new float[totalCtx, ed];
                combinedTimes = new float[totalCtx];
                int row = 0;

                if (globalToken != null)
                {
                    for (int d = 0; d < ed; d++)
                        combinedHidden[row, d] = globalToken[d] + ContextTypeEmbedding[2, d];
                    combinedTimes[row] = 0f;
                    row++;
                }

                for (int i = 0; i < numNews; i++)
                {
                    for (int d = 0; d < ed; d++)
                        combinedHidden[row, d] = newsHidden[i, d] + ContextTypeEmbedding[0, d];
                    combinedTimes[row] = newsTimes != null ? newsTimes[i] : 0f;
                    row++;
                }

                for (int i = 0; i < numPriceCtx; i++)
                {
                    for (int d = 0; d < ed; d++)
                        combinedHidden[row, d] = priceCtxHidden[i, d] + ContextTypeEmbedding[1, d];
                    combinedTimes[row] = priceCtxTimes != null ? priceCtxTimes[i] : 0f;
                    row++;
                }
            }

            cache.NumGlobalContext = numGlobal;
            cache.NumStoredNewsContext = 0;
            cache.NumLiveNewsContext = numNews;
            cache.NumNewsContext = numNews;
            cache.NumPriceContext = numPriceCtx;
            cache.PriceContextHidden = priceCtxHidden;
            cache.TextFinalHidden = combinedHidden;
            cache.StoryArrivalTimes = combinedTimes;

            var priceHidden = ForwardPriceDecoderWithCache(
                input.PriceSequence,
                0,
                input.PriceSequence.GetLength(0),
                combinedHidden,
                combinedTimes,
                cache,
                isTraining,
                dropoutRng);

            cache.PriceFinalHidden = priceHidden;
            return ProjectToOutputs(priceHidden, cache);
        }
        public float[,] EncodePriceHistory(float[,] histPrices)
        {
            int sl = histPrices.GetLength(0);
            var emb = EmbedPriceSequence(histPrices, 0, sl);
            bool[,] mask = _config.Price.UseDecoderOnly ? CreateCausalMask(sl) : null;
            var x = emb;
            foreach (var block in PriceBlocks)
            {
                block.SetContext(null); block.SetTimeData(null, null); block.SetTraining(false);
                x = block.Forward(x, mask);
            }
            return x;
        }

        // 
        // Multi-head output projection
        // 

        /// <summary>
        /// Projects price decoder hidden states through all output heads.
        /// Pre-activation logits for Range and Quality are stored in cache
        /// (cache.RangeLogits, cache.QualityLogits) when cache is not null,
        /// so the backward pass can chain through the activation derivatives.
        /// This overload is used at inference time (no cache).
        /// </summary>
        internal (float[,] regression, float[,] range, float[,] quality, float[,] direction, float[,] midDirection, float[,] confidence) ProjectToOutputs(float[,] hidden)
            => ProjectToOutputs(hidden, null);

        internal (float[,] regression, float[,] range, float[,] quality, float[,] direction, float[,] midDirection, float[,] confidence) ProjectToOutputs(float[,] hidden, MmtacForwardCache cache)
        {
            int sl = hidden.GetLength(0);
            int ed = _config.Price.EmbeddingDim;
            int rDim = MmtacOutputConfig.RegressionOutputCount; // 3

            //  Regression: High, Low, Close - linear 
            var regression = _accel.ProjectOutputBatch(hidden, RegressionProjection, RegressionBias, sl, rDim);

            //  Range - softplus: log(1 + exp(x)) 
            var range = new float[sl, 1];
            var rangeLogits = new float[sl];
            for (int t = 0; t < sl; t++)
            {
                float l = RangeBias[0];
                for (int k = 0; k < ed; k++) l += RangeProjection[0, k] * hidden[t, k];
                rangeLogits[t] = l;
                // Numerically stable softplus: for x > 20, softplus(x) ≈ x
                range[t, 0] = l > 20f ? l : MathF.Log(1f + MathF.Exp(l));
            }

            //  Quality - sigmoid 
            var quality = new float[sl, 1];
            var qualityLogits = new float[sl];
            for (int t = 0; t < sl; t++)
            {
                float l = QualityBias[0];
                for (int k = 0; k < ed; k++) l += QualityProjection[0, k] * hidden[t, k];
                qualityLogits[t] = l;
                quality[t, 0] = Sigmoid(l);
            }

            //  Direction - sigmoid 
            var direction = new float[sl, 1];
            for (int t = 0; t < sl; t++)
            {
                float l = DirectionBias[0];
                for (int k = 0; k < ed; k++) l += DirectionProjection[0, k] * hidden[t, k];
                direction[t, 0] = Sigmoid(l);
            }

            //  MidWindowDirection - sigmoid 
            var midDirection = new float[sl, 1];
            for (int t = 0; t < sl; t++)
            {
                float l = MidDirectionBias[0];
                for (int k = 0; k < ed; k++) l += MidDirectionProjection[0, k] * hidden[t, k];
                midDirection[t, 0] = Sigmoid(l);
            }

            //  Confidence - sigmoid (optional) 
            float[,] confidence = null;
            if (_config.Output.UseConfidenceHead)
            {
                confidence = new float[sl, 1];
                for (int t = 0; t < sl; t++)
                {
                    float l = ConfidenceBias[0];
                    for (int k = 0; k < ed; k++) l += ConfidenceProjection[0, k] * hidden[t, k];
                    confidence[t, 0] = Sigmoid(l);
                }
            }

            // Store pre-activation logits in cache for backward pass
            if (cache != null)
            {
                cache.RangeLogits = rangeLogits;
                cache.QualityLogits = qualityLogits;
            }

            return (regression, range, quality, direction, midDirection, confidence);
        }

        // 
        // Memory management
        // 

        public void PruneNewsMemory(int maxSize)
        {
            if (maxSize <= 0)
            {
                NewsMemory.Clear();
                return;
            }

            if (NewsMemory.Count <= maxSize)
                return;

            if (!PruningConfig.UseAttentionBasedPruning)
            {
                NewsMemory = NewsMemory
                    .OrderByDescending(e => e.AbsoluteTimestamp)
                    .Take(maxSize)
                    .ToList();
                return;
            }

            int reserve = 0;
            if (PruningConfig.NewEntryReserveFraction > 0f)
            {
                reserve = Math.Max(1, (int)(maxSize * PruningConfig.NewEntryReserveFraction));
                reserve = Math.Min(reserve, maxSize);
            }

            var byTime = NewsMemory.OrderByDescending(e => e.AbsoluteTimestamp).ToList();
            var kept = byTime
                .Take(reserve)
                .Concat(
                    byTime.Skip(reserve)
                        .OrderByDescending(e => e.QueryCount >= PruningConfig.MinQueryCountForPruning
                            ? e.AttentionScore
                            : float.MinValue)
                        .Take(maxSize - reserve))
                .ToList();

            NewsMemory = kept;
        }
        internal void PricePruneMemory(int maxSize)
        {
            if (maxSize <= 0)
            {
                PriceMemory.Clear();
                return;
            }

            if (PriceMemory.Count <= maxSize)
                return;

            if (!PruningConfig.UseAttentionBasedPruning)
            {
                PriceMemory = PriceMemory
                    .OrderByDescending(e => e.AbsoluteTimestamp)
                    .Take(maxSize)
                    .ToList();

                return;
            }

            int reserve = 0;

            if (PruningConfig.NewEntryReserveFraction > 0f)
            {
                reserve = Math.Max(1, (int)(maxSize * PruningConfig.NewEntryReserveFraction));
                reserve = Math.Min(reserve, maxSize);
            }

            var byTime = PriceMemory
                .OrderByDescending(e => e.AbsoluteTimestamp)
                .ToList();

            var kept = byTime
                .Take(reserve)
                .Concat(
                    byTime.Skip(reserve)
                        .OrderByDescending(e =>
                            e.QueryCount >= PruningConfig.MinQueryCountForPruning
                                ? e.AttentionScore
                                : float.MinValue)
                        .Take(maxSize - reserve))
                .ToList();

            PriceMemory = kept;
        }

        internal void PricePruneMemoryAfterAppend(int maxSize, int appendedEntryCount)
        {
            if (maxSize <= 0)
            {
                PriceMemory.Clear();
                return;
            }

            if (PriceMemory.Count <= maxSize)
                return;

            if (appendedEntryCount <= 0)
            {
                PricePruneMemory(maxSize);
                return;
            }

            var indexed = PriceMemory
                .Select((entry, index) => new IndexedPriceMemoryEntry
                {
                    Entry = entry,
                    Index = index
                })
                .Where(x => x.Entry != null && x.Entry.HiddenState != null)
                .ToList();

            if (indexed.Count == 0)
            {
                PriceMemory.Clear();
                return;
            }

            if (indexed.Count <= maxSize)
            {
                PriceMemory = indexed
                    .OrderBy(x => x.Entry.AbsoluteTimestamp)
                    .ThenBy(x => x.Index)
                    .Select(x => x.Entry)
                    .ToList();

                return;
            }

            int protectedCount = Math.Clamp(appendedEntryCount, 0, Math.Min(maxSize, indexed.Count));

            var kept = new List<IndexedPriceMemoryEntry>(maxSize);
            var keptIndexes = new HashSet<int>();

            void Keep(IndexedPriceMemoryEntry item)
            {
                if (kept.Count >= maxSize)
                    return;

                if (keptIndexes.Add(item.Index))
                    kept.Add(item);
            }

            // Protect the entries that were just appended by the current sample.
            // Use append order, not timestamp, because rolling/sliding datasets can
            // contain overlapping absolute timestamps.
            foreach (var item in indexed
                .OrderByDescending(x => x.Index)
                .Take(protectedCount)
                .OrderBy(x => x.Index))
            {
                Keep(item);
            }

            int reserveTarget = protectedCount;

            if (PruningConfig.NewEntryReserveFraction > 0f)
            {
                int fractionalReserve = Math.Max(
                    1,
                    (int)MathF.Ceiling(maxSize * PruningConfig.NewEntryReserveFraction));

                fractionalReserve = Math.Clamp(fractionalReserve, 0, maxSize);
                reserveTarget = Math.Max(reserveTarget, fractionalReserve);
            }

            reserveTarget = Math.Clamp(reserveTarget, 0, maxSize);

            var newestFirst = indexed
                .OrderByDescending(x => x.Entry.AbsoluteTimestamp)
                .ThenByDescending(x => x.Index)
                .ToList();

            // Satisfy configured newest-entry reserve without evicting the protected
            // just-appended sequence.
            foreach (var item in newestFirst)
            {
                if (kept.Count >= reserveTarget)
                    break;

                Keep(item);
            }

            if (!PruningConfig.UseAttentionBasedPruning)
            {
                foreach (var item in newestFirst)
                {
                    if (kept.Count >= maxSize)
                        break;

                    Keep(item);
                }
            }
            else
            {
                foreach (var item in indexed
                    .Where(x => !keptIndexes.Contains(x.Index))
                    .OrderByDescending(x =>
                        x.Entry.QueryCount >= PruningConfig.MinQueryCountForPruning
                            ? x.Entry.AttentionScore
                            : float.MinValue)
                    .ThenByDescending(x => x.Entry.AbsoluteTimestamp)
                    .ThenByDescending(x => x.Index))
                {
                    if (kept.Count >= maxSize)
                        break;

                    Keep(item);
                }
            }

            // Keep chronological order for price context. Cross-attention applies RoPE
            // to context keys, so stable temporal ordering matters.
            PriceMemory = kept
                .OrderBy(x => x.Entry.AbsoluteTimestamp)
                .ThenBy(x => x.Index)
                .Select(x => x.Entry)
                .ToList();
        }

        private sealed class IndexedPriceMemoryEntry
        {
            public PriceMemoryEntry Entry { get; set; }
            public int Index { get; set; }
        }
        private void UpdateMemoryAttentionScores(float[,] priceHidden, float[,] ctxH, int total, int globalOffset, int storedNewsMemoryCount, int liveNewsCount, int storedPriceMemoryCount)
        {
            if (ctxH == null || total == 0)
            {
                return;
            }

            int ed = _config.Price.EmbeddingDim;
            int last = priceHidden.GetLength(0) - 1;
            float alpha = PruningConfig.AttentionScoreAlpha;
            float scale = 1.0f / MathF.Sqrt(ed);

            var scores = _accel.ComputeMemoryAttentionScores(priceHidden, last, ctxH, total, scale);

            float max = scores.Length > 0 ? scores.Max() : float.MinValue;
            float sum = 0f;

            for (int s = 0; s < total; s++)
            {
                scores[s] = MathF.Exp(scores[s] - max);
                sum += scores[s];
            }

            if (sum > 0f)
            {
                for (int s = 0; s < total; s++)
                {
                    scores[s] /= sum;
                }
            }

            // Context layout in PredictWithMemory:
            // [global?] [stored news memory] [live news from this call] [stored price memory]

            int storedNewsOffset = globalOffset;

            for (int i = 0; i < storedNewsMemoryCount && i < NewsMemory.Count; i++)
            {
                int ctxIdx = storedNewsOffset + i;

                if (ctxIdx >= total)
                {
                    break;
                }

                var e = NewsMemory[i];
                e.AttentionScore = alpha * scores[ctxIdx] + (1f - alpha) * e.AttentionScore;
                e.QueryCount++;
            }

            int storedPriceOffset = globalOffset + storedNewsMemoryCount + liveNewsCount;

            for (int i = 0; i < storedPriceMemoryCount && i < PriceMemory.Count; i++)
            {
                int ctxIdx = storedPriceOffset + i;

                if (ctxIdx >= total)
                {
                    break;
                }

                var e = PriceMemory[i];
                e.AttentionScore = alpha * scores[ctxIdx] + (1f - alpha) * e.AttentionScore;
                e.QueryCount++;
            }
        }

        private void ApplyContextTypeEmbeddings(float[,] cH, int[] types)
            => _accel.ApplyContextTypeEmbedding(cH, ContextTypeEmbedding, types);

        // 
        // Text encoder helpers
        // 

        private (float[,] hidden, float[] times) EncodeStories(NewsStory[] stories)
        {
            if (stories == null)
                throw new ArgumentNullException(nameof(stories));

            int storyCount = stories.Length;
            int ed = _config.Text.EmbeddingDim;
            var hidden = new float[storyCount, ed];
            var times = new float[storyCount];

            for (int i = 0; i < storyCount; i++)
            {
                var story = stories[i];
                if (object.ReferenceEquals(story, null))
                {
                    times[i] = 0f;
                    continue;
                }

                times[i] = story.ArrivalTime;

                if (story.TokenIds == null || story.TokenIds.Length == 0)
                    continue;

                var tokenHidden = ForwardTextEncoder(story.TokenIds);
                int tokenCount = tokenHidden.GetLength(0);
                if (tokenCount == 0)
                    continue;

                float inv = 1.0f / tokenCount;
                for (int d = 0; d < ed; d++)
                {
                    float sum = 0f;
                    for (int t = 0; t < tokenCount; t++)
                        sum += tokenHidden[t, d];
                    hidden[i, d] = sum * inv;
                }
            }

            return (hidden, times);
        }

        internal (float[,] hidden, float[] times) EncodeStoriesForMemory(NewsStory[] stories)
        {
            return EncodeStories(stories);
        }

        internal (float[,] hidden, float[] times) EncodeStoriesWithCache(NewsStory[] stories, MmtacForwardCache cache)
        {
            if (stories == null)
                throw new ArgumentNullException(nameof(stories));
            if (cache == null)
                throw new ArgumentNullException(nameof(cache));

            int storyCount = stories.Length;
            int ed = _config.Text.EmbeddingDim;
            var hidden = new float[storyCount, ed];
            var times = new float[storyCount];

            cache.StoryCaches = new List<MmtacForwardCache>(storyCount);
            cache.StoryTokenCounts = new int[storyCount];

            for (int i = 0; i < storyCount; i++)
            {
                var story = stories[i];

                var storyCache = new MmtacForwardCache(_config.Text.NumLayers, _config.Price.NumLayers);
                cache.StoryCaches.Add(storyCache);

                if (object.ReferenceEquals(story, null))
                {
                    times[i] = 0f;
                    storyCache.TextTokenIds = Array.Empty<int>();
                    storyCache.TextEmbedded = new float[0, ed];
                    storyCache.TextFinalHidden = new float[0, ed];
                    cache.StoryTokenCounts[i] = 0;
                    continue;
                }

                times[i] = story.ArrivalTime;

                if (story.TokenIds == null || story.TokenIds.Length == 0)
                {
                    storyCache.TextTokenIds = Array.Empty<int>();
                    storyCache.TextEmbedded = new float[0, ed];
                    storyCache.TextFinalHidden = new float[0, ed];
                    cache.StoryTokenCounts[i] = 0;
                    continue;
                }

                var tokenHidden = ForwardTextEncoderWithCache(story.TokenIds, storyCache);
                int tokenCount = tokenHidden.GetLength(0);
                cache.StoryTokenCounts[i] = tokenCount;

                if (tokenCount == 0)
                    continue;

                float inv = 1.0f / tokenCount;
                for (int d = 0; d < ed; d++)
                {
                    float sum = 0f;
                    for (int t = 0; t < tokenCount; t++)
                        sum += tokenHidden[t, d];
                    hidden[i, d] = sum * inv;
                }
            }

            cache.TextFinalHidden = hidden;
            return (hidden, times);
        }

        private float[,] ForwardTextEncoder(int[] tokenIds)
        {
            int ed = _config.Text.EmbeddingDim;
            if (tokenIds == null || tokenIds.Length == 0)
                return new float[0, ed];

            int seqLen = tokenIds.Length;
            var x = EmbedTextTokens(tokenIds, seqLen);
            bool[,] mask = _config.Text.UseDecoderOnly ? CreateCausalMask(seqLen) : null;

            foreach (var block in TextBlocks)
                x = block.Forward(x, mask);

            return x;
        }

        private float[,] ForwardTextEncoderWithCache(int[] tokenIds, MmtacForwardCache cache)
        {
            if (cache == null)
                throw new ArgumentNullException(nameof(cache));

            int ed = _config.Text.EmbeddingDim;
            if (tokenIds == null || tokenIds.Length == 0)
            {
                cache.TextTokenIds = Array.Empty<int>();
                cache.TextEmbedded = new float[0, ed];
                cache.TextFinalHidden = new float[0, ed];
                return cache.TextFinalHidden;
            }

            int seqLen = tokenIds.Length;
            var emb = EmbedTextTokens(tokenIds, seqLen);
            cache.TextEmbedded = emb;
            cache.TextTokenIds = tokenIds;

            bool[,] mask = _config.Text.UseDecoderOnly ? CreateCausalMask(seqLen) : null;
            var x = emb;

            for (int layer = 0; layer < _config.Text.NumLayers; layer++)
            {
                var block = TextBlocks[layer];
                cache.TextLayerInputs.Add(x);

                var ac = cache.TextAttentionCaches[layer];
                ac.Input = x;

                var q = ComputeProjection(x, block.Attention.WQ, block.Attention.BiasQ);
                var k = ComputeProjection(x, block.Attention.WK, block.Attention.BiasK);
                var v = ComputeProjection(x, block.Attention.WV, block.Attention.BiasV);
                RotaryPositionEmbedding.ApplyInPlace(q, k, _config.Text.NumHeads);
                ac.Q = q;
                ac.K = k;
                ac.V = v;

                float scale = 1.0f / MathF.Sqrt(ed / _config.Text.NumHeads);
                var attnOut = _accel.MultiHeadAttentionForward(q, k, v, _config.Text.NumHeads, scale, mask);
                ac.AttentionOutput = attnOut;

                var projected = ComputeProjection(attnOut, block.Attention.WO, block.Attention.BiasO);
                var residual1 = _accel.MatrixAdd(x, projected);
                var ln1c = cache.TextLN1Caches[layer];
                var (normed1, mean1, var1, normalized1) = _accel.LayerNormForward(residual1, block.LN1Gamma, block.LN1Beta);
                ln1c.Input = residual1;
                ln1c.Mean = mean1;
                ln1c.Variance = var1;
                ln1c.Normalized = normalized1;

                cache.TextFFNInputs.Add(normed1);
                var ffnOut = _accel.FFNForwardBatch(normed1, seqLen, ed, block.FeedForwardNetwork.ForwardPassOnly);
                cache.TextFFNOutputs.Add(ffnOut);

                var residual2 = _accel.MatrixAdd(normed1, ffnOut);
                var ln2c = cache.TextLN2Caches[layer];
                var (normed2, mean2, var2, normalized2) = _accel.LayerNormForward(residual2, block.LN2Gamma, block.LN2Beta);
                ln2c.Input = residual2;
                ln2c.Mean = mean2;
                ln2c.Variance = var2;
                ln2c.Normalized = normalized2;

                x = normed2;
            }

            cache.TextFinalHidden = x;
            return x;
        }
        // 
        // Content-aware cross-attention wrapper
        // 
        internal float[,] ForwardPriceDecoderForMemoryBuild(float[,] priceSequence, float[,] contextHidden, float[] contextTimes, int globalBypassCount = 0)
        {
            return ForwardPriceDecoder(priceSequence, contextHidden, contextTimes, globalBypassCount);
        }
        private float[,] ContentAwareCrossAttentionWithCache(float[,] Q, float[,] K, float[,] V, float[,] timeDiffs, float[] keyTimesFromRef, float[,] queryEmbeddings, float[,] keyEmbeddings, TacamtBlock block, BlockCache bc, bool isTraining, Random dropoutRng, int globalBypassCount)
        {
            return _accel.ContentAwareCrossAttentionWithCache(Q, K, V, timeDiffs, keyTimesFromRef, queryEmbeddings, keyEmbeddings, block, bc, _config.Price.EmbeddingDim, _config.Price.NumHeads, isTraining, dropoutRng);
        }

        // 
        // Embedding helpers
        // 

        private float[,] EmbedTextTokens(int[] tokenIds, int sl)
        {
            if (tokenIds == null || tokenIds.Length == 0)
                return null;

            int actualLen = Math.Min(sl, tokenIds.Length);

            var embedded = new float[actualLen, _config.Text.EmbeddingDim];

            for (int i = 0; i < actualLen; i++)
            {
                int tok = tokenIds[i];

                if (tok < 0 || tok >= TextTokenEmbedding.GetLength(0))
                    throw new ArgumentException($"Token out of range: {tok}");

                for (int j = 0; j < _config.Text.EmbeddingDim; j++)
                    embedded[i, j] = TextTokenEmbedding[tok, j];
            }

            return embedded;
        }

        private float[,] EmbedPriceSequence(float[,] ps, int rowStart, int rowCount)
        {
            if (ps == null)
                throw new ArgumentNullException(nameof(ps));

            int rows = ps.GetLength(0);
            int cols = ps.GetLength(1);

            if (cols != _config.Price.InputFeatureDim)
            {
                throw new ArgumentException(
                    $"PriceSequence feature dimension ({cols}) must equal Price.InputFeatureDim ({_config.Price.InputFeatureDim}).",
                    nameof(ps));
            }

            if (rowStart < 0 || rowCount <= 0 || rowStart + rowCount > rows)
            {
                throw new ArgumentOutOfRangeException(
                    nameof(rowStart),
                    $"Invalid price row slice: start={rowStart}, count={rowCount}, rows={rows}.");
            }

            var projected = _accel.BatchDotProduct(PriceInputProjection, ps, rowStart, rowCount);
            return _accel.MatrixAddBias(projected, PriceInputProjectionBias);
        }

        // 
        // Utilities
        // 

        internal float[,] ComputeProjection(float[,] input, float[,] w, float[] b)
        {
            var p = _accel.BatchDotProduct(w, input);
            return _accel.MatrixAddBias(p, b);
        }

        private bool[,] CreateCausalMask(int sl) => _accel.CreateCausalMask(sl);

        protected float SampleGaussian()
        {
            float u1 = 1f - _random.NextSingle(), u2 = 1f - _random.NextSingle();
            return MathF.Sqrt(-2f * MathF.Log(u1)) * MathF.Cos(2f * MathF.PI * u2);
        }

        private static float Sigmoid(float x)
        {
            if (x >= 0) { float ex = MathF.Exp(-x); return 1f / (1f + ex); }
            else { float ex = MathF.Exp(x); return ex / (1f + ex); }
        }

        private float[,] InitWeights(int rows, int cols, float std)
        {
            var w = new float[rows, cols];
            for (int i = 0; i < rows; i++)
                for (int j = 0; j < cols; j++)
                    w[i, j] = SampleGaussian() * std;
            return w;
        }

        private static IAccelerationManager CreateAccel(RuntimeConfig rt)
        {
            return rt.AccelerationType switch
            {
                AccelerationType.GPU or AccelerationType.CUDA
                    => new AccelerationGPU(rt.AccelerationType, rt.AccelerationDeviceId),
                AccelerationType.CPU => new AccelerationCPU(),
                AccelerationType.MultiThreadCPU => new AccelerationMutliThreadCPU(),
                _ => throw new NotSupportedException($"Unsupported AccelerationType: {rt.AccelerationType}")
            };
        }
        internal float[,] ForwardPriceDecoderForMemoryBuild(float[,] priceSequence, float[,] contextHidden, float[] contextTimes)
        {
            return ForwardPriceDecoder(priceSequence, contextHidden, contextTimes);
        }
        // 
        // Init helpers
        // 

        private void InitTextEncoder()
        {
            int v = _config.Text.VocabSize;
            int ed = _config.Text.EmbeddingDim;
            float std = MathF.Sqrt(2.0f / (v + ed));
            TextTokenEmbedding = InitWeights(v, ed, std);
            TextBlocks = new TransformerBlock[_config.Text.NumLayers];

            for (int i = 0; i < _config.Text.NumLayers; i++)
            {
                TextBlocks[i] = new TransformerBlock(ed, _config.Text.NumHeads, _config.Text.FeedForwardDim, _config.Runtime.FFNActivationType, CostFunctionType.mse, ActivationDistribution.Normal, _config.Reg.L2RegulationLamda, _config.Reg.GradientClippingThreshold, _config.Runtime.AccelerationType, _config.Runtime.AccelerationDeviceId, _accel, _random);
            }
        }

        private void InitPriceDecoder()
        {
            int fd = _config.Price.InputFeatureDim;
            int ed = _config.Price.EmbeddingDim;
            float std = MathF.Sqrt(2.0f / (fd + ed));
            PriceInputProjection = InitWeights(ed, fd, std);
            PriceInputProjectionBias = new float[ed];

            PriceBlocks = new TacamtBlock[_config.Price.NumLayers];
            for (int i = 0; i < _config.Price.NumLayers; i++)
                PriceBlocks[i] = new TacamtBlock(ed, _config.Price.NumHeads, _config.Price.FeedForwardDim,
                    _config.Runtime.FFNActivationType, _accel, _random,
                    l2Lambda: _config.Reg.L2RegulationLamda,
                    decayProjectionDim: _config.Decay.ProjectionDim,
                    decayHiddenDim: _config.Decay.HiddenDim,
                    decayMemAttnDropout: _config.Decay.MemAttentionDropout,
                    decayMLPDropout: _config.Decay.MlpDropout,
                    decayWeightDecay: _config.Decay.WeightDecay,
                    decayTimeBases: _config.Decay.TimeEncodingBases);
        }

        private void InitGlobalProjection()
        {
            int gd = _config.Global.GlobalFeatureDim;
            if (gd <= 0) return;
            int ed = _config.Price.EmbeddingDim;
            float std = MathF.Sqrt(2.0f / (gd + ed));
            GlobalFeatureProjection = InitWeights(ed, gd, std);
            GlobalFeatureBias = new float[ed];
        }

        private void InitContextTypeEmbedding()
        {
            int ed = _config.Price.EmbeddingDim;
            float typeStd = MathF.Sqrt(2.0f / ed) * 0.1f;
            ContextTypeEmbedding = new float[MmtacConfig.ContextTypeCount, ed];
            for (int t = 0; t < MmtacConfig.ContextTypeCount; t++)
                for (int d = 0; d < ed; d++)
                    ContextTypeEmbedding[t, d] = SampleGaussian() * typeStd;
        }

        private void InitOutputHeads()
        {
            int ed = _config.Price.EmbeddingDim;
            int rDim = MmtacOutputConfig.RegressionOutputCount; // 3
            float regStd = MathF.Sqrt(2.0f / (ed + rDim));
            float clsStd = MathF.Sqrt(2.0f / (ed + 1));

            // Regression (High, Low, Close) - linear
            RegressionProjection = InitWeights(rDim, ed, regStd);
            RegressionBias = new float[rDim];

            // Range - softplus
            RangeProjection = InitWeights(1, ed, clsStd);
            RangeBias = new float[1];

            // Quality - sigmoid
            QualityProjection = InitWeights(1, ed, clsStd);
            QualityBias = new float[1];

            // Direction - sigmoid
            DirectionProjection = InitWeights(1, ed, clsStd);
            DirectionBias = new float[1];

            // MidWindowDirection - sigmoid
            MidDirectionProjection = InitWeights(1, ed, clsStd);
            MidDirectionBias = new float[1];

            // Confidence (optional) - sigmoid
            if (_config.Output.UseConfidenceHead)
            {
                ConfidenceProjection = InitWeights(1, ed, clsStd);
                ConfidenceBias = new float[1];
            }
        }

        // 
        // Save / Load
        // 

        public void Save(string dir)
        {
            Directory.CreateDirectory(dir);

            var cd = new Dictionary<string, object>
            {
                ["TextVocabSize"] = _config.Text.VocabSize,
                ["TextMaxSeqLen"] = _config.Text.MaxSequenceLength,
                ["TextEmbeddingDim"] = _config.Text.EmbeddingDim,
                ["TextNumHeads"] = _config.Text.NumHeads,
                ["TextNumLayers"] = _config.Text.NumLayers,
                ["TextFeedForwardDim"] = _config.Text.FeedForwardDim,
                ["TextUseDecoderOnly"] = _config.Text.UseDecoderOnly,
                ["TextFreeze"] = _config.Text.Freeze,

                ["PriceInputFeatureDim"] = _config.Price.InputFeatureDim,
                ["PriceMaxSeqLen"] = _config.Price.MaxSequenceLength,
                ["PriceEmbeddingDim"] = _config.Price.EmbeddingDim,
                ["PriceNumHeads"] = _config.Price.NumHeads,
                ["PriceNumLayers"] = _config.Price.NumLayers,
                ["PriceFeedForwardDim"] = _config.Price.FeedForwardDim,
                ["PriceUseDecoderOnly"] = _config.Price.UseDecoderOnly,

                ["GlobalFeatureDim"] = _config.Global.GlobalFeatureDim,
                ["GlobalBypassDecay"] = _config.Global.BypassDecay,

                ["UseConfidenceHead"] = _config.Output.UseConfidenceHead,
                ["DirectionLossWeight"] = _config.Output.DirectionLossWeight,
                ["MidDirectionLossWeight"] = _config.Output.MidDirectionLossWeight,
                ["RangeLossWeight"] = _config.Output.RangeLossWeight,
                ["QualityLossWeight"] = _config.Output.QualityLossWeight,
                ["CloseDirectionConsistencyWeight"] = _config.Output.CloseDirectionConsistencyWeight,
                ["CloseDirectionConsistencyMargin"] = _config.Output.CloseDirectionConsistencyMargin,

                ["FFNActivationType"] = (int)_config.Runtime.FFNActivationType,
                ["AccelerationType"] = (int)_config.Runtime.AccelerationType,
                ["AccelerationDeviceId"] = _config.Runtime.AccelerationDeviceId,

                ["L2RegulationLamda"] = _config.Reg.L2RegulationLamda,
                ["GradientClipThreshold"] = _config.Reg.GradientClippingThreshold,

                ["DecayEnabled"] = _config.Decay.Enabled,
                ["DecayProjectionDim"] = _config.Decay.ProjectionDim,
                ["DecayHiddenDim"] = _config.Decay.HiddenDim,
                ["DecayTimeEncodingBases"] = _config.Decay.TimeEncodingBases,
                ["DecayMemAttnDropout"] = _config.Decay.MemAttentionDropout,
                ["DecayMlpDropout"] = _config.Decay.MlpDropout,
                ["DecayWeightDecay"] = _config.Decay.WeightDecay,

                ["PriceContextEnabled"] = _config.PriceContext.Enabled,
                ["PriceContextMinHistory"] = _config.PriceContext.MinHistoryLength,
                ["PriceContextMinCurrent"] = _config.PriceContext.MinCurrentLength,

                ["PruningAlpha"] = PruningConfig.AttentionScoreAlpha,
                ["PruningMinQueryCount"] = PruningConfig.MinQueryCountForPruning,
                ["PruningReserveFraction"] = PruningConfig.NewEntryReserveFraction,
                ["PruningUseAttentionBased"] = PruningConfig.UseAttentionBasedPruning
            };

            File.WriteAllText(
                Path.Combine(dir, "config.json"),
                JsonSerializer.Serialize(cd, new JsonSerializerOptions { WriteIndented = true }));

            using (var s = new FileStream(Path.Combine(dir, "weights.bin"), FileMode.Create))
            using (var w = new BinaryWriter(s))
            {
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

                if (GlobalFeatureProjection != null)
                {
                    WM(w, GlobalFeatureProjection);
                    WV(w, GlobalFeatureBias);
                }

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

                WM(w, RegressionProjection);
                WV(w, RegressionBias);

                WM(w, RangeProjection);
                WV(w, RangeBias);

                WM(w, QualityProjection);
                WV(w, QualityBias);

                WM(w, DirectionProjection);
                WV(w, DirectionBias);

                WM(w, MidDirectionProjection);
                WV(w, MidDirectionBias);

                if (_config.Output.UseConfidenceHead)
                {
                    WM(w, ConfidenceProjection);
                    WV(w, ConfidenceBias);
                }

                WM(w, ContextTypeEmbedding);

                w.Flush();
            }

            for (int i = 0; i < _config.Text.NumLayers; i++)
                TextBlocks[i].FeedForwardNetwork.Save(Path.Combine(dir, $"text_ffn_{i}"));

            for (int i = 0; i < _config.Price.NumLayers; i++)
                PriceBlocks[i].FeedForwardNetwork.Save(Path.Combine(dir, $"price_ffn_{i}"));

            SaveMemory(dir);
            Tokenizer?.Save(Path.Combine(dir, "tokenizer"));
        }
        public static MmtacModel Load(string dir)
        {
            var d = JsonSerializer.Deserialize<Dictionary<string, JsonElement>>(
                File.ReadAllText(Path.Combine(dir, "config.json")));

            if (d == null)
                throw new InvalidOperationException("config.json could not be deserialized.");

            var cfg = new MmtacConfig
            {
                Text = new TextEncoderConfig
                {
                    VocabSize = d["TextVocabSize"].GetInt32(),
                    MaxSequenceLength = d["TextMaxSeqLen"].GetInt32(),
                    EmbeddingDim = d["TextEmbeddingDim"].GetInt32(),
                    NumHeads = d["TextNumHeads"].GetInt32(),
                    NumLayers = d["TextNumLayers"].GetInt32(),
                    FeedForwardDim = d["TextFeedForwardDim"].GetInt32(),
                    UseDecoderOnly = d["TextUseDecoderOnly"].GetBoolean(),
                    Freeze = d["TextFreeze"].GetBoolean()
                },

                Price = new PriceDecoderConfig
                {
                    InputFeatureDim = d["PriceInputFeatureDim"].GetInt32(),
                    MaxSequenceLength = d["PriceMaxSeqLen"].GetInt32(),
                    EmbeddingDim = d["PriceEmbeddingDim"].GetInt32(),
                    NumHeads = d["PriceNumHeads"].GetInt32(),
                    NumLayers = d["PriceNumLayers"].GetInt32(),
                    FeedForwardDim = d["PriceFeedForwardDim"].GetInt32(),
                    UseDecoderOnly = d["PriceUseDecoderOnly"].GetBoolean()
                },

                Global = new MmtacGlobalConfig
                {
                    GlobalFeatureDim = d.ContainsKey("GlobalFeatureDim")
                        ? d["GlobalFeatureDim"].GetInt32()
                        : 0,

                    BypassDecay = d.ContainsKey("GlobalBypassDecay")
                        && d["GlobalBypassDecay"].GetBoolean()
                },

                Output = new MmtacOutputConfig
                {
                    UseConfidenceHead = d["UseConfidenceHead"].GetBoolean(),

                    DirectionLossWeight = d.ContainsKey("DirectionLossWeight")
                        ? d["DirectionLossWeight"].GetSingle()
                        : 1f,

                    MidDirectionLossWeight = d.ContainsKey("MidDirectionLossWeight")
                        ? d["MidDirectionLossWeight"].GetSingle()
                        : 0.5f,

                    RangeLossWeight = d.ContainsKey("RangeLossWeight")
                        ? d["RangeLossWeight"].GetSingle()
                        : 1f,

                    QualityLossWeight = d.ContainsKey("QualityLossWeight")
                        ? d["QualityLossWeight"].GetSingle()
                        : 1f,

                    CloseDirectionConsistencyWeight = d.ContainsKey("CloseDirectionConsistencyWeight")
                        ? d["CloseDirectionConsistencyWeight"].GetSingle()
                        : 1.0f,

                    CloseDirectionConsistencyMargin = d.ContainsKey("CloseDirectionConsistencyMargin")
                        ? d["CloseDirectionConsistencyMargin"].GetSingle()
                        : 0.02f
                },

                Runtime = new RuntimeConfig
                {
                    FFNActivationType = (ActivationType)d["FFNActivationType"].GetInt32(),
                    AccelerationType = (AccelerationType)d["AccelerationType"].GetInt32(),
                    AccelerationDeviceId = d["AccelerationDeviceId"].GetInt32()
                },

                Reg = new RegularizationConfig
                {
                    L2RegulationLamda = d["L2RegulationLamda"].GetSingle(),
                    GradientClippingThreshold = d["GradientClipThreshold"].GetSingle()
                },

                Decay = new DecayNetworkConfig
                {
                    Enabled = d.ContainsKey("DecayEnabled")
                        ? d["DecayEnabled"].GetBoolean()
                        : true,

                    ProjectionDim = d.ContainsKey("DecayProjectionDim")
                        ? d["DecayProjectionDim"].GetInt32()
                        : 16,

                    HiddenDim = d.ContainsKey("DecayHiddenDim")
                        ? d["DecayHiddenDim"].GetInt32()
                        : 32,

                    TimeEncodingBases = d.ContainsKey("DecayTimeEncodingBases")
                        ? d["DecayTimeEncodingBases"].GetInt32()
                        : 16,

                    MemAttentionDropout = d.ContainsKey("DecayMemAttnDropout")
                        ? d["DecayMemAttnDropout"].GetSingle()
                        : 0.2f,

                    MlpDropout = d.ContainsKey("DecayMlpDropout")
                        ? d["DecayMlpDropout"].GetSingle()
                        : 0.2f,

                    WeightDecay = d.ContainsKey("DecayWeightDecay")
                        ? d["DecayWeightDecay"].GetSingle()
                        : 0f
                },

                PriceContext = new PriceContextConfig
                {
                    Enabled = !d.ContainsKey("PriceContextEnabled")
                        || d["PriceContextEnabled"].GetBoolean(),

                    MinHistoryLength = d.ContainsKey("PriceContextMinHistory")
                        ? d["PriceContextMinHistory"].GetInt32()
                        : 5,

                    MinCurrentLength = d.ContainsKey("PriceContextMinCurrent")
                        ? d["PriceContextMinCurrent"].GetInt32()
                        : 5
                },

                Pruning = new MemoryPruningConfig
                {
                    AttentionScoreAlpha = d.ContainsKey("PruningAlpha")
                        ? d["PruningAlpha"].GetSingle()
                        : new MemoryPruningConfig().AttentionScoreAlpha,

                    MinQueryCountForPruning = d.ContainsKey("PruningMinQueryCount")
                        ? d["PruningMinQueryCount"].GetInt32()
                        : new MemoryPruningConfig().MinQueryCountForPruning,

                    NewEntryReserveFraction = d.ContainsKey("PruningReserveFraction")
                        ? d["PruningReserveFraction"].GetSingle()
                        : new MemoryPruningConfig().NewEntryReserveFraction,

                    UseAttentionBasedPruning = d.ContainsKey("PruningUseAttentionBased")
                        ? d["PruningUseAttentionBased"].GetBoolean()
                        : new MemoryPruningConfig().UseAttentionBasedPruning
                }
            };

            cfg.Validate();

            var m = new MmtacModel(cfg);

            using (var s = new FileStream(Path.Combine(dir, "weights.bin"), FileMode.Open))
            using (var r = new BinaryReader(s))
            {
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

                if (cfg.Global.GlobalFeatureDim > 0)
                {
                    RM(r, m.GlobalFeatureProjection);
                    RV(r, m.GlobalFeatureBias);
                }

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

                RM(r, m.RegressionProjection);
                RV(r, m.RegressionBias);

                RM(r, m.RangeProjection);
                RV(r, m.RangeBias);

                RM(r, m.QualityProjection);
                RV(r, m.QualityBias);

                RM(r, m.DirectionProjection);
                RV(r, m.DirectionBias);

                RM(r, m.MidDirectionProjection);
                RV(r, m.MidDirectionBias);

                if (cfg.Output.UseConfidenceHead)
                {
                    RM(r, m.ConfidenceProjection);
                    RV(r, m.ConfidenceBias);
                }

                if (s.Position < s.Length)
                    RM(r, m.ContextTypeEmbedding);
            }

            for (int i = 0; i < cfg.Text.NumLayers; i++)
            {
                var loadedFfn = NeuralNetwork.Load(
                    Path.Combine(dir, $"text_ffn_{i}"),
                    cfg.Runtime.AccelerationType);

                m.TextBlocks[i].ReplaceFeedForwardNetwork(loadedFfn);
            }

            for (int i = 0; i < cfg.Price.NumLayers; i++)
            {
                m.PriceBlocks[i].FeedForwardNetwork = NeuralNetwork.Load(
                    Path.Combine(dir, $"price_ffn_{i}"),
                    cfg.Runtime.AccelerationType);
            }

            m.LoadMemory(dir);

            var tokDir = Path.Combine(dir, "tokenizer");

            if (Directory.Exists(tokDir))
            {
                try
                {
                    m.Tokenizer = BPETokenizer.Load(tokDir);
                }
                catch
                {
                    // Keep model loading tolerant of tokenizer corruption/missing optional tokenizer files.
                }
            }

            return m;
        }
        private void SaveMemory(string dir)
        {
            int ed = _config.Price.EmbeddingDim;
            using var ns = new FileStream(Path.Combine(dir, "news_memory.bin"), FileMode.Create);
            using var nw = new BinaryWriter(ns);
            nw.Write(LastPriceTimestamp); nw.Write(NewsMemory.Count);
            foreach (var e in NewsMemory)
            {
                nw.Write(e.AbsoluteTimestamp); nw.Write(e.AttentionScore); nw.Write(e.QueryCount);
                for (int d = 0; d < ed; d++) nw.Write(e.HiddenState[d]);
            }
            using var ps = new FileStream(Path.Combine(dir, "price_memory.bin"), FileMode.Create);
            using var pw = new BinaryWriter(ps);
            pw.Write(PriceMemory.Count);
            foreach (var e in PriceMemory)
            {
                pw.Write(e.AbsoluteTimestamp); pw.Write(e.AttentionScore); pw.Write(e.QueryCount);
                for (int d = 0; d < ed; d++) pw.Write(e.HiddenState[d]);
            }
        }

        private void LoadMemory(string dir)
        {
            int ed = _config.Price.EmbeddingDim;
            var np = Path.Combine(dir, "news_memory.bin");
            if (File.Exists(np)) using (var s = new FileStream(np, FileMode.Open)) using (var r = new BinaryReader(s))
            {
                LastPriceTimestamp = r.ReadDouble(); int c = r.ReadInt32();
                NewsMemory = new List<NewsMemoryEntry>(c);
                for (int i = 0; i < c; i++)
                {
                    var e = new NewsMemoryEntry { AbsoluteTimestamp = r.ReadDouble(), AttentionScore = r.ReadSingle(), QueryCount = r.ReadInt32(), HiddenState = new float[ed] };
                    for (int d = 0; d < ed; d++) e.HiddenState[d] = r.ReadSingle();
                    NewsMemory.Add(e);
                }
            }
            var pp = Path.Combine(dir, "price_memory.bin");
            if (File.Exists(pp)) using (var s = new FileStream(pp, FileMode.Open)) using (var r = new BinaryReader(s))
            {
                int c = r.ReadInt32();
                PriceMemory = new List<PriceMemoryEntry>(c);
                for (int i = 0; i < c; i++)
                {
                    var e = new PriceMemoryEntry { AbsoluteTimestamp = r.ReadDouble(), AttentionScore = r.ReadSingle(), QueryCount = r.ReadInt32(), HiddenState = new float[ed] };
                    for (int d = 0; d < ed; d++) e.HiddenState[d] = r.ReadSingle();
                    PriceMemory.Add(e);
                }
            }
        }

        //  Binary I/O helpers 
        static void WM(BinaryWriter w, float[,] m)
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
        static void WV(BinaryWriter w, float[] v)
        {
            w.Write(v.Length);
            for (int i = 0; i < v.Length; i++)
            {
                w.Write(v[i]);
            }
        }
        static void WA(BinaryWriter w, MultiHeadAttention a)
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
        static void RM(BinaryReader r, float[,] m)
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
        static void RV(BinaryReader r, float[] v)
        {
            int l = r.ReadInt32();
            for (int i = 0; i < l; i++)
            {
                v[i] = r.ReadSingle();
            }
        }
        static void RA(BinaryReader r, MultiHeadAttention a)
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
    }
}