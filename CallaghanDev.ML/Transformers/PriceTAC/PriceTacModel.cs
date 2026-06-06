using CallaghanDev.ML.AccelerationManagers;
using CallaghanDev.ML.AccelerationManagers.GPU;
using CallaghanDev.ML.Enums;
using CallaghanDev.ML.Transformers.Configuration;
using CallaghanDev.ML.Transformers.MultiTypeTransformer;
using CallaghanDev.ML.Transformers.TACAMT;
using System.Collections.Concurrent;
using System.Text.Json;

namespace CallaghanDev.ML.Transformers.PriceTAC
{
    /// <summary>
    /// Price-only MMTAC implementation.
    ///
    /// Keeps the MMTAC price path exactly: causal price self-attention, price-memory/global
    /// cross-attention, learned content-aware time-dependent decay, price-context training,
    /// rolling price memory, and all output heads.  There are no language-side
    /// encoders, tokenizers, inputs, or memories in this implementation.
    /// </summary>
    public sealed class PriceTacModel
    {
        private readonly PriceTacConfig _config;
        private readonly Random _random;
        private readonly RotaryPositionEmbedding _rotaryPositionEmbedding;
        private readonly ConcurrentDictionary<int, bool[,]> _causalMaskCache = new ConcurrentDictionary<int, bool[,]>();

        private IAccelerationManager _accel;

        public PriceTacConfig Config => _config;
        public IAccelerationManager AccelerationManager => _accel;

        // Price decoder weights
        public float[,] PriceInputProjection { get; set; }
        public float[] PriceInputProjectionBias { get; set; }
        public TacamtBlock[] PriceBlocks { get; set; }

        // Global feature projection
        public float[,] GlobalFeatureProjection { get; set; }
        public float[] GlobalFeatureBias { get; set; }

        // Shape: [3, PriceEmbeddingDim]. Row 0 is unused; row 1=price-memory; row 2=global.
        public float[,] ContextTypeEmbedding { get; set; }

        // Output head weights
        public float[,] RegressionProjection { get; set; }
        public float[] RegressionBias { get; set; }
        public float[,] RangeProjection { get; set; }
        public float[] RangeBias { get; set; }
        public float[,] QualityProjection { get; set; }
        public float[] QualityBias { get; set; }
        public float[,] DirectionProjection { get; set; }
        public float[] DirectionBias { get; set; }
        public float[,] MidDirectionProjection { get; set; }
        public float[] MidDirectionBias { get; set; }
        public float[,] ConfidenceProjection { get; set; }
        public float[] ConfidenceBias { get; set; }

        // Persistent rolling price memory
        public List<PriceMemoryEntry> PriceMemory { get; set; } = new List<PriceMemoryEntry>();
        public double LastPriceTimestamp { get; set; } = 0;
        public MemoryPruningConfig PruningConfig { get; set; } = new MemoryPruningConfig();

        public PriceTacModel(PriceTacConfig config, Random random = null)
        {
            if (config == null)
                throw new ArgumentNullException(nameof(config));

            config.Validate();

            _config = config;
            _random = random ?? new Random();
            _accel = AccelerationFactory.Create(config.Runtime);
            _rotaryPositionEmbedding = new RotaryPositionEmbedding(_accel);

            PruningConfig = new MemoryPruningConfig
            {
                AttentionScoreAlpha = config.Pruning.AttentionScoreAlpha,
                MinQueryCountForPruning = config.Pruning.MinQueryCountForPruning,
                NewEntryReserveFraction = config.Pruning.NewEntryReserveFraction,
                UseAttentionBasedPruning = config.Pruning.UseAttentionBasedPruning
            };

            InitPriceDecoder();
            InitGlobalProjection();
            InitContextTypeEmbedding();
            InitOutputHeads();
        }

        public (float[,] regression, float[,] range, float[,] quality, float[,] direction, float[,] midDirection, float[,] confidence) Forward(PriceTacInput input)
        {
            if (input == null)
                throw new ArgumentNullException(nameof(input));
            if (input.PriceSequence == null)
                throw new ArgumentNullException(nameof(input.PriceSequence));
            if (input.PriceSequence.GetLength(0) <= 0)
                throw new ArgumentException("PriceSequence must contain at least one row.", nameof(input));

            float[] globalToken = null;
            if (_config.Global.GlobalFeatureDim > 0 && input.GlobalFeatures != null)
                globalToken = EmbedGlobalFeatures(input.GlobalFeatures);

            BuildContext(null, null, globalToken, out var contextHidden, out var contextTimes, out int numGlobal, out int _unusedPriceContextCount);

            var priceHidden = ForwardPriceDecoder(input.PriceSequence, contextHidden, contextTimes, numGlobal);
            return ProjectToOutputs(priceHidden);
        }

        public PriceTacPrediction PredictNext(PriceTacInput input)
        {
            var (reg, range, quality, dir, midDir, conf) = Forward(input);
            int last = reg.GetLength(0) - 1;
            return new PriceTacPrediction
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

        internal (float[,] regression, float[,] range, float[,] quality, float[,] direction, float[,] midDirection, float[,] confidence) ForwardWithCache(PriceTacInput input, PriceTacForwardCache cache, bool isTraining = false, Random dropoutRng = null)
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

        internal (float[,] regression, float[,] range, float[,] quality, float[,] direction, float[,] midDirection, float[,] confidence) ForwardWithCache(PriceTacInput input, int rowStart, int rowCount, PriceTacForwardCache cache, bool isTraining = false, Random dropoutRng = null)
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

            float[] globalToken = null;
            if (_config.Global.GlobalFeatureDim > 0 && input.GlobalFeatures != null)
            {
                globalToken = EmbedGlobalFeatures(input.GlobalFeatures);
                cache.GlobalTokenEmbedded = globalToken;
                cache.GlobalRawInput = input.GlobalFeatures;
            }

            BuildContext(null, null, globalToken, out var contextHidden, out var contextTimes, out int numGlobal, out int numPrice);

            cache.NumGlobalContext = numGlobal;
            cache.NumPriceContext = numPrice;
            cache.ContextHidden = contextHidden;
            cache.ContextTimes = contextTimes;

            var priceHidden = ForwardPriceDecoderWithCache(input.PriceSequence, rowStart, rowCount, contextHidden, contextTimes, cache, isTraining, dropoutRng);
            cache.PriceFinalHidden = priceHidden;
            return ProjectToOutputs(priceHidden, cache);
        }

        public PriceTacPrediction PredictWithMemory(PriceTacInput input, double currentAbsoluteTimestamp, double timeUnitsPerPosition = 1.0, int maxPriceMemorySize = 200)
        {
            if (input == null)
                throw new ArgumentNullException(nameof(input));
            if (input.PriceSequence == null)
                throw new ArgumentNullException(nameof(input.PriceSequence));
            if (input.PriceSequence.GetLength(0) <= 0)
                throw new ArgumentException("PriceSequence must contain at least one row.", nameof(input));
            if (input.PriceSequence.GetLength(1) != _config.Price.InputFeatureDim)
                throw new ArgumentException($"PriceSequence feature dimension ({input.PriceSequence.GetLength(1)}) must equal Price.InputFeatureDim ({_config.Price.InputFeatureDim}).", nameof(input));
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
            int storedPriceMemoryCount = PriceMemory?.Count ?? 0;

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

            BuildManualContext(ctxH, ctxT, ctxTypes, embDim, out var contextHidden, out var contextTimes);

            var priceHidden = ForwardPriceDecoder(input.PriceSequence, contextHidden, contextTimes, globalOffset);
            var (reg, range, quality, dir, midDir, conf) = ProjectToOutputs(priceHidden);
            int last = reg.GetLength(0) - 1;

            UpdateMemoryAttentionScores(priceHidden, contextHidden, ctxH.Count, globalOffset, storedPriceMemoryCount);

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

            LastPriceTimestamp = currentAbsoluteTimestamp + Math.Max(0, priceSeqLen - 1) * timeUnitsPerPosition;
            PricePruneMemoryAfterAppend(maxPriceMemorySize, priceSeqLen);

            return new PriceTacPrediction
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

        public void ClearPriceMemory() => PriceMemory.Clear();

        public void ClearAllMemory()
        {
            ClearPriceMemory();
            LastPriceTimestamp = 0.0;
        }

        internal float[] EmbedGlobalFeatures(float[] globalFeatures)
        {
            if (globalFeatures == null)
                throw new ArgumentNullException(nameof(globalFeatures));

            int gd = _config.Global.GlobalFeatureDim;
            int ed = _config.Price.EmbeddingDim;

            if (gd <= 0)
                throw new InvalidOperationException("GlobalFeatureDim is zero, but EmbedGlobalFeatures was called.");
            if (globalFeatures.Length != gd)
                throw new ArgumentException($"globalFeatures.Length ({globalFeatures.Length}) must equal GlobalFeatureDim ({gd}).", nameof(globalFeatures));

            return _accel.ProjectGlobalFeatures(globalFeatures, GlobalFeatureProjection, GlobalFeatureBias);
        }

        internal void BuildContext(float[,] priceContextHidden, float[] priceContextTimes, float[] preEmbeddedGlobal, out float[,] contextHidden, out float[] contextTimes, out int numGlobal, out int numPrice)
        {
            var result = _accel.BuildMmtacContextWithPrice(
                null,
                null,
                preEmbeddedGlobal,
                priceContextHidden,
                priceContextTimes,
                ContextTypeEmbedding);

            contextHidden = result.contextHidden;
            contextTimes = result.contextTimes;
            numGlobal = result.numGlobal;
            numPrice = result.numPrice;
        }

        internal void BuildManualContext(List<float[]> ctxH, List<float> ctxT, List<int> ctxTypes, int embDim, out float[,] contextHidden, out float[] contextTimes)
        {
            contextHidden = null;
            contextTimes = null;

            if (ctxH == null || ctxH.Count == 0)
                return;

            contextHidden = new float[ctxH.Count, embDim];
            contextTimes = new float[ctxH.Count];

            for (int i = 0; i < ctxH.Count; i++)
            {
                int copyDim = Math.Min(embDim, ctxH[i].Length);
                for (int d = 0; d < copyDim; d++)
                    contextHidden[i, d] = ctxH[i][d];

                contextTimes[i] = ctxT[i];
            }

            _accel.ApplyContextTypeEmbedding(contextHidden, ContextTypeEmbedding, ctxTypes.ToArray());
        }

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

            var cache = new PriceTacForwardCache(_config.Price.NumLayers)
            {
                NumGlobalContext = Math.Max(0, globalBypassCount)
            };

            return ForwardPriceDecoderWithCache(priceSequence, 0, seqLen, contextHidden, contextTimes, cache, isTraining: false, dropoutRng: null);
        }

        public float[,] ForwardPriceDecoderWithCache(float[,] priceSequence, int rowStart, int rowCount, float[,] contextHidden, float[] contextTimes, PriceTacForwardCache cache, bool isTraining = true, Random dropoutRng = null)
        {
            if (priceSequence == null)
                throw new ArgumentNullException(nameof(priceSequence));
            if (cache == null)
                throw new ArgumentNullException(nameof(cache));
            if (rowStart < 0 || rowCount <= 0 || rowStart + rowCount > priceSequence.GetLength(0))
                throw new ArgumentOutOfRangeException($"Invalid price row slice: start={rowStart}, count={rowCount}.");

            int seqLen = rowCount;
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

            var emb = EmbedPriceSequence(priceSequence, rowStart, seqLen);
            cache.PriceEmbedded = emb;
            cache.PriceContinuousInput = priceSequence;
            cache.PriceContinuousInputRowStart = rowStart;
            cache.PriceContinuousInputRowCount = seqLen;

            bool[,] selfMask = _config.Price.UseDecoderOnly ? CreateCausalMask(seqLen) : null;
            var x = emb;

            for (int layer = 0; layer < _config.Price.NumLayers; layer++)
            {
                var block = PriceBlocks[layer];
                var bc = cache.PriceBlockCaches[layer];
                bc.BlockInput = x;

                var (selfQ, selfK, selfV) = _accel.ProjectQKV(
                    x,
                    block.SelfAttention.WQ, block.SelfAttention.BiasQ,
                    block.SelfAttention.WK, block.SelfAttention.BiasK,
                    block.SelfAttention.WV, block.SelfAttention.BiasV);

                _rotaryPositionEmbedding.ApplyInPlace(selfQ, selfK, nh);
                bc.SelfQ = selfQ;
                bc.SelfK = selfK;
                bc.SelfV = selfV;

                var selfAttnOutput = _accel.MultiHeadAttentionForward(selfQ, selfK, selfV, nh, scale, selfMask);
                bc.SelfAttnOutput = selfAttnOutput;

                var selfProjected = ComputeProjection(selfAttnOutput, block.SelfAttention.WO, block.SelfAttention.BiasO);
                var (normedSelf, selfMean, selfVar, selfNorm, selfResidual) = _accel.ResidualLayerNormForward(x, selfProjected, block.LNSelfGamma, block.LNSelfBeta);
                bc.SelfResidualInput = selfResidual;
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

                    if (contextTimes != null)
                    {
                        timeDiffs = _accel.ComputeTimeDiffMatrix(seqLen, contextTimes);
                        keyTimes = contextTimes;

                        if (_config.Global.BypassDecay && cache.NumGlobalContext > 0)
                        {
                            int bypass = Math.Min(cache.NumGlobalContext, contextTimes.Length);
                            _accel.ZeroMatrixColumns(timeDiffs, bypass);
                        }

                        bc.TimeDiffs = timeDiffs;
                        bc.KeyTimesFromRef = keyTimes;
                    }

                    var crossQ = ComputeProjection(normedSelf, block.CrossAttention.WQ, block.CrossAttention.BiasQ);
                    var (crossK, crossV) = _accel.ProjectKV(
                        contextHidden,
                        block.CrossAttention.WK,
                        block.CrossAttention.BiasK,
                        block.CrossAttention.WV,
                        block.CrossAttention.BiasV);

                    bc.CrossQ = crossQ;
                    bc.CrossK = crossK;
                    bc.CrossV = crossV;

                    var crossAttnOutput = _accel.ContentAwareCrossAttentionWithCache(
                        crossQ,
                        crossK,
                        crossV,
                        timeDiffs,
                        keyTimes,
                        normedSelf,
                        contextHidden,
                        block,
                        bc,
                        _config.Price.EmbeddingDim,
                        _config.Price.NumHeads,
                        enableDecayBias: _config.Decay.Enabled,
                        isTraining: isTraining,
                        dropoutRng: dropoutRng);

                    bc.CrossAttnOutput = crossAttnOutput;

                    var crossProjected = ComputeProjection(crossAttnOutput, block.CrossAttention.WO, block.CrossAttention.BiasO);
                    var (ncr, crossMean, crossVar, crossNorm, crossResidual) = _accel.ResidualLayerNormForward(normedSelf, crossProjected, block.LnCrossGamma, block.LnCrossBeta);
                    bc.CrossResidualInput = crossResidual;
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

                var ffnOutput = _accel.FFNForwardBatch(normedCross, seqLen, ed, block.FeedForwardNetwork.ForwardPassOnly);
                bc.FFNInputRows = null;
                bc.FFNOutput = ffnOutput;

                var (normedFfn, ffnMean, ffnVar, ffnNorm, ffnResidual) = _accel.ResidualLayerNormForward(normedCross, ffnOutput, block.LNFFNGamma, block.LNFFNBeta);
                bc.FFNResidualInput = ffnResidual;
                bc.LNFFNCache.Input = ffnResidual;
                bc.LNFFNCache.Mean = ffnMean;
                bc.LNFFNCache.Variance = ffnVar;
                bc.LNFFNCache.Normalized = ffnNorm;
                x = normedFfn;
            }

            return x;
        }

        internal (float[,] reg, float[,] range, float[,] quality, float[,] dir, float[,] midDir, float[,] conf) ForwardWithPriceContextAndCache(PriceTacInput input, float[,] priceCtxHidden, float[] priceCtxTimes, PriceTacForwardCache cache, bool isTraining = true, Random dropoutRng = null)
        {
            if (input == null)
                throw new ArgumentNullException(nameof(input));
            if (input.PriceSequence == null)
                throw new ArgumentNullException(nameof(input.PriceSequence));

            return ForwardWithPriceContextSliceAndCache(input, 0, input.PriceSequence.GetLength(0), priceCtxHidden, priceCtxTimes, cache, isTraining, dropoutRng);
        }

        internal (float[,] reg, float[,] range, float[,] quality, float[,] dir, float[,] midDir, float[,] conf) ForwardWithPriceContextSliceAndCache(PriceTacInput input, int rowStart, int rowCount, float[,] priceCtxHidden, float[] priceCtxTimes, PriceTacForwardCache cache, bool isTraining = true, Random dropoutRng = null)
        {
            if (input == null)
                throw new ArgumentNullException(nameof(input));
            if (input.PriceSequence == null)
                throw new ArgumentNullException(nameof(input.PriceSequence));
            if (cache == null)
                throw new ArgumentNullException(nameof(cache));
            if (rowStart < 0 || rowCount <= 0 || rowStart + rowCount > input.PriceSequence.GetLength(0))
                throw new ArgumentOutOfRangeException($"Invalid price row slice: start={rowStart}, count={rowCount}.");

            cache.Reset();

            int ed = _config.Price.EmbeddingDim;

            float[] globalToken = null;
            if (_config.Global.GlobalFeatureDim > 0 && input.GlobalFeatures != null)
            {
                globalToken = EmbedGlobalFeatures(input.GlobalFeatures);
                cache.GlobalTokenEmbedded = globalToken;
                cache.GlobalRawInput = input.GlobalFeatures;
            }

            int numPriceCtx = priceCtxHidden != null ? priceCtxHidden.GetLength(0) : 0;

            if (priceCtxHidden != null && priceCtxHidden.GetLength(1) != ed)
                throw new ArgumentException("priceCtxHidden embedding dimension does not match Price.EmbeddingDim.", nameof(priceCtxHidden));
            if (priceCtxTimes != null && priceCtxTimes.Length != numPriceCtx)
                throw new ArgumentException("priceCtxTimes length must match priceCtxHidden row count.", nameof(priceCtxTimes));

            BuildContext(priceCtxHidden, priceCtxTimes, globalToken, out var combinedHidden, out var combinedTimes, out int numGlobal, out numPriceCtx);

            cache.NumGlobalContext = numGlobal;
            cache.NumPriceContext = numPriceCtx;
            cache.PriceContextHidden = priceCtxHidden;
            cache.ContextHidden = combinedHidden;
            cache.ContextTimes = combinedTimes;

            var priceHidden = ForwardPriceDecoderWithCache(input.PriceSequence, rowStart, rowCount, combinedHidden, combinedTimes, cache, isTraining, dropoutRng);
            cache.PriceFinalHidden = priceHidden;
            return ProjectToOutputs(priceHidden, cache);
        }

        public float[,] EncodePriceHistory(float[,] histPrices)
        {
            if (histPrices == null)
                throw new ArgumentNullException(nameof(histPrices));

            return EncodePriceHistory(histPrices, 0, histPrices.GetLength(0));
        }

        public float[,] EncodePriceHistory(float[,] histPrices, int rowStart, int rowCount)
        {
            if (histPrices == null)
                throw new ArgumentNullException(nameof(histPrices));
            if (rowStart < 0 || rowCount <= 0 || rowStart + rowCount > histPrices.GetLength(0))
                throw new ArgumentOutOfRangeException($"Invalid history row slice: start={rowStart}, count={rowCount}.");

            var emb = EmbedPriceSequence(histPrices, rowStart, rowCount);
            bool[,] mask = _config.Price.UseDecoderOnly ? CreateCausalMask(rowCount) : null;
            var x = emb;

            foreach (var block in PriceBlocks)
            {
                block.SetContext(null);
                block.SetTimeData(null, null);
                block.SetTraining(false);
                x = block.Forward(x, mask);
            }

            return x;
        }

        internal (float[,] regression, float[,] range, float[,] quality, float[,] direction, float[,] midDirection, float[,] confidence) ProjectToOutputs(float[,] hidden)
        {
            return ProjectToOutputs(hidden, null);
        }

        internal (float[,] regression, float[,] range, float[,] quality, float[,] direction, float[,] midDirection, float[,] confidence) ProjectToOutputs(float[,] hidden, PriceTacForwardCache cache)
        {
            var result = _accel.ProjectMmtacOutputHeads(
                hidden,
                RegressionProjection,
                RegressionBias,
                RangeProjection,
                RangeBias,
                QualityProjection,
                QualityBias,
                DirectionProjection,
                DirectionBias,
                MidDirectionProjection,
                MidDirectionBias,
                ConfidenceProjection,
                ConfidenceBias,
                _config.Output.UseConfidenceHead);

            if (cache != null)
            {
                cache.RegressionLogits = result.regressionLogits;
                cache.RangeLogits = result.rangeLogits;
                cache.QualityLogits = result.qualityLogits;
            }

            return (result.regression, result.range, result.quality, result.direction, result.midDirection, result.confidence);
        }

        #region Memory management

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

            var byTime = PriceMemory.OrderByDescending(e => e.AbsoluteTimestamp).ToList();
            var kept = byTime
                .Take(reserve)
                .Concat(byTime.Skip(reserve)
                    .OrderByDescending(e => e.QueryCount >= PruningConfig.MinQueryCountForPruning ? e.AttentionScore : float.MinValue)
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
                .Select((entry, index) => new IndexedPriceMemoryEntry { Entry = entry, Index = index })
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

            foreach (var item in indexed.OrderByDescending(x => x.Index).Take(protectedCount).OrderBy(x => x.Index))
                Keep(item);

            int reserveTarget = protectedCount;
            if (PruningConfig.NewEntryReserveFraction > 0f)
            {
                int fractionalReserve = Math.Max(1, (int)MathF.Ceiling(maxSize * PruningConfig.NewEntryReserveFraction));
                fractionalReserve = Math.Clamp(fractionalReserve, 0, maxSize);
                reserveTarget = Math.Max(reserveTarget, fractionalReserve);
            }

            reserveTarget = Math.Clamp(reserveTarget, 0, maxSize);

            var newestFirst = indexed.OrderByDescending(x => x.Entry.AbsoluteTimestamp).ThenByDescending(x => x.Index).ToList();

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
                    .OrderByDescending(x => x.Entry.QueryCount >= PruningConfig.MinQueryCountForPruning ? x.Entry.AttentionScore : float.MinValue)
                    .ThenByDescending(x => x.Entry.AbsoluteTimestamp)
                    .ThenByDescending(x => x.Index))
                {
                    if (kept.Count >= maxSize)
                        break;
                    Keep(item);
                }
            }

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

        private void UpdateMemoryAttentionScores(float[,] priceHidden, float[,] ctxH, int total, int globalOffset, int storedPriceMemoryCount)
        {
            if (ctxH == null || total == 0)
                return;

            int ed = _config.Price.EmbeddingDim;
            int last = priceHidden.GetLength(0) - 1;
            float alpha = PruningConfig.AttentionScoreAlpha;
            float scale = 1.0f / MathF.Sqrt(ed);

            var scores = _accel.ComputeMemoryAttentionScores(priceHidden, last, ctxH, total, scale);
            scores = _accel.SoftmaxVector(scores);

            int storedPriceOffset = globalOffset;
            for (int i = 0; i < storedPriceMemoryCount && i < PriceMemory.Count; i++)
            {
                int ctxIdx = storedPriceOffset + i;
                if (ctxIdx >= total)
                    break;

                var e = PriceMemory[i];
                e.AttentionScore = alpha * scores[ctxIdx] + (1f - alpha) * e.AttentionScore;
                e.QueryCount++;
            }
        }

        #endregion

        internal float[,] ForwardPriceDecoderForMemoryBuild(float[,] priceSequence, float[,] contextHidden, float[] contextTimes, int globalBypassCount = 0)
        {
            return ForwardPriceDecoder(priceSequence, contextHidden, contextTimes, globalBypassCount);
        }

        private float[,] EmbedPriceSequence(float[,] ps, int rowStart, int rowCount)
        {
            if (ps == null)
                throw new ArgumentNullException(nameof(ps));

            int rows = ps.GetLength(0);
            int cols = ps.GetLength(1);

            if (cols != _config.Price.InputFeatureDim)
                throw new ArgumentException($"PriceSequence feature dimension ({cols}) must equal Price.InputFeatureDim ({_config.Price.InputFeatureDim}).", nameof(ps));
            if (rowStart < 0 || rowCount <= 0 || rowStart + rowCount > rows)
                throw new ArgumentOutOfRangeException(nameof(rowStart), $"Invalid price row slice: start={rowStart}, count={rowCount}, rows={rows}.");

            return _accel.BatchDotProductAddBias(PriceInputProjection, ps, rowStart, rowCount, PriceInputProjectionBias);
        }

        internal float[,] ComputeProjection(float[,] input, float[,] w, float[] b)
        {
            return _accel.BatchDotProductAddBias(w, input, b);
        }

        private bool[,] CreateCausalMask(int sl)
        {
            if (sl <= 0)
                return _accel.CreateCausalMask(sl);

            return _causalMaskCache.GetOrAdd(sl, length => _accel.CreateCausalMask(length));
        }

        protected float SampleGaussian()
        {
            float u1 = 1f - _random.NextSingle();
            float u2 = 1f - _random.NextSingle();
            return MathF.Sqrt(-2f * MathF.Log(u1)) * MathF.Cos(2f * MathF.PI * u2);
        }

        public static double WindowStartTimestampFromWindowEnd(double windowEndAbsoluteTimestamp, int sequenceLength, double timeUnitsPerPosition)
        {
            if (sequenceLength <= 0)
                throw new ArgumentOutOfRangeException(nameof(sequenceLength), "sequenceLength must be positive.");
            if (timeUnitsPerPosition == 0.0)
                throw new ArgumentOutOfRangeException(nameof(timeUnitsPerPosition), "Must be non-zero.");

            return windowEndAbsoluteTimestamp - Math.Max(0, sequenceLength - 1) * timeUnitsPerPosition;
        }

        public PriceTacPrediction PredictWithMemoryAtWindowEnd(PriceTacInput input, double windowEndAbsoluteTimestamp, double timeUnitsPerPosition = 1.0, int maxPriceMemorySize = 200)
        {
            if (input == null)
                throw new ArgumentNullException(nameof(input));
            if (input.PriceSequence == null)
                throw new ArgumentNullException(nameof(input.PriceSequence));

            double windowStartTimestamp = WindowStartTimestampFromWindowEnd(windowEndAbsoluteTimestamp, input.PriceSequence.GetLength(0), timeUnitsPerPosition);
            return PredictWithMemory(input, windowStartTimestamp, timeUnitsPerPosition, maxPriceMemorySize);
        }

        private float[,] InitWeights(int rows, int cols, float std)
        {
            var w = new float[rows, cols];
            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < cols; j++)
                    w[i, j] = SampleGaussian() * std;
            }
            return w;
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
            {
                PriceBlocks[i] = new TacamtBlock(
                    ed,
                    _config.Price.NumHeads,
                    _config.Price.FeedForwardDim,
                    _config.Runtime.FFNActivationType,
                    _accel,
                    _random,
                    l2Lambda: _config.Reg.L2RegulationLamda,
                    decayProjectionDim: _config.Decay.ProjectionDim,
                    decayHiddenDim: _config.Decay.HiddenDim,
                    decayMemAttnDropout: _config.Decay.MemAttentionDropout,
                    decayMLPDropout: _config.Decay.MlpDropout,
                    decayWeightDecay: _config.Decay.WeightDecay,
                    decayTimeBases: _config.Decay.TimeEncodingBases,
                    accelerationType: _config.Runtime.AccelerationType,
                    accelerationDeviceId: _config.Runtime.AccelerationDeviceId);
            }
        }

        private void InitGlobalProjection()
        {
            int gd = _config.Global.GlobalFeatureDim;
            if (gd <= 0)
                return;

            int ed = _config.Price.EmbeddingDim;
            float std = MathF.Sqrt(2.0f / (gd + ed));
            GlobalFeatureProjection = InitWeights(ed, gd, std);
            GlobalFeatureBias = new float[ed];
        }

        private void InitContextTypeEmbedding()
        {
            int ed = _config.Price.EmbeddingDim;
            float typeStd = MathF.Sqrt(2.0f / ed) * 0.1f;
            ContextTypeEmbedding = new float[PriceTacConfig.ContextTypeCount, ed];
            for (int t = 0; t < PriceTacConfig.ContextTypeCount; t++)
            {
                for (int d = 0; d < ed; d++)
                    ContextTypeEmbedding[t, d] = SampleGaussian() * typeStd;
            }
        }

        private void InitOutputHeads()
        {
            int ed = _config.Price.EmbeddingDim;
            int rDim = PriceTacOutputConfig.RegressionOutputCount;
            float regStd = MathF.Sqrt(2.0f / (ed + rDim));
            float clsStd = MathF.Sqrt(2.0f / (ed + 1));

            RegressionProjection = InitWeights(rDim, ed, regStd);
            RegressionBias = new float[rDim];
            RangeProjection = InitWeights(1, ed, clsStd);
            RangeBias = new float[1];
            QualityProjection = InitWeights(1, ed, clsStd);
            QualityBias = new float[1];
            DirectionProjection = InitWeights(1, ed, clsStd);
            DirectionBias = new float[1];
            MidDirectionProjection = InitWeights(1, ed, clsStd);
            MidDirectionBias = new float[1];

            if (_config.Output.UseConfidenceHead)
            {
                ConfidenceProjection = InitWeights(1, ed, clsStd);
                ConfidenceBias = new float[1];
            }
        }

        #region Save/Load

        public void Save(string dir)
        {
            Directory.CreateDirectory(dir);

            var cd = new Dictionary<string, object>
            {
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

            File.WriteAllText(Path.Combine(dir, "config.json"), JsonSerializer.Serialize(cd, new JsonSerializerOptions { WriteIndented = true }));

            using (var s = new FileStream(Path.Combine(dir, "weights.bin"), FileMode.Create))
            using (var w = new BinaryWriter(s))
            {
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

            for (int i = 0; i < _config.Price.NumLayers; i++)
                PriceBlocks[i].FeedForwardNetwork.Save(Path.Combine(dir, $"price_ffn_{i}"));

            SaveMemory(dir);
        }

        public static PriceTacModel Load(string dir, AccelerationType accelerationType, int accellerationDeviceIndex)
        {
            var d = JsonSerializer.Deserialize<Dictionary<string, JsonElement>>(File.ReadAllText(Path.Combine(dir, "config.json")));
            if (d == null)
                throw new InvalidOperationException("config.json could not be deserialized.");

            var cfg = new PriceTacConfig
            {
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
                Global = new PriceTacGlobalConfig
                {
                    GlobalFeatureDim = d.ContainsKey("GlobalFeatureDim") ? d["GlobalFeatureDim"].GetInt32() : 0,
                    BypassDecay = d.ContainsKey("GlobalBypassDecay") && d["GlobalBypassDecay"].GetBoolean()
                },
                Output = new PriceTacOutputConfig
                {
                    UseConfidenceHead = d["UseConfidenceHead"].GetBoolean(),
                    DirectionLossWeight = d.ContainsKey("DirectionLossWeight") ? d["DirectionLossWeight"].GetSingle() : 1f,
                    MidDirectionLossWeight = d.ContainsKey("MidDirectionLossWeight") ? d["MidDirectionLossWeight"].GetSingle() : 0.5f,
                    RangeLossWeight = d.ContainsKey("RangeLossWeight") ? d["RangeLossWeight"].GetSingle() : 1f,
                    QualityLossWeight = d.ContainsKey("QualityLossWeight") ? d["QualityLossWeight"].GetSingle() : 1f,
                    CloseDirectionConsistencyWeight = d.ContainsKey("CloseDirectionConsistencyWeight") ? d["CloseDirectionConsistencyWeight"].GetSingle() : 1.0f,
                    CloseDirectionConsistencyMargin = d.ContainsKey("CloseDirectionConsistencyMargin") ? d["CloseDirectionConsistencyMargin"].GetSingle() : 0.02f
                },
                Runtime = new RuntimeConfig
                {
                    FFNActivationType = (ActivationType)d["FFNActivationType"].GetInt32(),
                    AccelerationType = accelerationType,
                    AccelerationDeviceId = accellerationDeviceIndex
                },
                Reg = new RegularizationConfig
                {
                    L2RegulationLamda = d["L2RegulationLamda"].GetSingle(),
                    GradientClippingThreshold = d["GradientClipThreshold"].GetSingle()
                },
                Decay = new DecayNetworkConfig
                {
                    Enabled = d.ContainsKey("DecayEnabled") ? d["DecayEnabled"].GetBoolean() : true,
                    ProjectionDim = d.ContainsKey("DecayProjectionDim") ? d["DecayProjectionDim"].GetInt32() : 16,
                    HiddenDim = d.ContainsKey("DecayHiddenDim") ? d["DecayHiddenDim"].GetInt32() : 32,
                    TimeEncodingBases = d.ContainsKey("DecayTimeEncodingBases") ? d["DecayTimeEncodingBases"].GetInt32() : 16,
                    MemAttentionDropout = d.ContainsKey("DecayMemAttnDropout") ? d["DecayMemAttnDropout"].GetSingle() : 0.2f,
                    MlpDropout = d.ContainsKey("DecayMlpDropout") ? d["DecayMlpDropout"].GetSingle() : 0.2f,
                    WeightDecay = d.ContainsKey("DecayWeightDecay") ? d["DecayWeightDecay"].GetSingle() : 0f
                },
                PriceContext = new PriceContextConfig
                {
                    Enabled = !d.ContainsKey("PriceContextEnabled") || d["PriceContextEnabled"].GetBoolean(),
                    MinHistoryLength = d.ContainsKey("PriceContextMinHistory") ? d["PriceContextMinHistory"].GetInt32() : 5,
                    MinCurrentLength = d.ContainsKey("PriceContextMinCurrent") ? d["PriceContextMinCurrent"].GetInt32() : 5
                },
                Pruning = new MemoryPruningConfig
                {
                    AttentionScoreAlpha = d.ContainsKey("PruningAlpha") ? d["PruningAlpha"].GetSingle() : new MemoryPruningConfig().AttentionScoreAlpha,
                    MinQueryCountForPruning = d.ContainsKey("PruningMinQueryCount") ? d["PruningMinQueryCount"].GetInt32() : new MemoryPruningConfig().MinQueryCountForPruning,
                    NewEntryReserveFraction = d.ContainsKey("PruningReserveFraction") ? d["PruningReserveFraction"].GetSingle() : new MemoryPruningConfig().NewEntryReserveFraction,
                    UseAttentionBasedPruning = d.ContainsKey("PruningUseAttentionBased") ? d["PruningUseAttentionBased"].GetBoolean() : new MemoryPruningConfig().UseAttentionBasedPruning
                }
            };

            cfg.Validate();
            var m = new PriceTacModel(cfg);
            m.SetAccellerator(accelerationType, accellerationDeviceIndex);

            using (var s = new FileStream(Path.Combine(dir, "weights.bin"), FileMode.Open))
            using (var r = new BinaryReader(s))
            {
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

            for (int i = 0; i < cfg.Price.NumLayers; i++)
                m.PriceBlocks[i].FeedForwardNetwork = NeuralNetwork.Load(Path.Combine(dir, $"price_ffn_{i}"), cfg.Runtime.AccelerationType);

            m.LoadMemory(dir);
            return m;
        }

        private void SaveMemory(string dir)
        {
            int ed = _config.Price.EmbeddingDim;
            using var ps = new FileStream(Path.Combine(dir, "price_memory.bin"), FileMode.Create);
            using var pw = new BinaryWriter(ps);
            pw.Write(LastPriceTimestamp);
            pw.Write(PriceMemory.Count);
            foreach (var e in PriceMemory)
            {
                pw.Write(e.AbsoluteTimestamp);
                pw.Write(e.AttentionScore);
                pw.Write(e.QueryCount);
                for (int d = 0; d < ed; d++)
                    pw.Write(e.HiddenState[d]);
            }
        }

        private void LoadMemory(string dir)
        {
            int ed = _config.Price.EmbeddingDim;
            var pp = Path.Combine(dir, "price_memory.bin");
            if (!File.Exists(pp))
                return;

            using var s = new FileStream(pp, FileMode.Open);
            using var r = new BinaryReader(s);

            // PriceTAC writes LastPriceTimestamp first.  If loading a very old file
            // that omitted it, callers should re-run inference to rebuild memory.
            LastPriceTimestamp = r.ReadDouble();
            int c = r.ReadInt32();
            PriceMemory = new List<PriceMemoryEntry>(c);

            for (int i = 0; i < c; i++)
            {
                var e = new PriceMemoryEntry
                {
                    AbsoluteTimestamp = r.ReadDouble(),
                    AttentionScore = r.ReadSingle(),
                    QueryCount = r.ReadInt32(),
                    HiddenState = new float[ed]
                };

                for (int d = 0; d < ed; d++)
                    e.HiddenState[d] = r.ReadSingle();

                PriceMemory.Add(e);
            }
        }

        static void WM(BinaryWriter w, float[,] m)
        {
            int r = m.GetLength(0), c = m.GetLength(1);
            w.Write(r);
            w.Write(c);
            for (int i = 0; i < r; i++)
                for (int j = 0; j < c; j++)
                    w.Write(m[i, j]);
        }

        static void WV(BinaryWriter w, float[] v)
        {
            w.Write(v.Length);
            for (int i = 0; i < v.Length; i++)
                w.Write(v[i]);
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
                for (int j = 0; j < cols; j++)
                    m[i, j] = r.ReadSingle();
        }

        static void RV(BinaryReader r, float[] v)
        {
            int l = r.ReadInt32();
            for (int i = 0; i < l; i++)
                v[i] = r.ReadSingle();
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

        #endregion

        public void SetAccellerator(AccelerationType accelerationType, int accellerationDeviceIndex)
        {
            _accel = AccelerationFactory.Create(accelerationType, accellerationDeviceIndex);
        }
    }
}
