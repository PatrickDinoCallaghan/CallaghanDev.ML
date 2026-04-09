using CallaghanDev.ML.AccelerationManagers;
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
    ///   Regression  : High, Low, Close          — linear,   MSE
    ///   Range       : High - Low                — softplus, MSE
    ///   Quality     : reliability score [0,1]   — sigmoid,  MSE
    ///   Direction   : P(close_t+1 > close_t)    — sigmoid,  BCE
    ///   MidDir      : P(mid-window up-move)      — sigmoid,  BCE
    ///   Confidence  : optional reliability       — sigmoid,  BCE
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
            config.Validate();
            _config = config;
            _random = random ?? new Random();
            _accel = CreateAccel(config.Runtime);

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
            if (tokenizer.VocabSize > _config.Text.VocabSize)
                throw new ArgumentException($"Tokenizer vocab size ({tokenizer.VocabSize}) exceeds config TextVocabSize ({_config.Text.VocabSize}).");
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
            float[,] sh = null; float[] st = null;
            if (input.NewsStories != null && input.NewsStories.Length > 0)
                (sh, st) = EncodeStories(input.NewsStories);

            float[,] contextHidden; float[] contextTimes;
            BuildContext(sh, st, input.GlobalFeatures, null, null, out contextHidden, out contextTimes);

            var priceHidden = ForwardPriceDecoder(input.PriceSequence, contextHidden, contextTimes);
            return ProjectToOutputs(priceHidden);
        }

        /// <summary>Convenience overload — returns only the last-timestep prediction.</summary>
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
            int sl = input.PriceSequence.GetLength(0);
            return ForwardWithCache(input, rowStart: 0, rowCount: sl, cache, isTraining, dropoutRng);
        }

        internal (float[,] regression, float[,] range, float[,] quality, float[,] direction, float[,] midDirection, float[,] confidence) ForwardWithCache(MultimodalInput input, int rowStart, int rowCount, MmtacForwardCache cache, bool isTraining = false, Random dropoutRng = null)
        {
            if (input == null)
            {
                throw new ArgumentNullException(nameof(input));
            }
            if (input.PriceSequence == null)
            {
                throw new ArgumentNullException("input.PriceSequence");
            }
            if (rowStart + rowCount > input.PriceSequence.GetLength(0))
            {
                throw new ArgumentException("rowStart + rowCount exceeds PriceSequence row count.");
            }


            float[,] sh = null; float[] st = null;

            if (input.NewsStories != null && input.NewsStories.Length > 0)
            {
                (sh, st) = EncodeStoriesWithCache(input.NewsStories, cache);
            }
            else
            {
                cache.TextFinalHidden = null;
                cache.TextTokenIds = null;
            }
            cache.StoryArrivalTimes = st;

            //  Embed global features 
            float[] globalToken = null;
            if (_config.Global.GlobalFeatureDim > 0 && input.GlobalFeatures != null)
            {
                globalToken = EmbedGlobalFeatures(input.GlobalFeatures);
                cache.GlobalTokenEmbedded = globalToken;
                cache.GlobalRawInput = input.GlobalFeatures;
            }

            //  Build combined context matrix 
            float[,] contextHidden; float[] contextTimes;
            BuildContext(sh, st, null, globalToken, cache, out contextHidden, out contextTimes);

            cache.TextFinalHidden = contextHidden;
            cache.StoryArrivalTimes = contextTimes;

            //  Price decoder 
            var ph = ForwardPriceDecoderWithCache(
                input.PriceSequence, rowStart, rowCount,
                contextHidden, contextTimes, cache, isTraining, dropoutRng);

            cache.PriceFinalHidden = ph;
            return ProjectToOutputs(ph);
        }

        // 
        // Persistent memory inference
        // 

        public ModelPrediction PredictWithMemory(MultimodalInput input, double currentAbsoluteTimestamp, double timeUnitsPerPosition = 1.0, int maxNewsMemorySize = 100, int maxPriceMemorySize = 200)
        {
            int embDim = _config.Price.EmbeddingDim;
            var ctxH = new List<float[]>();
            var ctxT = new List<float>();
            var ctxTypes = new List<int>();

            // Global token goes first (index 0), matching BuildContext ordering and the
            // globalOffset expected by UpdateMemoryAttentionScores.
            float[] globalToken = null;
            if (_config.Global.GlobalFeatureDim > 0 && input.GlobalFeatures != null)
            {
                globalToken = EmbedGlobalFeatures(input.GlobalFeatures);
                ctxH.Add(globalToken);
                ctxT.Add(0f);
                ctxTypes.Add(2);
            }

            // News memory — pass the raw hidden state without any pre-scaling.
            // Temporal decay is handled entirely by the ContentAwareDecayNetwork inside
            // TacamtBlock.CrossAttentionForward, which receives the time-diff matrix
            // built from ctxT by ForwardPriceDecoder. Pre-multiplying the vector here
            // was redundant and prevented the trained decay network from expressing its
            // learned per-content decay behaviour.
            foreach (var e in NewsMemory)
            {
                float relTime = -(float)((currentAbsoluteTimestamp - e.AbsoluteTimestamp) / timeUnitsPerPosition);
                var v = new float[embDim];
                for (int d = 0; d < embDim; d++)
                    v[d] = e.HiddenState[d];
                ctxH.Add(v);
                ctxT.Add(relTime);
                ctxTypes.Add(0);
            }

            // Live news stories arriving with this call.
            float[,] newSH = null;
            if (input.NewsStories != null && input.NewsStories.Length > 0)
            {
                float[] nst;
                (newSH, nst) = EncodeStories(input.NewsStories);
                for (int i = 0; i < nst.Length; i++)
                {
                    var v = new float[embDim];
                    for (int d = 0; d < embDim; d++)
                        v[d] = newSH[i, d];
                    ctxH.Add(v);
                    ctxT.Add(nst[i]);
                    ctxTypes.Add(0);
                }
            }

            // Price memory — raw hidden states, same reasoning as news memory above.
            foreach (var e in PriceMemory)
            {
                float relTime = -(float)((currentAbsoluteTimestamp - e.AbsoluteTimestamp) / timeUnitsPerPosition);
                var v = new float[embDim];
                for (int d = 0; d < embDim; d++)
                    v[d] = e.HiddenState[d];
                ctxH.Add(v);
                ctxT.Add(relTime);
                ctxTypes.Add(1);
            }

            float[,] cH = null;
            float[] cT = null;
            int totalCtx = ctxH.Count;

            if (totalCtx > 0)
            {
                cH = new float[totalCtx, embDim];
                cT = new float[totalCtx];
                for (int i = 0; i < totalCtx; i++)
                {
                    int cd = Math.Min(embDim, ctxH[i].Length);
                    for (int d = 0; d < cd; d++)
                        cH[i, d] = ctxH[i][d];
                    cT[i] = ctxT[i];
                }
                // Bake context-type offsets into the rows (news=0, price=1, global=2).
                ApplyContextTypeEmbeddings(cH, ctxTypes.ToArray());

                // Mirror the BypassDecay logic in ForwardPriceDecoderWithCache: zero the
                // time-diff columns for the global token so the decay network ignores it.
                // ForwardPriceDecoder recomputes td internally, so we set it on the block
                // indirectly — the bypass is already handled inside ForwardPriceDecoder.
            }

            var priceHidden = ForwardPriceDecoder(input.PriceSequence, cH, cT);
            var (reg, range, quality, dir, midDir, conf) = ProjectToOutputs(priceHidden);
            int last = reg.GetLength(0) - 1;

            UpdateMemoryAttentionScores(priceHidden, cH, totalCtx, globalToken != null ? 1 : 0);

            // Commit live news stories to memory using their raw (unscaled) hidden states.
            if (newSH != null)
            {
                for (int i = 0; i < input.NewsStories.Length; i++)
                {
                    var hv = new float[embDim];
                    for (int d = 0; d < embDim; d++)
                        hv[d] = newSH[i, d];
                    NewsMemory.Add(new NewsMemoryEntry
                    {
                        HiddenState = hv,
                        AbsoluteTimestamp = currentAbsoluteTimestamp + input.NewsStories[i].ArrivalTime * timeUnitsPerPosition
                    });
                }
            }

            int psl = input.PriceSequence.GetLength(0);
            for (int t = 0; t < psl; t++)
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

            LastPriceTimestamp = currentAbsoluteTimestamp + Math.Max(0, psl - 1) * timeUnitsPerPosition;

            PruneNewsMemory(maxNewsMemorySize);
            PricePruneMemory(maxPriceMemorySize);

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
        public void ClearNewsMemory() => NewsMemory.Clear();
        public void ClearPriceMemory() => PriceMemory.Clear();
        public void ClearAllMemory() { ClearNewsMemory(); ClearPriceMemory(); }

        // 
        // Global feature embedding
        // 

        internal float[] EmbedGlobalFeatures(float[] globalFeatures)
        {
            if (globalFeatures == null) throw new ArgumentNullException(nameof(globalFeatures));
            int gd = _config.Global.GlobalFeatureDim;
            int ed = _config.Price.EmbeddingDim;
            if (globalFeatures.Length < gd)
                throw new ArgumentException($"globalFeatures.Length ({globalFeatures.Length}) < GlobalFeatureDim ({gd}).");

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

        internal void BuildContext(
            float[,] newsHidden, float[] newsTimes,
            float[] globalFeatures,
            float[] preEmbeddedGlobal,
            MmtacForwardCache cache,
            out float[,] contextHidden,
            out float[] contextTimes)
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

        private float[,] ForwardPriceDecoder(float[,] priceSequence, float[,] contextHidden, float[] contextTimes)
        {
            int sl = priceSequence.GetLength(0);
            var emb = EmbedPriceSequence(priceSequence, 0, sl);
            bool[,] mask = _config.Price.UseDecoderOnly ? CreateCausalMask(sl) : null;

            float[,] td = null; float[] ktr = null;
            if (contextHidden != null && contextTimes != null)
            {
                td = _accel.ComputeTimeDiffMatrix(sl, contextTimes);
                ktr = contextTimes;
                // Decay bypass: zero the global token column(s) so decay has no time signal for it.
                if (_config.Global.BypassDecay && _config.Global.GlobalFeatureDim > 0)
                    for (int qi = 0; qi < sl; qi++)
                        td[qi, 0] = 0f;
            }

            var x = emb;
            foreach (var block in PriceBlocks)
            {
                block.SetContext(contextHidden);
                block.SetTimeData(td, ktr);
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
            if (priceSequence == null) throw new ArgumentNullException(nameof(priceSequence));

            int sl = rowCount;
            int ed = _config.Price.EmbeddingDim;
            int nh = _config.Price.NumHeads;
            int hd = ed / nh;
            float scale = 1.0f / MathF.Sqrt(hd);

            var emb = EmbedPriceSequence(priceSequence, rowStart, sl);
            cache.PriceEmbedded = emb;
            cache.PriceContinuousInput = priceSequence;

            bool[,] selfMask = _config.Price.UseDecoderOnly ? CreateCausalMask(sl) : null;
            var x = emb;

            for (int layer = 0; layer < _config.Price.NumLayers; layer++)
            {
                var block = PriceBlocks[layer];
                var bc = cache.PriceBlockCaches[layer];
                bc.BlockInput = x;

                //  Self attention 
                var sQ = ComputeProjection(x, block.SelfAttention.WQ, block.SelfAttention.BiasQ);
                var sK = ComputeProjection(x, block.SelfAttention.WK, block.SelfAttention.BiasK);
                var sV = ComputeProjection(x, block.SelfAttention.WV, block.SelfAttention.BiasV);
                RotaryPositionEmbedding.ApplyInPlace(sQ, sK, nh);
                bc.SelfQ = sQ; bc.SelfK = sK; bc.SelfV = sV;

                var sao = _accel.MultiHeadAttentionForward(sQ, sK, sV, nh, scale, selfMask);
                bc.SelfAttnOutput = sao;

                var sp = ComputeProjection(sao, block.SelfAttention.WO, block.SelfAttention.BiasO);
                var sr = _accel.MatrixAdd(x, sp);
                bc.SelfResidualInput = sr;

                var (ns, sm, sv, sn) = _accel.LayerNormForward(sr, block.LNSelfGamma, block.LNSelfBeta);
                bc.LNSelfCache.Input = sr; bc.LNSelfCache.Mean = sm; bc.LNSelfCache.Variance = sv; bc.LNSelfCache.Normalized = sn;
                bc.NormedSelf = ns;

                //  Cross attention 
                float[,] nc;
                if (contextHidden != null)
                {
                    float[,] td = null; float[] ktr = null;
                    if (contextTimes != null)
                    {
                        td = _accel.ComputeTimeDiffMatrix(sl, contextTimes);
                        ktr = contextTimes;
                        if (_config.Global.BypassDecay && cache.NumGlobalContext > 0)
                            for (int qi = 0; qi < sl; qi++)
                                for (int gi = 0; gi < cache.NumGlobalContext; gi++)
                                    td[qi, gi] = 0f;
                        bc.TimeDiffs = td;
                        bc.KeyTimesFromRef = ktr;
                    }

                    var cQ = ComputeProjection(ns, block.CrossAttention.WQ, block.CrossAttention.BiasQ);
                    var cK = ComputeProjection(contextHidden, block.CrossAttention.WK, block.CrossAttention.BiasK);
                    var cV = ComputeProjection(contextHidden, block.CrossAttention.WV, block.CrossAttention.BiasV);
                    RotaryPositionEmbedding.ApplyInPlace(cQ, cK, nh);
                    bc.CrossQ = cQ; bc.CrossK = cK; bc.CrossV = cV;

                    var cao = ContentAwareCrossAttentionWithCache(
                        cQ, cK, cV, td, ktr, ns, contextHidden, block, bc,
                        isTraining, dropoutRng, cache.NumGlobalContext);

                    bc.CrossAttnOutput = cao;
                    var cp = ComputeProjection(cao, block.CrossAttention.WO, block.CrossAttention.BiasO);
                    var cr = _accel.MatrixAdd(ns, cp);
                    bc.CrossResidualInput = cr;

                    var (ncr, cm, cvr, cn) = _accel.LayerNormForward(cr, block.LnCrossGamma, block.LnCrossBeta);
                    bc.LNCrossCache.Input = cr; bc.LNCrossCache.Mean = cm; bc.LNCrossCache.Variance = cvr; bc.LNCrossCache.Normalized = cn;
                    bc.NormedCross = ncr;
                    nc = ncr;
                }
                else
                {
                    bc.CrossQ = bc.CrossK = bc.CrossV = bc.CrossAttnOutput = null;
                    bc.TimeDiffs = null; bc.DecayCache = null;
                    var (ncr, cm, cvr, cn) = _accel.LayerNormForward(ns, block.LnCrossGamma, block.LnCrossBeta);
                    bc.LNCrossCache.Input = ns; bc.LNCrossCache.Mean = cm; bc.LNCrossCache.Variance = cvr; bc.LNCrossCache.Normalized = cn;
                    bc.NormedCross = ncr;
                    nc = ncr;
                }

                //  FFN 
                var ffnIn = new float[sl][];
                for (int i = 0; i < sl; i++)
                {
                    var ir = new float[ed];
                    for (int j = 0; j < ed; j++) ir[j] = nc[i, j];
                    ffnIn[i] = ir;
                }
                var ffOut = _accel.FFNForwardBatch(nc, sl, ed, block.FeedForwardNetwork.ForwardPassOnly);
                bc.FFNInputRows = ffnIn; bc.FFNOutput = ffOut;

                var fr = _accel.MatrixAdd(nc, ffOut);
                bc.FFNResidualInput = fr;

                var (nf, fm, fv, fn) = _accel.LayerNormForward(fr, block.LNFFNGamma, block.LNFFNBeta);
                bc.LNFFNCache.Input = fr; bc.LNFFNCache.Mean = fm; bc.LNFFNCache.Variance = fv; bc.LNFFNCache.Normalized = fn;
                x = nf;
            }

            return x;
        }

        // 
        // Price-context training forward
        // 

        internal (float[,] reg, float[,] range, float[,] quality, float[,] dir, float[,] midDir, float[,] conf) ForwardWithPriceContextAndCache( MultimodalInput input, float[,] priceCtxHidden, float[] priceCtxTimes, MmtacForwardCache cache, bool isTraining = true, Random dropoutRng = null)
        {
            int ed = _config.Price.EmbeddingDim;

            float[,] newsHidden = null; float[] newsTimes = null; int numNews = 0;
            if (input.NewsStories != null && input.NewsStories.Length > 0)
            {
                (newsHidden, newsTimes) = EncodeStoriesWithCache(input.NewsStories, cache);
                numNews = newsHidden.GetLength(0);
            }
            else { cache.TextFinalHidden = null; cache.TextTokenIds = null; }

            float[] globalToken = null;
            if (_config.Global.GlobalFeatureDim > 0 && input.GlobalFeatures != null)
            {
                globalToken = EmbedGlobalFeatures(input.GlobalFeatures);
                cache.GlobalTokenEmbedded = globalToken;
                cache.GlobalRawInput = input.GlobalFeatures;
            }

            int numGlobal = globalToken != null ? 1 : 0;
            int numPriceCtx = priceCtxHidden != null ? priceCtxHidden.GetLength(0) : 0;
            int totalCtx = numGlobal + numNews + numPriceCtx;

            float[,] combinedHidden = null; float[] combinedTimes = null;
            if (totalCtx > 0)
            {
                combinedHidden = new float[totalCtx, ed];
                combinedTimes = new float[totalCtx];
                int row = 0;

                if (globalToken != null)
                {
                    for (int d = 0; d < ed; d++)
                        combinedHidden[row, d] = globalToken[d] + ContextTypeEmbedding[2, d];
                    combinedTimes[row] = 0f; row++;
                }
                for (int i = 0; i < numNews; i++)
                {
                    for (int d = 0; d < ed; d++)
                        combinedHidden[row, d] = newsHidden[i, d] + ContextTypeEmbedding[0, d];
                    combinedTimes[row] = newsTimes != null ? newsTimes[i] : 0f; row++;
                }
                for (int i = 0; i < numPriceCtx; i++)
                {
                    for (int d = 0; d < ed; d++)
                        combinedHidden[row, d] = priceCtxHidden[i, d] + ContextTypeEmbedding[1, d];
                    combinedTimes[row] = priceCtxTimes != null ? priceCtxTimes[i] : 0f; row++;
                }
            }

            cache.NumGlobalContext = numGlobal;
            cache.NumNewsContext = numNews;
            cache.NumPriceContext = numPriceCtx;
            cache.PriceContextHidden = priceCtxHidden;
            cache.TextFinalHidden = combinedHidden;
            cache.StoryArrivalTimes = combinedTimes;

            var priceHidden = ForwardPriceDecoderWithCache(
                input.PriceSequence, 0, input.PriceSequence.GetLength(0),
                combinedHidden, combinedTimes, cache, isTraining, dropoutRng);

            cache.PriceFinalHidden = priceHidden;
            return ProjectToOutputs(priceHidden);
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

            //  Regression: High, Low, Close — linear 
            var regression = _accel.ProjectOutputBatch(hidden, RegressionProjection, RegressionBias, sl, rDim);

            //  Range — softplus: log(1 + exp(x)) 
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

            //  Quality — sigmoid 
            var quality = new float[sl, 1];
            var qualityLogits = new float[sl];
            for (int t = 0; t < sl; t++)
            {
                float l = QualityBias[0];
                for (int k = 0; k < ed; k++) l += QualityProjection[0, k] * hidden[t, k];
                qualityLogits[t] = l;
                quality[t, 0] = Sigmoid(l);
            }

            //  Direction — sigmoid 
            var direction = new float[sl, 1];
            for (int t = 0; t < sl; t++)
            {
                float l = DirectionBias[0];
                for (int k = 0; k < ed; k++) l += DirectionProjection[0, k] * hidden[t, k];
                direction[t, 0] = Sigmoid(l);
            }

            //  MidWindowDirection — sigmoid 
            var midDirection = new float[sl, 1];
            for (int t = 0; t < sl; t++)
            {
                float l = MidDirectionBias[0];
                for (int k = 0; k < ed; k++) l += MidDirectionProjection[0, k] * hidden[t, k];
                midDirection[t, 0] = Sigmoid(l);
            }

            //  Confidence — sigmoid (optional) 
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
            if (NewsMemory.Count <= maxSize) return;
            if (!PruningConfig.UseAttentionBasedPruning)
            {
                NewsMemory = NewsMemory.OrderByDescending(e => e.AbsoluteTimestamp).Take(maxSize).ToList();
                return;
            }
           // int reserve = Math.Max(1, (int)(maxSize * PruningConfig.NewEntryReserveFraction));
            int reserve = PruningConfig.NewEntryReserveFraction <= 0f ? 0 : Math.Max(1, (int)(maxSize * PruningConfig.NewEntryReserveFraction));
            var byTime = NewsMemory.OrderByDescending(e => e.AbsoluteTimestamp).ToList();
            var kept = byTime.Take(reserve)
                             .Concat(byTime.Skip(reserve)
                                 .OrderByDescending(e => e.QueryCount >= PruningConfig.MinQueryCountForPruning
                                     ? e.AttentionScore : float.MinValue)
                                 .Take(maxSize - reserve))
                             .ToList();
            NewsMemory = kept;
        }

        internal void PricePruneMemory(int maxSize)
        {
            if (PriceMemory.Count <= maxSize) return;
            if (!PruningConfig.UseAttentionBasedPruning)
            {
                PriceMemory = PriceMemory.OrderByDescending(e => e.AbsoluteTimestamp).Take(maxSize).ToList();
                return;
            }

            int reserve = PruningConfig.NewEntryReserveFraction <= 0f ? 0 : Math.Max(1, (int)(maxSize * PruningConfig.NewEntryReserveFraction));
            var byTime = PriceMemory.OrderByDescending(e => e.AbsoluteTimestamp).ToList();
            var kept = byTime.Take(reserve)
                             .Concat(byTime.Skip(reserve)
                                 .OrderByDescending(e => e.QueryCount >= PruningConfig.MinQueryCountForPruning
                                     ? e.AttentionScore : float.MaxValue)
                                 .Take(maxSize - reserve))
                             .ToList();
            PriceMemory = kept;
        }

        private void UpdateMemoryAttentionScores(float[,] priceHidden, float[,] ctxH, int total, int globalOffset)
        {
            if (ctxH == null || total == 0) return;
            int ed = _config.Price.EmbeddingDim;
            int last = priceHidden.GetLength(0) - 1;
            float alpha = PruningConfig.AttentionScoreAlpha;
            float scale = 1.0f / MathF.Sqrt(ed);

            var scores = _accel.ComputeMemoryAttentionScores(priceHidden, last, ctxH, total, scale);
            float max = scores.Length > 0 ? scores.Max() : float.MinValue;
            float sum = 0;
            for (int s = 0; s < total; s++) { scores[s] = MathF.Exp(scores[s] - max); sum += scores[s]; }
            if (sum > 0) for (int s = 0; s < total; s++) scores[s] /= sum;

            int newsCount = NewsMemory?.Count ?? 0;
            int priceCount = PriceMemory?.Count ?? 0;
            int newsOffset = globalOffset;

            for (int i = 0; i < newsCount && (newsOffset + i) < total; i++)
            {
                var e = NewsMemory[i];
                e.AttentionScore = alpha * scores[newsOffset + i] + (1 - alpha) * e.AttentionScore;
                e.QueryCount++;
            }

            int priceOffset = globalOffset + newsCount;
            for (int i = 0; i < priceCount && (priceOffset + i) < total; i++)
            {
                var e = PriceMemory[i];
                e.AttentionScore = alpha * scores[priceOffset + i] + (1 - alpha) * e.AttentionScore;
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
            int ns = stories.Length;
            int ed = _config.Text.EmbeddingDim;
            var hidden = new float[ns, ed];
            var times = new float[ns];
            for (int i = 0; i < ns; i++)
            {
                var h = ForwardTextEncoder(stories[i].TokenIds);
                int tok = h.GetLength(0);
                for (int d = 0; d < ed; d++) { float s = 0f; for (int t = 0; t < tok; t++) s += h[t, d]; hidden[i, d] = s / tok; }
                times[i] = stories[i].ArrivalTime;
            }
            return (hidden, times);
        }

        internal (float[,] hidden, float[] times) EncodeStoriesForMemory(NewsStory[] stories)
            => EncodeStories(stories);

        private (float[,] hidden, float[] times) EncodeStoriesWithCache(NewsStory[] stories, MmtacForwardCache cache)
        {
            int ns = stories.Length;
            int ed = _config.Text.EmbeddingDim;
            var hidden = new float[ns, ed];
            var times = new float[ns];

            cache.StoryCaches = new List<MmtacForwardCache>();
            cache.StoryTokenCounts = new int[ns];

            for (int i = 0; i < ns; i++)
            {
                var sc = new MmtacForwardCache(_config.Text.NumLayers, _config.Price.NumLayers);
                var h = ForwardTextEncoderWithCache(stories[i].TokenIds, sc);
                int tok = h.GetLength(0);
                cache.StoryTokenCounts[i] = tok;
                for (int d = 0; d < ed; d++) { float s = 0f; for (int t = 0; t < tok; t++) s += h[t, d]; hidden[i, d] = s / tok; }
                times[i] = stories[i].ArrivalTime;
                cache.StoryCaches.Add(sc);
            }
            cache.TextFinalHidden = hidden;
            return (hidden, times);
        }

        private float[,] ForwardTextEncoder(int[] tokenIds)
        {
            int sl = tokenIds.Length;
            var emb = EmbedTextTokens(tokenIds, sl);
            bool[,] mask = _config.Text.UseDecoderOnly ? CreateCausalMask(sl) : null;
            var x = emb;
            foreach (var b in TextBlocks) x = b.Forward(x, mask);
            return x;
        }

        private float[,] ForwardTextEncoderWithCache(int[] tokenIds, MmtacForwardCache cache)
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
                var b = TextBlocks[layer];
                cache.TextLayerInputs.Add(x);

                var ac = cache.TextAttentionCaches[layer];
                ac.Input = x;

                var Q = ComputeProjection(x, b.Attention.WQ, b.Attention.BiasQ);
                var K = ComputeProjection(x, b.Attention.WK, b.Attention.BiasK);
                var V = ComputeProjection(x, b.Attention.WV, b.Attention.BiasV);
                RotaryPositionEmbedding.ApplyInPlace(Q, K, _config.Text.NumHeads);
                ac.Q = Q; ac.K = K; ac.V = V;

                float s = 1.0f / MathF.Sqrt(ed / _config.Text.NumHeads);
                var attnOut = _accel.MultiHeadAttentionForward(Q, K, V, _config.Text.NumHeads, s, mask);
                ac.AttentionOutput = attnOut;

                var projected = ComputeProjection(attnOut, b.Attention.WO, b.Attention.BiasO);
                var residual1 = _accel.MatrixAdd(x, projected);
                var ln1c = cache.TextLN1Caches[layer];
                var (normed1, m1, v1, n1) = _accel.LayerNormForward(residual1, b.LN1Gamma, b.LN1Beta);
                ln1c.Input = residual1; ln1c.Mean = m1; ln1c.Variance = v1; ln1c.Normalized = n1;

                cache.TextFFNInputs.Add(normed1);
                var ffnOut = _accel.FFNForwardBatch(normed1, sl, ed, b.FeedForwardNetwork.ForwardPassOnly);
                cache.TextFFNOutputs.Add(ffnOut);

                var residual2 = _accel.MatrixAdd(normed1, ffnOut);
                var ln2c = cache.TextLN2Caches[layer];
                var (normed2, m2, v2, n2) = _accel.LayerNormForward(residual2, b.LN2Gamma, b.LN2Beta);
                ln2c.Input = residual2; ln2c.Mean = m2; ln2c.Variance = v2; ln2c.Normalized = n2;

                x = normed2;
            }
            cache.TextFinalHidden = x;
            return x;
        }

        // 
        // Content-aware cross-attention wrapper
        // 

        private float[,] ContentAwareCrossAttentionWithCache(
            float[,] Q, float[,] K, float[,] V,
            float[,] timeDiffs, float[] keyTimesFromRef,
            float[,] queryEmbeddings, float[,] keyEmbeddings,
            TacamtBlock block, BlockCache bc,
            bool isTraining, Random dropoutRng,
            int globalBypassCount)
        {
            return _accel.ContentAwareCrossAttentionWithCache(
                Q, K, V, timeDiffs, keyTimesFromRef, queryEmbeddings, keyEmbeddings,
                block, bc, _config.Price.EmbeddingDim, _config.Price.NumHeads,
                isTraining, dropoutRng);
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

        // 
        // Init helpers
        // 

        private void InitTextEncoder()
        {
            TextTokenEmbedding = new float[_config.Text.VocabSize, _config.Text.EmbeddingDim];
            int v = _config.Text.VocabSize;
            int ed = _config.Text.EmbeddingDim;
            TextBlocks = new TransformerBlock[_config.Text.NumLayers];
            for (int i = 0; i < _config.Text.NumLayers; i++)
                TextBlocks[i] = new TransformerBlock(
                    ed, _config.Text.NumHeads, _config.Text.FeedForwardDim,
                    _config.Runtime.FFNActivationType,
                    CostFunctionType.mse, ActivationDistribution.Normal,
                    _config.Reg.L2RegulationLamda, _config.Reg.GradientClippingThreshold,
                    _config.Runtime.AccelerationType, _config.Runtime.AccelerationDeviceId,
                    _accel, _random);
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

            // Regression (High, Low, Close) — linear
            RegressionProjection = InitWeights(rDim, ed, regStd);
            RegressionBias = new float[rDim];

            // Range — softplus
            RangeProjection = InitWeights(1, ed, clsStd);
            RangeBias = new float[1];

            // Quality — sigmoid
            QualityProjection = InitWeights(1, ed, clsStd);
            QualityBias = new float[1];

            // Direction — sigmoid
            DirectionProjection = InitWeights(1, ed, clsStd);
            DirectionBias = new float[1];

            // MidWindowDirection — sigmoid
            MidDirectionProjection = InitWeights(1, ed, clsStd);
            MidDirectionBias = new float[1];

            // Confidence (optional) — sigmoid
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

                ["FFNActivationType"] = (int)_config.Runtime.FFNActivationType,
                ["AccelerationType"] = (int)_config.Runtime.AccelerationType,
                ["AccelerationDeviceId"] = _config.Runtime.AccelerationDeviceId,

                ["L2RegulationLamda"] = _config.Reg.L2RegulationLamda,
                ["GradientClipThreshold"] = _config.Reg.GradientClippingThreshold,

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

            File.WriteAllText(Path.Combine(dir, "config.json"),
                JsonSerializer.Serialize(cd, new JsonSerializerOptions { WriteIndented = true }));

            using var s = new FileStream(Path.Combine(dir, "weights.bin"), FileMode.Create);
            using var w = new BinaryWriter(s);

            WM(w, TextTokenEmbedding);
            for (int i = 0; i < _config.Text.NumLayers; i++)
            {
                var b = TextBlocks[i];
                WA(w, b.Attention);
                WV(w, b.LN1Gamma); WV(w, b.LN1Beta);
                WV(w, b.LN2Gamma); WV(w, b.LN2Beta);
            }

            WM(w, PriceInputProjection); WV(w, PriceInputProjectionBias);
            if (GlobalFeatureProjection != null) { WM(w, GlobalFeatureProjection); WV(w, GlobalFeatureBias); }

            for (int i = 0; i < _config.Price.NumLayers; i++)
            {
                var b = PriceBlocks[i];
                WA(w, b.SelfAttention);
                WV(w, b.LNSelfGamma); WV(w, b.LNSelfBeta);
                WA(w, b.CrossAttention);
                WV(w, b.LnCrossGamma); WV(w, b.LnCrossBeta);
                WV(w, b.LNFFNGamma); WV(w, b.LNFFNBeta);
                b.DecayNetwork.WriteTo(w);
            }

            // Output heads
            WM(w, RegressionProjection); WV(w, RegressionBias);
            WM(w, RangeProjection); WV(w, RangeBias);
            WM(w, QualityProjection); WV(w, QualityBias);
            WM(w, DirectionProjection); WV(w, DirectionBias);
            WM(w, MidDirectionProjection); WV(w, MidDirectionBias);
            if (_config.Output.UseConfidenceHead) { WM(w, ConfidenceProjection); WV(w, ConfidenceBias); }

            WM(w, ContextTypeEmbedding);

            for (int i = 0; i < _config.Text.NumLayers; i++) TextBlocks[i].FeedForwardNetwork.Save(Path.Combine(dir, $"text_ffn_{i}"));
            for (int i = 0; i < _config.Price.NumLayers; i++) PriceBlocks[i].FeedForwardNetwork.Save(Path.Combine(dir, $"price_ffn_{i}"));

            SaveMemory(dir);
            Tokenizer?.Save(Path.Combine(dir, "tokenizer"));
        }

        public static MmtacModel Load(string dir)
        {
            var d = JsonSerializer.Deserialize<Dictionary<string, JsonElement>>(File.ReadAllText(Path.Combine(dir, "config.json")));

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
                    GlobalFeatureDim = d.ContainsKey("GlobalFeatureDim") ? d["GlobalFeatureDim"].GetInt32() : 0,
                    BypassDecay = d.ContainsKey("GlobalBypassDecay") && d["GlobalBypassDecay"].GetBoolean()
                },
                Output = new MmtacOutputConfig
                {
                    UseConfidenceHead = d["UseConfidenceHead"].GetBoolean(),
                    DirectionLossWeight = d.ContainsKey("DirectionLossWeight") ? d["DirectionLossWeight"].GetSingle() : 1f,
                    MidDirectionLossWeight = d.ContainsKey("MidDirectionLossWeight") ? d["MidDirectionLossWeight"].GetSingle() : 0.5f,
                    RangeLossWeight = d.ContainsKey("RangeLossWeight") ? d["RangeLossWeight"].GetSingle() : 1f,
                    QualityLossWeight = d.ContainsKey("QualityLossWeight") ? d["QualityLossWeight"].GetSingle() : 1f
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
                }
            };
            cfg.Validate();

            var m = new MmtacModel(cfg);
            if (d.ContainsKey("PruningAlpha")) m.PruningConfig.AttentionScoreAlpha = d["PruningAlpha"].GetSingle();
            if (d.ContainsKey("PruningMinQueryCount")) m.PruningConfig.MinQueryCountForPruning = d["PruningMinQueryCount"].GetInt32();
            if (d.ContainsKey("PruningReserveFraction")) m.PruningConfig.NewEntryReserveFraction = d["PruningReserveFraction"].GetSingle();
            if (d.ContainsKey("PruningUseAttentionBased")) m.PruningConfig.UseAttentionBasedPruning = d["PruningUseAttentionBased"].GetBoolean();

            using (var s = new FileStream(Path.Combine(dir, "weights.bin"), FileMode.Open))
            using (var r = new BinaryReader(s))
            {
                RM(r, m.TextTokenEmbedding);
                for (int i = 0; i < cfg.Text.NumLayers; i++)
                {
                    var b = m.TextBlocks[i];
                    RA(r, b.Attention);
                    RV(r, b.LN1Gamma); RV(r, b.LN1Beta);
                    RV(r, b.LN2Gamma); RV(r, b.LN2Beta);
                }
                RM(r, m.PriceInputProjection); RV(r, m.PriceInputProjectionBias);
                if (cfg.Global.GlobalFeatureDim > 0) { RM(r, m.GlobalFeatureProjection); RV(r, m.GlobalFeatureBias); }
                for (int i = 0; i < cfg.Price.NumLayers; i++)
                {
                    var b = m.PriceBlocks[i];
                    RA(r, b.SelfAttention);
                    RV(r, b.LNSelfGamma); RV(r, b.LNSelfBeta);
                    RA(r, b.CrossAttention);
                    RV(r, b.LnCrossGamma); RV(r, b.LnCrossBeta);
                    RV(r, b.LNFFNGamma); RV(r, b.LNFFNBeta);
                    b.DecayNetwork = ContentAwareDecayNetwork.ReadFrom(r);
                }
                RM(r, m.RegressionProjection); RV(r, m.RegressionBias);
                RM(r, m.RangeProjection); RV(r, m.RangeBias);
                RM(r, m.QualityProjection); RV(r, m.QualityBias);
                RM(r, m.DirectionProjection); RV(r, m.DirectionBias);
                RM(r, m.MidDirectionProjection); RV(r, m.MidDirectionBias);
                if (cfg.Output.UseConfidenceHead) { RM(r, m.ConfidenceProjection); RV(r, m.ConfidenceBias); }
                if (s.Position < s.Length) RM(r, m.ContextTypeEmbedding);
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
            if (Directory.Exists(tokDir)) try { m.Tokenizer = BPETokenizer.Load(tokDir); } catch { }

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
        static void WV(BinaryWriter w, float[] v) { w.Write(v.Length); for (int i = 0; i < v.Length; i++) w.Write(v[i]); }
        static void WA(BinaryWriter w, MultiHeadAttention a) { WM(w, a.WQ); WM(w, a.WK); WM(w, a.WV); WM(w, a.WO); WV(w, a.BiasQ); WV(w, a.BiasK); WV(w, a.BiasV); WV(w, a.BiasO); }
        static void RM(BinaryReader r, float[,] m) { int rows = r.ReadInt32(), cols = r.ReadInt32(); for (int i = 0; i < rows; i++) for (int j = 0; j < cols; j++) m[i, j] = r.ReadSingle(); }
        static void RV(BinaryReader r, float[] v) { int l = r.ReadInt32(); for (int i = 0; i < l; i++) v[i] = r.ReadSingle(); }
        static void RA(BinaryReader r, MultiHeadAttention a) { RM(r, a.WQ); RM(r, a.WK); RM(r, a.WV); RM(r, a.WO); RV(r, a.BiasQ); RV(r, a.BiasK); RV(r, a.BiasV); RV(r, a.BiasO); }
    }
}