using CallaghanDev.ML.AccelerationManagers;
using CallaghanDev.ML.Transformers.Cache;
using CallaghanDev.ML.Transformers.Configuration;
using CallaghanDev.ML.Transformers.CrossAttentionMultimodal;
using System;
using System.Collections.Generic;
using System.Linq;

namespace CallaghanDev.ML.Transformers.TACAMT
{
    public class Trainer
    {
        private readonly Model _model;
        private readonly MultimodalTransformerConfig _config;
        private readonly TrainingConfig _trainConfig;
        private readonly Gradients _gradients;
        private readonly IAccelerationManager _accel;
        private readonly Random _random;
        private readonly List<List<float[,]>> _textFFNWeightGrads;
        private readonly List<List<float[]>> _textFFNBiasGrads;
        private readonly List<List<float[,]>> _priceFFNWeightGrads;
        private readonly List<List<float[]>> _priceFFNBiasGrads;
        private readonly Random _dropoutRng;

        public Trainer(Model model, TrainingConfig trainConfig)
        {
            _model = model;
            _config = model.Config;
            _trainConfig = trainConfig;
            _gradients = new Gradients(_config);
            _accel = model.AccelerationManager;
            _random = new Random();
            _dropoutRng = new Random(_random.Next());
            _textFFNWeightGrads = new List<List<float[,]>>();
            _textFFNBiasGrads = new List<List<float[]>>();

            for (int i = 0; i < _config.Text.NumLayers; i++)
            {
                var (w, b) = model.TextBlocks[i].FeedForwardNetwork.CreateGradientStorage();
                _textFFNWeightGrads.Add(w);
                _textFFNBiasGrads.Add(b);
            }

            _priceFFNWeightGrads = new List<List<float[,]>>(); _priceFFNBiasGrads = new List<List<float[]>>();

            for (int i = 0; i < _config.Price.NumLayers; i++)
            {
                var (w, b) = model.PriceBlocks[i].FeedForwardNetwork.CreateGradientStorage();
                _priceFFNWeightGrads.Add(w);
                _priceFFNBiasGrads.Add(b);
            }
        }

        //TODO: Clean this up
        public void Train(NewsStory[][] storiesPerSample, float[][,] priceInputs, float[][,] priceTargets, float[][] confTargets = null)
        {
            int n = storiesPerSample.Length;

            int totalEpochs = _trainConfig.Epochs;

            for (int ep = 0; ep < _trainConfig.Epochs; ep++)
            {
                float lr = ComputeLearningRate(ep, totalEpochs);

                if (_trainConfig.Verbose)
                {
                    Console.WriteLine($"\n=== Epoch {ep + 1}/{_trainConfig.Epochs} ===");
                }

                var sh = Enumerable.Range(0, n).OrderBy(_ => _random.Next()).ToArray();

                float el = 0;

                int nb = 0;

                for (int i = 0; i < sh.Length; i += _trainConfig.BatchSize)
                {
                    int bs = Math.Min(_trainConfig.BatchSize, sh.Length - i);

                    float bl = TrainBatch(sh.Skip(i).Take(bs).ToArray(), storiesPerSample, priceInputs, priceTargets, confTargets, lr);

                    el += bl;
                    nb++;

                    if (_trainConfig.Verbose && nb % 10 == 0)
                    {
                        Console.WriteLine($"  Batch {nb}: Loss = {bl:F6}");
                    }
                }
                if (_trainConfig.Verbose)
                {
                    Console.WriteLine($"  Epoch {ep + 1} Average Loss: {(nb > 0 ? el / nb : 0):F6}");
                }
            }
        }

        public void Train(int[][] textSequences, float[][,] priceInputs, float[][,] priceTargets, float[][] confTargets = null)
        {
            Train(textSequences.Select(t => t != null && t.Length > 0 ? new[] { new NewsStory(t, 0f) } : null).ToArray(), priceInputs, priceTargets, confTargets);
        }


        //TODO: See if you cant speed this up with some accelleration magic
        public void TrainSequential(NewsStory[][] stories, float[][,] priceInputs, float[][,] priceTargets, double[] timestamps, double timeUnitsPerPosition = 1.0, int maxNewsMemory = 100, int maxPriceMemory = 200, float[][] confTargets = null)
        {
            // Guard: timeUnitsPerPosition=0 would cause divide-by-zero when computing relative times
            if (timeUnitsPerPosition == 0.0)
            {
                throw new ArgumentOutOfRangeException(nameof(timeUnitsPerPosition));
            }

            int n = stories.Length;
            int totalEpochs = _trainConfig.Epochs;
            int embDim = _config.Price.EmbeddingDim;

            // Precompute reciprocal so inner loops multiply instead of divide
            float invTime = (float)(1.0 / timeUnitsPerPosition);

            // Reusable heap buffers allocated once and grown as needed
            // These live outside both loops so they survive across epochs and samples.
            // Only reallocated when a larger size is needed.
            float[,] priceCtxHiddenBuf = null;
            float[] priceCtxTimesBuf = null;
            float[,] newsMemHiddenBuf = null;
            float[] newsMemTimesBuf = null;
            float[,] combinedHiddenBuf = null;
            float[] combinedTimesBuf = null;

            // Grow-only 2-D buffer helper: reallocates only if null, too few rows, or wrong cols
            static void Ensure2DBuffer(ref float[,] buf, int rows, int cols)
            {
                if (rows <= 0)
                {
                    return;
                }
                if (buf == null || buf.GetLength(0) < rows || buf.GetLength(1) != cols)
                {
                    buf = new float[rows, cols];
                }
            }

            // Grow-only 1-D buffer helper: reallocates only if null or too short
            static void Ensure1DBuffer(ref float[] buf, int len)
            {
                if (len <= 0)
                {
                    return;
                }
                if (buf == null || buf.Length < len)
                {
                    buf = new float[len];
                }
            }

            // One cache object reused across all samples in every epoch 
            // Reset() is called per-sample; the object itself is never re-newed.
            var cache = new MultimodalForwardCache(_config.Text.NumLayers, _config.Price.NumLayers);

            for (int ep = 0; ep < totalEpochs; ep++)
            {
                // Compute learning rate for this epoch (may apply schedule/decay)
                float lr = ComputeLearningRate(ep, totalEpochs);

                if (_trainConfig.Verbose)
                {
                    Console.WriteLine($"\n=== Epoch {ep + 1}/{totalEpochs} (Sequential)[{_accel.GetType().Name}] ===");
                }

                // Clear rolling memory at the START of each epoch so memory builds
                // fresh from sample 0 of this epoch rather than carrying over stale
                // hidden states from the previous epoch.
                _model.ClearAllMemory();

                float epochLoss = 0f;
                int validCount = 0;

                for (int idx = 0; idx < n; idx++)
                {
                    var ps = priceInputs[idx];
                    var pt = priceTargets[idx];

                    // Skip samples with missing price data
                    if (ps == null || pt == null)
                    {
                        continue;
                    }

                    int sl = ps.GetLength(0);

                    // Need at least 2 timesteps: one input (t) and one target (t+1)
                    if (sl < 2)
                    {
                        continue;
                    }

                    double currentTs = timestamps[idx];

                    try
                    {
                        // Input / target window
                        // Input  : rows [0 .. sl-2]  (all but the last timestep)
                        // Target : rows [1 .. sl-1]  (all but the first timestep)
                        // Passed as offset+count rather than pre-sliced to avoid allocation.
                        float[,] inp = ps;
                        int inpStart = 0;
                        int inpCount = sl - 1;

                        float[,] tgt = pt;
                        int tgtStart = 1;
                        int tgtCount = sl - 1;

                        // Confidence targets (optional)
                        // Shift by 1 to align with the target window; skip if too short.
                        float[] ct = null;
                        var src = confTargets?[idx];


                        if (src != null && src.Length >= sl)
                        {
                            ct = new float[sl - 1];
                            Array.Copy(src, 1, ct, 0, sl - 1);
                        }

                        // Snapshot current memory counts before this sample
                        // Memory is read-only during forward/backward; new entries are
                        // added AFTER the parameter update (see end of this iteration).
                        int newsMemCount = _model.NewsMemory?.Count ?? 0;
                        int priceMemCount = _model.PriceMemory?.Count ?? 0;

                        // Clear per-sample tensors in the cache (leaves list capacities intact)
                        cache.Reset();

                        // Pack price memory into contiguous buffers.
                        float[,] priceCtxHidden = null;
                        float[] priceCtxTimes = null;

                        if (priceMemCount > 0)
                        {
                            Ensure2DBuffer(ref priceCtxHiddenBuf, priceMemCount, embDim);
                            Ensure1DBuffer(ref priceCtxTimesBuf, priceMemCount);

                            // Point locals at the reusable buffers
                            priceCtxHidden = priceCtxHiddenBuf;
                            priceCtxTimes = priceCtxTimesBuf;

                            for (int i = 0; i < priceMemCount; i++)
                            {
                                var entry = _model.PriceMemory[i];
                                var hs = entry.HiddenState;

                                for (int d = 0; d < embDim; d++)
                                {
                                    priceCtxHidden[i, d] = hs[d];
                                }

                                // Negative = in the past relative to currentTs; scale by invTime
                                priceCtxTimes[i] = -(float)(currentTs - entry.AbsoluteTimestamp) * invTime;
                            }
                        }

                        // Pack news memory into contiguous buffers
                        float[,] newsMemHidden = null;
                        float[] newsMemTimes = null;

                        if (newsMemCount > 0)
                        {
                            Ensure2DBuffer(ref newsMemHiddenBuf, newsMemCount, embDim);
                            Ensure1DBuffer(ref newsMemTimesBuf, newsMemCount);

                            newsMemHidden = newsMemHiddenBuf;
                            newsMemTimes = newsMemTimesBuf;

                            for (int i = 0; i < newsMemCount; i++)
                            {
                                var entry = _model.NewsMemory[i];
                                var hs = entry.HiddenState;

                                for (int d = 0; d < embDim; d++)
                                {
                                    newsMemHidden[i, d] = hs[d];
                                }

                                newsMemTimes[i] = -(float)(currentTs - entry.AbsoluteTimestamp) * invTime;
                            }
                        }

                        // Encode any fresh news stories arriving at this timestep
                        NewsStory[] currentStories = stories[idx];
                        NewsStory[] adjustedStories = (currentStories != null && currentStories.Length > 0) ? currentStories : null;

                        float[,] pred, conf;

                        if (newsMemCount > 0 || priceMemCount > 0)
                        {
                            // Encode fresh stories through the text encoder
                            float[,] freshNewsHidden = null;
                            float[] freshNewsTimes = null;
                            int freshNewsCount = 0;

                            if (adjustedStories != null)
                            {
                                // Encodes each story to a hidden vector; result is [numStories, embDim]
                                (freshNewsHidden, freshNewsTimes) = _model.EncodeStoriesWithCache(adjustedStories, cache);
                                freshNewsCount = freshNewsHidden.GetLength(0);
                            }

                            // Build combined context: [newsMemory | freshNews | priceMemory]
                            int totalCtx = newsMemCount + freshNewsCount + priceMemCount;

                            float[,] combinedHidden = null;
                            float[] combinedTimes = null;

                            if (totalCtx > 0)
                            {
                                Ensure2DBuffer(ref combinedHiddenBuf, totalCtx, embDim);
                                Ensure1DBuffer(ref combinedTimesBuf, totalCtx);

                                combinedHidden = combinedHiddenBuf;
                                combinedTimes = combinedTimesBuf;

                                int ci = 0;

                                // Paste news memory entries first
                                for (int i = 0; i < newsMemCount; i++, ci++)
                                {
                                    for (int d = 0; d < embDim; d++)
                                    {
                                        combinedHidden[ci, d] = newsMemHidden[i, d];
                                    }
                                    combinedTimes[ci] = newsMemTimes[i];
                                }

                                //  aste freshly encoded stories
                                for (int i = 0; i < freshNewsCount; i++, ci++)
                                {
                                    for (int d = 0; d < embDim; d++)
                                    {
                                        combinedHidden[ci, d] = freshNewsHidden[i, d];
                                    }
                                    combinedTimes[ci] = freshNewsTimes[i];
                                }

                                // Paste price memory entries last
                                for (int i = 0; i < priceMemCount; i++, ci++)
                                {
                                    for (int d = 0; d < embDim; d++)
                                    {
                                        combinedHidden[ci, d] = priceCtxHidden[i, d];
                                    }
                                    combinedTimes[ci] = priceCtxTimes[i];
                                }

                                // Apply context-type embeddings (from v1)
                                // Adds a learned type vector to every context token so the
                                // decoder can distinguish news tokens from price tokens.
                                // Index 0 = news type embedding, index 1 = price type embedding.
                                int newsTotal = newsMemCount + freshNewsCount;

                                for (int i = 0; i < newsTotal; i++)
                                {
                                    for (int d = 0; d < embDim; d++)
                                    {
                                        combinedHidden[i, d] += _model.ContextTypeEmbedding[0, d];
                                    }
                                }

                                for (int i = 0; i < priceMemCount; i++)
                                {
                                    for (int d = 0; d < embDim; d++)
                                    {
                                        combinedHidden[newsTotal + i, d] += _model.ContextTypeEmbedding[1, d];
                                    }
                                }

                                // Store context metadata on cache for backward pass 
                                // BackwardPass reads these to know how the context was split.
                                cache.NumNewsContext = newsMemCount + freshNewsCount;
                                cache.NumPriceContext = priceMemCount;
                                cache.PriceContextHidden = priceCtxHidden;
                                cache.TextFinalHidden = combinedHidden;
                                cache.StoryArrivalTimes = combinedTimes;
                            }

                            // Forward pass through the price decoder
                            // Uses offset+count variant to avoid pre-slicing inp
                            var priceHidden = _model.ForwardPriceDecoderWithCache(inp, inpStart, inpCount, combinedHidden, combinedTimes, cache, isTraining: true, dropoutRng: _dropoutRng);

                            // Store final hidden for memory update below
                            cache.PriceFinalHidden = priceHidden;

                            // Project decoder output to prediction + confidence heads
                            (pred, conf) = _model.ProjectToOutput(priceHidden);
                        }
                        else
                        {
                            // No memory context at all: run the simpler joint forward pass
                            (pred, conf) = _model.ForwardWithCache(adjustedStories, inp, inpStart, inpCount, cache, isTraining: true, dropoutRng: _dropoutRng);
                        }

                        // Backward pass
                        // Zero accumulated gradients from any previous iteration first
                        ZeroAllGradients();

                        float loss = BackwardPass(pred, conf, tgt, tgtStart, tgtCount, ct, cache);

                        // Skip parameter update if loss is NaN or Inf (numerically unstable step)
                        if (!float.IsFinite(loss))
                        {
                            ZeroAllGradients();
                            continue;
                        }

                        // Optionally clip gradient global norm to prevent exploding gradients
                        if (_trainConfig.UseGradientClipping)
                        {
                            ClipGradients(_trainConfig.GradientClipThreshold);
                        }

                        // Apply SGD / Adam step
                        UpdateAllParameters(lr);

                        epochLoss += loss;
                        validCount++;

                        // Update rolling memory AFTER parameter update
                        // This is the key difference from TrainSequential (old): memory
                        // accumulates within the epoch so each subsequent sample sees the
                        // hidden states produced by all previous samples this epoch.

                        // Add fresh news stories to news memory
                        if (adjustedStories != null)
                        {
                            // Re-encode with updated weights so memory stays consistent
                            // with the current model state after this step's update.
                            (var encodedNews, _) = _model.EncodeStories(adjustedStories);

                            for (int i = 0; i < adjustedStories.Length; i++)
                            {
                                var hv = new float[embDim];

                                for (int d = 0; d < embDim; d++)
                                {
                                    hv[d] = encodedNews[i, d];
                                }

                                _model.NewsMemory.Add(new NewsMemoryEntry
                                {
                                    HiddenState = hv,
                                    // Convert story arrival time from relative positions to absolute timestamp
                                    AbsoluteTimestamp = currentTs + adjustedStories[i].ArrivalTime * timeUnitsPerPosition
                                });
                            }
                        }

                        // Add each price decoder hidden state to price memory
                        int priceSeqLen = cache.PriceFinalHidden.GetLength(0);
                        for (int t = 0; t < priceSeqLen; t++)
                        {
                            var pv = new float[embDim];
                            for (int d = 0; d < embDim; d++)
                            {
                                pv[d] = cache.PriceFinalHidden[t, d];
                            }

                            _model.PriceMemory.Add(new PriceMemoryEntry
                            {
                                HiddenState = pv,
                                // Each position t in the sequence maps to one time unit forward
                                AbsoluteTimestamp = currentTs + t * timeUnitsPerPosition
                            });
                        }

                        // Record the timestamp of the last processed price sequence
                        _model.LastPriceTimestamp = currentTs;

                        // Trim memory to configured maximums (oldest entries dropped first)
                        _model.PruneNewsMemory(maxNewsMemory);
                        _model.PricePruneMemory(maxPriceMemory);

                        if (_trainConfig.Verbose && validCount % 50 == 0)
                        {
                            Console.WriteLine($"  Sample {validCount}: Loss = {loss:F6}, NewsMemory = {_model.NewsMemory.Count}, PriceMemory = {_model.PriceMemory.Count}");
                        }
                    }
                    catch (Exception ex)
                    {
                        // Swallow per-sample exceptions so one bad sample doesn't abort the epoch.
                        // Zero gradients so a partial backward doesn't corrupt the next update.
                        ZeroAllGradients();

                        if (_trainConfig.Verbose)
                        {
                            Console.WriteLine($"  WARNING: {ex.Message}");
                            Console.WriteLine($"  Info: {ex.StackTrace}");
                        }
                    }
                }

                if (_trainConfig.Verbose)
                {
                    Console.WriteLine($"  Epoch {ep + 1} Average Loss: {(validCount > 0 ? epochLoss / validCount : 0):F6}");
                }
            }
        }


        private float TrainBatch(int[] bi, NewsStory[][] allS, float[][,] allP, float[][,] allT, float[][] allC, float lr)
        {
            ZeroAllGradients();

            float tl = 0;
            int vc = 0;

            int minSplitLen = _config.PriceContext.MinHistoryLength + _config.PriceContext.MinCurrentLength + 1;

            foreach (int idx in bi)
            {
                var ps = allP[idx]; int sl = ps.GetLength(0);

                if (sl < 2)
                {
                    continue;
                }
                try
                {
                    bool canSplit = sl >= minSplitLen;

                    if (canSplit)
                    {
                        float loss = TrainWithPriceContext(idx, allS, allP, allT, allC);
                        if (float.IsNaN(loss) || float.IsInfinity(loss))
                        {
                            continue;
                        }
                        tl += loss; vc++;
                    }
                    else
                    {
                        var inp = SliceRows(ps, 0, sl - 1);
                        var tgt = SliceRows(allT[idx], 1, sl);
                        float[] ct = allC?[idx] != null ? allC[idx].Skip(1).Take(sl - 1).ToArray() : null;

                        var cache = new MultimodalForwardCache(_config.Text.NumLayers, _config.Price.NumLayers);

                        var (pred, conf) = _model.ForwardWithCache(allS[idx], inp, cache, isTraining: true, dropoutRng: _dropoutRng);

                        float loss = BackwardPass(pred, conf, tgt, ct, cache);

                        if (float.IsNaN(loss) || float.IsInfinity(loss))
                        {
                            continue;
                        }
                        tl += loss; vc++;
                    }
                }
                catch (Exception ex)
                {
                    if (_trainConfig.Verbose) Console.WriteLine($"  WARNING: {ex.Message}");
                }
            }

            if (vc == 0)
            {
                return 0f;
            }
            ScaleAllGradients(1.0f / vc);

            if (_trainConfig.UseGradientClipping)
            {
                ClipGradients(_trainConfig.GradientClipThreshold);
            }

            UpdateAllParameters(lr);

            return tl / vc;
        }

        private float TrainWithPriceContext(int idx, NewsStory[][] allS, float[][,] allP, float[][,] allT, float[][] allC)
        {
            var ps = allP[idx]; int sl = ps.GetLength(0);
            int featureDim = ps.GetLength(1);

            int minHist = _config.PriceContext.MinHistoryLength;
            int maxHist = sl - _config.PriceContext.MinCurrentLength - 1;
            int splitPoint = minHist + _random.Next(maxHist - minHist + 1);

            int histLen = splitPoint;
            int currentLen = sl - splitPoint;

            var histPrices = SliceRows(ps, 0, histLen);
            var currentInput = SliceRows(ps, splitPoint, sl - 1);
            var currentTarget = SliceRows(allT[idx], splitPoint + 1, sl);
            int currentSeqLen = currentInput.GetLength(0);

            if (currentSeqLen < 2)
            {
                return 0f;
            }

            float[] confTgt = null;

            if (allC?[idx] != null)
            {
                confTgt = allC[idx].Skip(splitPoint + 1).Take(currentSeqLen).ToArray();
            }

            var priceCtxHidden = _model.EncodePriceHistory(histPrices);

            var priceCtxTimes = new float[histLen];
             

            // Positive age: oldest history gets the largest age, newest gets age 1
            for (int t = 0; t < histLen; t++)
            {
                priceCtxTimes[t] = histLen - t;
            }

            NewsStory[] adjustedStories = null;

            if (allS[idx] != null && allS[idx].Length > 0)
            {
                adjustedStories = new NewsStory[allS[idx].Length];

                for (int i = 0; i < allS[idx].Length; i++)
                {
                    adjustedStories[i] = new NewsStory(allS[idx][i].TokenIds, allS[idx][i].ArrivalTime - splitPoint);
                }
            }

            var cache = new MultimodalForwardCache(_config.Text.NumLayers, _config.Price.NumLayers);

            var (pred, conf) = _model.ForwardWithPriceContextAndCache(adjustedStories, currentInput, priceCtxHidden, priceCtxTimes, cache, isTraining: true, dropoutRng: _dropoutRng);

            return BackwardPass(pred, conf, currentTarget, confTgt, cache);
        }

        #region Validation

        public float Validate(NewsStory[][] stories, float[][,] priceInputs, float[][,] priceTargets)
        {
            float totalLoss = 0f;
            int totalCount = 0;
            int minSplitLength = _config.PriceContext.MinHistoryLength + _config.PriceContext.MinCurrentLength + 1;

            for (int i = 0; i < stories.Length; i++)
            {
                float[,] inputPrices = priceInputs[i];
                int sequenceLength = inputPrices.GetLength(0);

                if (sequenceLength < 2) continue;

                bool canUseContextSplit = (sequenceLength >= minSplitLength);

                if (canUseContextSplit)
                {
                    int minHistory = _config.PriceContext.MinHistoryLength;
                    int maxHistory = sequenceLength - _config.PriceContext.MinCurrentLength - 1;
                    int splitPoint = (minHistory + maxHistory) / 2;

                    float[,] historyPrices = SliceRows(inputPrices, 0, splitPoint);
                    float[,] currentInput = SliceRows(inputPrices, splitPoint, sequenceLength - 1);
                    float[,] currentTarget = SliceRows(priceTargets[i], splitPoint + 1, sequenceLength);

                    int currentSeqLength = currentInput.GetLength(0);

                    if (currentSeqLength < 2)
                    {
                        continue;
                    }

                    float[,] priceContextHidden = _model.EncodePriceHistory(historyPrices);

                    float[] priceContextTimes = new float[splitPoint];

                    for (int t = 0; t < splitPoint; t++)
                    {
                        // oldest gets largest age, newest gets age 1
                        priceContextTimes[t] = (splitPoint - t);
                    }

                    NewsStory[] adjustedStories = null;
                    if (stories[i] != null && stories[i].Length > 0)
                    {
                        adjustedStories = new NewsStory[stories[i].Length];
                        for (int s = 0; s < stories[i].Length; s++)
                        {
                            adjustedStories[s] = new NewsStory(stories[i][s].TokenIds, stories[i][s].ArrivalTime - splitPoint);
                        }
                    }

                    var cache = new MultimodalForwardCache(_config.Text.NumLayers, _config.Price.NumLayers);

                    var (predictions, _) = _model.ForwardWithPriceContextAndCache(adjustedStories, currentInput, priceContextHidden, priceContextTimes, cache, isTraining: false);

                    for (int t = 0; t < currentSeqLength; t++)
                    {
                        for (int j = 0; j < _config.Output.OutputDim; j++)
                        {
                            float diff = predictions[t, j] - currentTarget[t, j];
                            totalLoss += diff * diff;
                        }
                    }

                    totalCount += currentSeqLength;
                }
                else
                {
                    int effectiveLength = sequenceLength - 1;

                    float[,] validationInput = SliceRows(inputPrices, 0, effectiveLength);
                    float[,] validationTarget = SliceRows(priceTargets[i], 1, effectiveLength + 1);

                    var cache = new MultimodalForwardCache(_config.Text.NumLayers, _config.Price.NumLayers);

                    var (predictions, _) = _model.ForwardWithCache(stories[i], validationInput, cache);

                    for (int t = 0; t < effectiveLength; t++)
                    {
                        for (int j = 0; j < _config.Output.OutputDim; j++)
                        {
                            float diff = predictions[t, j] - validationTarget[t, j];
                            totalLoss += diff * diff;
                        }
                    }

                    totalCount += effectiveLength;
                }
            }

            return totalCount > 0 ? totalLoss / (totalCount * _config.Output.OutputDim) : 0f;
        }

        public float Validate(int[][] texts, float[][,] pi, float[][,] pt)
        {
            return Validate(texts.Select(t => t != null && t.Length > 0 ? new[] { new NewsStory(t, 0f) } : null).ToArray(), pi, pt);
        }

        #endregion

        private float BackwardPass(float[,] pred, float[,] conf, float[,] tgt, float[] confTgt, MultimodalForwardCache cache)
        {
            int sl = pred.GetLength(0);

            return BackwardPass(pred, conf, tgt, tgtRowStart: 0, tgtRowCount: sl, confTgt, cache);
        }

        private float BackwardPass(float[,] pred, float[,] conf, float[,] tgt, int tgtRowStart, int tgtRowCount, float[] confTgt, MultimodalForwardCache cache)
        {
            if (tgtRowStart < 0 || tgtRowCount < 0)
            {
                throw new ArgumentOutOfRangeException();
            }

            if (tgtRowStart + tgtRowCount > tgt.GetLength(0))
            {
                throw new ArgumentException("Invalid target slice.");
            }

            int sequenceLength = tgtRowCount;
            int outputDim = _config.Output.OutputDim;
            int embeddingDim = _config.Price.EmbeddingDim;

            // Offset-aware MSE (since accelerator assumes row 0)
            float mse = 0f;
            float[,] dPred = new float[sequenceLength, outputDim];

            for (int t = 0; t < sequenceLength; t++)
            {
                int tgtRow = tgtRowStart + t;

                for (int j = 0; j < outputDim; j++)
                {
                    float diff = pred[t, j] - tgt[tgtRow, j];

                    mse += diff * diff;

                    dPred[t, j] = 2f * diff / sequenceLength;
                }
            }

            mse /= sequenceLength;

            // Projection backward (unchanged)
            float[,] dHidden = _accel.BackpropOutputProjection(
                dPred,
                cache.PriceFinalHidden,
                _model.OutputProjection,
                _gradients.OutputProjectionGrad,
                _gradients.OutputBiasGrad,
                sequenceLength,
                outputDim,
                embeddingDim);

            float confidenceLoss = 0f;

            if (_config.Output.UseConfidenceHead && conf != null)
            {
                for (int t = 0; t < sequenceLength; t++)
                {
                    int tgtRow = tgtRowStart + t;

                    float prediction = conf[t, 0];
                    float target;

                    if (confTgt != null)
                    {
                        target = confTgt[t];
                    }
                    else
                    {
                        float sumSq = 0f;

                        for (int j = 0; j < outputDim; j++)
                        {
                            float diff = pred[t, j] - tgt[tgtRow, j];
                            sumSq += diff * diff;
                        }

                        float rmse = MathF.Sqrt(sumSq / outputDim);

                        target = MathF.Exp(-5f * rmse);
                    }

                    float clampedPrediction = Math.Clamp(prediction, 1e-7f, 1f - 1e-7f);

                    confidenceLoss -= target * MathF.Log(clampedPrediction) + (1f - target) * MathF.Log(1f - clampedPrediction);

                    float dLoss = (prediction - target) * _trainConfig.ConfidenceLossWeight / sequenceLength;

                    for (int e = 0; e < embeddingDim; e++)
                    {
                        _gradients.ConfidenceProjectionGrad[0, e] += dLoss * cache.PriceFinalHidden[t, e];

                        dHidden[t, e] += dLoss * _model.ConfidenceProjection[0, e];
                    }

                    _gradients.ConfidenceBiasGrad[0] += dLoss;
                }

                confidenceLoss = confidenceLoss / sequenceLength;
            }

            bool hasContext = (cache.TextFinalHidden != null);

            float[,] dSharedHidden = BackpropPriceDecoder(dHidden, cache, hasContext);

            if (!_config.Text.Freeze && hasContext && dSharedHidden != null && cache.StoryCaches != null)
            {
                BackpropMultiStoryTextEncoder(dSharedHidden, cache);
            }

            return mse + (_trainConfig.ConfidenceLossWeight * confidenceLoss);
        }
        private float[,] BackpropPriceDecoder(float[,] dOut, MultimodalForwardCache cache, bool hasContext)
        {
            int embeddingDim = _config.Price.EmbeddingDim;
            int numHeads = _config.Price.NumHeads;
            int headDim = embeddingDim / numHeads;

            if (embeddingDim % numHeads != 0)
            {
                throw new ArgumentException("Embedding dim must be divisible by numHeads");
            }

            float scale = 1.0f / MathF.Sqrt(headDim);

            int numNewsContext = cache.NumNewsContext;
            int numPriceContext = cache.NumPriceContext;
            int totalContext = 0;

            if (hasContext && cache.TextFinalHidden != null)
            {
                totalContext = cache.TextFinalHidden.GetLength(0);
            }

            if (totalContext > 0 && numNewsContext == 0 && numPriceContext == 0)
            {
                numNewsContext = totalContext;
            }

            float[,] dNewsHidden = null;
            if (numNewsContext > 0)
            {
                dNewsHidden = new float[numNewsContext, embeddingDim];
            }

            float[,] dX = dOut;

            for (int layer = _config.Price.NumLayers - 1; layer >= 0; layer--)
            {
                var block = _model.PriceBlocks[layer];
                var blockCache = cache.PriceBlockCaches[layer];
                var blockGrads = _gradients.PriceBlockGrads[layer];
                int sequenceLength = dX.GetLength(0);

                // FFN sublayer
                var (dFfnResidual, dGammaFfn, dBetaFfn) = _accel.LayerNormBackward(dX, blockCache.LNFFNCache.Normalized, block.LNFFNGamma, blockCache.LNFFNCache.Input, blockCache.LNFFNCache.Mean, blockCache.LNFFNCache.Variance);

                _accel.VectorAccumulate(blockGrads.LNFFNGrads.GammaGrad, dGammaFfn);
                _accel.VectorAccumulate(blockGrads.LNFFNGrads.BetaGrad, dBetaFfn);

                var dFfnInput = new float[sequenceLength, embeddingDim];

                for (int i = 0; i < sequenceLength; i++)
                {
                    var gradRow = new float[embeddingDim];
                    for (int j = 0; j < embeddingDim; j++)
                    {
                        gradRow[j] = dFfnResidual[i, j];
                    }

                    block.FeedForwardNetwork.ForwardPassOnly(blockCache.FFNInputRows[i]);

                    var inputGrad = block.FeedForwardNetwork.ComputeInputGradient(gradRow, _priceFFNWeightGrads[layer], _priceFFNBiasGrads[layer]);

                    for (int j = 0; j < embeddingDim; j++)
                    {
                        dFfnInput[i, j] = inputGrad[j];
                    }
                }

                // Residual: dFfnInput += dFfnResidual  - no new matrix alloc
                _accel.MatrixAddInPlace(dFfnInput, dFfnResidual);

                // Cross-attention sublayer
                var (dCrossResidual, dGammaCross, dBetaCross) = _accel.LayerNormBackward(dFfnInput, blockCache.LNCrossCache.Normalized, block.LnCrossGamma, blockCache.LNCrossCache.Input, blockCache.LNCrossCache.Mean, blockCache.LNCrossCache.Variance);

                _accel.VectorAccumulate(blockGrads.LNCrossGrads.GammaGrad, dGammaCross);
                _accel.VectorAccumulate(blockGrads.LNCrossGrads.BetaGrad, dBetaCross);

                float[,] dSelfInput;

                if (hasContext && blockCache.CrossQ != null && totalContext > 0)
                {
                    var crossAttnGrads = blockGrads.CrossAttnGrads;

                    var dCrossCombined = new float[sequenceLength, embeddingDim];
                    _accel.BackpropLinearProjection(blockCache.CrossAttnOutput, dCrossResidual, block.CrossAttention.WO, crossAttnGrads.WO_Grad, crossAttnGrads.BiasO_Grad, dCrossCombined);

                    var (dQ, dK, dV, dDecayBias) = BackpropTimeDecayedAttn(blockCache.CrossQ, blockCache.CrossK, blockCache.CrossV, dCrossCombined, blockCache.CrossAttentionWeights, blockCache.TimeDiffs, block);
                    RotaryPositionEmbedding.ApplyBackwardInPlace(dQ, dK, numHeads);
                    if (blockCache.DecayCache != null && dDecayBias != null)
                    {
                        var (decayParamGrads, dQueryEmb, dKeyEmb) = block.DecayNetwork.Backward(dDecayBias, blockCache.DecayCache);

                        AccumulateDecayGrads(blockGrads.DecayGrads, decayParamGrads);

                        if (dNewsHidden != null)
                        {
                            for (int i = 0; i < numNewsContext; i++)
                            {
                                for (int j = 0; j < embeddingDim; j++)
                                {
                                    dNewsHidden[i, j] += dKeyEmb[i, j];
                                }
                            }
                        }

                        // Route decay gradients for price-context keys into ContextTypeEmbeddingGrad[1].
                        // dKeyEmb rows [numNewsContext .. numNewsContext+numPriceContext-1] correspond to
                        // price memory entries; these were previously dropped, causing the price type
                        // embedding gradient to be systematically underestimated.
                        if (numPriceContext > 0)
                        {
                            for (int i = 0; i < numPriceContext; i++)
                            {
                                for (int j = 0; j < embeddingDim; j++)
                                {
                                    _gradients.ContextTypeEmbeddingGrad[1, j] += dKeyEmb[numNewsContext + i, j];
                                }
                            }
                        }

                        for (int i = 0; i < sequenceLength; i++)
                        {
                            for (int j = 0; j < embeddingDim; j++)
                            {
                                dCrossResidual[i, j] += dQueryEmb[i, j];
                            }
                        }

                    }

                    var dFromQ = new float[sequenceLength, embeddingDim];
                    _accel.BackpropLinearProjection(blockCache.NormedSelf, dQ, block.CrossAttention.WQ, crossAttnGrads.WQ_Grad, crossAttnGrads.BiasQ_Grad, dFromQ);

                    var dCtxFromK = new float[totalContext, embeddingDim];
                    var dCtxFromV = new float[totalContext, embeddingDim];

                    _accel.BackpropLinearProjection(cache.TextFinalHidden, dK, block.CrossAttention.WK, crossAttnGrads.WK_Grad, crossAttnGrads.BiasK_Grad, dCtxFromK);

                    _accel.BackpropLinearProjection(cache.TextFinalHidden, dV, block.CrossAttention.WV, crossAttnGrads.WV_Grad, crossAttnGrads.BiasV_Grad, dCtxFromV);

                    if (dNewsHidden != null)
                    {
                        for (int i = 0; i < numNewsContext; i++)
                        {
                            for (int j = 0; j < embeddingDim; j++)
                            {
                                dNewsHidden[i, j] += dCtxFromK[i, j] + dCtxFromV[i, j];
                            }
                        }
                    }

                    if (numPriceContext > 0)
                    {
                        for (int i = 0; i < numPriceContext; i++)
                        {
                            for (int j = 0; j < embeddingDim; j++)
                            {
                                _gradients.ContextTypeEmbeddingGrad[1, j] += dCtxFromK[numNewsContext + i, j] + dCtxFromV[numNewsContext + i, j];
                            }
                        }
                    }

                    // Residual: dFromQ += dCrossResidual  - no new matrix alloc
                    _accel.MatrixAddInPlace(dFromQ, dCrossResidual);
                    dSelfInput = dFromQ;
                }
                else
                {
                    dSelfInput = dCrossResidual;
                }

                // Self-attention sublayer
                var (dSelfResidual, dGammaSelf, dBetaSelf) = _accel.LayerNormBackward(dSelfInput, blockCache.LNSelfCache.Normalized, block.LNSelfGamma, blockCache.LNSelfCache.Input, blockCache.LNSelfCache.Mean, blockCache.LNSelfCache.Variance);

                _accel.VectorAccumulate(blockGrads.LNSelfGrads.GammaGrad, dGammaSelf);
                _accel.VectorAccumulate(blockGrads.LNSelfGrads.BetaGrad, dBetaSelf);

                var selfAttnGrads = blockGrads.SelfAttnGrads;
                var dSelfCombined = new float[sequenceLength, embeddingDim];

                _accel.BackpropLinearProjection(blockCache.SelfAttnOutput, dSelfResidual, block.SelfAttention.WO, selfAttnGrads.WO_Grad, selfAttnGrads.BiasO_Grad, dSelfCombined);
                var (dQSelf, dKSelf, dVSelf) = _accel.MultiHeadAttentionBackward(
                    blockCache.SelfQ, blockCache.SelfK, blockCache.SelfV,
                    dSelfCombined, numHeads, scale, _config.Price.UseDecoderOnly);

                RotaryPositionEmbedding.ApplyBackwardInPlace(dQSelf, dKSelf, numHeads);
                var dBlockInput = new float[sequenceLength, embeddingDim];

                _accel.BackpropLinearProjection(blockCache.BlockInput, dQSelf, block.SelfAttention.WQ, selfAttnGrads.WQ_Grad, selfAttnGrads.BiasQ_Grad, dBlockInput);
                _accel.BackpropLinearProjection(blockCache.BlockInput, dKSelf, block.SelfAttention.WK, selfAttnGrads.WK_Grad, selfAttnGrads.BiasK_Grad, dBlockInput);
                _accel.BackpropLinearProjection(blockCache.BlockInput, dVSelf, block.SelfAttention.WV, selfAttnGrads.WV_Grad, selfAttnGrads.BiasV_Grad, dBlockInput);

                // Residual: dBlockInput += dSelfResidual  - no new matrix alloc
                _accel.MatrixAddInPlace(dBlockInput, dSelfResidual);
                dX = dBlockInput;
            }

            // Input projection backward - uses accelerator instead of deleted manual method
            _accel.BackpropInputProjection(dX, cache.PriceContinuousInput, _gradients.PriceInputProjectionGrad, _gradients.PriceInputProjectionBiasGrad, dX.GetLength(0), _config.Price.EmbeddingDim, _config.Price.InputFeatureDim);

            if (dNewsHidden != null)
            {
                for (int i = 0; i < numNewsContext; i++)
                {
                    for (int j = 0; j < embeddingDim; j++)
                    {
                        _gradients.ContextTypeEmbeddingGrad[0, j] += dNewsHidden[i, j];
                    }
                }
            }

            return dNewsHidden;
        }

        private (float[,] dQ, float[,] dK, float[,] dV, float[,,] dDecayBias) BackpropTimeDecayedAttn(float[,] Q, float[,] K, float[,] V, float[,] dOutput, float[][,] attnW, float[,] timeDiffs, TacamtBlock block)
        {
            int psl = Q.GetLength(0); int tsl = K.GetLength(0); int ed = _config.Price.EmbeddingDim; int nh = _config.Price.NumHeads; int hd = ed / nh;

            float s = 1.0f / MathF.Sqrt(hd);

            var dQ = new float[psl, ed];
            var dK = new float[tsl, ed];
            var dV = new float[tsl, ed];


            float[,,] dDB = timeDiffs != null ? new float[psl, tsl, nh] : null;

            for (int h = 0; h < nh; h++)
            {
                int si = h * hd;
                var w = attnW[h];

                for (int p = 0; p < psl; p++)
                {
                    var dw = new float[tsl];
                    for (int sv = 0; sv < tsl; sv++)
                    {
                        float d = 0;
                        for (int dd = 0; dd < hd; dd++)
                        {
                            int di = si + dd;
                            dV[sv, di] += w[p, sv] * dOutput[p, di];
                            d += dOutput[p, di] * V[sv, di];
                        }
                        dw[sv] = d;
                    }

                    float dotWD = 0;

                    for (int sv = 0; sv < tsl; sv++)
                    {
                        dotWD += w[p, sv] * dw[sv];
                    }

                    var ds = new float[tsl];

                    for (int sv = 0; sv < tsl; sv++)
                    {
                        ds[sv] = w[p, sv] * (dw[sv] - dotWD);
                    }

                    for (int sv = 0; sv < tsl; sv++)
                    {
                        float dsc = ds[sv];

                        for (int dd = 0; dd < hd; dd++)
                        {
                            int di = si + dd;
                            dQ[p, di] += dsc * s * K[sv, di];
                            dK[sv, di] += dsc * s * Q[p, di];
                        }
                        if (dDB != null)
                        {
                            dDB[p, sv, h] += dsc;
                        }
                    }
                }
            }

            return (dQ, dK, dV, dDB);
        }

        private void AccumulateDecayGrads(ContentAwareDecayGradients tgt, ContentAwareDecayGradients src)
        {
            int nh = src.B2Grad.Length;
            int cd = src.QueryProjectionGrad.GetLength(2);
            int pd = src.QueryProjectionGrad.GetLength(1);
            int hd = src.B1Grad.GetLength(1);
            int mid = src.W1Grad.GetLength(2);
            int ntb = src.TimeLogFreqGrad.GetLength(1);
            int rawDim = src.TimeProjGrad.GetLength(2);

            for (int h = 0; h < nh; h++)
            {
                tgt.LogBaseDecayRateGrad[h] += src.LogBaseDecayRateGrad[h];
                tgt.B2Grad[h] += src.B2Grad[h];

                for (int p = 0; p < pd; p++)
                {
                    for (int d = 0; d < cd; d++)
                    {
                        tgt.QueryProjectionGrad[h, p, d] += src.QueryProjectionGrad[h, p, d];
                        tgt.KeyProjectionGrad[h, p, d] += src.KeyProjectionGrad[h, p, d];
                    }

                    tgt.QueryProjectionBiasGrad[h, p] += src.QueryProjectionBiasGrad[h, p];
                    tgt.KeyProjectionBiasGrad[h, p] += src.KeyProjectionBiasGrad[h, p];

                    for (int r = 0; r < rawDim; r++)
                    {
                        tgt.TimeProjGrad[h, p, r] += src.TimeProjGrad[h, p, r];
                    }

                    tgt.TimeProjBiasGrad[h, p] += src.TimeProjBiasGrad[h, p];

                    for (int q = 0; q < pd; q++)
                    {
                        tgt.MemAttnOutputWGrad[h, p, q] += src.MemAttnOutputWGrad[h, p, q];
                    }

                    tgt.MemAttnOutputBGrad[h, p] += src.MemAttnOutputBGrad[h, p];
                }

                for (int b = 0; b < ntb; b++)
                {
                    tgt.TimeLogFreqGrad[h, b] += src.TimeLogFreqGrad[h, b];
                }

                for (int j = 0; j < hd; j++)
                {
                    tgt.B1Grad[h, j] += src.B1Grad[h, j];
                    tgt.W2Grad[h, j] += src.W2Grad[h, j];
                    for (int k = 0; k < mid; k++)
                    {
                        tgt.W1Grad[h, j, k] += src.W1Grad[h, j, k];
                    }
                }
            }
        }

        private void BackpropMultiStoryTextEncoder(float[,] dSH, MultimodalForwardCache cache)
        {
            int ns = cache.StoryCaches.Count;
            int ed = _config.Text.EmbeddingDim;

            for (int s = 0; s < ns; s++)
            {
                int tc = cache.StoryTokenCounts[s];
                float inv = 1.0f / tc;
                var dTH = new float[tc, ed];

                for (int t = 0; t < tc; t++)
                {
                    for (int d = 0; d < ed; d++)
                    {
                        dTH[t, d] = dSH[s, d] * inv;
                    }
                }

                BackpropTextEncoder(dTH, cache.StoryCaches[s]);
            }
        }

        private void BackpropTextEncoder(float[,] dTH, MultimodalForwardCache cache)
        {
            int ed = _config.Text.EmbeddingDim;
            int nh = _config.Text.NumHeads;
            int hd = ed / nh;
            float s = 1.0f / MathF.Sqrt(hd);

            var dX = dTH;

            for (int layer = _config.Text.NumLayers - 1; layer >= 0; layer--)
            {
                var b = _model.TextBlocks[layer];
                var ag = _gradients.TextAttnGrads[layer];

                // FFN sublayer backward
                var l2c = cache.TextLN2Caches[layer];
                var (dFR, dG2, dB2) = _accel.LayerNormBackward(dX, l2c.Normalized, b.LN2Gamma, l2c.Input, l2c.Mean, l2c.Variance);

                var lg2 = _gradients.TextLN2Grads[layer];
                _accel.VectorAccumulate(lg2.GammaGrad, dG2);
                _accel.VectorAccumulate(lg2.BetaGrad, dB2);

                int sl = dFR.GetLength(0);
                var dFFNInput = new float[sl, ed];

                for (int i = 0; i < sl; i++)
                {
                    var rowGrad = new float[ed];

                    for (int j = 0; j < ed; j++)
                    {
                        rowGrad[j] = dFR[i, j];
                    }


                    var inputRow = new float[ed];

                    for (int j = 0; j < ed; j++)
                    {
                        inputRow[j] = cache.TextFFNInputs[layer][i, j];
                    }

                    b.FeedForwardNetwork.ForwardPassOnly(inputRow);

                    var di = b.FeedForwardNetwork.ComputeInputGradient(
                        rowGrad,
                        _textFFNWeightGrads[layer],
                        _textFFNBiasGrads[layer]
                    );

                    for (int j = 0; j < ed; j++)
                    {
                        dFFNInput[i, j] = di[j];
                    }
                }

                _accel.MatrixAddInPlace(dFFNInput, dFR);

                // Self-attention sublayer backward
                var l1c = cache.TextLN1Caches[layer];

                var (dAR, dG1, dB1) = _accel.LayerNormBackward(dFFNInput, l1c.Normalized, b.LN1Gamma, l1c.Input, l1c.Mean, l1c.Variance);

                var lg1 = _gradients.TextLN1Grads[layer];

                _accel.VectorAccumulate(lg1.GammaGrad, dG1);
                _accel.VectorAccumulate(lg1.BetaGrad, dB1);

                var ac = cache.TextAttentionCaches[layer];
                var dC = new float[sl, ed];

                _accel.BackpropLinearProjection(ac.AttentionOutput, dAR, b.Attention.WO, ag.WO_Grad, ag.BiasO_Grad, dC);

                var (dQ, dK, dV) = _accel.MultiHeadAttentionBackward(ac.Q, ac.K, ac.V, dC, nh, s, _config.Text.UseDecoderOnly);

                RotaryPositionEmbedding.ApplyBackwardInPlace(dQ, dK, nh);

                var dI = new float[sl, ed];

                _accel.BackpropLinearProjection(ac.Input, dQ, b.Attention.WQ, ag.WQ_Grad, ag.BiasQ_Grad, dI);
                _accel.BackpropLinearProjection(ac.Input, dK, b.Attention.WK, ag.WK_Grad, ag.BiasK_Grad, dI);
                _accel.BackpropLinearProjection(ac.Input, dV, b.Attention.WV, ag.WV_Grad, ag.BiasV_Grad, dI);

                // Residual: dI += dAR  - no new matrix alloc
                _accel.MatrixAddInPlace(dI, dAR);

                dX = dI;
            }

            // Token embedding scatter - via accelerator instead of manual loop
            _accel.AccumulateTokenEmbeddingGrad(_gradients.TextEmbeddingGrad, dX, cache.TextTokenIds, dX.GetLength(0), ed);
        }

        private void UpdateAllParameters(float lr)
        {
            if (!_config.Text.Freeze)
            {
                _accel.MatrixUpdate(_model.TextTokenEmbedding, _gradients.TextEmbeddingGrad, lr);

                for (int i = 0; i < _config.Text.NumLayers; i++)
                {
                    var b = _model.TextBlocks[i];
                    var ag = _gradients.TextAttnGrads[i];

                    _accel.MatrixUpdate(b.Attention.WQ, ag.WQ_Grad, lr);
                    _accel.MatrixUpdate(b.Attention.WK, ag.WK_Grad, lr);
                    _accel.MatrixUpdate(b.Attention.WV, ag.WV_Grad, lr);
                    _accel.MatrixUpdate(b.Attention.WO, ag.WO_Grad, lr);

                    _accel.VectorUpdate(b.Attention.BiasQ, ag.BiasQ_Grad, lr);
                    _accel.VectorUpdate(b.Attention.BiasK, ag.BiasK_Grad, lr);
                    _accel.VectorUpdate(b.Attention.BiasV, ag.BiasV_Grad, lr);
                    _accel.VectorUpdate(b.Attention.BiasO, ag.BiasO_Grad, lr);

                    var l1 = _gradients.TextLN1Grads[i];
                    var l2 = _gradients.TextLN2Grads[i];
                    _accel.VectorUpdate(b.LN1Gamma, l1.GammaGrad, lr);
                    _accel.VectorUpdate(b.LN1Beta, l1.BetaGrad, lr);
                    _accel.VectorUpdate(b.LN2Gamma, l2.GammaGrad, lr);
                    _accel.VectorUpdate(b.LN2Beta, l2.BetaGrad, lr);

                    b.FeedForwardNetwork.ApplyExternalGradients(_textFFNWeightGrads[i], _textFFNBiasGrads[i], lr);
                }
            }

            _accel.MatrixUpdate(_model.PriceInputProjection, _gradients.PriceInputProjectionGrad, lr);
            _accel.VectorUpdate(_model.PriceInputProjectionBias, _gradients.PriceInputProjectionBiasGrad, lr);

            for (int i = 0; i < _config.Price.NumLayers; i++)
            {
                var b = _model.PriceBlocks[i];
                var bg = _gradients.PriceBlockGrads[i];

                foreach (var (attn, ag) in new[] { (b.SelfAttention, bg.SelfAttnGrads), (b.CrossAttention, bg.CrossAttnGrads) })
                {
                    _accel.MatrixUpdate(attn.WQ, ag.WQ_Grad, lr);
                    _accel.MatrixUpdate(attn.WK, ag.WK_Grad, lr);
                    _accel.MatrixUpdate(attn.WV, ag.WV_Grad, lr);
                    _accel.MatrixUpdate(attn.WO, ag.WO_Grad, lr);

                    _accel.VectorUpdate(attn.BiasQ, ag.BiasQ_Grad, lr);
                    _accel.VectorUpdate(attn.BiasK, ag.BiasK_Grad, lr);
                    _accel.VectorUpdate(attn.BiasV, ag.BiasV_Grad, lr);
                    _accel.VectorUpdate(attn.BiasO, ag.BiasO_Grad, lr);
                }

                if (bg.DecayGrads != null) UpdateDecayNetwork(b.DecayNetwork, bg.DecayGrads, lr);

                _accel.VectorUpdate(b.LNSelfGamma, bg.LNSelfGrads.GammaGrad, lr);
                _accel.VectorUpdate(b.LNSelfBeta, bg.LNSelfGrads.BetaGrad, lr);
                _accel.VectorUpdate(b.LnCrossGamma, bg.LNCrossGrads.GammaGrad, lr);
                _accel.VectorUpdate(b.LnCrossBeta, bg.LNCrossGrads.BetaGrad, lr);
                _accel.VectorUpdate(b.LNFFNGamma, bg.LNFFNGrads.GammaGrad, lr);
                _accel.VectorUpdate(b.LNFFNBeta, bg.LNFFNGrads.BetaGrad, lr);

                b.FeedForwardNetwork.ApplyExternalGradients(_priceFFNWeightGrads[i], _priceFFNBiasGrads[i], lr);
            }

            _accel.MatrixUpdate(_model.OutputProjection, _gradients.OutputProjectionGrad, lr);
            _accel.VectorUpdate(_model.OutputBias, _gradients.OutputBiasGrad, lr);

            if (_config.Output.UseConfidenceHead)
            {
                _accel.MatrixUpdate(_model.ConfidenceProjection, _gradients.ConfidenceProjectionGrad, lr);
                _accel.VectorUpdate(_model.ConfidenceBias, _gradients.ConfidenceBiasGrad, lr);
            }

            int cteEd = _config.Price.EmbeddingDim;

            for (int t = 0; t < 2; t++)
            {
                for (int d = 0; d < cteEd; d++)
                {
                    _model.ContextTypeEmbedding[t, d] -= lr * _gradients.ContextTypeEmbeddingGrad[t, d];
                }
            }
        }

        private void UpdateDecayNetwork(ContentAwareDecayNetwork net, ContentAwareDecayGradients g, float lr)
        {
            int nh = net.NumHeads;
            int cd = net.ContentDim;
            int pd = net.ProjectionDim;
            int hd = net.HiddenDim;
            int mid = net.MLPInputDim;
            int ntb = net.NumTimeBases;
            int rawDim = net.TimeRawDim;

            for (int h = 0; h < nh; h++)
            {
                net.LogBaseDecayRate[h] -= lr * g.LogBaseDecayRateGrad[h];
                net.B2[h] -= lr * g.B2Grad[h];

                for (int p = 0; p < pd; p++)
                {
                    for (int d = 0; d < cd; d++)
                    {
                        net.QueryProjection[h, p, d] -= lr * g.QueryProjectionGrad[h, p, d];
                        net.KeyProjection[h, p, d] -= lr * g.KeyProjectionGrad[h, p, d];
                    }
                    net.QueryProjectionBias[h, p] -= lr * g.QueryProjectionBiasGrad[h, p];
                    net.KeyProjectionBias[h, p] -= lr * g.KeyProjectionBiasGrad[h, p];

                    for (int r = 0; r < rawDim; r++) net.TimeProj[h, p, r] -= lr * g.TimeProjGrad[h, p, r];
                    net.TimeProjBias[h, p] -= lr * g.TimeProjBiasGrad[h, p];

                    for (int q = 0; q < pd; q++) net.MemAttnOutputW[h, p, q] -= lr * g.MemAttnOutputWGrad[h, p, q];
                    net.MemAttnOutputB[h, p] -= lr * g.MemAttnOutputBGrad[h, p];
                }

                for (int b = 0; b < ntb; b++) net.TimeLogFreq[h, b] -= lr * g.TimeLogFreqGrad[h, b];

                for (int j = 0; j < hd; j++)
                {
                    net.B1[h, j] -= lr * g.B1Grad[h, j];
                    net.W2[h, j] -= lr * g.W2Grad[h, j];
                    for (int k = 0; k < mid; k++) net.W1[h, j, k] -= lr * g.W1Grad[h, j, k];
                }
            }
        }

        private void ZeroAllGradients()
        {
            _gradients.Zero();

            for (int i = 0; i < _config.Text.NumLayers; i++)
            {
                foreach (var w in _textFFNWeightGrads[i]) _accel.ZeroMatrix(w);
                // ZeroVector via accelerator - consistent with matrix zeroing above
                foreach (var b in _textFFNBiasGrads[i]) _accel.ZeroVector(b);
            }

            for (int i = 0; i < _config.Price.NumLayers; i++)
            {
                foreach (var w in _priceFFNWeightGrads[i]) _accel.ZeroMatrix(w);
                foreach (var b in _priceFFNBiasGrads[i]) _accel.ZeroVector(b);
            }
        }

        private void ClipGradients(float th)
        {
            float n = ComputeGradientNorm();
            if (n > th) ScaleAllGradients(th / n);
        }

        private float ComputeGradientNorm()
        {
            float sum = 0;

            if (!_config.Text.Freeze)
            {
                sum += _accel.MatrixSquaredNorm(_gradients.TextEmbeddingGrad);

                foreach (var g in _gradients.TextAttnGrads)
                {
                    sum += _accel.MatrixSquaredNorm(g.WQ_Grad);
                    sum += _accel.MatrixSquaredNorm(g.WK_Grad);
                    sum += _accel.MatrixSquaredNorm(g.WV_Grad);
                    sum += _accel.MatrixSquaredNorm(g.WO_Grad);
                    sum += _accel.VectorSquaredNorm(g.BiasQ_Grad);
                    sum += _accel.VectorSquaredNorm(g.BiasK_Grad);
                    sum += _accel.VectorSquaredNorm(g.BiasV_Grad);
                    sum += _accel.VectorSquaredNorm(g.BiasO_Grad);
                }

                foreach (var g in _gradients.TextLN1Grads)
                {
                    sum += _accel.VectorSquaredNorm(g.GammaGrad);
                    sum += _accel.VectorSquaredNorm(g.BetaGrad);
                }

                foreach (var g in _gradients.TextLN2Grads)
                {
                    sum += _accel.VectorSquaredNorm(g.GammaGrad);
                    sum += _accel.VectorSquaredNorm(g.BetaGrad);
                }

                for (int i = 0; i < _config.Text.NumLayers; i++)
                {
                    foreach (var w in _textFFNWeightGrads[i]) sum += _accel.MatrixSquaredNorm(w);
                    foreach (var b in _textFFNBiasGrads[i]) sum += _accel.VectorSquaredNorm(b);
                }
            }

            sum += _accel.MatrixSquaredNorm(_gradients.PriceInputProjectionGrad);
            sum += _accel.VectorSquaredNorm(_gradients.PriceInputProjectionBiasGrad);

            foreach (var bg in _gradients.PriceBlockGrads)
            {
                foreach (var g in new[] { bg.SelfAttnGrads, bg.CrossAttnGrads })
                {
                    sum += _accel.MatrixSquaredNorm(g.WQ_Grad);
                    sum += _accel.MatrixSquaredNorm(g.WK_Grad);
                    sum += _accel.MatrixSquaredNorm(g.WV_Grad);
                    sum += _accel.MatrixSquaredNorm(g.WO_Grad);
                    sum += _accel.VectorSquaredNorm(g.BiasQ_Grad);
                    sum += _accel.VectorSquaredNorm(g.BiasK_Grad);
                    sum += _accel.VectorSquaredNorm(g.BiasV_Grad);
                    sum += _accel.VectorSquaredNorm(g.BiasO_Grad);
                }

                sum += _accel.VectorSquaredNorm(bg.LNSelfGrads.GammaGrad);
                sum += _accel.VectorSquaredNorm(bg.LNSelfGrads.BetaGrad);
                sum += _accel.VectorSquaredNorm(bg.LNCrossGrads.GammaGrad);
                sum += _accel.VectorSquaredNorm(bg.LNCrossGrads.BetaGrad);
                sum += _accel.VectorSquaredNorm(bg.LNFFNGrads.GammaGrad);
                sum += _accel.VectorSquaredNorm(bg.LNFFNGrads.BetaGrad);

                if (bg.DecayGrads != null) sum += DecayGradNorm(bg.DecayGrads);
            }

            for (int i = 0; i < _config.Price.NumLayers; i++)
            {
                foreach (var w in _priceFFNWeightGrads[i]) sum += _accel.MatrixSquaredNorm(w);
                foreach (var b in _priceFFNBiasGrads[i]) sum += _accel.VectorSquaredNorm(b);
            }

            sum += _accel.MatrixSquaredNorm(_gradients.OutputProjectionGrad);
            sum += _accel.VectorSquaredNorm(_gradients.OutputBiasGrad);

            if (_config.Output.UseConfidenceHead)
            {
                sum += _accel.MatrixSquaredNorm(_gradients.ConfidenceProjectionGrad);
                sum += _accel.VectorSquaredNorm(_gradients.ConfidenceBiasGrad);
            }

            sum += _accel.MatrixSquaredNorm(_gradients.ContextTypeEmbeddingGrad);

            return MathF.Sqrt(sum);
        }

        private float DecayGradNorm(ContentAwareDecayGradients g)
        {
            // All 3-D tensors now go through MatrixSquaredNorm3D on the accelerator;
            // 2-D tensors use the existing MatrixSquaredNorm; vectors use VectorSquaredNorm.
            return _accel.VectorSquaredNorm(g.LogBaseDecayRateGrad)
                 + _accel.VectorSquaredNorm(g.B2Grad)
                 + _accel.MatrixSquaredNorm3D(g.QueryProjectionGrad)
                 + _accel.MatrixSquaredNorm(g.QueryProjectionBiasGrad)
                 + _accel.MatrixSquaredNorm3D(g.KeyProjectionGrad)
                 + _accel.MatrixSquaredNorm(g.KeyProjectionBiasGrad)
                 + _accel.MatrixSquaredNorm(g.TimeLogFreqGrad)
                 + _accel.MatrixSquaredNorm3D(g.TimeProjGrad)
                 + _accel.MatrixSquaredNorm(g.TimeProjBiasGrad)
                 + _accel.MatrixSquaredNorm3D(g.MemAttnOutputWGrad)
                 + _accel.MatrixSquaredNorm(g.MemAttnOutputBGrad)
                 + _accel.MatrixSquaredNorm3D(g.W1Grad)
                 + _accel.MatrixSquaredNorm(g.B1Grad)
                 + _accel.MatrixSquaredNorm(g.W2Grad);
        }

        private void ScaleAllGradients(float s)
        {
            if (!_config.Text.Freeze)
            {
                _accel.MatrixScaleInPlace(_gradients.TextEmbeddingGrad, s);

                foreach (var g in _gradients.TextAttnGrads)
                {
                    _accel.MatrixScaleInPlace(g.WQ_Grad, s);
                    _accel.MatrixScaleInPlace(g.WK_Grad, s);
                    _accel.MatrixScaleInPlace(g.WV_Grad, s);
                    _accel.MatrixScaleInPlace(g.WO_Grad, s);
                    _accel.VectorScaleInPlace(g.BiasQ_Grad, s);
                    _accel.VectorScaleInPlace(g.BiasK_Grad, s);
                    _accel.VectorScaleInPlace(g.BiasV_Grad, s);
                    _accel.VectorScaleInPlace(g.BiasO_Grad, s);
                }

                foreach (var g in _gradients.TextLN1Grads)
                {
                    _accel.VectorScaleInPlace(g.GammaGrad, s);
                    _accel.VectorScaleInPlace(g.BetaGrad, s);
                }

                foreach (var g in _gradients.TextLN2Grads)
                {
                    _accel.VectorScaleInPlace(g.GammaGrad, s);
                    _accel.VectorScaleInPlace(g.BetaGrad, s);
                }

                for (int i = 0; i < _config.Text.NumLayers; i++)
                {
                    foreach (var w in _textFFNWeightGrads[i]) _accel.MatrixScaleInPlace(w, s);
                    foreach (var b in _textFFNBiasGrads[i]) _accel.VectorScaleInPlace(b, s);
                }
            }

            _accel.MatrixScaleInPlace(_gradients.PriceInputProjectionGrad, s);
            _accel.VectorScaleInPlace(_gradients.PriceInputProjectionBiasGrad, s);

            foreach (var bg in _gradients.PriceBlockGrads)
            {
                foreach (var g in new[] { bg.SelfAttnGrads, bg.CrossAttnGrads })
                {
                    _accel.MatrixScaleInPlace(g.WQ_Grad, s);
                    _accel.MatrixScaleInPlace(g.WK_Grad, s);
                    _accel.MatrixScaleInPlace(g.WV_Grad, s);
                    _accel.MatrixScaleInPlace(g.WO_Grad, s);
                    _accel.VectorScaleInPlace(g.BiasQ_Grad, s);
                    _accel.VectorScaleInPlace(g.BiasK_Grad, s);
                    _accel.VectorScaleInPlace(g.BiasV_Grad, s);
                    _accel.VectorScaleInPlace(g.BiasO_Grad, s);
                }

                _accel.VectorScaleInPlace(bg.LNSelfGrads.GammaGrad, s);
                _accel.VectorScaleInPlace(bg.LNSelfGrads.BetaGrad, s);
                _accel.VectorScaleInPlace(bg.LNCrossGrads.GammaGrad, s);
                _accel.VectorScaleInPlace(bg.LNCrossGrads.BetaGrad, s);
                _accel.VectorScaleInPlace(bg.LNFFNGrads.GammaGrad, s);
                _accel.VectorScaleInPlace(bg.LNFFNGrads.BetaGrad, s);

                // ScaleDecayGrads - all 3-D tensor loops replaced by Matrix3DScaleInPlace
                if (bg.DecayGrads != null) ScaleDecayGrads(bg.DecayGrads, s);
            }

            for (int i = 0; i < _config.Price.NumLayers; i++)
            {
                foreach (var w in _priceFFNWeightGrads[i]) _accel.MatrixScaleInPlace(w, s);
                foreach (var b in _priceFFNBiasGrads[i]) _accel.VectorScaleInPlace(b, s);
            }

            _accel.MatrixScaleInPlace(_gradients.OutputProjectionGrad, s);
            _accel.VectorScaleInPlace(_gradients.OutputBiasGrad, s);

            if (_config.Output.UseConfidenceHead)
            {
                _accel.MatrixScaleInPlace(_gradients.ConfidenceProjectionGrad, s);
                _accel.VectorScaleInPlace(_gradients.ConfidenceBiasGrad, s);
            }

            _accel.MatrixScaleInPlace(_gradients.ContextTypeEmbeddingGrad, s);
        }

        private void ScaleDecayGrads(ContentAwareDecayGradients g, float s)
        {
            // 1-D arrays
            _accel.VectorScaleInPlace(g.LogBaseDecayRateGrad, s);
            _accel.VectorScaleInPlace(g.B2Grad, s);

            // 2-D arrays
            _accel.MatrixScaleInPlace(g.QueryProjectionBiasGrad, s);
            _accel.MatrixScaleInPlace(g.KeyProjectionBiasGrad, s);
            _accel.MatrixScaleInPlace(g.TimeLogFreqGrad, s);
            _accel.MatrixScaleInPlace(g.TimeProjBiasGrad, s);
            _accel.MatrixScaleInPlace(g.MemAttnOutputBGrad, s);
            _accel.MatrixScaleInPlace(g.B1Grad, s);
            _accel.MatrixScaleInPlace(g.W2Grad, s);

            // 3-D tensors - replaced manual triple-loops with Matrix3DScaleInPlace
            _accel.Matrix3DScaleInPlace(g.QueryProjectionGrad, s);
            _accel.Matrix3DScaleInPlace(g.KeyProjectionGrad, s);
            _accel.Matrix3DScaleInPlace(g.TimeProjGrad, s);
            _accel.Matrix3DScaleInPlace(g.MemAttnOutputWGrad, s);
            _accel.Matrix3DScaleInPlace(g.W1Grad, s);
        }

        private float[,] SliceRows(float[,] m, int s, int e)
        {
            int c = m.GetLength(1);
            int n = e - s;
            var r = new float[n, c];
            for (int i = 0; i < n; i++)
            {
                for (int j = 0; j < c; j++)
                {
                    r[i, j] = m[s + i, j];
                }
            }
            return r;
        }

        /// <summary>
        /// Linear warm-up followed by cosine decay.
        /// Warm-up (ep &lt; warmUpEpochs): ramps from baseLR/warmUpEpochs to baseLR.
        /// Cosine decay: smoothly anneals to minLR (= baseLR * 0.01).
        /// </summary>
        private float ComputeLearningRate(int currentEpoch, int totalEpochs)
        {
            float baseLR = _trainConfig.LearningRate;
            int warmUpEpochs = Math.Clamp((int)(totalEpochs * 0.1f), 1, 5);
            float minLR = baseLR * 0.01f;

            if (currentEpoch < warmUpEpochs)
            {
                return baseLR * (currentEpoch + 1f) / warmUpEpochs;
            }

            int decayEpochs = Math.Max(1, totalEpochs - warmUpEpochs);
            float progress = (float)(currentEpoch - warmUpEpochs) / decayEpochs;
            return minLR + 0.5f * (baseLR - minLR) * (1f + MathF.Cos(MathF.PI * progress));
        }
    }
}