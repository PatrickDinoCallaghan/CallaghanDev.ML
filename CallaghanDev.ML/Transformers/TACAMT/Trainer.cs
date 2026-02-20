using CallaghanDev.ML.AccelerationManagers;
using CallaghanDev.ML.Transformers.Cache;
using CallaghanDev.ML.Transformers.CrossAttentionMultimodal;
using System;
using System.Collections.Generic;
using System.Linq;

namespace CallaghanDev.ML.Transformers.TACAMT
{
    public class Trainer
    {
        private readonly Model _model;
        private readonly Config _config;
        private readonly MultimodalTrainingConfig _trainConfig;
        private readonly Gradients _gradients;
        private readonly IAccelerationManager _accel;
        private readonly Random _random;
        private readonly List<List<float[,]>> _textFFNWeightGrads;
        private readonly List<List<float[]>> _textFFNBiasGrads;
        private readonly List<List<float[,]>> _priceFFNWeightGrads;
        private readonly List<List<float[]>> _priceFFNBiasGrads;
        private readonly Random _dropoutRng;

        public Trainer(Model model, MultimodalTrainingConfig trainConfig)
        {
            _model = model; _config = model.Config; _trainConfig = trainConfig;
            _gradients = new Gradients(_config); _accel = model.AccelerationManager;
            _random = new Random();
            _dropoutRng = new Random(_random.Next());
            _textFFNWeightGrads = new List<List<float[,]>>(); _textFFNBiasGrads = new List<List<float[]>>();
            for (int i = 0; i < _config.TextNumLayers; i++) { var (w, b) = model.TextBlocks[i].FeedForwardNetwork.CreateGradientStorage(); _textFFNWeightGrads.Add(w); _textFFNBiasGrads.Add(b); }
            _priceFFNWeightGrads = new List<List<float[,]>>(); _priceFFNBiasGrads = new List<List<float[]>>();
            for (int i = 0; i < _config.PriceNumLayers; i++) { var (w, b) = model.PriceBlocks[i].FeedForwardNetwork.CreateGradientStorage(); _priceFFNWeightGrads.Add(w); _priceFFNBiasGrads.Add(b); }
        }

        //TODO: Clean this up
        public void Train(NewsStory[][] storiesPerSample, float[][,] priceInputs, float[][,] priceTargets, float[][] confTargets = null)
        {
            int n = storiesPerSample.Length;
            // float lr = _trainConfig.LearningRate;

            int totalEpochs = _trainConfig.Epochs;

            for (int ep = 0; ep < _trainConfig.Epochs; ep++)
            {
                float lr = ComputeLearningRate(ep, totalEpochs);

                if (_trainConfig.Verbose) 
                { 
                    Console.WriteLine($"\n=== Epoch {ep + 1}/{_trainConfig.Epochs} ==="); 
                }

                var sh = Enumerable.Range(0, n).OrderBy(_ => _random.Next()).ToArray(); float el = 0; int nb = 0;

                for (int i = 0; i < sh.Length; i += _trainConfig.BatchSize)
                {
                    int bs = Math.Min(_trainConfig.BatchSize, sh.Length - i);

                    float bl = TrainBatch(sh.Skip(i).Take(bs).ToArray(), storiesPerSample, priceInputs, priceTargets, confTargets, lr);

                    el += bl; nb++;

                    if (_trainConfig.Verbose && nb % 10 == 0) 
                    { 
                        Console.WriteLine($"  Batch {nb}: Loss = {bl:F6}"); 
                    }
                }
                if (_trainConfig.Verbose)
                { 
                    Console.WriteLine($"  Epoch {ep + 1} Average Loss: {(nb > 0 ? el / nb : 0):F6}");
                }
                if (_trainConfig.UseLearningRateDecay) 
                { 
                    lr *= _trainConfig.LearningRateDecay; 
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
            int n = stories.Length;
            int totalEpochs = _trainConfig.Epochs;
            int embDim = _config.PriceEmbeddingDim;

            for (int ep = 0; ep < _trainConfig.Epochs; ep++)
            {

                float lr = ComputeLearningRate(ep, totalEpochs);


                if (_trainConfig.Verbose) 
                { 
                    Console.WriteLine($"\n=== Epoch {ep + 1}/{_trainConfig.Epochs} (Sequential) ==="); 
                }

                // Clear memory at the start of each epoch so the model re-learns
                // the full temporal sequence from scratch each pass
                _model.ClearAllMemory();

                float epochLoss = 0; int validCount = 0;

                for (int idx = 0; idx < n; idx++)
                {
                    var ps = priceInputs[idx];
                    int sl = ps.GetLength(0);
                    if (sl < 2) continue;

                    double currentTs = timestamps[idx];

                    try
                    {
                        //Assemble context from persistent memory (same as PredictWithMemory) ===
                        var ctxH = new List<float[]>();
                        var ctxT = new List<float>();

                        // Add accumulated news memory
                        if (_model.NewsMemory != null)
                        {
                            foreach (var e in _model.NewsMemory)
                            {
                                ctxH.Add(e.HiddenState);
                                ctxT.Add(-(float)((currentTs - e.AbsoluteTimestamp) / timeUnitsPerPosition));
                            }
                        }

                        // Encode and add current news stories
                        float[,] newsSH = null;
                        float[] newsTimes = null;
                        NewsStory[] currentStories = stories[idx];
                        if (currentStories != null && currentStories.Length > 0)
                        {
                            // Encode stories (detached — we'll get fresh gradients through the cached path)
                            for (int i = 0; i < currentStories.Length; i++)
                            {
                                // Use arrival time relative to price sequence
                                ctxT.Add(currentStories[i].ArrivalTime);
                            }
                        }

                        // Add accumulated price memory
                        if (_model.PriceMemory != null)
                        {
                            foreach (var e in _model.PriceMemory)
                            {
                                ctxH.Add(e.HiddenState);
                                ctxT.Add(-(float)((currentTs - e.AbsoluteTimestamp) / timeUnitsPerPosition));
                            }
                        }

                        //Forward with cache (training mode) ===
                        var inp = SliceRows(ps, 0, sl - 1);
                        var tgt = SliceRows(priceTargets[idx], 1, sl);
                        float[] ct = confTargets?[idx] != null ? confTargets[idx].Skip(1).Take(sl - 1).ToArray() : null;

                        // Build combined context matrix from memory entries
                        int newsMemCount = _model.NewsMemory?.Count ?? 0;
                        int priceMemCount = _model.PriceMemory?.Count ?? 0;
                        int numNewStories = currentStories?.Length ?? 0;

                        var cache = new MultimodalForwardCache(_config.TextNumLayers, _config.PriceNumLayers);

                        // Use ForwardWithPriceContextAndCache if we have memory, otherwise standard forward
                        float[,] priceCtxHidden = null;
                        float[] priceCtxTimes = null;

                        if (priceMemCount > 0)
                        {
                            priceCtxHidden = new float[priceMemCount, embDim];
                            priceCtxTimes = new float[priceMemCount];

                            for (int i = 0; i < priceMemCount; i++)
                            {
                                var entry = _model.PriceMemory[i];

                                for (int d = 0; d < embDim; d++)
                                {
                                    priceCtxHidden[i, d] = entry.HiddenState[d];
                                }

                                priceCtxTimes[i] = -(float)((currentTs - entry.AbsoluteTimestamp) / timeUnitsPerPosition);
                            }
                        }

                        // Adjust story arrival times relative to current window if we have news memory
                        NewsStory[] adjustedStories = null;
                        if (currentStories != null && currentStories.Length > 0)
                        {
                            adjustedStories = currentStories;
                        }

                        // Add news memory entries as additional context alongside current stories
                        // We need to combine news memory + new stories + price memory
                        float[,] newsMemHidden = null;
                        float[] newsMemTimes = null;
                        if (newsMemCount > 0)
                        {
                            newsMemHidden = new float[newsMemCount, embDim];
                            newsMemTimes = new float[newsMemCount];
                            for (int i = 0; i < newsMemCount; i++)
                            {
                                var entry = _model.NewsMemory[i];
                                for (int d = 0; d < embDim; d++)
                                {
                                    newsMemHidden[i, d] = entry.HiddenState[d];
                                }
                                newsMemTimes[i] = -(float)((currentTs - entry.AbsoluteTimestamp) / timeUnitsPerPosition);
                            }
                        }

                        float[,] pred, conf;

                        if (newsMemCount > 0 || priceMemCount > 0)
                        {
                            // Encode current stories if any
                            float[,] freshNewsHidden = null;
                            float[] freshNewsTimes = null;
                            int freshNewsCount = 0;

                            if (adjustedStories != null && adjustedStories.Length > 0)
                            {
                                (freshNewsHidden, freshNewsTimes) = _model.EncodeStoriesWithCache(adjustedStories, cache);

                                freshNewsCount = freshNewsHidden.GetLength(0);
                            }
                            else
                            {
                                cache.TextFinalHidden = null; cache.TextTokenIds = null;
                            }

                            // Assemble full combined context: [newsMemory] + [freshNews] + [priceMemory]
                            int totalCtx = newsMemCount + freshNewsCount + priceMemCount;
                            float[,] combinedHidden = null;
                            float[] combinedTimes = null;

                            if (totalCtx > 0)
                            {
                                combinedHidden = new float[totalCtx, embDim];
                                combinedTimes = new float[totalCtx];
                                int ci = 0;

                                for (int i = 0; i < newsMemCount; i++)
                                {
                                    for (int d = 0; d < embDim; d++) 
                                    { 
                                        combinedHidden[ci, d] = newsMemHidden[i, d];
                                    }

                                    combinedTimes[ci] = newsMemTimes[i]; ci++;
                                }
                                for (int i = 0; i < freshNewsCount; i++)
                                {
                                    for (int d = 0; d < embDim; d++) 
                                    { 
                                        combinedHidden[ci, d] = freshNewsHidden[i, d]; 
                                    }
                                    combinedTimes[ci] = freshNewsTimes[i]; ci++;
                                }
                                for (int i = 0; i < priceMemCount; i++)
                                {
                                    for (int d = 0; d < embDim; d++) 
                                    { 
                                        combinedHidden[ci, d] = priceCtxHidden[i, d]; 
                                    }
                                    combinedTimes[ci] = priceCtxTimes[i]; ci++;
                                }

                                // Apply context type embeddings
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
                            }

                            cache.NumNewsContext = newsMemCount + freshNewsCount;
                            cache.NumPriceContext = priceMemCount;
                            cache.PriceContextHidden = priceCtxHidden;
                            cache.TextFinalHidden = combinedHidden;
                            cache.StoryArrivalTimes = combinedTimes;

                            var priceHidden = _model.ForwardPriceDecoderWithCache(inp, combinedHidden, combinedTimes, cache, isTraining: true, dropoutRng: _dropoutRng);
                            cache.PriceFinalHidden = priceHidden;
                            (pred, conf) = _model.ProjectToOutput(priceHidden);
                        }
                        else
                        {
                            // No memory yet — standard forward (type embedding applied inside ForwardWithCache)
                            (pred, conf) = _model.ForwardWithCache(adjustedStories, inp, cache, isTraining: true, dropoutRng: _dropoutRng);
                        }

                        // Backward pass
                        ZeroAllGradients();

                        float loss = BackwardPass(pred, conf, tgt, ct, cache);

                        if (float.IsNaN(loss) || float.IsInfinity(loss))
                        {
                            ZeroAllGradients();
                            continue;
                        }
                        if (_trainConfig.UseGradientClipping)
                        {
                            ClipGradients(_trainConfig.GradientClipThreshold);

                        }

                        UpdateAllParameters(lr);

                        epochLoss += loss; validCount++;

                        // Update persistent memory (same as PredictWithMemory)
                        // Store new news hidden states
                        if (currentStories != null && currentStories.Length > 0)
                        {
                            float[,] encodedNews;
                            float[] encodedTimes;
                            (encodedNews, encodedTimes) = _model.EncodeStories(currentStories);
                            for (int i = 0; i < currentStories.Length; i++)
                            {
                                var hv = new float[embDim];
                                for (int d = 0; d < embDim; d++) hv[d] = encodedNews[i, d];
                                _model.NewsMemory.Add(new NewsMemoryEntry
                                {
                                    HiddenState = hv,
                                    AbsoluteTimestamp = currentTs + currentStories[i].ArrivalTime * timeUnitsPerPosition
                                });
                            }
                        }

                        // Store price hidden states from forward pass
                        int priceSeqLen = cache.PriceFinalHidden.GetLength(0);
                        for (int t = 0; t < priceSeqLen; t++)
                        {
                            var pv = new float[embDim];
                            for (int d = 0; d < embDim; d++) pv[d] = cache.PriceFinalHidden[t, d];
                            _model.PriceMemory.Add(new PriceMemoryEntry
                            {
                                HiddenState = pv,
                                AbsoluteTimestamp = currentTs + t * timeUnitsPerPosition
                            });
                        }

                        _model.LastPriceTimestamp = currentTs;

                        _model.PruneNewsMemory(maxNewsMemory);
                        _model.PricePruneMemory(maxPriceMemory);

                        if (_trainConfig.Verbose && validCount % 50 == 0)
                        {
                            Console.WriteLine($"  Sample {validCount}: Loss = {loss:F6}, NewsMemory = {_model.NewsMemory.Count}, PriceMemory = {_model.PriceMemory.Count}");
                        }
                    }
                    catch (Exception ex)
                    {
                        ZeroAllGradients();
                        if (_trainConfig.Verbose)
                        {
                            Console.WriteLine($"  WARNING: {ex.Message}");
                        }
                    }
                }

                if (_trainConfig.Verbose)
                {
                    Console.WriteLine($"  Epoch {ep + 1} Average Loss: {(validCount > 0 ? epochLoss / validCount : 0):F6}");
                }
                if (_trainConfig.UseLearningRateDecay) 
                { 
                    lr *= _trainConfig.LearningRateDecay;
                }
            }
        }

        private float TrainBatch(int[] bi, NewsStory[][] allS, float[][,] allP, float[][,] allT, float[][] allC, float lr)
        {
            ZeroAllGradients();

            float tl = 0;
            int vc = 0;

            int minSplitLen = _config.PriceContextMinHistoryLength + _config.PriceContextMinCurrentLength + 1;


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
                        var inp = SliceRows(ps, 0, sl - 1); var tgt = SliceRows(allT[idx], 1, sl);

                        float[] ct = allC?[idx] != null ? allC[idx].Skip(1).Take(sl - 1).ToArray() : null;

                        var cache = new MultimodalForwardCache(_config.TextNumLayers, _config.PriceNumLayers);

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
                    if (_trainConfig.Verbose)
                    {
                        Console.WriteLine($"  WARNING: {ex.Message}");
                    }
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

            int minHist = _config.PriceContextMinHistoryLength;
            int maxHist = sl - _config.PriceContextMinCurrentLength - 1;
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

            // Encode price history (DETACHED)
            var priceCtxHidden = _model.EncodePriceHistory(histPrices);

            var priceCtxTimes = new float[histLen];

            for (int t = 0; t < histLen; t++)
            {
                priceCtxTimes[t] = -(histLen - t);
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

            //  Forward with combined context (training mode: dropout active)
            // Type embeddings are applied inside ForwardWithPriceContextAndCache
            var cache = new MultimodalForwardCache(_config.TextNumLayers, _config.PriceNumLayers);
            var (pred, conf) = _model.ForwardWithPriceContextAndCache(adjustedStories, currentInput, priceCtxHidden, priceCtxTimes, cache, isTraining: true, dropoutRng: _dropoutRng);

            // Backward pass
            return BackwardPass(pred, conf, currentTarget, confTgt, cache);
        }

        #region Validation

        public float Validate(NewsStory[][] stories, float[][,] priceInputs, float[][,] priceTargets)
        {
            float totalLoss = 0f;
            int totalCount = 0;

            int minSplitLength = _config.PriceContextMinHistoryLength + _config.PriceContextMinCurrentLength + 1;

            for (int i = 0; i < stories.Length; i++)
            {
                float[,] inputPrices = priceInputs[i];
                int sequenceLength = inputPrices.GetLength(0);

                if (sequenceLength < 2)
                {
                    continue;
                }

                bool canUseContextSplit = (sequenceLength >= minSplitLength);

                if (canUseContextSplit)
                {
                    int minHistory = _config.PriceContextMinHistoryLength;
                    int maxHistory = sequenceLength - _config.PriceContextMinCurrentLength - 1;

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
                        priceContextTimes[t] = -(splitPoint - t);
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

                    var cache = new MultimodalForwardCache(_config.TextNumLayers, _config.PriceNumLayers);

                    var (predictions, _) = _model.ForwardWithPriceContextAndCache(adjustedStories, currentInput, priceContextHidden, priceContextTimes, cache, isTraining: false);

                    for (int t = 0; t < currentSeqLength; t++)
                    {
                        for (int j = 0; j < _config.OutputDim; j++)
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

                    var cache = new MultimodalForwardCache(_config.TextNumLayers, _config.PriceNumLayers);

                    var (predictions, _) = _model.ForwardWithCache(stories[i], validationInput, cache);

                    for (int t = 0; t < effectiveLength; t++)
                    {
                        for (int j = 0; j < _config.OutputDim; j++)
                        {
                            float diff = predictions[t, j] - validationTarget[t, j];

                            totalLoss += diff * diff;
                        }
                    }

                    totalCount += effectiveLength;
                }
            }

            if (totalCount > 0)
            {
                return totalLoss / (totalCount * _config.OutputDim);
            }

            return 0f;
        }

        public float Validate(int[][] texts, float[][,] pi, float[][,] pt)
        { return Validate(texts.Select(t => t != null && t.Length > 0 ? new[] { new NewsStory(t, 0f) } : null).ToArray(), pi, pt); }

        #endregion

        private float BackwardPass(float[,] pred, float[,] conf, float[,] tgt, float[] confTgt, MultimodalForwardCache cache)
        {
            int sequenceLength = pred.GetLength(0);
            int outputDim = _config.OutputDim;
            int embeddingDim = _config.PriceEmbeddingDim;

            float mse = 0f;

            float[,] dPred = new float[sequenceLength, outputDim];

            for (int t = 0; t < sequenceLength; t++)
            {
                for (int j = 0; j < outputDim; j++)
                {
                    float diff = pred[t, j] - tgt[t, j];

                    mse += diff * diff;

                    dPred[t, j] = (2f * diff) / (sequenceLength * outputDim);
                }
            }

            mse = mse / (sequenceLength * outputDim);

            float[,] dHidden = new float[sequenceLength, embeddingDim];

            for (int t = 0; t < sequenceLength; t++)
            {
                for (int v = 0; v < outputDim; v++)
                {
                    for (int e = 0; e < embeddingDim; e++)
                    {
                        _gradients.OutputProjectionGrad[v, e] +=
                            cache.PriceFinalHidden[t, e] * dPred[t, v];
                    }

                    _gradients.OutputBiasGrad[v] += dPred[t, v];
                }

                for (int e = 0; e < embeddingDim; e++)
                {
                    float grad = 0f;

                    for (int v = 0; v < outputDim; v++)
                    {
                        grad += dPred[t, v] * _model.OutputProjection[v, e];
                    }

                    dHidden[t, e] += grad;
                }
            }

            float confidenceLoss = 0f;

            if (_config.UseConfidenceHead && conf != null)
            {
                for (int t = 0; t < sequenceLength; t++)
                {
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
                            float diff = pred[t, j] - tgt[t, j];
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

                confidenceLoss /= sequenceLength;
            }

            bool hasContext = (cache.TextFinalHidden != null);

            float[,] dSharedHidden = BackpropPriceDecoder(dHidden, cache, hasContext);

            if (!_config.FreezeTextEncoder && hasContext && dSharedHidden != null && cache.StoryCaches != null)
            {
                BackpropMultiStoryTextEncoder(dSharedHidden, cache);
            }

            return mse + (_trainConfig.ConfidenceLossWeight * confidenceLoss);
        }
        private float[,] BackpropPriceDecoder(float[,] dOut, MultimodalForwardCache cache, bool hasContext)
        {
            int embeddingDim = _config.PriceEmbeddingDim;
            int numHeads = _config.PriceNumHeads;
            int headDim = embeddingDim / numHeads;
            if (embeddingDim % numHeads != 0)
                throw new ArgumentException("Embedding dim must be divisible by numHeads");

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

            for (int layer = _config.PriceNumLayers - 1; layer >= 0; layer--)
            {
                var block = _model.PriceBlocks[layer];
                var blockCache = cache.PriceBlockCaches[layer];
                var blockGrads = _gradients.PriceBlockGrads[layer];

                int sequenceLength = dX.GetLength(0);

                var (dFfnResidual, dGammaFfn, dBetaFfn) =
                    _accel.LayerNormBackward(
                        dX,
                        blockCache.LNFFNCache.Normalized,
                        block.LNFFNGamma,
                        blockCache.LNFFNCache.Input,
                        blockCache.LNFFNCache.Mean,
                        blockCache.LNFFNCache.Variance);

                for (int j = 0; j < embeddingDim; j++)
                {
                    blockGrads.LNFFNGrads.GammaGrad[j] += dGammaFfn[j];
                    blockGrads.LNFFNGrads.BetaGrad[j] += dBetaFfn[j];
                }

                float[,] dFfnInput = new float[sequenceLength, embeddingDim];

                for (int i = 0; i < sequenceLength; i++)
                {
                    float[] gradRow = new float[embeddingDim];

                    for (int j = 0; j < embeddingDim; j++)
                    {
                        gradRow[j] = dFfnResidual[i, j];
                    }

                    block.FeedForwardNetwork.ForwardPassOnly(blockCache.FFNInputRows[i]);

                    float[] inputGrad =
                        block.FeedForwardNetwork.ComputeInputGradient(
                            gradRow,
                            _priceFFNWeightGrads[layer],
                            _priceFFNBiasGrads[layer]);

                    for (int j = 0; j < embeddingDim; j++)
                    {
                        dFfnInput[i, j] = inputGrad[j];
                    }
                }

                float[,] dAfterFfn = _accel.MatrixAdd(dFfnResidual, dFfnInput);

                var (dCrossResidual, dGammaCross, dBetaCross) =
                    _accel.LayerNormBackward(
                        dAfterFfn,
                        blockCache.LNCrossCache.Normalized,
                        block.LNCrossGamma,
                        blockCache.LNCrossCache.Input,
                        blockCache.LNCrossCache.Mean,
                        blockCache.LNCrossCache.Variance);

                for (int j = 0; j < embeddingDim; j++)
                {
                    blockGrads.LNCrossGrads.GammaGrad[j] += dGammaCross[j];
                    blockGrads.LNCrossGrads.BetaGrad[j] += dBetaCross[j];
                }

                float[,] dSelfInput;

                if (hasContext && blockCache.CrossQ != null && totalContext > 0)
                {
                    var crossAttnGrads = blockGrads.CrossAttnGrads;

                    float[,] dCrossCombined = new float[sequenceLength, embeddingDim];

                    _accel.BackpropLinearProjection(
                        blockCache.CrossAttnOutput,
                        dCrossResidual,
                        block.CrossAttention.WO,
                        crossAttnGrads.WO_Grad,
                        crossAttnGrads.BiasO_Grad,
                        dCrossCombined);

                    var (dQ, dK, dV, dDecayBias) =
                        BackpropTimeDecayedAttn(
                            blockCache.CrossQ,
                            blockCache.CrossK,
                            blockCache.CrossV,
                            dCrossCombined,
                            blockCache.CrossAttentionWeights,
                            blockCache.TimeDiffs,
                            block);

                    if (blockCache.DecayCache != null && dDecayBias != null)
                    {
                        var (decayParamGrads, dQueryEmb, dKeyEmb) =
                            block.DecayNetwork.Backward(dDecayBias, blockCache.DecayCache);

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

                        for (int i = 0; i < sequenceLength; i++)
                        {
                            for (int j = 0; j < embeddingDim; j++)
                            {
                                dCrossResidual[i, j] += dQueryEmb[i, j];
                            }
                        }
                    }

                    float[,] dFromQ = new float[sequenceLength, embeddingDim];

                    _accel.BackpropLinearProjection(
                        blockCache.NormedSelf,
                        dQ,
                        block.CrossAttention.WQ,
                        crossAttnGrads.WQ_Grad,
                        crossAttnGrads.BiasQ_Grad,
                        dFromQ);

                    float[,] dCtxFromK = new float[totalContext, embeddingDim];
                    float[,] dCtxFromV = new float[totalContext, embeddingDim];

                    _accel.BackpropLinearProjection(
                        cache.TextFinalHidden,
                        dK,
                        block.CrossAttention.WK,
                        crossAttnGrads.WK_Grad,
                        crossAttnGrads.BiasK_Grad,
                        dCtxFromK);

                    _accel.BackpropLinearProjection(
                        cache.TextFinalHidden,
                        dV,
                        block.CrossAttention.WV,
                        crossAttnGrads.WV_Grad,
                        crossAttnGrads.BiasV_Grad,
                        dCtxFromV);

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
                                _gradients.ContextTypeEmbeddingGrad[1, j] +=
                                    dCtxFromK[numNewsContext + i, j] +
                                    dCtxFromV[numNewsContext + i, j];
                            }
                        }
                    }

                    dSelfInput = _accel.MatrixAdd(dCrossResidual, dFromQ);
                }
                else
                {
                    dSelfInput = dCrossResidual;
                }

                var (dSelfResidual, dGammaSelf, dBetaSelf) =
                    _accel.LayerNormBackward(
                        dSelfInput,
                        blockCache.LNSelfCache.Normalized,
                        block.LNSelfGamma,
                        blockCache.LNSelfCache.Input,
                        blockCache.LNSelfCache.Mean,
                        blockCache.LNSelfCache.Variance);

                for (int j = 0; j < embeddingDim; j++)
                {
                    blockGrads.LNSelfGrads.GammaGrad[j] += dGammaSelf[j];
                    blockGrads.LNSelfGrads.BetaGrad[j] += dBetaSelf[j];
                }

                var selfAttnGrads = blockGrads.SelfAttnGrads;

                float[,] dSelfCombined = new float[sequenceLength, embeddingDim];

                _accel.BackpropLinearProjection(
                    blockCache.SelfAttnOutput,
                    dSelfResidual,
                    block.SelfAttention.WO,
                    selfAttnGrads.WO_Grad,
                    selfAttnGrads.BiasO_Grad,
                    dSelfCombined);

                var (dQSelf, dKSelf, dVSelf) =
                    _accel.MultiHeadAttentionBackward(
                        blockCache.SelfQ,
                        blockCache.SelfK,
                        blockCache.SelfV,
                        dSelfCombined,
                        numHeads,
                        scale,
                        _config.PriceUseDecoderOnly);

                float[,] dBlockInput = new float[sequenceLength, embeddingDim];

                _accel.BackpropLinearProjection(
                    blockCache.BlockInput,
                    dQSelf,
                    block.SelfAttention.WQ,
                    selfAttnGrads.WQ_Grad,
                    selfAttnGrads.BiasQ_Grad,
                    dBlockInput);

                _accel.BackpropLinearProjection(
                    blockCache.BlockInput,
                    dKSelf,
                    block.SelfAttention.WK,
                    selfAttnGrads.WK_Grad,
                    selfAttnGrads.BiasK_Grad,
                    dBlockInput);

                _accel.BackpropLinearProjection(
                    blockCache.BlockInput,
                    dVSelf,
                    block.SelfAttention.WV,
                    selfAttnGrads.WV_Grad,
                    selfAttnGrads.BiasV_Grad,
                    dBlockInput);

                dX = _accel.MatrixAdd(dSelfResidual, dBlockInput);
            }

            BackpropPriceInputProjection(dX, cache.PriceContinuousInput);

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

        private (float[,] dQ, float[,] dK, float[,] dV, float[,,] dDecayBias) BackpropTimeDecayedAttn(float[,] Q, float[,] K, float[,] V, float[,] dOutput, float[][,] attnW, float[,] timeDiffs, TransformerBlock block)
        {
            int psl = Q.GetLength(0), tsl = K.GetLength(0), ed = _config.PriceEmbeddingDim, nh = _config.PriceNumHeads, hd = ed / nh;
            float scale = 1.0f / MathF.Sqrt(hd);
            var dQ = new float[psl, ed]; var dK = new float[tsl, ed]; var dV = new float[tsl, ed];
            float[,,] dDB = timeDiffs != null ? new float[psl, tsl, nh] : null;

            for (int h = 0; h < nh; h++)
            {

                int si = h * hd;
                var w = attnW[h];

                for (int p = 0; p < psl; p++)
                {
                    var dw = new float[tsl];

                    for (int s = 0; s < tsl; s++)
                    {
                        float d = 0;

                        for (int dd = 0; dd < hd; dd++)
                        {
                            int di = si + dd;

                            dV[s, di] += w[p, s] * dOutput[p, di];
                            d += dOutput[p, di] * V[s, di];
                        }

                        dw[s] = d;
                    }

                    float dotWD = 0;

                    for (int s = 0; s < tsl; s++)
                    {
                        dotWD += w[p, s] * dw[s];
                    }

                    var ds = new float[tsl];

                    for (int s = 0; s < tsl; s++)
                    {
                        ds[s] = w[p, s] * (dw[s] - dotWD);
                    }

                    for (int s = 0; s < tsl; s++)
                    {
                        float dsc = ds[s];

                        for (int dd = 0; dd < hd; dd++)
                        {
                            int di = si + dd;

                            dQ[p, di] += dsc * scale * K[s, di];
                            dK[s, di] += dsc * scale * Q[p, di];
                        }

                        if (dDB != null)
                        {
                            dDB[p, s, h] += dsc;
                        }
                    }
                }

            }
            return (dQ, dK, dV, dDB);
        }

        private void AccumulateDecayGrads(ContentAwareDecayGradients tgt, ContentAwareDecayGradients src)
        {
            int nh = src.B2Grad.Length,
                cd = src.QueryProjectionGrad.GetLength(2),
                pd = src.QueryProjectionGrad.GetLength(1),
                hd = src.B1Grad.GetLength(1),
                mid = src.W1Grad.GetLength(2);

            int ntb = src.TimeLogFreqGrad.GetLength(1),
                rawDim = src.TimeProjGrad.GetLength(2);

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
            int ed = _config.TextEmbeddingDim;

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
            int ed = _config.TextEmbeddingDim;
            int nh = _config.TextNumHeads;
            int hd = ed / nh;

            var dX = dTH;

            for (int layer = _config.TextNumLayers - 1; layer >= 0; layer--)
            {
                var b = _model.TextBlocks[layer];
                var ag = _gradients.TextAttnGrads[layer];

                var l2c = cache.TextLN2Caches[layer];

                var (dFR, dG2, dB2) = _accel.LayerNormBackward(dX, l2c.Normalized, b.LN2Gamma, l2c.Input, l2c.Mean, l2c.Variance);

                var lg2 = _gradients.TextLN2Grads[layer];

                for (int j = 0; j < dG2.Length; j++)
                {
                    lg2.GammaGrad[j] += dG2[j];
                    lg2.BetaGrad[j] += dB2[j];
                }

                int sl = dFR.GetLength(0);
                var dN1 = new float[sl, ed];

                for (int i = 0; i < sl; i++)
                {
                    var dr = new float[ed];

                    for (int j = 0; j < ed; j++)
                    {
                        dr[j] = dFR[i, j];
                    }

                    b.FeedForwardNetwork.ForwardPassOnly(cache.TextFFNInputs[layer][i]);

                    var di = b.FeedForwardNetwork.ComputeInputGradient(dr, _textFFNWeightGrads[layer], _textFFNBiasGrads[layer]);

                    for (int j = 0; j < ed; j++)
                    {
                        dN1[i, j] = di[j];
                    }
                }

                dN1 = _accel.MatrixAdd(dFR, dN1);

                var l1c = cache.TextLN1Caches[layer];

                var (dAR, dG1, dB1) = _accel.LayerNormBackward(dN1, l1c.Normalized, b.LN1Gamma, l1c.Input, l1c.Mean, l1c.Variance);

                var lg1 = _gradients.TextLN1Grads[layer];

                for (int j = 0; j < dG1.Length; j++)
                {
                    lg1.GammaGrad[j] += dG1[j];
                    lg1.BetaGrad[j] += dB1[j];
                }

                var ac = cache.TextAttentionCaches[layer];
                float s = 1.0f / MathF.Sqrt(hd);

                var dC = new float[sl, ed];

                _accel.BackpropLinearProjection(ac.AttentionOutput, dAR, b.Attention.WO, ag.WO_Grad, ag.BiasO_Grad, dC);

                var (dQ, dK, dV) = _accel.MultiHeadAttentionBackward(
                    ac.Q,
                    ac.K,
                    ac.V,
                    dC,
                    nh,
                    s,
                    _config.TextUseDecoderOnly
                );

                var dI = new float[sl, ed];

                _accel.BackpropLinearProjection(ac.Input, dQ, b.Attention.WQ, ag.WQ_Grad, ag.BiasQ_Grad, dI);
                _accel.BackpropLinearProjection(ac.Input, dK, b.Attention.WK, ag.WK_Grad, ag.BiasK_Grad, dI);
                _accel.BackpropLinearProjection(ac.Input, dV, b.Attention.WV, ag.WV_Grad, ag.BiasV_Grad, dI);

                dX = _accel.MatrixAdd(dAR, dI);
            }

            int tsl = dX.GetLength(0);

            for (int i = 0; i < tsl; i++)
            {
                int tid = cache.TextTokenIds[i];

                for (int j = 0; j < ed; j++)
                {
                    _gradients.TextEmbeddingGrad[tid, j] += dX[i, j];
                }
            }
        }
        private void BackpropPriceInputProjection(float[,] dX, float[,] ci)
        {
            int sl = dX.GetLength(0);
            int ed = _config.PriceEmbeddingDim;
            int id = _config.PriceInputFeatureDim;

            for (int i = 0; i < sl; i++)
            {
                for (int e = 0; e < ed; e++)
                {
                    float d = dX[i, e];

                    for (int f = 0; f < id; f++)
                    {
                        _gradients.PriceInputProjectionGrad[e, f] += d * ci[i, f];
                    }

                    _gradients.PriceInputProjectionBiasGrad[e] += d;
                }
            }
        }

        private void UpdateAllParameters(float lr)
        {
            if (!_config.FreezeTextEncoder)
            {
                _accel.MatrixUpdate(_model.TextTokenEmbedding, _gradients.TextEmbeddingGrad, lr);

                for (int i = 0; i < _config.TextNumLayers; i++)
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

                    b.FeedForwardNetwork.ApplyExternalGradients(
                        _textFFNWeightGrads[i],
                        _textFFNBiasGrads[i],
                        lr
                    );
                }
            }

            _accel.MatrixUpdate(_model.PriceInputProjection, _gradients.PriceInputProjectionGrad, lr);
            _accel.VectorUpdate(_model.PriceInputProjectionBias, _gradients.PriceInputProjectionBiasGrad, lr);

            for (int i = 0; i < _config.PriceNumLayers; i++)
            {
                var b = _model.PriceBlocks[i];
                var bg = _gradients.PriceBlockGrads[i];

                foreach (var (attn, ag) in new[]
                {
            (b.SelfAttention, bg.SelfAttnGrads),
            (b.CrossAttention, bg.CrossAttnGrads)
        })
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

                if (bg.DecayGrads != null)
                {
                    UpdateDecayNetwork(b.DecayNetwork, bg.DecayGrads, lr);
                }

                _accel.VectorUpdate(b.LNSelfGamma, bg.LNSelfGrads.GammaGrad, lr);
                _accel.VectorUpdate(b.LNSelfBeta, bg.LNSelfGrads.BetaGrad, lr);

                _accel.VectorUpdate(b.LNCrossGamma, bg.LNCrossGrads.GammaGrad, lr);
                _accel.VectorUpdate(b.LNCrossBeta, bg.LNCrossGrads.BetaGrad, lr);

                _accel.VectorUpdate(b.LNFFNGamma, bg.LNFFNGrads.GammaGrad, lr);
                _accel.VectorUpdate(b.LNFFNBeta, bg.LNFFNGrads.BetaGrad, lr);

                b.FeedForwardNetwork.ApplyExternalGradients(
                    _priceFFNWeightGrads[i],
                    _priceFFNBiasGrads[i],
                    lr
                );
            }

            _accel.MatrixUpdate(_model.OutputProjection, _gradients.OutputProjectionGrad, lr);
            _accel.VectorUpdate(_model.OutputBias, _gradients.OutputBiasGrad, lr);

            if (_config.UseConfidenceHead)
            {
                _accel.MatrixUpdate(_model.ConfidenceProjection, _gradients.ConfidenceProjectionGrad, lr);
                _accel.VectorUpdate(_model.ConfidenceBias, _gradients.ConfidenceBiasGrad, lr);
            }

            int cteEd = _config.PriceEmbeddingDim;

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
            int nh = net.NumHeads,
                cd = net.ContentDim,
                pd = net.ProjectionDim,
                hd = net.HiddenDim,
                mid = net.MLPInputDim;

            int ntb = net.NumTimeBases,
                rawDim = net.TimeRawDim;

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

                    for (int r = 0; r < rawDim; r++)
                    {
                        net.TimeProj[h, p, r] -= lr * g.TimeProjGrad[h, p, r];
                    }

                    net.TimeProjBias[h, p] -= lr * g.TimeProjBiasGrad[h, p];

                    for (int q = 0; q < pd; q++)
                    {
                        net.MemAttnOutputW[h, p, q] -= lr * g.MemAttnOutputWGrad[h, p, q];
                    }

                    net.MemAttnOutputB[h, p] -= lr * g.MemAttnOutputBGrad[h, p];
                }

                for (int b = 0; b < ntb; b++)
                {
                    net.TimeLogFreq[h, b] -= lr * g.TimeLogFreqGrad[h, b];
                }

                for (int j = 0; j < hd; j++)
                {
                    net.B1[h, j] -= lr * g.B1Grad[h, j];
                    net.W2[h, j] -= lr * g.W2Grad[h, j];

                    for (int k = 0; k < mid; k++)
                    {
                        net.W1[h, j, k] -= lr * g.W1Grad[h, j, k];
                    }
                }
            }
        }
        private void ZeroAllGradients()
        {
            _gradients.Zero();

            for (int i = 0; i < _config.TextNumLayers; i++)
            {
                foreach (var w in _textFFNWeightGrads[i])
                {
                    _accel.ZeroMatrix(w);
                }

                foreach (var b in _textFFNBiasGrads[i])
                {
                    Array.Clear(b, 0, b.Length);
                }
            }

            for (int i = 0; i < _config.PriceNumLayers; i++)
            {
                foreach (var w in _priceFFNWeightGrads[i])
                {
                    _accel.ZeroMatrix(w);
                }

                foreach (var b in _priceFFNBiasGrads[i])
                {
                    Array.Clear(b, 0, b.Length);
                }
            }
        }


        private void ClipGradients(float th)
        {
            float n = ComputeGradientNorm();

            if (n > th)
            {
                ScaleAllGradients(th / n);
            }

        }

        private float ComputeGradientNorm()
        {
            float sum = 0;

            if (!_config.FreezeTextEncoder)
            {
                sum += _accel.MatrixSquaredNorm(_gradients.TextEmbeddingGrad);

                foreach (var g in _gradients.TextAttnGrads)
                {
                    sum += _accel.MatrixSquaredNorm(g.WQ_Grad);
                    sum += _accel.MatrixSquaredNorm(g.WK_Grad);
                    sum += _accel.MatrixSquaredNorm(g.WV_Grad);
                    sum += _accel.MatrixSquaredNorm(g.WO_Grad);

                    sum += VN(g.BiasQ_Grad);
                    sum += VN(g.BiasK_Grad);
                    sum += VN(g.BiasV_Grad);
                    sum += VN(g.BiasO_Grad);
                }

                foreach (var g in _gradients.TextLN1Grads)
                {
                    sum += VN(g.GammaGrad);
                    sum += VN(g.BetaGrad);
                }

                foreach (var g in _gradients.TextLN2Grads)
                {
                    sum += VN(g.GammaGrad);
                    sum += VN(g.BetaGrad);
                }

                for (int i = 0; i < _config.TextNumLayers; i++)
                {
                    foreach (var w in _textFFNWeightGrads[i])
                    {
                        sum += _accel.MatrixSquaredNorm(w);
                    }

                    foreach (var b in _textFFNBiasGrads[i])
                    {
                        sum += VN(b);
                    }
                }
            }

            sum += _accel.MatrixSquaredNorm(_gradients.PriceInputProjectionGrad);
            sum += VN(_gradients.PriceInputProjectionBiasGrad);

            foreach (var bg in _gradients.PriceBlockGrads)
            {
                foreach (var g in new[] { bg.SelfAttnGrads, bg.CrossAttnGrads })
                {
                    sum += _accel.MatrixSquaredNorm(g.WQ_Grad);
                    sum += _accel.MatrixSquaredNorm(g.WK_Grad);
                    sum += _accel.MatrixSquaredNorm(g.WV_Grad);
                    sum += _accel.MatrixSquaredNorm(g.WO_Grad);

                    sum += VN(g.BiasQ_Grad);
                    sum += VN(g.BiasK_Grad);
                    sum += VN(g.BiasV_Grad);
                    sum += VN(g.BiasO_Grad);
                }

                sum += VN(bg.LNSelfGrads.GammaGrad);
                sum += VN(bg.LNSelfGrads.BetaGrad);
                sum += VN(bg.LNCrossGrads.GammaGrad);
                sum += VN(bg.LNCrossGrads.BetaGrad);
                sum += VN(bg.LNFFNGrads.GammaGrad);
                sum += VN(bg.LNFFNGrads.BetaGrad);

                if (bg.DecayGrads != null)
                {
                    sum += DecayGradNorm(bg.DecayGrads);
                }
            }

            for (int i = 0; i < _config.PriceNumLayers; i++)
            {
                foreach (var w in _priceFFNWeightGrads[i])
                {
                    sum += _accel.MatrixSquaredNorm(w);
                }

                foreach (var b in _priceFFNBiasGrads[i])
                {
                    sum += VN(b);
                }
            }

            sum += _accel.MatrixSquaredNorm(_gradients.OutputProjectionGrad);
            sum += VN(_gradients.OutputBiasGrad);

            if (_config.UseConfidenceHead)
            {
                sum += _accel.MatrixSquaredNorm(_gradients.ConfidenceProjectionGrad);
                sum += VN(_gradients.ConfidenceBiasGrad);
            }

            sum += MN2(_gradients.ContextTypeEmbeddingGrad);

            return MathF.Sqrt(sum);
        }

        private float DecayGradNorm(ContentAwareDecayGradients g)
        {
            return VN(g.LogBaseDecayRateGrad)
                 + VN(g.B2Grad)
                 + MN3(g.QueryProjectionGrad)
                 + MN2(g.QueryProjectionBiasGrad)
                 + MN3(g.KeyProjectionGrad)
                 + MN2(g.KeyProjectionBiasGrad)
                 + MN2(g.TimeLogFreqGrad)
                 + MN3(g.TimeProjGrad)
                 + MN2(g.TimeProjBiasGrad)
                 + MN3(g.MemAttnOutputWGrad)
                 + MN2(g.MemAttnOutputBGrad)
                 + MN3(g.W1Grad)
                 + MN2(g.B1Grad)
                 + MN2(g.W2Grad);
        }

        private void ScaleAllGradients(float s)
        {
            if (!_config.FreezeTextEncoder)
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

                for (int i = 0; i < _config.TextNumLayers; i++)
                {
                    foreach (var w in _textFFNWeightGrads[i])
                    {
                        _accel.MatrixScaleInPlace(w, s);
                    }

                    foreach (var b in _textFFNBiasGrads[i])
                    {
                        _accel.VectorScaleInPlace(b, s);
                    }
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

                if (bg.DecayGrads != null)
                {
                    ScaleDecayGrads(bg.DecayGrads, s);
                }
            }

            for (int i = 0; i < _config.PriceNumLayers; i++)
            {
                foreach (var w in _priceFFNWeightGrads[i])
                {
                    _accel.MatrixScaleInPlace(w, s);
                }

                foreach (var b in _priceFFNBiasGrads[i])
                {
                    _accel.VectorScaleInPlace(b, s);
                }
            }

            _accel.MatrixScaleInPlace(_gradients.OutputProjectionGrad, s);
            _accel.VectorScaleInPlace(_gradients.OutputBiasGrad, s);

            if (_config.UseConfidenceHead)
            {
                _accel.MatrixScaleInPlace(_gradients.ConfidenceProjectionGrad, s);
                _accel.VectorScaleInPlace(_gradients.ConfidenceBiasGrad, s);
            }

            _accel.MatrixScaleInPlace(_gradients.ContextTypeEmbeddingGrad, s);
        }

        private void ScaleDecayGrads(ContentAwareDecayGradients g, float s)
        {
            int nh = g.B2Grad.Length,
                pd = g.QueryProjectionBiasGrad.GetLength(1),
                cd = g.QueryProjectionGrad.GetLength(2),
                hd = g.B1Grad.GetLength(1),
                mid = g.W1Grad.GetLength(2);

            int ntb = g.TimeLogFreqGrad.GetLength(1),
                rawDim = g.TimeProjGrad.GetLength(2);

            for (int h = 0; h < nh; h++)
            {
                g.LogBaseDecayRateGrad[h] *= s;
                g.B2Grad[h] *= s;

                for (int p = 0; p < pd; p++)
                {
                    for (int d = 0; d < cd; d++)
                    {
                        g.QueryProjectionGrad[h, p, d] *= s;
                        g.KeyProjectionGrad[h, p, d] *= s;
                    }

                    g.QueryProjectionBiasGrad[h, p] *= s;
                    g.KeyProjectionBiasGrad[h, p] *= s;

                    for (int r = 0; r < rawDim; r++)
                    {
                        g.TimeProjGrad[h, p, r] *= s;
                    }

                    g.TimeProjBiasGrad[h, p] *= s;

                    for (int q = 0; q < pd; q++)
                    {
                        g.MemAttnOutputWGrad[h, p, q] *= s;
                    }

                    g.MemAttnOutputBGrad[h, p] *= s;
                }

                for (int b = 0; b < ntb; b++)
                {
                    g.TimeLogFreqGrad[h, b] *= s;
                }

                for (int j = 0; j < hd; j++)
                {
                    g.B1Grad[h, j] *= s;
                    g.W2Grad[h, j] *= s;

                    for (int k = 0; k < mid; k++)
                    {
                        g.W1Grad[h, j, k] *= s;
                    }
                }
            }
        }

        private float VN(float[] v)
        {
            float s = 0;
            for (int i = 0; i < v.Length; i++)
            {
                s += v[i] * v[i];
            }
            return s;
        }
        private float MN2(float[,] m)
        {
            float s = 0;

            int d0 = m.GetLength(0),
                d1 = m.GetLength(1);

            for (int i = 0; i < d0; i++)
            {
                for (int j = 0; j < d1; j++)
                {
                    s += m[i, j] * m[i, j];
                }
            }

            return s;
        }

        private float MN3(float[,,] m)
        {
            float s = 0;

            int d0 = m.GetLength(0),
                d1 = m.GetLength(1),
                d2 = m.GetLength(2);

            for (int i = 0; i < d0; i++)
            {
                for (int j = 0; j < d1; j++)
                {
                    for (int k = 0; k < d2; k++)
                    {
                        s += m[i, j, k] * m[i, j, k];
                    }
                }
            }

            return s;
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
        /// Computes the learning rate for the current epoch using a linear warm-up
        /// followed by cosine decay.
        ///
        /// Schedule:
        ///   Warm-up phase  (ep &lt; WarmUpEpochs):
        ///     lr = baseLR * (ep + 1) / WarmUpEpochs
        ///     Ramps linearly from baseLR/WarmUpEpochs up to baseLR.
        ///     Prevents large, destabilising updates at the start of training when
        ///     gradients are noisy and weights are far from a good basin.
        ///
        ///   Cosine decay phase (ep >= WarmUpEpochs):
        ///     progress = (ep - WarmUpEpochs) / max(1, totalEpochs - WarmUpEpochs)
        ///     lr = minLR + 0.5 * (baseLR - minLR) * (1 + cos(pi * progress))
        ///     Smoothly anneals from baseLR down to minLR (default: baseLR * 0.01).
        ///     The cosine shape avoids the aggressive early drop of linear decay
        ///     and still reaches a very small LR by the final epoch, allowing the
        ///     model to settle into a sharp minimum without oscillating.
        ///
        /// If WarmUpEpochs is 0 (not set on MultimodalTrainingConfig), warm-up is
        /// skipped and cosine decay runs for the full training duration.
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
