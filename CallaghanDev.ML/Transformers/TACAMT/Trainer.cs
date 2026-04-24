using CallaghanDev.ML.AccelerationManagers;
using CallaghanDev.ML.Transformers.Cache;
using CallaghanDev.ML.Transformers.Configuration;
using CallaghanDev.ML.Transformers.CrossAttentionMultimodal;
using CallaghanDev.ML.Transformers.MultiTypeTransformer;
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
            _random = new Random(12345);
            _dropoutRng = new Random(67890);
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
            if (storiesPerSample == null)
                throw new ArgumentNullException(nameof(storiesPerSample));
            if (priceInputs == null)
                throw new ArgumentNullException(nameof(priceInputs));
            if (priceTargets == null)
                throw new ArgumentNullException(nameof(priceTargets));

            int n = storiesPerSample.Length;

            int totalEpochs = _trainConfig.Epochs;

            for (int ep = 0; ep < _trainConfig.Epochs; ep++)
            {
                float lr = ComputeLearningRate(ep, totalEpochs);

                if (_trainConfig.Verbose)
                {
                    Console.WriteLine($"\n=== Epoch {ep + 1}/{_trainConfig.Epochs} ===");
                }
                var sh = Enumerable.Range(0, n).ToArray();

                float el = 0f;
                int nb = 0;

                for (int i = 0; i < sh.Length; i += _trainConfig.BatchSize)
                {
                    int bs = Math.Min(_trainConfig.BatchSize, sh.Length - i);

                    float bl = TrainBatch(
                        sh.Skip(i).Take(bs).ToArray(),
                        storiesPerSample,
                        priceInputs,
                        priceTargets,
                        confTargets,
                        lr,
                        ep);

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
            if (textSequences == null)
                throw new ArgumentNullException(nameof(textSequences));

            var stories = new NewsStory[textSequences.Length][];

            for (int i = 0; i < textSequences.Length; i++)
                stories[i] = new[] { new NewsStory((int[])textSequences[i].Clone(), 0f) };

            Train(stories, priceInputs, priceTargets, confTargets);
        }

        //TODO: See if you cant speed this up with some accelleration magic
        public void TrainSequential(
            NewsStory[][] stories,
            float[][,] priceInputs,
            float[][,] priceTargets,
            double[] timestamps,
            double timeUnitsPerPosition = 1.0,
            int maxNewsMemory = 100,
            int maxPriceMemory = 200,
            float[][] confTargets = null)
        {
            if (stories == null)
                throw new ArgumentNullException(nameof(stories));

            if (priceInputs == null)
                throw new ArgumentNullException(nameof(priceInputs));

            if (priceTargets == null)
                throw new ArgumentNullException(nameof(priceTargets));

            if (timestamps == null)
                throw new ArgumentNullException(nameof(timestamps));

            if (timeUnitsPerPosition == 0.0)
                throw new ArgumentOutOfRangeException(nameof(timeUnitsPerPosition));

            int n = stories.Length;

            if (priceInputs.Length < n || priceTargets.Length < n || timestamps.Length < n)
                throw new ArgumentException("stories, priceInputs, priceTargets, and timestamps must have compatible lengths.");

            int totalEpochs = _trainConfig.Epochs;
            int embeddingDim = _config.Price.EmbeddingDim;
            float invTime = (float)(1.0 / timeUnitsPerPosition);

            var cache = new MultimodalForwardCache(_config.Text.NumLayers, _config.Price.NumLayers);

            for (int ep = 0; ep < totalEpochs; ep++)
            {
                float lr = ComputeLearningRate(ep, totalEpochs);

                if (_trainConfig.Verbose)
                    Console.WriteLine($"\n=== Epoch {ep + 1}/{totalEpochs} (Sequential)[{_accel.GetType().Name}] ===");

                _model.ClearAllMemory();

                float epochLoss = 0f;
                int validCount = 0;

                for (int idx = 0; idx < n; idx++)
                {
                    var priceSequence = priceInputs[idx];
                    var targetSequence = priceTargets[idx];

                    if (priceSequence == null || targetSequence == null)
                        continue;

                    int seqLen = priceSequence.GetLength(0);

                    if (seqLen < 2)
                        continue;

                    double currentTs = timestamps[idx];

                    try
                    {
                        int inputStart = 0;
                        int inputCount = seqLen - 1;

                        int targetStart = 1;
                        int targetCount = seqLen - 1;

                        float[] confTarget = null;
                        var confSource = confTargets?[idx];

                        if (confSource != null && confSource.Length >= seqLen)
                        {
                            confTarget = new float[seqLen - 1];
                            Array.Copy(confSource, 1, confTarget, 0, seqLen - 1);
                        }

                        int newsMemCount = _model.NewsMemory?.Count ?? 0;
                        int priceMemCount = _model.PriceMemory?.Count ?? 0;

                        cache.Reset();

                        NewsStory[] currentStories =
                            stories[idx] != null && stories[idx].Length > 0
                                ? stories[idx]
                                : null;

                        float[,] predictions;
                        float[,] confidence;
                        float[,] combinedHidden = null;
                        float[] combinedTimes = null;

                        if (newsMemCount > 0 || priceMemCount > 0)
                        {
                            float[,] freshNewsHidden = null;
                            float[] freshNewsTimes = null;
                            int freshNewsCount = 0;

                            if (currentStories != null)
                            {
                                (freshNewsHidden, freshNewsTimes) = _model.EncodeStoriesWithCache(currentStories, cache);
                                freshNewsCount = freshNewsHidden.GetLength(0);
                            }
                            else
                            {
                                cache.StoryCaches = new List<MultimodalForwardCache>();
                                cache.StoryTokenCounts = Array.Empty<int>();
                            }

                            int totalCtx = newsMemCount + freshNewsCount + priceMemCount;

                            if (totalCtx > 0)
                            {
                                combinedHidden = new float[totalCtx, embeddingDim];
                                combinedTimes = new float[totalCtx];

                                var typeIndices = new int[totalCtx];

                                int ctx = 0;

                                for (int i = 0; i < newsMemCount; i++, ctx++)
                                {
                                    var entry = _model.NewsMemory[i];

                                    for (int d = 0; d < embeddingDim; d++)
                                        combinedHidden[ctx, d] = entry.HiddenState[d];

                                    combinedTimes[ctx] = (float)((entry.AbsoluteTimestamp - currentTs) * invTime);
                                    typeIndices[ctx] = 0;
                                }

                                for (int i = 0; i < freshNewsCount; i++, ctx++)
                                {
                                    for (int d = 0; d < embeddingDim; d++)
                                        combinedHidden[ctx, d] = freshNewsHidden[i, d];

                                    combinedTimes[ctx] = freshNewsTimes[i];
                                    typeIndices[ctx] = 0;
                                }

                                for (int i = 0; i < priceMemCount; i++, ctx++)
                                {
                                    var entry = _model.PriceMemory[i];

                                    for (int d = 0; d < embeddingDim; d++)
                                        combinedHidden[ctx, d] = entry.HiddenState[d];

                                    combinedTimes[ctx] = (float)((entry.AbsoluteTimestamp - currentTs) * invTime);
                                    typeIndices[ctx] = 1;
                                }

                                _accel.ApplyContextTypeEmbedding(combinedHidden, _model.ContextTypeEmbedding, typeIndices);

                                cache.NumNewsContext = newsMemCount + freshNewsCount;
                                cache.NumPriceContext = priceMemCount;
                                cache.PriceContextHidden = null;
                                cache.TextFinalHidden = combinedHidden;
                                cache.StoryArrivalTimes = combinedTimes;
                            }

                            var priceHidden = _model.ForwardPriceDecoderWithCache(
                                priceSequence,
                                inputStart,
                                inputCount,
                                combinedHidden,
                                combinedTimes,
                                cache,
                                isTraining: true,
                                dropoutRng: _dropoutRng);

                            cache.PriceFinalHidden = priceHidden;

                            (predictions, confidence) = _model.ProjectToOutput(priceHidden);
                        }
                        else
                        {
                            (predictions, confidence) = _model.ForwardWithCache(
                                currentStories,
                                priceSequence,
                                inputStart,
                                inputCount,
                                cache,
                                isTraining: true,
                                dropoutRng: _dropoutRng);
                        }

                        ZeroAllGradients();

                        float loss = BackwardPass(
                            predictions,
                            confidence,
                            targetSequence,
                            targetStart,
                            targetCount,
                            confTarget,
                            cache);

                        if (!float.IsFinite(loss))
                        {
                            ZeroAllGradients();
                            continue;
                        }

                        if (_trainConfig.UseGradientClipping)
                            ClipGradients(_trainConfig.GradientClipThreshold);

                        UpdateAllParameters(lr);

                        if (combinedHidden != null)
                            _model.UpdateMemoryAttentionScores(cache.PriceFinalHidden, combinedHidden, combinedHidden.GetLength(0));

                        epochLoss += loss;
                        validCount++;

                        if (currentStories != null)
                        {
                            _model.UpdateNewsMemory(
                                currentStories,
                                currentTs,
                                timeUnitsPerPosition,
                                maxNewsMemory);
                        }

                        int hiddenSeqLen = cache.PriceFinalHidden.GetLength(0);

                        for (int t = 0; t < hiddenSeqLen; t++)
                        {
                            var hidden = new float[embeddingDim];

                            for (int d = 0; d < embeddingDim; d++)
                                hidden[d] = cache.PriceFinalHidden[t, d];

                            _model.PriceMemory.Add(new PriceMemoryEntry
                            {
                                HiddenState = hidden,
                                AbsoluteTimestamp = currentTs + t * timeUnitsPerPosition
                            });
                        }

                        _model.LastPriceTimestamp = currentTs + Math.Max(0, hiddenSeqLen - 1) * timeUnitsPerPosition;

                        _model.PruneNewsMemory(maxNewsMemory);
                        _model.PricePruneMemory(maxPriceMemory);

                        if (_trainConfig.Verbose && validCount % 50 == 0)
                        {
                            Console.WriteLine(
                                $"  Sample {validCount}: Loss = {loss:F6}, " +
                                $"NewsMemory = {_model.NewsMemory.Count}, PriceMemory = {_model.PriceMemory.Count}");
                        }
                    }
                    catch (Exception ex)
                    {
                        ZeroAllGradients();

                        if (_trainConfig.Verbose)
                        {
                            Console.WriteLine($"  WARNING: {ex.Message}");
                            Console.WriteLine($"  Info: {ex.StackTrace}");
                        }
                    }
                }

                if (_trainConfig.Verbose)
                    Console.WriteLine($"  Epoch {ep + 1} Average Loss: {(validCount > 0 ? epochLoss / validCount : 0):F6}");
            }
        }
        private float TrainBatch(
            int[] batchIndices,
            NewsStory[][] allStories,
            float[][,] allPriceInputs,
            float[][,] allPriceTargets,
            float[][] allConfidenceTargets,
            float lr,
            int epoch)
        {
            var batchGradients = new Gradients(_config);
            var (batchTextFFNWeightGrads, batchTextFFNBiasGrads) = CreateTextFFNGradientStorage();
            var (batchPriceFFNWeightGrads, batchPriceFFNBiasGrads) = CreatePriceFFNGradientStorage();

            float totalLoss = 0f;
            int validCount = 0;

            int minSplitLen =
                _config.PriceContext.MinHistoryLength +
                _config.PriceContext.MinCurrentLength +
                1;

            foreach (int idx in batchIndices)
            {
                ZeroAllGradients();

                try
                {
                    if (idx < 0 ||
                        idx >= allStories.Length ||
                        idx >= allPriceInputs.Length ||
                        idx >= allPriceTargets.Length)
                    {
                        continue;
                    }

                    var stories = allStories[idx];
                    var priceInput = allPriceInputs[idx];
                    var priceTarget = allPriceTargets[idx];

                    if (priceInput == null || priceTarget == null)
                        continue;

                    int seqLen = priceInput.GetLength(0);

                    if (seqLen < 2)
                        continue;

                    float loss;

                    bool canSplit =
                        _config.PriceContext.Enabled &&
                        seqLen >= minSplitLen;

                    Random sampleDropoutRng = CreateSampleDropoutRandom(stories, priceInput, epoch);

                    if (canSplit)
                    {
                        loss = TrainWithPriceContext(
                            idx,
                            allStories,
                            allPriceInputs,
                            allPriceTargets,
                            allConfidenceTargets,
                            epoch,
                            sampleDropoutRng);
                    }
                    else
                    {
                        float[,] input = SliceRows(priceInput, 0, seqLen - 1);
                        float[,] target = SliceRows(priceTarget, 1, seqLen);

                        float[] confTarget = null;

                        if (allConfidenceTargets != null &&
                            idx < allConfidenceTargets.Length &&
                            allConfidenceTargets[idx] != null)
                        {
                            confTarget = allConfidenceTargets[idx]
                                .Skip(1)
                                .Take(seqLen - 1)
                                .ToArray();
                        }

                        var cache = new MultimodalForwardCache(
                            _config.Text.NumLayers,
                            _config.Price.NumLayers);

                        var (pred, conf) = _model.ForwardWithCache(
                            stories,
                            input,
                            cache,
                            isTraining: true,
                            dropoutRng: sampleDropoutRng);

                        loss = BackwardPass(pred, conf, target, confTarget, cache);
                    }

                    if (!float.IsFinite(loss))
                    {
                        ZeroAllGradients();
                        continue;
                    }

                    AccumulateLiveGradientsInto(
                        batchGradients,
                        batchTextFFNWeightGrads,
                        batchTextFFNBiasGrads,
                        batchPriceFFNWeightGrads,
                        batchPriceFFNBiasGrads);

                    totalLoss += loss;
                    validCount++;
                }
                catch (Exception ex)
                {
                    ZeroAllGradients();

                    if (_trainConfig.Verbose)
                        Console.WriteLine($"  WARNING: {ex.Message}");
                }
                finally
                {
                    ZeroAllGradients();
                }
            }

            if (validCount == 0)
                return 0f;

            LoadAccumulatedGradientsIntoLive(
                batchGradients,
                batchTextFFNWeightGrads,
                batchTextFFNBiasGrads,
                batchPriceFFNWeightGrads,
                batchPriceFFNBiasGrads);

            ScaleAllGradients(1.0f / validCount);

            if (_trainConfig.UseGradientClipping)
                ClipGradients(_trainConfig.GradientClipThreshold);

            UpdateAllParameters(lr);
            ZeroAllGradients();

            return totalLoss / validCount;
        }
        private float TrainWithPriceContext(
            int idx,
            NewsStory[][] allStories,
            float[][,] allPriceInputs,
            float[][,] allPriceTargets,
            float[][] allConfidenceTargets,
            int epoch,
            Random dropoutRng)
        {
            if (!_config.PriceContext.Enabled)
                return 0f;

            var priceSequence = allPriceInputs[idx];
            int seqLen = priceSequence.GetLength(0);

            int minHistory = _config.PriceContext.MinHistoryLength;
            int maxHistory = seqLen - _config.PriceContext.MinCurrentLength - 1;

            if (maxHistory < minHistory)
                return 0f;

            NewsStory[] sourceStories =
                allStories != null && idx >= 0 && idx < allStories.Length
                    ? allStories[idx]
                    : null;

            int splitPoint = ChooseDeterministicSplitPoint(
                sourceStories,
                priceSequence,
                minHistory,
                maxHistory,
                epoch);

            int historyLen = splitPoint;

            var historyPrices = SliceRows(priceSequence, 0, historyLen);
            var currentInput = SliceRows(priceSequence, splitPoint, seqLen - 1);
            var currentTarget = SliceRows(allPriceTargets[idx], splitPoint + 1, seqLen);

            int currentSeqLen = currentInput.GetLength(0);

            if (currentSeqLen <= 0)
                return 0f;

            float[] confidenceTarget = null;

            if (allConfidenceTargets != null &&
                idx < allConfidenceTargets.Length &&
                allConfidenceTargets[idx] != null)
            {
                confidenceTarget = allConfidenceTargets[idx]
                    .Skip(splitPoint + 1)
                    .Take(currentSeqLen)
                    .ToArray();
            }

            var priceCtxHidden = _model.EncodePriceHistory(historyPrices);
            var priceCtxTimes = new float[historyLen];

            for (int t = 0; t < historyLen; t++)
                priceCtxTimes[t] = t - splitPoint;

            NewsStory[] adjustedStories = null;

            if (sourceStories != null && sourceStories.Length > 0)
            {
                adjustedStories = new NewsStory[sourceStories.Length];

                for (int i = 0; i < sourceStories.Length; i++)
                {
                    adjustedStories[i] = new NewsStory(
                        sourceStories[i].TokenIds,
                        sourceStories[i].ArrivalTime - splitPoint);
                }
            }

            var cache = new MultimodalForwardCache(_config.Text.NumLayers, _config.Price.NumLayers);

            var (pred, conf) = _model.ForwardWithPriceContextAndCache(
                adjustedStories,
                currentInput,
                priceCtxHidden,
                priceCtxTimes,
                cache,
                isTraining: true,
                dropoutRng: dropoutRng);

            return BackwardPass(pred, conf, currentTarget, confidenceTarget, cache);
        }
        #region Validation

        public float Validate(NewsStory[][] stories, float[][,] priceInputs, float[][,] priceTargets)
        {
            float totalLoss = 0f;
            int totalCount = 0;

            int minSplitLength =
                _config.PriceContext.MinHistoryLength +
                _config.PriceContext.MinCurrentLength +
                1;

            for (int i = 0; i < stories.Length; i++)
            {
                float[,] inputPrices = priceInputs[i];

                if (inputPrices == null || priceTargets[i] == null)
                    continue;

                int sequenceLength = inputPrices.GetLength(0);

                if (sequenceLength < 2)
                    continue;

                bool canUseContextSplit =
                    _config.PriceContext.Enabled &&
                    sequenceLength >= minSplitLength;

                if (canUseContextSplit)
                {
                    int minHistory = _config.PriceContext.MinHistoryLength;
                    int maxHistory = sequenceLength - _config.PriceContext.MinCurrentLength - 1;
                    int splitPoint = (minHistory + maxHistory) / 2;

                    float[,] historyPrices = SliceRows(inputPrices, 0, splitPoint);
                    float[,] currentInput = SliceRows(inputPrices, splitPoint, sequenceLength - 1);
                    float[,] currentTarget = SliceRows(priceTargets[i], splitPoint + 1, sequenceLength);

                    int currentSeqLength = currentInput.GetLength(0);

                    if (currentSeqLength <= 0)
                        continue;

                    float[,] priceContextHidden = _model.EncodePriceHistory(historyPrices);
                    float[] priceContextTimes = new float[splitPoint];

                    for (int t = 0; t < splitPoint; t++)
                        priceContextTimes[t] = t - splitPoint;

                    NewsStory[] adjustedStories = null;

                    if (stories[i] != null && stories[i].Length > 0)
                    {
                        adjustedStories = new NewsStory[stories[i].Length];

                        for (int s = 0; s < stories[i].Length; s++)
                        {
                            adjustedStories[s] = new NewsStory(
                                stories[i][s].TokenIds,
                                stories[i][s].ArrivalTime - splitPoint);
                        }
                    }

                    var cache = new MultimodalForwardCache(_config.Text.NumLayers, _config.Price.NumLayers);

                    var (predictions, _) = _model.ForwardWithPriceContextAndCache(
                        adjustedStories,
                        currentInput,
                        priceContextHidden,
                        priceContextTimes,
                        cache,
                        isTraining: false);

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

            return totalCount > 0
                ? totalLoss / (totalCount * _config.Output.OutputDim)
                : 0f;
        }
        public float Validate(int[][] texts, float[][,] pi, float[][,] pt)
        {
            if (texts == null)
                throw new ArgumentNullException(nameof(texts));

            var stories = new NewsStory[texts.Length][];

            for (int i = 0; i < texts.Length; i++)
                stories[i] = new[] { new NewsStory(texts[i], 0f) };

            return Validate(stories, pi, pt);
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

            int totalCount = sequenceLength * outputDim;

            for (int t = 0; t < sequenceLength; t++)
            {
                int tgtRow = tgtRowStart + t;

                for (int j = 0; j < outputDim; j++)
                {
                    float diff = pred[t, j] - tgt[tgtRow, j];

                    mse += diff * diff;

                    dPred[t, j] = 2f * diff / totalCount;
                }
            }

            mse /= totalCount;

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

        private void BackpropMultiStoryTextEncoder(float[,] dStoryHidden, MultimodalForwardCache cache)
        {
            if (dStoryHidden == null || cache?.StoryCaches == null || cache.StoryCaches.Count == 0)
                return;

            int storyCount = cache.StoryCaches.Count;
            int embeddingDim = _config.Text.EmbeddingDim;

            // In sequential memory training, context is:
            // [existing news memory | fresh news stories | price memory]
            //
            // StoryCaches only contains fresh news stories, so gradients for fresh stories
            // start after existing news-memory rows.
            int contextOffset = 0;

            if (cache.NumNewsContext > storyCount)
                contextOffset = cache.NumNewsContext - storyCount;

            if (contextOffset + storyCount > dStoryHidden.GetLength(0))
                contextOffset = Math.Max(0, dStoryHidden.GetLength(0) - storyCount);

            for (int s = 0; s < storyCount; s++)
            {
                int sourceRow = contextOffset + s;
                int tokenCount = cache.StoryTokenCounts[s];

                if (tokenCount <= 0)
                    continue;

                float invTokenCount = 1.0f / tokenCount;
                var dTokenHidden = new float[tokenCount, embeddingDim];

                for (int t = 0; t < tokenCount; t++)
                {
                    for (int d = 0; d < embeddingDim; d++)
                        dTokenHidden[t, d] = dStoryHidden[sourceRow, d] * invTokenCount;
                }

                BackpropTextEncoder(dTokenHidden, cache.StoryCaches[s]);
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

            float minLogRate = net.GetMinLogBaseDecayRate();
            float maxLogRate = net.GetMaxLogBaseDecayRate();

            for (int h = 0; h < nh; h++)
            {
                net.LogBaseDecayRate[h] -= lr * g.LogBaseDecayRateGrad[h];
                if (net.LogBaseDecayRate[h] < minLogRate) net.LogBaseDecayRate[h] = minLogRate;
                if (net.LogBaseDecayRate[h] > maxLogRate) net.LogBaseDecayRate[h] = maxLogRate;

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
                        net.TimeProj[h, p, r] -= lr * g.TimeProjGrad[h, p, r];

                    net.TimeProjBias[h, p] -= lr * g.TimeProjBiasGrad[h, p];

                    for (int q = 0; q < pd; q++)
                        net.MemAttnOutputW[h, p, q] -= lr * g.MemAttnOutputWGrad[h, p, q];

                    net.MemAttnOutputB[h, p] -= lr * g.MemAttnOutputBGrad[h, p];
                }

                for (int b = 0; b < ntb; b++)
                    net.TimeLogFreq[h, b] -= lr * g.TimeLogFreqGrad[h, b];

                for (int j = 0; j < hd; j++)
                {
                    net.B1[h, j] -= lr * g.B1Grad[h, j];
                    net.W2[h, j] -= lr * g.W2Grad[h, j];

                    for (int k = 0; k < mid; k++)
                        net.W1[h, j, k] -= lr * g.W1Grad[h, j, k];
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
        private (List<List<float[,]>> weightGrads, List<List<float[]>> biasGrads) CreateTextFFNGradientStorage()
        {
            var weights = new List<List<float[,]>>();
            var biases = new List<List<float[]>>();

            for (int i = 0; i < _config.Text.NumLayers; i++)
            {
                var (w, b) = _model.TextBlocks[i].FeedForwardNetwork.CreateGradientStorage();
                weights.Add(w);
                biases.Add(b);
            }

            return (weights, biases);
        }

        private (List<List<float[,]>> weightGrads, List<List<float[]>> biasGrads) CreatePriceFFNGradientStorage()
        {
            var weights = new List<List<float[,]>>();
            var biases = new List<List<float[]>>();

            for (int i = 0; i < _config.Price.NumLayers; i++)
            {
                var (w, b) = _model.PriceBlocks[i].FeedForwardNetwork.CreateGradientStorage();
                weights.Add(w);
                biases.Add(b);
            }

            return (weights, biases);
        }

        private void AccumulateLiveGradientsInto(
            Gradients target,
            List<List<float[,]>> targetTextFFNWeightGrads,
            List<List<float[]>> targetTextFFNBiasGrads,
            List<List<float[,]>> targetPriceFFNWeightGrads,
            List<List<float[]>> targetPriceFFNBiasGrads)
        {
            AccumulateGradients(target, _gradients);

            AccumulateFFNGradients(targetTextFFNWeightGrads, targetTextFFNBiasGrads, _textFFNWeightGrads, _textFFNBiasGrads);
            AccumulateFFNGradients(targetPriceFFNWeightGrads, targetPriceFFNBiasGrads, _priceFFNWeightGrads, _priceFFNBiasGrads);
        }

        private void LoadAccumulatedGradientsIntoLive(
            Gradients source,
            List<List<float[,]>> sourceTextFFNWeightGrads,
            List<List<float[]>> sourceTextFFNBiasGrads,
            List<List<float[,]>> sourcePriceFFNWeightGrads,
            List<List<float[]>> sourcePriceFFNBiasGrads)
        {
            ZeroAllGradients();

            AccumulateGradients(_gradients, source);

            AccumulateFFNGradients(_textFFNWeightGrads, _textFFNBiasGrads, sourceTextFFNWeightGrads, sourceTextFFNBiasGrads);
            AccumulateFFNGradients(_priceFFNWeightGrads, _priceFFNBiasGrads, sourcePriceFFNWeightGrads, sourcePriceFFNBiasGrads);
        }

        private void AccumulateGradients(Gradients target, Gradients source)
        {
            if (!_config.Text.Freeze)
            {
                _accel.MatrixAccumulate(target.TextEmbeddingGrad, source.TextEmbeddingGrad);

                for (int i = 0; i < _config.Text.NumLayers; i++)
                {
                    AccumulateAttentionGradients(target.TextAttnGrads[i], source.TextAttnGrads[i]);
                    AccumulateLayerNormGradients(target.TextLN1Grads[i], source.TextLN1Grads[i]);
                    AccumulateLayerNormGradients(target.TextLN2Grads[i], source.TextLN2Grads[i]);
                }
            }

            _accel.MatrixAccumulate(target.PriceInputProjectionGrad, source.PriceInputProjectionGrad);
            _accel.VectorAccumulate(target.PriceInputProjectionBiasGrad, source.PriceInputProjectionBiasGrad);

            for (int i = 0; i < _config.Price.NumLayers; i++)
            {
                AccumulateCrossAttentionBlockGradients(target.PriceBlockGrads[i], source.PriceBlockGrads[i]);
            }

            _accel.MatrixAccumulate(target.OutputProjectionGrad, source.OutputProjectionGrad);
            _accel.VectorAccumulate(target.OutputBiasGrad, source.OutputBiasGrad);

            if (_config.Output.UseConfidenceHead &&
                target.ConfidenceProjectionGrad != null &&
                source.ConfidenceProjectionGrad != null)
            {
                _accel.MatrixAccumulate(target.ConfidenceProjectionGrad, source.ConfidenceProjectionGrad);
                _accel.VectorAccumulate(target.ConfidenceBiasGrad, source.ConfidenceBiasGrad);
            }

            _accel.MatrixAccumulate(target.ContextTypeEmbeddingGrad, source.ContextTypeEmbeddingGrad);
        }

        private void AccumulateAttentionGradients(AttentionGradients target, AttentionGradients source)
        {
            _accel.MatrixAccumulate(target.WQ_Grad, source.WQ_Grad);
            _accel.MatrixAccumulate(target.WK_Grad, source.WK_Grad);
            _accel.MatrixAccumulate(target.WV_Grad, source.WV_Grad);
            _accel.MatrixAccumulate(target.WO_Grad, source.WO_Grad);

            _accel.VectorAccumulate(target.BiasQ_Grad, source.BiasQ_Grad);
            _accel.VectorAccumulate(target.BiasK_Grad, source.BiasK_Grad);
            _accel.VectorAccumulate(target.BiasV_Grad, source.BiasV_Grad);
            _accel.VectorAccumulate(target.BiasO_Grad, source.BiasO_Grad);
        }

        private void AccumulateLayerNormGradients(LayerNormGradients target, LayerNormGradients source)
        {
            _accel.VectorAccumulate(target.GammaGrad, source.GammaGrad);
            _accel.VectorAccumulate(target.BetaGrad, source.BetaGrad);
        }

        private void AccumulateCrossAttentionBlockGradients(CrossAttentionBlockGradients target, CrossAttentionBlockGradients source)
        {
            AccumulateAttentionGradients(target.SelfAttnGrads, source.SelfAttnGrads);
            AccumulateLayerNormGradients(target.LNSelfGrads, source.LNSelfGrads);

            AccumulateAttentionGradients(target.CrossAttnGrads, source.CrossAttnGrads);
            AccumulateLayerNormGradients(target.LNCrossGrads, source.LNCrossGrads);

            if (target.DecayGrads != null && source.DecayGrads != null)
                AccumulateDecayGrads(target.DecayGrads, source.DecayGrads);

            AccumulateLayerNormGradients(target.LNFFNGrads, source.LNFFNGrads);
        }

        private void AccumulateFFNGradients(
            List<List<float[,]>> targetWeights,
            List<List<float[]>> targetBiases,
            List<List<float[,]>> sourceWeights,
            List<List<float[]>> sourceBiases)
        {
            if (targetWeights != null && sourceWeights != null)
            {
                int layerCount = Math.Min(targetWeights.Count, sourceWeights.Count);

                for (int layer = 0; layer < layerCount; layer++)
                {
                    int matrixCount = Math.Min(targetWeights[layer].Count, sourceWeights[layer].Count);

                    for (int i = 0; i < matrixCount; i++)
                        _accel.MatrixAccumulate(targetWeights[layer][i], sourceWeights[layer][i]);
                }
            }

            if (targetBiases != null && sourceBiases != null)
            {
                int layerCount = Math.Min(targetBiases.Count, sourceBiases.Count);

                for (int layer = 0; layer < layerCount; layer++)
                {
                    int vectorCount = Math.Min(targetBiases[layer].Count, sourceBiases[layer].Count);

                    for (int i = 0; i < vectorCount; i++)
                        _accel.VectorAccumulate(targetBiases[layer][i], sourceBiases[layer][i]);
                }
            }
        }

        private Random CreateSampleDropoutRandom(NewsStory[] stories, float[,] priceInput, int epoch)
        {
            return new Random(CreateStableSampleSeed(stories, priceInput, epoch, 7919));
        }

        private int ChooseDeterministicSplitPoint(
            NewsStory[] stories,
            float[,] priceSequence,
            int minHistory,
            int maxHistory,
            int epoch)
        {
            if (maxHistory <= minHistory)
                return minHistory;

            int seed = CreateStableSampleSeed(stories, priceSequence, epoch, 104729);
            int span = maxHistory - minHistory + 1;

            return minHistory + (int)((uint)seed % (uint)span);
        }

        private static int CreateStableSampleSeed(NewsStory[] stories, float[,] priceInput, int epoch, int salt)
        {
            unchecked
            {
                int hash = unchecked((int)2166136261);

                MixHash(ref hash, salt);
                MixHash(ref hash, epoch);

                if (stories == null)
                {
                    MixHash(ref hash, -1);
                }
                else
                {
                    MixHash(ref hash, stories.Length);

                    for (int s = 0; s < stories.Length; s++)
                    {
                        var story = stories[s];

                        if (story == null)
                        {
                            MixHash(ref hash, -2);
                            continue;
                        }

                        MixHash(ref hash, FloatToStableBits(story.ArrivalTime));

                        if (story.TokenIds == null)
                        {
                            MixHash(ref hash, -3);
                        }
                        else
                        {
                            MixHash(ref hash, story.TokenIds.Length);

                            for (int i = 0; i < story.TokenIds.Length; i++)
                                MixHash(ref hash, story.TokenIds[i]);
                        }
                    }
                }

                if (priceInput == null)
                {
                    MixHash(ref hash, -4);
                }
                else
                {
                    int rows = priceInput.GetLength(0);
                    int cols = priceInput.GetLength(1);

                    MixHash(ref hash, rows);
                    MixHash(ref hash, cols);

                    for (int r = 0; r < rows; r++)
                    {
                        for (int c = 0; c < cols; c++)
                            MixHash(ref hash, FloatToStableBits(priceInput[r, c]));
                    }
                }

                hash &= 0x7fffffff;

                return hash == 0 ? 1 : hash;
            }
        }

        private static void MixHash(ref int hash, int value)
        {
            unchecked
            {
                hash ^= value;
                hash *= 16777619;
            }
        }

        private static int FloatToStableBits(float value)
        {
            if (value == 0f)
                value = 0f;

            return BitConverter.ToInt32(BitConverter.GetBytes(value), 0);
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