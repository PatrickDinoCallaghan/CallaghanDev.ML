using CallaghanDev.ML.AccelerationManagers;
using CallaghanDev.ML.Enums;
using CallaghanDev.ML.Transformers.Cache;
using CallaghanDev.ML.Transformers.Configuration;
using CallaghanDev.ML.Transformers.MultiTypeTransformer;
using CallaghanDev.ML.Transformers.TACAMT;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace CallaghanDev.ML.Transformers.MMTAC
{
    /// <summary>
    /// Trainer for MmtacModel.
    ///
    /// Loss function:
    ///   totalLoss = MSE(High, Low, Close)
    ///             + λ_range    * MSE(Range,   softplus activation)
    ///             + λ_quality  * MSE(Quality,  sigmoid activation)
    ///             + λ_dir      * BCE(Direction)
    ///             + λ_midDir   * BCE(MidWindowDirection)
    ///             + λ_conf     * BCE(Confidence)   [optional]
    /// </summary>
    public class MmtacTrainer
    {
        private readonly MmtacModel _model;
        private readonly MmtacConfig _config;
        private readonly TrainingConfig _trainConfig;
        private readonly MmtacGradients _gradients;
        private readonly IAccelerationManager _accel;
        private readonly Random _random;
        private readonly Random _dropoutRng;

        private readonly List<List<float[,]>> _textFFNWeightGrads;
        private readonly List<List<float[]>> _textFFNBiasGrads;
        private readonly List<List<float[,]>> _priceFFNWeightGrads;
        private readonly List<List<float[]>> _priceFFNBiasGrads;

        public MmtacTrainer(MmtacModel model, TrainingConfig trainConfig)
        {
            _model = model;
            _config = model.Config;
            _trainConfig = trainConfig;
            _gradients = new MmtacGradients(_config);
            _accel = model.AccelerationManager;
            _random = new Random();
            _dropoutRng = new Random(_random.Next());

            _textFFNWeightGrads = new List<List<float[,]>>();
            _textFFNBiasGrads = new List<List<float[]>>();
            _priceFFNWeightGrads = new List<List<float[,]>>();
            _priceFFNBiasGrads = new List<List<float[]>>();

            for (int i = 0; i < _config.Text.NumLayers; i++)
            {
                var (w, b) = model.TextBlocks[i].FeedForwardNetwork.CreateGradientStorage();
                _textFFNWeightGrads.Add(w); _textFFNBiasGrads.Add(b);
            }
            for (int i = 0; i < _config.Price.NumLayers; i++)
            {
                var (w, b) = model.PriceBlocks[i].FeedForwardNetwork.CreateGradientStorage();
                _priceFFNWeightGrads.Add(w); _priceFFNBiasGrads.Add(b);
            }
        }

        // 
        // Standard batch training
        // 
        public void Train(MultimodalInput[] inputs, ModelTarget[][] targets, float[][] confTargets = null)
        {
            _trainConfig.Validate();
            int n = inputs.Length;

            for (int ep = 0; ep < _trainConfig.Epochs; ep++)
            {
                float lr = ComputeLR(ep);
                if (_trainConfig.Verbose) Console.WriteLine($"\n=== Epoch {ep + 1}/{_trainConfig.Epochs} ===");

                var shuffled = Enumerable.Range(0, n).OrderBy(_ => _random.Next()).ToArray();
                float epochLoss = 0f; int batchCount = 0;

                for (int i = 0; i < shuffled.Length; i += _trainConfig.BatchSize)
                {
                    var batch = shuffled.Skip(i).Take(_trainConfig.BatchSize).ToArray();
                    float bl = TrainBatch(batch, inputs, targets, confTargets, lr);
                    epochLoss += bl; batchCount++;
                    if (_trainConfig.Verbose && batchCount % 10 == 0)
                    {
                        Console.WriteLine($"  Batch {batchCount}: Loss = {bl:F6}");
                    }
                }

                if (_trainConfig.Verbose)
                {
                    Console.WriteLine($"  Epoch {ep + 1} Avg Loss: {(batchCount > 0 ? epochLoss / batchCount : 0f):F6}");
                }
            }
        }

        // 
        // Sequential training with rolling memory
        // 
        public void TrainSequential(MultimodalInput[] inputs, ModelTarget[][] targets, double[] timestamps, double timeUnitsPerPosition = 1.0, int maxNewsMemory = 100, int maxPriceMemory = 200, float[][] confTargets = null)
        {
            _trainConfig.Validate();

            if (inputs == null) throw new ArgumentNullException(nameof(inputs));
            if (targets == null) throw new ArgumentNullException(nameof(targets));
            if (timestamps == null) throw new ArgumentNullException(nameof(timestamps));
            if (inputs.Length != targets.Length || inputs.Length != timestamps.Length)
            {
                throw new ArgumentException("inputs, targets, and timestamps must all have the same length.");
            }
            if (timeUnitsPerPosition == 0.0)
            {
                throw new ArgumentOutOfRangeException(nameof(timeUnitsPerPosition), "Must be non-zero.");
            }

            int n = inputs.Length;
            int embDim = _config.Price.EmbeddingDim;
            float invTime = (float)(1.0 / timeUnitsPerPosition);

            for (int ep = 0; ep < _trainConfig.Epochs; ep++)
            {
                float lr = ComputeLR(ep);
                if (_trainConfig.Verbose)
                    Console.WriteLine($"\n=== Epoch {ep + 1}/{_trainConfig.Epochs} ===");

                _model.ClearAllMemory();
                float epochLoss = 0f;
                int validCount = 0;

                int[] ordered = Enumerable.Range(0, n)
                    .OrderBy(i => timestamps[i])
                    .ToArray();

                foreach (int idx in ordered)
                {
                    var inp = inputs[idx];
                    if (inp?.PriceSequence == null)
                        continue;

                    int sl = inp.PriceSequence.GetLength(0);
                    if (sl < 2)
                        continue;

                    double currentTs = timestamps[idx];

                    try
                    {
                        var priceinp = SliceRows(inp.PriceSequence, 0, sl - 1);
                        var cache = new MmtacForwardCache(_config.Text.NumLayers, _config.Price.NumLayers);

                        // Context layout must match PredictWithMemory:
                        // [global?] [stored news memory] [live news from this sample] [stored price memory]
                        var ctxH = new List<float[]>();
                        var ctxT = new List<float>();
                        var ctxTypes = new List<int>();

                        float[] globalToken = null;
                        if (_config.Global.GlobalFeatureDim > 0 && inp.GlobalFeatures != null)
                        {
                            globalToken = _model.EmbedGlobalFeatures(inp.GlobalFeatures);
                            ctxH.Add(globalToken);
                            ctxT.Add(0f);
                            ctxTypes.Add(2);

                            cache.GlobalTokenEmbedded = globalToken;
                            cache.GlobalRawInput = inp.GlobalFeatures;
                        }

                        int storedNewsCount = _model.NewsMemory.Count;
                        foreach (var e in _model.NewsMemory)
                        {
                            var v = new float[embDim];
                            for (int d = 0; d < embDim; d++)
                                v[d] = e.HiddenState[d];

                            ctxH.Add(v);
                            ctxT.Add(-(float)((currentTs - e.AbsoluteTimestamp) * invTime));
                            ctxTypes.Add(0);
                        }

                        float[,] liveNewsHidden = null;
                        float[] liveNewsTimes = null;
                        int liveNewsCount = 0;

                        if (inp.NewsStories != null && inp.NewsStories.Length > 0)
                        {
                            (liveNewsHidden, liveNewsTimes) = _model.EncodeStoriesWithCache(inp.NewsStories, cache);
                            liveNewsCount = liveNewsHidden.GetLength(0);

                            for (int i = 0; i < liveNewsCount; i++)
                            {
                                var v = new float[embDim];
                                for (int d = 0; d < embDim; d++)
                                {
                                    v[d] = liveNewsHidden[i, d];
                                }

                                ctxH.Add(v);
                                ctxT.Add(liveNewsTimes[i]);
                                ctxTypes.Add(0);
                            }
                        }
                        else
                        {
                            cache.StoryCaches = null;
                            cache.StoryTokenCounts = null;
                        }

                        int storedPriceCount = _model.PriceMemory.Count;
                        foreach (var e in _model.PriceMemory)
                        {
                            var v = new float[embDim];
                            for (int d = 0; d < embDim; d++)
                            {
                                v[d] = e.HiddenState[d];
                            }

                            ctxH.Add(v);
                            ctxT.Add(-(float)((currentTs - e.AbsoluteTimestamp) * invTime));
                            ctxTypes.Add(1);
                        }

                        float[,] combinedHidden = null;
                        float[] combinedTimes = null;

                        if (ctxH.Count > 0)
                        {
                            combinedHidden = new float[ctxH.Count, embDim];
                            combinedTimes = new float[ctxH.Count];

                            for (int ci = 0; ci < ctxH.Count; ci++)
                            {
                                int cd = Math.Min(embDim, ctxH[ci].Length);
                                for (int d = 0; d < cd; d++)
                                    combinedHidden[ci, d] = ctxH[ci][d];

                                combinedTimes[ci] = ctxT[ci];
                            }

                            ApplyContextTypeEmbeddingsInPlace(combinedHidden, ctxTypes.ToArray());
                        }

                        cache.NumGlobalContext = globalToken != null ? 1 : 0;
                        cache.NumStoredNewsContext = storedNewsCount;
                        cache.NumLiveNewsContext = liveNewsCount;
                        cache.NumNewsContext = storedNewsCount + liveNewsCount;
                        cache.NumPriceContext = storedPriceCount;
                        cache.TextFinalHidden = combinedHidden;
                        cache.StoryArrivalTimes = combinedTimes;

                        var ph = _model.ForwardPriceDecoderWithCache(
                            priceinp,
                            0,
                            priceinp.GetLength(0),
                            combinedHidden,
                            combinedTimes,
                            cache,
                            true,
                            _dropoutRng);

                        cache.PriceFinalHidden = ph;

                        var (reg, range, quality, dir, midDir, conf) = _model.ProjectToOutputs(ph, cache);

                        int tsl = sl - 1;
                        int rDim = MmtacOutputConfig.RegressionOutputCount;
                        var tgtReg = new float[tsl, rDim];
                        var tgtRange = new float[tsl, 1];
                        var tgtQuality = new float[tsl, 1];
                        var tgtDir = new float[tsl, 1];
                        var tgtMid = new float[tsl, 1];

                        for (int t = 0; t < tsl; t++)
                        {
                            var tgt = targets[idx][t + 1];
                            var ra = tgt.ToRegressionArray();

                            for (int j = 0; j < rDim; j++)
                                tgtReg[t, j] = ra[j];

                            tgtRange[t, 0] = tgt.Range;
                            tgtQuality[t, 0] = tgt.Quality;
                            tgtDir[t, 0] = tgt.Direction;
                            tgtMid[t, 0] = tgt.MidWindowDirection;


                        }
                        var prevClose = new float[tsl];
                        for (int t = 0; t < tsl; t++)
                        {
                            prevClose[t] = targets[idx][t].Close;
                        }

                        float[] ct = confTargets?[idx]?.Skip(1).Take(tsl).ToArray();

                        ZeroAllGradients();

                        float loss = BackwardPass(
      reg, range, quality, dir, midDir, conf,
      tgtReg, tgtRange, tgtQuality, tgtDir, tgtMid,
      prevClose, ct, cache);

                        if (!float.IsNaN(loss) && !float.IsInfinity(loss))
                        {
                            if (_trainConfig.UseGradientClipping)
                                ClipGradients(_trainConfig.GradientClipThreshold);

                            UpdateAllParameters(lr);
                            epochLoss += loss;
                            validCount++;

                            CommitObservedSampleToMemory(
                                inp,
                                currentTs,
                                timeUnitsPerPosition,
                                maxNewsMemory,
                                maxPriceMemory);
                        }

                        if (_trainConfig.Verbose && validCount % 50 == 0)
                        {
                            Console.WriteLine($"  Sample {validCount}: Loss = {loss:F6}");
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
                    Console.WriteLine($"  Epoch {ep + 1} Avg Loss: {(validCount > 0 ? epochLoss / validCount : 0f):F6}");
                }
            }
        }
   

        // 
        // TrainBatch
        // 

        private float TrainBatch(int[] batchIndices, MultimodalInput[] allInputs, ModelTarget[][] allTargets, float[][] allConf, float lr)
        {
            ZeroAllGradients();

            float tl = 0f;
            int vc = 0;
            int minSplit = _config.PriceContext.MinHistoryLength + _config.PriceContext.MinCurrentLength + 1;

            foreach (int idx in batchIndices)
            {
                int sl = allInputs[idx].PriceSequence.GetLength(0);
                if (sl < 2)
                {
                    continue;
                }

                try
                {
                    float loss = (_config.PriceContext.Enabled && sl >= minSplit) ? TrainWithPriceContext(idx, allInputs, allTargets, allConf) : TrainSimple(idx, allInputs, allTargets, allConf);

                    if (!float.IsNaN(loss) && !float.IsInfinity(loss))
                    {
                        tl += loss;
                        vc++;
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

            if (vc == 0) return 0f;
            ScaleAllGradients(1.0f / vc);
            if (_trainConfig.UseGradientClipping) ClipGradients(_trainConfig.GradientClipThreshold);
            UpdateAllParameters(lr);
            return tl / vc;
        }

        private float TrainSimple(int idx, MultimodalInput[] allInputs, ModelTarget[][] allTargets, float[][] allConf)
        {
            var inp = allInputs[idx];
            int sl = inp.PriceSequence.GetLength(0);
            var priceInp = SliceRows(inp.PriceSequence, 0, sl - 1);
            int eff = priceInp.GetLength(0);

            BuildTargetArrays(allTargets[idx], 1, eff,
                out var tgtReg, out var tgtRange, out var tgtQuality, out var tgtDir, out var tgtMid);
            var prevClose = BuildPreviousCloseArray(allTargets[idx], 1, eff);
            float[] ct = allConf?[idx]?.Skip(1).Take(eff).ToArray();

            var wrappedInput = new MultimodalInput
            {
                PredictionTimestamp = inp.PredictionTimestamp,
                PriceSequence = priceInp,
                GlobalFeatures = inp.GlobalFeatures,
                NewsStories = inp.NewsStories
            };

            var cache = new MmtacForwardCache(_config.Text.NumLayers, _config.Price.NumLayers);
            var (reg, range, quality, dir, midDir, conf) = _model.ForwardWithCache(wrappedInput, cache, isTraining: true, dropoutRng: _dropoutRng);
            // Re-run ProjectToOutputs with cache so logits are stored
            (reg, range, quality, dir, midDir, conf) = _model.ProjectToOutputs(cache.PriceFinalHidden, cache);
            return BackwardPass(reg, range, quality, dir, midDir, conf,
                            tgtReg, tgtRange, tgtQuality, tgtDir, tgtMid, prevClose, ct, cache);
        }

        private float TrainWithPriceContext(int idx, MultimodalInput[] allInputs, ModelTarget[][] allTargets, float[][] allConf)
        {
            var inp = allInputs[idx];
            int sl = inp.PriceSequence.GetLength(0);
            int minH = _config.PriceContext.MinHistoryLength;
            int maxH = sl - _config.PriceContext.MinCurrentLength - 1;
            int sp = minH + _random.Next(maxH - minH + 1);

            var hist = SliceRows(inp.PriceSequence, 0, sp);
            var current = SliceRows(inp.PriceSequence, sp, sl - 1);
            int csl = current.GetLength(0);
            if (csl < 2) return 0f;

            BuildTargetArrays(allTargets[idx], sp + 1, csl, out var tgtReg, out var tgtRange, out var tgtQuality, out var tgtDir, out var tgtMid);

            var prevClose = BuildPreviousCloseArray(allTargets[idx], sp + 1, csl);
            float[] ct = allConf?[idx]?.Skip(sp + 1).Take(csl).ToArray();

            var priceCtxH = _model.EncodePriceHistory(hist);
            var priceCtxT = Enumerable.Range(0, sp).Select(t => -(float)(sp - t)).ToArray();

            var wrappedInput = new MultimodalInput
            {
                PredictionTimestamp = inp.PredictionTimestamp,
                PriceSequence = current,
                GlobalFeatures = inp.GlobalFeatures,
                NewsStories = AdjustStoryTimes(inp.NewsStories, sp)
            };

            var cache = new MmtacForwardCache(_config.Text.NumLayers, _config.Price.NumLayers);
            var (reg, range, quality, dir, midDir, conf) = _model.ForwardWithPriceContextAndCache(
                wrappedInput, priceCtxH, priceCtxT, cache, isTraining: true, dropoutRng: _dropoutRng);
            // Ensure logits cached
            (reg, range, quality, dir, midDir, conf) = _model.ProjectToOutputs(cache.PriceFinalHidden, cache);
            return BackwardPass(reg, range, quality, dir, midDir, conf,
                           tgtReg, tgtRange, tgtQuality, tgtDir, tgtMid, prevClose, ct, cache);
        }

        // 
        // Backward pass
        // 

        private float BackwardPass(float[,] reg,  float[,] range,  float[,] quality,  float[,] dir, float[,] midDir, float[,] conf,  float[,] tgtReg,  float[,] tgtRange, float[,] tgtQuality, float[,] tgtDir, float[,] tgtMid, float[] prevClose, float[] confTgt, MmtacForwardCache cache)
        {
            int sl = reg.GetLength(0);
            int rDim = MmtacOutputConfig.RegressionOutputCount;
            int ed = _config.Price.EmbeddingDim;

            float mseLoss = 0f;
            var dReg = new float[sl, rDim];
            for (int t = 0; t < sl; t++)
            {
                for (int j = 0; j < rDim; j++)
                {
                    float diff = reg[t, j] - tgtReg[t, j];
                    mseLoss += diff * diff;
                    dReg[t, j] = 2f * diff / sl;
                }
            }
            mseLoss /= sl;

            // Auxiliary consistency loss:
            // Encourage predicted Close to move in the same direction as the Direction target
            // relative to the previous observed close.
            float closeDirConsistencyLoss = 0f;
            float closeDirWeight = _config.Output.CloseDirectionConsistencyWeight;
            float closeDirMargin = _config.Output.CloseDirectionConsistencyMargin;

            if (closeDirWeight > 0f && prevClose != null)
            {
                for (int t = 0; t < sl; t++)
                {
                    float sign = tgtDir[t, 0] >= 0.5f ? 1f : -1f;

                    // Positive when Close moved in the correct direction by at least the margin.
                    float z = sign * (reg[t, 2] - prevClose[t] - sign * closeDirMargin);

                    // softplus(-z) = log(1 + exp(-z))
                    float lossTerm;
                    if (z > 20f)
                    {
                        lossTerm = MathF.Exp(-z);
                    }
                    else if (z < -20f)
                    {
                        lossTerm = -z;
                    }
                    else
                    {
                        lossTerm = MathF.Log(1f + MathF.Exp(-z));
                    }

                    closeDirConsistencyLoss += lossTerm;

                    // d/dz softplus(-z) = -sigmoid(-z)
                    float sigmoidNegZ;
                    if (z >= 0f)
                    {
                        float ez = MathF.Exp(-z);
                        sigmoidNegZ = ez / (1f + ez);
                    }
                    else
                    {
                        float ez = MathF.Exp(z);
                        sigmoidNegZ = 1f / (1f + ez);
                    }

                    float dClose = -sign * sigmoidNegZ * closeDirWeight / sl;
                    dReg[t, 2] += dClose;
                }

                closeDirConsistencyLoss /= sl;
            }
            var dHidden = _accel.BackpropOutputProjection(dReg, cache.PriceFinalHidden, _model.RegressionProjection, _gradients.RegressionProjectionGrad, _gradients.RegressionBiasGrad, sl, rDim, ed);

            float rangeLoss = 0f;
            for (int t = 0; t < sl; t++)
            {
                float diff = range[t, 0] - tgtRange[t, 0];
                rangeLoss += diff * diff;

                float dOutput = 2f * diff / sl * _config.Output.RangeLossWeight;
                float logit = cache.RangeLogits[t];
                float sigLogit = logit >= 0 ? 1f / (1f + MathF.Exp(-logit)) : MathF.Exp(logit) / (1f + MathF.Exp(logit));

                float dLogit = dOutput * sigLogit;
                _gradients.RangeBiasGrad[0] += dLogit;

                for (int k = 0; k < ed; k++)
                {
                    _gradients.RangeProjectionGrad[0, k] += dLogit * cache.PriceFinalHidden[t, k];
                    dHidden[t, k] += dLogit * _model.RangeProjection[0, k];
                }
            }
            rangeLoss /= sl;

            float qualityLoss = 0f;
            for (int t = 0; t < sl; t++)
            {
                float p = quality[t, 0];
                float diff = p - tgtQuality[t, 0];
                qualityLoss += diff * diff;

                float dOutput = 2f * diff / sl * _config.Output.QualityLossWeight;
                float dLogit = dOutput * p * (1f - p);
                _gradients.QualityBiasGrad[0] += dLogit;

                for (int k = 0; k < ed; k++)
                {
                    _gradients.QualityProjectionGrad[0, k] += dLogit * cache.PriceFinalHidden[t, k];
                    dHidden[t, k] += dLogit * _model.QualityProjection[0, k];
                }
            }
            qualityLoss /= sl;

            float dirLoss = 0f;
            for (int t = 0; t < sl; t++)
            {
                float p = dir[t, 0];
                float y = tgtDir[t, 0];
                float pc = Math.Clamp(p, 1e-7f, 1f - 1e-7f);
                dirLoss -= y * MathF.Log(pc) + (1f - y) * MathF.Log(1f - pc);

                float dLogit = (p - y) * _config.Output.DirectionLossWeight / sl;
                _gradients.DirectionBiasGrad[0] += dLogit;

                for (int k = 0; k < ed; k++)
                {
                    _gradients.DirectionProjectionGrad[0, k] += dLogit * cache.PriceFinalHidden[t, k];
                    dHidden[t, k] += dLogit * _model.DirectionProjection[0, k];
                }
            }
            dirLoss /= sl;

            float midDirLoss = 0f;
            for (int t = 0; t < sl; t++)
            {
                float p = midDir[t, 0];
                float y = tgtMid[t, 0];
                float pc = Math.Clamp(p, 1e-7f, 1f - 1e-7f);
                midDirLoss -= y * MathF.Log(pc) + (1f - y) * MathF.Log(1f - pc);

                float dLogit = (p - y) * _config.Output.MidDirectionLossWeight / sl;
                _gradients.MidDirectionBiasGrad[0] += dLogit;

                for (int k = 0; k < ed; k++)
                {
                    _gradients.MidDirectionProjectionGrad[0, k] += dLogit * cache.PriceFinalHidden[t, k];
                    dHidden[t, k] += dLogit * _model.MidDirectionProjection[0, k];
                }
            }
            midDirLoss /= sl;

            float confLoss = 0f;
            float confWeight = (_config.Output.UseConfidenceHead && _trainConfig.ConfidenceLossWeight <= 0f) ? 1f : _trainConfig.ConfidenceLossWeight;

            if (_config.Output.UseConfidenceHead && conf != null && confWeight > 0f)
            {
                for (int t = 0; t < sl; t++)
                {
                    float p = conf[t, 0];
                    float y = confTgt != null ? confTgt[t] : MathF.Exp(-5f * MathF.Sqrt(
                            Enumerable.Range(0, rDim).Sum(j =>
                            {
                                float d2 = reg[t, j] - tgtReg[t, j];
                                return d2 * d2;
                            }) / rDim));

                    float pc = Math.Clamp(p, 1e-7f, 1f - 1e-7f);
                    confLoss -= y * MathF.Log(pc) + (1f - y) * MathF.Log(1f - pc);

                    float dLogit = (p - y) * confWeight / sl;
                    _gradients.ConfidenceBiasGrad[0] += dLogit;

                    for (int k = 0; k < ed; k++)
                    {
                        _gradients.ConfidenceProjectionGrad[0, k] += dLogit * cache.PriceFinalHidden[t, k];
                        dHidden[t, k] += dLogit * _model.ConfidenceProjection[0, k];
                    }
                }
                confLoss /= sl;
            }

            var dNewsHidden = BackpropPriceDecoder(dHidden, cache);

            if (!_config.Text.Freeze && cache.TextFinalHidden != null && dNewsHidden != null && cache.StoryCaches != null)
            {
                BackpropMultiStoryTextEncoder(dNewsHidden, cache);
            }

            if (_config.Global.GlobalFeatureDim > 0 && cache.GlobalTokenEmbedded != null && cache.GlobalRawInput != null && _gradients.GlobalFeatureProjectionGrad != null)
            {
                int gd = _config.Global.GlobalFeatureDim;
                for (int d = 0; d < ed; d++)
                {
                    float dg = _gradients.ContextTypeEmbeddingGrad[2, d];
                    _gradients.GlobalFeatureBiasGrad[d] += dg;

                    for (int g = 0; g < gd; g++)
                    {
                        _gradients.GlobalFeatureProjectionGrad[d, g] += dg * cache.GlobalRawInput[g];
                    }
                }
            }

            return mseLoss
          + closeDirWeight * closeDirConsistencyLoss
          + _config.Output.RangeLossWeight * rangeLoss
          + _config.Output.QualityLossWeight * qualityLoss
          + _config.Output.DirectionLossWeight * dirLoss
          + _config.Output.MidDirectionLossWeight * midDirLoss
          + confWeight * confLoss;
        }
        
        private float[,] BackpropPriceDecoder(float[,] dOut, MmtacForwardCache cache)
        {
            int ed = _config.Price.EmbeddingDim;
            int nh = _config.Price.NumHeads;
            int hd = ed / nh;
            float scale = 1.0f / MathF.Sqrt(hd);

            int numGlobal = cache.NumGlobalContext;
            int numStoredNews = cache.NumStoredNewsContext;
            int numLiveNews = cache.NumLiveNewsContext;
            int numNews = cache.NumNewsContext;
            int numPriceCtx = cache.NumPriceContext;
            int totalContext = cache.TextFinalHidden?.GetLength(0) ?? 0;
            bool hasContext = (totalContext > 0);

            if (totalContext > 0 && numNews == 0 && numPriceCtx == 0 && numGlobal == 0)
            {
                numNews = totalContext;
                numStoredNews = 0;
                numLiveNews = totalContext;
            }

            int liveNewsOffset = numGlobal + numStoredNews;
            int priceOffset = numGlobal + numNews;

            float[,] dLiveNewsHidden = numLiveNews > 0 ? new float[numLiveNews, ed] : null;

            var dX = dOut;

            for (int layer = _config.Price.NumLayers - 1; layer >= 0; layer--)
            {
                var block = _model.PriceBlocks[layer];
                var bc = cache.PriceBlockCaches[layer];
                var bg = _gradients.PriceBlockGrads[layer];
                int seqLen = dX.GetLength(0);

                // FFN sublayer
                var (dFfnRes, dGammaFFN, dBetaFFN) = _accel.LayerNormBackward(dX, bc.LNFFNCache.Normalized, block.LNFFNGamma, bc.LNFFNCache.Input, bc.LNFFNCache.Mean, bc.LNFFNCache.Variance);
                _accel.VectorAccumulate(bg.LNFFNGrads.GammaGrad, dGammaFFN);
                _accel.VectorAccumulate(bg.LNFFNGrads.BetaGrad, dBetaFFN);

                var dFfnIn = new float[seqLen, ed];
                for (int i = 0; i < seqLen; i++)
                {
                    var gr = new float[ed]; for (int j = 0; j < ed; j++) gr[j] = dFfnRes[i, j];
                    block.FeedForwardNetwork.ForwardPassOnly(bc.FFNInputRows[i]);
                    var di = block.FeedForwardNetwork.ComputeInputGradient(gr, _priceFFNWeightGrads[layer], _priceFFNBiasGrads[layer]);
                    for (int j = 0; j < ed; j++) dFfnIn[i, j] = di[j];
                }
                _accel.MatrixAddInPlace(dFfnIn, dFfnRes);

                // Cross-attention sublayer
                var (dCrossRes, dGammaCross, dBetaCross) = _accel.LayerNormBackward(dFfnIn, bc.LNCrossCache.Normalized, block.LnCrossGamma, bc.LNCrossCache.Input, bc.LNCrossCache.Mean, bc.LNCrossCache.Variance);
                _accel.VectorAccumulate(bg.LNCrossGrads.GammaGrad, dGammaCross);
                _accel.VectorAccumulate(bg.LNCrossGrads.BetaGrad, dBetaCross);

                float[,] dSelfIn;
                if (hasContext && bc.CrossQ != null && totalContext > 0)
                {
                    var crossGrads = bg.CrossAttnGrads;
                    var dCrossComb = new float[seqLen, ed];
                    _accel.BackpropLinearProjection(bc.CrossAttnOutput, dCrossRes, block.CrossAttention.WO, crossGrads.WO_Grad, crossGrads.BiasO_Grad, dCrossComb);

                    var (dQ, dK, dV, dDecayBias) = BackpropTimeDecayedAttn(bc.CrossQ, bc.CrossK, bc.CrossV, dCrossComb, bc.CrossAttentionWeights, bc.TimeDiffs, block);
                    RotaryPositionEmbedding.ApplyBackwardInPlace(dQ, dK, nh);
                    /*
                    if (bc.DecayCache != null && dDecayBias != null)
                    {
                        var (decayParamGrads, dQueryEmb, dKeyEmb) = block.DecayNetwork.Backward(dDecayBias, bc.DecayCache);
                        AccumulateDecayGrads(bg.DecayGrads, decayParamGrads);
                        if (dLiveNewsHidden != null)
                            for (int i = 0; i < numLiveNews; i++)
                                for (int j = 0; j < ed; j++)
                                    dLiveNewsHidden[i, j] += dKeyEmb[liveNewsOffset + i, j];
                        for (int i = 0; i < seqLen; i++)
                            for (int j = 0; j < ed; j++)
                                dCrossRes[i, j] += dQueryEmb[i, j];
                    }*/
                    if (bc.DecayCache != null && dDecayBias != null)
                    {
                        var (decayParamGrads, dQueryEmb, dKeyEmb) =
                            block.DecayNetwork.Backward(dDecayBias, bc.DecayCache);

                        AccumulateDecayGrads(bg.DecayGrads, decayParamGrads);

                        if (dKeyEmb != null)
                        {
                            // Global token rows
                            for (int gi = 0; gi < numGlobal; gi++)
                            {
                                for (int j = 0; j < ed; j++)
                                {
                                    _gradients.ContextTypeEmbeddingGrad[2, j] += dKeyEmb[gi, j];
                                }
                            }

                            // News rows: all news contributes to the shared news type embedding,
                            // and live news additionally backprops into the text encoder.
                            for (int i = 0; i < numNews; i++)
                            {
                                int ctxIdx = numGlobal + i;
                                bool isLive = i >= numStoredNews;
                                int liveIdx = i - numStoredNews;

                                for (int j = 0; j < ed; j++)
                                {
                                    float g = dKeyEmb[ctxIdx, j];
                                    _gradients.ContextTypeEmbeddingGrad[0, j] += g;

                                    if (isLive && dLiveNewsHidden != null)
                                    {
                                        dLiveNewsHidden[liveIdx, j] += g;
                                    }
                                }
                            }

                            // Stored price-memory rows contribute to the price-memory type embedding.
                            for (int i = 0; i < numPriceCtx; i++)
                            {
                                int ctxIdx = priceOffset + i;

                                for (int j = 0; j < ed; j++)
                                {
                                    _gradients.ContextTypeEmbeddingGrad[1, j] += dKeyEmb[ctxIdx, j];
                                }
                            }
                        }

                        if (dQueryEmb != null)
                        {
                            for (int i = 0; i < seqLen; i++)
                            {
                                for (int j = 0; j < ed; j++)
                                {
                                    dCrossRes[i, j] += dQueryEmb[i, j];
                                }
                            }
                        }
                    }
                    var dFromQ = new float[seqLen, ed];
                    var dCtxFromK = new float[totalContext, ed];
                    var dCtxFromV = new float[totalContext, ed];
                    _accel.BackpropLinearProjection(bc.NormedSelf, dQ, block.CrossAttention.WQ, crossGrads.WQ_Grad, crossGrads.BiasQ_Grad, dFromQ);
                    _accel.BackpropLinearProjection(cache.TextFinalHidden, dK, block.CrossAttention.WK, crossGrads.WK_Grad, crossGrads.BiasK_Grad, dCtxFromK);
                    _accel.BackpropLinearProjection(cache.TextFinalHidden, dV, block.CrossAttention.WV, crossGrads.WV_Grad, crossGrads.BiasV_Grad, dCtxFromV);

                    if (dLiveNewsHidden != null)
                        for (int i = 0; i < numLiveNews; i++)
                            for (int j = 0; j < ed; j++)
                                dLiveNewsHidden[i, j] += dCtxFromK[liveNewsOffset + i, j] + dCtxFromV[liveNewsOffset + i, j];

                    for (int i = 0; i < numPriceCtx; i++)
                        for (int j = 0; j < ed; j++)
                            _gradients.ContextTypeEmbeddingGrad[1, j] += dCtxFromK[priceOffset + i, j] + dCtxFromV[priceOffset + i, j];

                    if (numGlobal > 0)
                        for (int j = 0; j < ed; j++)
                            _gradients.ContextTypeEmbeddingGrad[2, j] += dCtxFromK[0, j] + dCtxFromV[0, j];

                    for (int i = 0; i < numNews; i++)
                        for (int j = 0; j < ed; j++)
                            _gradients.ContextTypeEmbeddingGrad[0, j] += dCtxFromK[numGlobal + i, j] + dCtxFromV[numGlobal + i, j];

                    _accel.MatrixAddInPlace(dFromQ, dCrossRes);
                    dSelfIn = dFromQ;
                }
                else
                {
                    dSelfIn = dCrossRes;
                }

                // Self-attention sublayer
                var (dSelfRes, dGammaSelf, dBetaSelf) = _accel.LayerNormBackward(dSelfIn, bc.LNSelfCache.Normalized, block.LNSelfGamma, bc.LNSelfCache.Input, bc.LNSelfCache.Mean, bc.LNSelfCache.Variance);
                _accel.VectorAccumulate(bg.LNSelfGrads.GammaGrad, dGammaSelf);
                _accel.VectorAccumulate(bg.LNSelfGrads.BetaGrad, dBetaSelf);

                var selfGrads = bg.SelfAttnGrads;
                var dSelfComb = new float[seqLen, ed];
                _accel.BackpropLinearProjection(bc.SelfAttnOutput, dSelfRes, block.SelfAttention.WO, selfGrads.WO_Grad, selfGrads.BiasO_Grad, dSelfComb);
                var (dQSelf, dKSelf, dVSelf) = _accel.MultiHeadAttentionBackward(bc.SelfQ, bc.SelfK, bc.SelfV, dSelfComb, nh, scale, _config.Price.UseDecoderOnly);
                RotaryPositionEmbedding.ApplyBackwardInPlace(dQSelf, dKSelf, nh);

                var dBlockIn = new float[seqLen, ed];
                _accel.BackpropLinearProjection(bc.BlockInput, dQSelf, block.SelfAttention.WQ, selfGrads.WQ_Grad, selfGrads.BiasQ_Grad, dBlockIn);
                _accel.BackpropLinearProjection(bc.BlockInput, dKSelf, block.SelfAttention.WK, selfGrads.WK_Grad, selfGrads.BiasK_Grad, dBlockIn);
                _accel.BackpropLinearProjection(bc.BlockInput, dVSelf, block.SelfAttention.WV, selfGrads.WV_Grad, selfGrads.BiasV_Grad, dBlockIn);
                _accel.MatrixAddInPlace(dBlockIn, dSelfRes);
                dX = dBlockIn;
            }

            _accel.BackpropInputProjection(dX, cache.PriceContinuousInput,
                _gradients.PriceInputProjectionGrad, _gradients.PriceInputProjectionBiasGrad,
                dX.GetLength(0), _config.Price.EmbeddingDim, _config.Price.InputFeatureDim);

            return dLiveNewsHidden;
        }

        private (float[,] dQ, float[,] dK, float[,] dV, float[,,] dDecayBias) BackpropTimeDecayedAttn(float[,] Q, float[,] K, float[,] V, float[,] dOutput, float[][,] attnW, float[,] timeDiffs, TacamtBlock block)
        {
            int psl = Q.GetLength(0), tsl = K.GetLength(0);
            int ed = _config.Price.EmbeddingDim, nh = _config.Price.NumHeads, hd = ed / nh;
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
                        float dot = 0f;
                        for (int dd = 0; dd < hd; dd++) 
                        {
                            int di = si + dd; 
                            dV[sv, di] += w[p, sv] * dOutput[p, di];
                            dot += dOutput[p, di] * V[sv, di]; 
                        }
                        dw[sv] = dot;
                    }
                    float dotWD = 0f;
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

        // 
        // Text encoder backward
        // 

        private void BackpropMultiStoryTextEncoder(float[,] dSH, MmtacForwardCache cache)
        {
            int ns = cache.StoryCaches.Count;
            int ed = _config.Text.EmbeddingDim;
            for (int s = 0; s < ns; s++)
            {
                int tc = cache.StoryTokenCounts[s];
                float inv = 1.0f / tc;
                var dTH = new float[tc, ed];
                for (int t = 0; t < tc; t++)
                    for (int d = 0; d < ed; d++)
                        dTH[t, d] = dSH[s, d] * inv;
                BackpropTextEncoder(dTH, cache.StoryCaches[s]);
            }
        }

        private void BackpropTextEncoder(float[,] dTH, MmtacForwardCache cache)
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
                var l2c = cache.TextLN2Caches[layer];

                var (dFR, dG2, dB2) = _accel.LayerNormBackward(dX, l2c.Normalized, b.LN2Gamma, l2c.Input, l2c.Mean, l2c.Variance);
                _accel.VectorAccumulate(_gradients.TextLN2Grads[layer].GammaGrad, dG2);
                _accel.VectorAccumulate(_gradients.TextLN2Grads[layer].BetaGrad, dB2);

                int sl = dFR.GetLength(0);
                var dFFNIn = new float[sl, ed];
                for (int i = 0; i < sl; i++)
                {
                    var rg = new float[ed]; for (int j = 0; j < ed; j++) rg[j] = dFR[i, j];
                    var ir = new float[ed]; for (int j = 0; j < ed; j++) ir[j] = cache.TextFFNInputs[layer][i, j];
                    b.FeedForwardNetwork.ForwardPassOnly(ir);
                    var di = b.FeedForwardNetwork.ComputeInputGradient(rg, _textFFNWeightGrads[layer], _textFFNBiasGrads[layer]);
                    for (int j = 0; j < ed; j++) dFFNIn[i, j] = di[j];
                }
                _accel.MatrixAddInPlace(dFFNIn, dFR);

                var l1c = cache.TextLN1Caches[layer];
                var (dAR, dG1, dB1) = _accel.LayerNormBackward(dFFNIn, l1c.Normalized, b.LN1Gamma, l1c.Input, l1c.Mean, l1c.Variance);
                _accel.VectorAccumulate(_gradients.TextLN1Grads[layer].GammaGrad, dG1);
                _accel.VectorAccumulate(_gradients.TextLN1Grads[layer].BetaGrad, dB1);

                var ac = cache.TextAttentionCaches[layer];
                var dC = new float[sl, ed];
                _accel.BackpropLinearProjection(ac.AttentionOutput, dAR, b.Attention.WO, ag.WO_Grad, ag.BiasO_Grad, dC);
                var (dQ, dK, dV) = _accel.MultiHeadAttentionBackward(ac.Q, ac.K, ac.V, dC, nh, s, _config.Text.UseDecoderOnly);
                RotaryPositionEmbedding.ApplyBackwardInPlace(dQ, dK, nh);

                var dI = new float[sl, ed];
                _accel.BackpropLinearProjection(ac.Input, dQ, b.Attention.WQ, ag.WQ_Grad, ag.BiasQ_Grad, dI);
                _accel.BackpropLinearProjection(ac.Input, dK, b.Attention.WK, ag.WK_Grad, ag.BiasK_Grad, dI);
                _accel.BackpropLinearProjection(ac.Input, dV, b.Attention.WV, ag.WV_Grad, ag.BiasV_Grad, dI);
                _accel.MatrixAddInPlace(dI, dAR);
                dX = dI;
            }
            _accel.AccumulateTokenEmbeddingGrad(_gradients.TextEmbeddingGrad, dX, cache.TextTokenIds, dX.GetLength(0), ed);
        }

        private void CommitObservedSampleToMemory(MultimodalInput inp, double currentTs, double timeUnitsPerPosition, int maxNewsMemory, int maxPriceMemory)
        {
            if (inp == null || inp.PriceSequence == null)
            {
                return;
            }

            int embDim = _config.Price.EmbeddingDim;
            float invTime = (float)(1.0 / timeUnitsPerPosition);

            float[,] liveNewsHiddenPost = null;
            float[] liveNewsTimesPost = null;

            if (inp.NewsStories != null && inp.NewsStories.Length > 0)
            {
                (liveNewsHiddenPost, liveNewsTimesPost) = _model.EncodeStoriesForMemory(inp.NewsStories);
            }

            var postCtxH = new List<float[]>();
            var postCtxT = new List<float>();
            var postCtxTypes = new List<int>();

            if (_config.Global.GlobalFeatureDim > 0 && inp.GlobalFeatures != null)
            {
                var postGlobalToken = _model.EmbedGlobalFeatures(inp.GlobalFeatures);
                postCtxH.Add(postGlobalToken);
                postCtxT.Add(0f);
                postCtxTypes.Add(2);
            }

            foreach (var e in _model.NewsMemory)
            {
                var v = new float[embDim];
                for (int d = 0; d < embDim; d++)
                {
                    v[d] = e.HiddenState[d];
                }

                postCtxH.Add(v);
                postCtxT.Add(-(float)((currentTs - e.AbsoluteTimestamp) * invTime));
                postCtxTypes.Add(0);
            }

            if (liveNewsHiddenPost != null)
            {
                int ns = liveNewsHiddenPost.GetLength(0);

                for (int i = 0; i < ns; i++)
                {
                    var v = new float[embDim];
                    for (int d = 0; d < embDim; d++)
                    {
                        v[d] = liveNewsHiddenPost[i, d];
                    }

                    postCtxH.Add(v);
                    postCtxT.Add(liveNewsTimesPost[i]);
                    postCtxTypes.Add(0);
                }
            }

            foreach (var e in _model.PriceMemory)
            {
                var v = new float[embDim];
                for (int d = 0; d < embDim; d++)
                {
                    v[d] = e.HiddenState[d];
                }

                postCtxH.Add(v);
                postCtxT.Add(-(float)((currentTs - e.AbsoluteTimestamp) * invTime));
                postCtxTypes.Add(1);
            }

            float[,] postCombinedHidden = null;
            float[] postCombinedTimes = null;

            if (postCtxH.Count > 0)
            {
                postCombinedHidden = new float[postCtxH.Count, embDim];
                postCombinedTimes = new float[postCtxH.Count];

                for (int ci = 0; ci < postCtxH.Count; ci++)
                {
                    int cd = Math.Min(embDim, postCtxH[ci].Length);
                    for (int d = 0; d < cd; d++)
                    {
                        postCombinedHidden[ci, d] = postCtxH[ci][d];
                    }

                    postCombinedTimes[ci] = postCtxT[ci];
                }

                ApplyContextTypeEmbeddingsInPlace(postCombinedHidden, postCtxTypes.ToArray());
            }

            // IMPORTANT: commit the FULL observed sample, not the training prefix.
            var phUpdated = _model.ForwardPriceDecoderForMemoryBuild(
                inp.PriceSequence,
                postCombinedHidden,
                postCombinedTimes);

            if (liveNewsHiddenPost != null)
            {
                int ns = liveNewsHiddenPost.GetLength(0);

                for (int i = 0; i < ns; i++)
                {
                    var hv = new float[embDim];
                    for (int d = 0; d < embDim; d++)
                    {
                        hv[d] = liveNewsHiddenPost[i, d];
                    }

                    _model.NewsMemory.Add(new NewsMemoryEntry
                    {
                        HiddenState = hv,
                        AbsoluteTimestamp = currentTs + inp.NewsStories[i].ArrivalTime * timeUnitsPerPosition
                    });
                }
            }

            for (int t = 0; t < phUpdated.GetLength(0); t++)
            {
                var pv = new float[embDim];
                for (int d = 0; d < embDim; d++)
                {
                    pv[d] = phUpdated[t, d];
                }

                _model.PriceMemory.Add(new PriceMemoryEntry
                {
                    HiddenState = pv,
                    AbsoluteTimestamp = currentTs + t * timeUnitsPerPosition
                });
            }

            _model.LastPriceTimestamp =
                currentTs + Math.Max(0, inp.PriceSequence.GetLength(0) - 1) * timeUnitsPerPosition;

            _model.PruneNewsMemory(maxNewsMemory);
            _model.PricePruneMemory(maxPriceMemory);
        }

        private ModelPrediction PredictWithCurrentMemoryNoCommit(MultimodalInput input, double currentAbsoluteTimestamp, double timeUnitsPerPosition = 1.0)
        {
            if (input == null) throw new ArgumentNullException(nameof(input));
            if (input.PriceSequence == null) throw new ArgumentNullException(nameof(input.PriceSequence));

            int embDim = _config.Price.EmbeddingDim;

            var ctxH = new List<float[]>();
            var ctxT = new List<float>();
            var ctxTypes = new List<int>();

            if (_config.Global.GlobalFeatureDim > 0 && input.GlobalFeatures != null)
            {
                var globalToken = _model.EmbedGlobalFeatures(input.GlobalFeatures);
                ctxH.Add(globalToken);
                ctxT.Add(0f);
                ctxTypes.Add(2);
            }

            foreach (var e in _model.NewsMemory)
            {
                float relTime = -(float)((currentAbsoluteTimestamp - e.AbsoluteTimestamp) / timeUnitsPerPosition);
                var v = new float[embDim];

                for (int d = 0; d < embDim; d++)
                {
                    v[d] = e.HiddenState[d];
                }

                ctxH.Add(v);
                ctxT.Add(relTime);
                ctxTypes.Add(0);
            }

            if (input.NewsStories != null && input.NewsStories.Length > 0)
            {
                var (newSH, nst) = _model.EncodeStoriesForMemory(input.NewsStories);

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

            foreach (var e in _model.PriceMemory)
            {
                float relTime = -(float)((currentAbsoluteTimestamp - e.AbsoluteTimestamp) / timeUnitsPerPosition);
                var v = new float[embDim];

                for (int d = 0; d < embDim; d++)
                {
                    v[d] = e.HiddenState[d];
                }

                ctxH.Add(v);
                ctxT.Add(relTime);
                ctxTypes.Add(1);
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

                ApplyContextTypeEmbeddingsInPlace(cH, ctxTypes.ToArray());
            }

            var priceHidden = _model.ForwardPriceDecoderForMemoryBuild(input.PriceSequence, cH, cT);
            var (reg, range, quality, dir, midDir, conf) = _model.ProjectToOutputs(priceHidden);

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
        #region Gradient utilities

        private void AccumulateDecayGrads(ContentAwareDecayGradients tgt, ContentAwareDecayGradients src)
        {
            int nh = src.B2Grad.Length, cd = src.QueryProjectionGrad.GetLength(2),
                pd = src.QueryProjectionGrad.GetLength(1), hd = src.B1Grad.GetLength(1),
                mid = src.W1Grad.GetLength(2), ntb = src.TimeLogFreqGrad.GetLength(1),
                rawDim = src.TimeProjGrad.GetLength(2);
            for (int h = 0; h < nh; h++)
            {
                tgt.LogBaseDecayRateGrad[h] += src.LogBaseDecayRateGrad[h];
                tgt.B2Grad[h] += src.B2Grad[h];
                for (int p = 0; p < pd; p++)
                {
                    for (int d = 0; d < cd; d++) { tgt.QueryProjectionGrad[h, p, d] += src.QueryProjectionGrad[h, p, d]; tgt.KeyProjectionGrad[h, p, d] += src.KeyProjectionGrad[h, p, d]; }
                    tgt.QueryProjectionBiasGrad[h, p] += src.QueryProjectionBiasGrad[h, p];
                    tgt.KeyProjectionBiasGrad[h, p] += src.KeyProjectionBiasGrad[h, p];
                    for (int r = 0; r < rawDim; r++) tgt.TimeProjGrad[h, p, r] += src.TimeProjGrad[h, p, r];
                    tgt.TimeProjBiasGrad[h, p] += src.TimeProjBiasGrad[h, p];
                    for (int q = 0; q < pd; q++) tgt.MemAttnOutputWGrad[h, p, q] += src.MemAttnOutputWGrad[h, p, q];
                    tgt.MemAttnOutputBGrad[h, p] += src.MemAttnOutputBGrad[h, p];
                }
                for (int bIdx = 0; bIdx < ntb; bIdx++) tgt.TimeLogFreqGrad[h, bIdx] += src.TimeLogFreqGrad[h, bIdx];
                for (int j = 0; j < hd; j++) { tgt.B1Grad[h, j] += src.B1Grad[h, j]; tgt.W2Grad[h, j] += src.W2Grad[h, j]; for (int k = 0; k < mid; k++) tgt.W1Grad[h, j, k] += src.W1Grad[h, j, k]; }
            }
        }

        private void ZeroAllGradients()
        {
            _gradients.Zero();
            for (int i = 0; i < _config.Text.NumLayers; i++) { foreach (var w in _textFFNWeightGrads[i]) _accel.ZeroMatrix(w); foreach (var b in _textFFNBiasGrads[i]) _accel.ZeroVector(b); }
            for (int i = 0; i < _config.Price.NumLayers; i++) { foreach (var w in _priceFFNWeightGrads[i]) _accel.ZeroMatrix(w); foreach (var b in _priceFFNBiasGrads[i]) _accel.ZeroVector(b); }
        }

        private void ScaleAllGradients(float scale)
        {
            ScaleMatrix(_gradients.TextEmbeddingGrad, scale);
            for (int i = 0; i < _config.Text.NumLayers; i++)
            {
                ScaleAttnGrads(_gradients.TextAttnGrads[i], scale);
                ScaleLNGrads(_gradients.TextLN1Grads[i], scale);
                ScaleLNGrads(_gradients.TextLN2Grads[i], scale);
                foreach (var w in _textFFNWeightGrads[i]) ScaleMatrix(w, scale);
                foreach (var b in _textFFNBiasGrads[i]) ScaleVector(b, scale);
            }
            ScaleMatrix(_gradients.PriceInputProjectionGrad, scale);
            ScaleVector(_gradients.PriceInputProjectionBiasGrad, scale);
            if (_gradients.GlobalFeatureProjectionGrad != null)
            {
                ScaleMatrix(_gradients.GlobalFeatureProjectionGrad, scale);
                ScaleVector(_gradients.GlobalFeatureBiasGrad, scale);
            }
            for (int i = 0; i < _config.Price.NumLayers; i++)
            {
                var bg = _gradients.PriceBlockGrads[i];
                ScaleAttnGrads(bg.SelfAttnGrads, scale); ScaleAttnGrads(bg.CrossAttnGrads, scale);
                ScaleLNGrads(bg.LNSelfGrads, scale); ScaleLNGrads(bg.LNCrossGrads, scale); ScaleLNGrads(bg.LNFFNGrads, scale);
                foreach (var w in _priceFFNWeightGrads[i]) ScaleMatrix(w, scale);
                foreach (var b in _priceFFNBiasGrads[i]) ScaleVector(b, scale);
            }
            ScaleMatrix(_gradients.RegressionProjectionGrad, scale); ScaleVector(_gradients.RegressionBiasGrad, scale);
            ScaleMatrix(_gradients.RangeProjectionGrad, scale); ScaleVector(_gradients.RangeBiasGrad, scale);
            ScaleMatrix(_gradients.QualityProjectionGrad, scale); ScaleVector(_gradients.QualityBiasGrad, scale);
            ScaleMatrix(_gradients.DirectionProjectionGrad, scale); ScaleVector(_gradients.DirectionBiasGrad, scale);
            ScaleMatrix(_gradients.MidDirectionProjectionGrad, scale); ScaleVector(_gradients.MidDirectionBiasGrad, scale);
            if (_gradients.ConfidenceProjectionGrad != null)
            {
                ScaleMatrix(_gradients.ConfidenceProjectionGrad, scale);
                ScaleVector(_gradients.ConfidenceBiasGrad, scale);
            }
            ScaleMatrix(_gradients.ContextTypeEmbeddingGrad, scale);
        }

        private void ClipGradients(float threshold)
        {
            float norm = ComputeGradientNorm();
            if (norm > threshold) ScaleAllGradients(threshold / norm);
        }

        private float ComputeGradientNorm()
        {
            float sum = 0f;
            if (!_config.Text.Freeze)
            {
                sum += _accel.MatrixSquaredNorm(_gradients.TextEmbeddingGrad);
                for (int i = 0; i < _config.Text.NumLayers; i++)
                    sum += AttnGradNorm(_gradients.TextAttnGrads[i]) + LNGradNorm(_gradients.TextLN1Grads[i]) + LNGradNorm(_gradients.TextLN2Grads[i]) + FFNGradNorm(i, text: true);
            }
            sum += _accel.MatrixSquaredNorm(_gradients.PriceInputProjectionGrad);
            for (int i = 0; i < _config.Price.NumLayers; i++)
            {
                var bg = _gradients.PriceBlockGrads[i];
                sum += AttnGradNorm(bg.SelfAttnGrads) + AttnGradNorm(bg.CrossAttnGrads)
                     + LNGradNorm(bg.LNSelfGrads) + LNGradNorm(bg.LNCrossGrads) + LNGradNorm(bg.LNFFNGrads)
                     + FFNGradNorm(i, text: false);
            }
            sum += _accel.MatrixSquaredNorm(_gradients.RegressionProjectionGrad) + _accel.VectorSquaredNorm(_gradients.RegressionBiasGrad);
            sum += _accel.MatrixSquaredNorm(_gradients.RangeProjectionGrad) + _accel.VectorSquaredNorm(_gradients.RangeBiasGrad);
            sum += _accel.MatrixSquaredNorm(_gradients.QualityProjectionGrad) + _accel.VectorSquaredNorm(_gradients.QualityBiasGrad);
            sum += _accel.MatrixSquaredNorm(_gradients.DirectionProjectionGrad) + _accel.VectorSquaredNorm(_gradients.DirectionBiasGrad);
            sum += _accel.MatrixSquaredNorm(_gradients.MidDirectionProjectionGrad) + _accel.VectorSquaredNorm(_gradients.MidDirectionBiasGrad);
            return MathF.Sqrt(sum);
        }

        private float AttnGradNorm(AttentionGradients ag) =>
            _accel.MatrixSquaredNorm(ag.WQ_Grad) + _accel.MatrixSquaredNorm(ag.WK_Grad)
          + _accel.MatrixSquaredNorm(ag.WV_Grad) + _accel.MatrixSquaredNorm(ag.WO_Grad);
        private float LNGradNorm(LayerNormGradients lg) =>
            _accel.VectorSquaredNorm(lg.GammaGrad) + _accel.VectorSquaredNorm(lg.BetaGrad);
        private float FFNGradNorm(int layer, bool text) =>
            (text ? _textFFNWeightGrads : _priceFFNWeightGrads)[layer].Sum(w => _accel.MatrixSquaredNorm(w));

        private void UpdateAllParameters(float lr)
        {
            if (!_config.Text.Freeze)
            {
                _accel.MatrixUpdate(_model.TextTokenEmbedding, _gradients.TextEmbeddingGrad, lr);
                for (int i = 0; i < _config.Text.NumLayers; i++)
                {
                    var b = _model.TextBlocks[i]; var ag = _gradients.TextAttnGrads[i];
                    UpdateAttn(b.Attention, ag, lr);
                    _accel.VectorUpdate(b.LN1Gamma, _gradients.TextLN1Grads[i].GammaGrad, lr); _accel.VectorUpdate(b.LN1Beta, _gradients.TextLN1Grads[i].BetaGrad, lr);
                    _accel.VectorUpdate(b.LN2Gamma, _gradients.TextLN2Grads[i].GammaGrad, lr); _accel.VectorUpdate(b.LN2Beta, _gradients.TextLN2Grads[i].BetaGrad, lr);
                    b.FeedForwardNetwork.ApplyExternalGradients(_textFFNWeightGrads[i], _textFFNBiasGrads[i], lr);
                }
            }

            _accel.MatrixUpdate(_model.PriceInputProjection, _gradients.PriceInputProjectionGrad, lr);
            _accel.VectorUpdate(_model.PriceInputProjectionBias, _gradients.PriceInputProjectionBiasGrad, lr);

            if (_model.GlobalFeatureProjection != null && _gradients.GlobalFeatureProjectionGrad != null)
            {
                _accel.MatrixUpdate(_model.GlobalFeatureProjection, _gradients.GlobalFeatureProjectionGrad, lr);
                _accel.VectorUpdate(_model.GlobalFeatureBias, _gradients.GlobalFeatureBiasGrad, lr);
            }

            for (int i = 0; i < _config.Price.NumLayers; i++)
            {
                var b = _model.PriceBlocks[i]; var bg = _gradients.PriceBlockGrads[i];
                UpdateAttn(b.SelfAttention, bg.SelfAttnGrads, lr);
                UpdateAttn(b.CrossAttention, bg.CrossAttnGrads, lr);
                if (bg.DecayGrads != null) UpdateDecayNetwork(b.DecayNetwork, bg.DecayGrads, lr);
                _accel.VectorUpdate(b.LNSelfGamma, bg.LNSelfGrads.GammaGrad, lr); _accel.VectorUpdate(b.LNSelfBeta, bg.LNSelfGrads.BetaGrad, lr);
                _accel.VectorUpdate(b.LnCrossGamma, bg.LNCrossGrads.GammaGrad, lr); _accel.VectorUpdate(b.LnCrossBeta, bg.LNCrossGrads.BetaGrad, lr);
                _accel.VectorUpdate(b.LNFFNGamma, bg.LNFFNGrads.GammaGrad, lr); _accel.VectorUpdate(b.LNFFNBeta, bg.LNFFNGrads.BetaGrad, lr);
                b.FeedForwardNetwork.ApplyExternalGradients(_priceFFNWeightGrads[i], _priceFFNBiasGrads[i], lr);
            }

            // Output heads
            _accel.MatrixUpdate(_model.RegressionProjection, _gradients.RegressionProjectionGrad, lr); _accel.VectorUpdate(_model.RegressionBias, _gradients.RegressionBiasGrad, lr);
            _accel.MatrixUpdate(_model.RangeProjection, _gradients.RangeProjectionGrad, lr); _accel.VectorUpdate(_model.RangeBias, _gradients.RangeBiasGrad, lr);
            _accel.MatrixUpdate(_model.QualityProjection, _gradients.QualityProjectionGrad, lr); _accel.VectorUpdate(_model.QualityBias, _gradients.QualityBiasGrad, lr);
            _accel.MatrixUpdate(_model.DirectionProjection, _gradients.DirectionProjectionGrad, lr); _accel.VectorUpdate(_model.DirectionBias, _gradients.DirectionBiasGrad, lr);
            _accel.MatrixUpdate(_model.MidDirectionProjection, _gradients.MidDirectionProjectionGrad, lr); _accel.VectorUpdate(_model.MidDirectionBias, _gradients.MidDirectionBiasGrad, lr);

            if (_config.Output.UseConfidenceHead && _model.ConfidenceProjection != null)
            {
                _accel.MatrixUpdate(_model.ConfidenceProjection, _gradients.ConfidenceProjectionGrad, lr);
                _accel.VectorUpdate(_model.ConfidenceBias, _gradients.ConfidenceBiasGrad, lr);
            }

            int cteEd = _config.Price.EmbeddingDim;
            for (int t = 0; t < MmtacConfig.ContextTypeCount; t++)
                for (int d = 0; d < cteEd; d++)
                    _model.ContextTypeEmbedding[t, d] -= lr * _gradients.ContextTypeEmbeddingGrad[t, d];
        }

        private void UpdateAttn(MultiHeadAttention attn, AttentionGradients ag, float lr)
        {
            _accel.MatrixUpdate(attn.WQ, ag.WQ_Grad, lr); _accel.MatrixUpdate(attn.WK, ag.WK_Grad, lr);
            _accel.MatrixUpdate(attn.WV, ag.WV_Grad, lr); _accel.MatrixUpdate(attn.WO, ag.WO_Grad, lr);
            _accel.VectorUpdate(attn.BiasQ, ag.BiasQ_Grad, lr); _accel.VectorUpdate(attn.BiasK, ag.BiasK_Grad, lr);
            _accel.VectorUpdate(attn.BiasV, ag.BiasV_Grad, lr); _accel.VectorUpdate(attn.BiasO, ag.BiasO_Grad, lr);
        }

        private void UpdateDecayNetwork(ContentAwareDecayNetwork net, ContentAwareDecayGradients g, float lr)
        {
            int nh = net.NumHeads, cd = net.ContentDim, pd = net.ProjectionDim, hd = net.HiddenDim, mid = net.MLPInputDim, ntb = net.NumTimeBases, rawDim = net.TimeRawDim;
            for (int h = 0; h < nh; h++)
            {
                net.LogBaseDecayRate[h] -= lr * g.LogBaseDecayRateGrad[h]; net.B2[h] -= lr * g.B2Grad[h];
                for (int p = 0; p < pd; p++)
                {
                    for (int d = 0; d < cd; d++) { net.QueryProjection[h, p, d] -= lr * g.QueryProjectionGrad[h, p, d]; net.KeyProjection[h, p, d] -= lr * g.KeyProjectionGrad[h, p, d]; }
                    net.QueryProjectionBias[h, p] -= lr * g.QueryProjectionBiasGrad[h, p]; net.KeyProjectionBias[h, p] -= lr * g.KeyProjectionBiasGrad[h, p];
                    for (int r = 0; r < rawDim; r++) net.TimeProj[h, p, r] -= lr * g.TimeProjGrad[h, p, r]; net.TimeProjBias[h, p] -= lr * g.TimeProjBiasGrad[h, p];
                    for (int q = 0; q < pd; q++) net.MemAttnOutputW[h, p, q] -= lr * g.MemAttnOutputWGrad[h, p, q]; net.MemAttnOutputB[h, p] -= lr * g.MemAttnOutputBGrad[h, p];
                }
                for (int b2 = 0; b2 < ntb; b2++) net.TimeLogFreq[h, b2] -= lr * g.TimeLogFreqGrad[h, b2];
                for (int j = 0; j < hd; j++) { net.B1[h, j] -= lr * g.B1Grad[h, j]; net.W2[h, j] -= lr * g.W2Grad[h, j]; for (int k = 0; k < mid; k++) net.W1[h, j, k] -= lr * g.W1Grad[h, j, k]; }
            }
        }

        #endregion

        #region Helpers

        private float ComputeLR(int epoch)
        {
            float baseLR = _trainConfig.LearningRate;
            if (_trainConfig.WarmupEpochs > 0 && epoch < _trainConfig.WarmupEpochs)
                return baseLR * (epoch + 1f) / _trainConfig.WarmupEpochs;
            if (_trainConfig.UseLearningRateDecay)
                return baseLR * MathF.Pow(_trainConfig.LearningRateDecay, epoch - _trainConfig.WarmupEpochs);
            return baseLR;
        }

        private static float[,] SliceRows(float[,] m, int start, int end)
        {
            int len = end - start, cols = m.GetLength(1);
            var out_ = new float[len, cols];
            for (int i = 0; i < len; i++) for (int j = 0; j < cols; j++) out_[i, j] = m[start + i, j];
            return out_;
        }

        private static void BuildTargetArrays(ModelTarget[] targets, int offset, int count, out float[,] tgtReg, out float[,] tgtRange, out float[,] tgtQuality, out float[,] tgtDir, out float[,] tgtMid)
        {
            int rDim = MmtacOutputConfig.RegressionOutputCount; // 3
            tgtReg = new float[count, rDim];
            tgtRange = new float[count, 1];
            tgtQuality = new float[count, 1];
            tgtDir = new float[count, 1];
            tgtMid = new float[count, 1];
            for (int t = 0; t < count; t++)
            {
                var tg = targets[offset + t];
                var ra = tg.ToRegressionArray(); // High, Low, Close
                for (int j = 0; j < rDim; j++) tgtReg[t, j] = ra[j];
                tgtRange[t, 0] = tg.Range;
                tgtQuality[t, 0] = tg.Quality;
                tgtDir[t, 0] = tg.Direction;
                tgtMid[t, 0] = tg.MidWindowDirection;
            }
        }

        private static NewsStory[] AdjustStoryTimes(NewsStory[] stories, int splitOffset)
        {
            if (stories == null || stories.Length == 0) return null;
            var out_ = new NewsStory[stories.Length];
            for (int i = 0; i < stories.Length; i++)
                out_[i] = new NewsStory(stories[i].TokenIds, stories[i].ArrivalTime - splitOffset);
            return out_;
        }

        private void ApplyContextTypeEmbeddingsInPlace(float[,] m, int[] types)
            => _accel.ApplyContextTypeEmbedding(m, _model.ContextTypeEmbedding, types);

        private void ScaleMatrix(float[,] m, float s) => _accel.MatrixScaleInPlace(m, s);
        private void ScaleVector(float[] v, float s) => _accel.VectorScaleInPlace(v, s);
        private void ScaleAttnGrads(AttentionGradients ag, float s)
        {
            ScaleMatrix(ag.WQ_Grad, s); ScaleMatrix(ag.WK_Grad, s);
            ScaleMatrix(ag.WV_Grad, s); ScaleMatrix(ag.WO_Grad, s);
            ScaleVector(ag.BiasQ_Grad, s); ScaleVector(ag.BiasK_Grad, s);
            ScaleVector(ag.BiasV_Grad, s); ScaleVector(ag.BiasO_Grad, s);
        }
        private void ScaleLNGrads(LayerNormGradients lg, float s)
        {
            ScaleVector(lg.GammaGrad, s); ScaleVector(lg.BetaGrad, s);
        }

        private static float[] BuildPreviousCloseArray(ModelTarget[] targets, int offset, int count)
        {
            if (targets == null)
            {
                throw new ArgumentNullException(nameof(targets));
            }
            if (offset <= 0)
            {
                throw new ArgumentOutOfRangeException(nameof(offset), "offset must be >= 1 so a previous close exists.");
            }
            if (offset + count > targets.Length)
            {
                throw new ArgumentException("offset + count exceeds targets length.");
            }

            var prevClose = new float[count];

            for (int t = 0; t < count; t++)
            {
                prevClose[t] = targets[offset + t - 1].Close;
            }

            return prevClose;
        }

        #endregion

        #region Validation

        /// <summary>
        /// Evaluates the model on a batch of samples using the legacy non-sequential validation path.
        /// When price-context mode is enabled and the sequence is long enough, the method splits each
        /// sample into history and current segments, builds the appropriate targets, runs inference,
        /// and accumulates a validation loss using <see cref="ComputeValLoss"/>.
        /// This method does not update model parameters.
        /// </summary>
        public float Validate(MultimodalInput[] inputs, ModelTarget[][] targets)
        {
            float total = 0f; int count = 0;
            int minSplit = _config.PriceContext.MinHistoryLength + _config.PriceContext.MinCurrentLength + 1;

            for (int i = 0; i < inputs.Length; i++)
            {
                int sl = inputs[i].PriceSequence.GetLength(0);
                if (sl < 2)
                {
                    continue;
                }

                try
                {
                    float[,] inp, tgtReg, tgtRange, tgtQuality, tgtDir, tgtMid;

                    if (_config.PriceContext.Enabled && sl >= minSplit)
                    {
                        int sp = (sl / 2);
                        var hist = SliceRows(inputs[i].PriceSequence, 0, sp);
                        inp = SliceRows(inputs[i].PriceSequence, sp, sl - 1);
                        int csl = inp.GetLength(0);
                        BuildTargetArrays(targets[i], sp + 1, csl, out tgtReg, out tgtRange, out tgtQuality, out tgtDir, out tgtMid);

                        var priceCtxH = _model.EncodePriceHistory(hist);
                        var priceCtxT = Enumerable.Range(0, sp).Select(t => -(float)(sp - t)).ToArray();
                        var wrappedInput = new MultimodalInput
                        {
                            PredictionTimestamp = inputs[i].PredictionTimestamp,
                            PriceSequence = inp,
                            GlobalFeatures = inputs[i].GlobalFeatures,
                            NewsStories = AdjustStoryTimes(inputs[i].NewsStories, sp)
                        };
                        var cache = new MmtacForwardCache(_config.Text.NumLayers, _config.Price.NumLayers);
                        var (reg, range, quality, dir, midDir, _) = _model.ForwardWithPriceContextAndCache(wrappedInput, priceCtxH, priceCtxT, cache, isTraining: false);
                        total += ComputeValLoss(reg, range, quality, dir, midDir, tgtReg, tgtRange, tgtQuality, tgtDir, tgtMid);
                        count += csl;
                    }
                    else
                    {
                        inp = SliceRows(inputs[i].PriceSequence, 0, sl - 1);
                        int eff = inp.GetLength(0);
                        BuildTargetArrays(targets[i], 1, eff, out tgtReg, out tgtRange, out tgtQuality, out tgtDir, out tgtMid);
                        var wrappedInput = new MultimodalInput
                        {
                            PredictionTimestamp = inputs[i].PredictionTimestamp,
                            PriceSequence = inp,
                            GlobalFeatures = inputs[i].GlobalFeatures,
                            NewsStories = inputs[i].NewsStories
                        };
                        var cache = new MmtacForwardCache(_config.Text.NumLayers, _config.Price.NumLayers);
                        var (reg, range, quality, dir, midDir, _) = _model.ForwardWithCache(wrappedInput, cache, isTraining: false);
                        total += ComputeValLoss(reg, range, quality, dir, midDir, tgtReg, tgtRange, tgtQuality, tgtDir, tgtMid);
                        count += eff;
                    }
                }
                catch { }
            }
            return count > 0 ? total / count : 0f;
        }

        /// <summary>
        /// Evaluates the model in chronological order using rolling memory, so validation follows the
        /// same causal memory accumulation pattern as sequential inference.
        /// For each sample, the method predicts from all rows except the final one, compares the result
        /// to the final target row using <see cref="ComputeSequentialValLoss"/>, and clears model memory
        /// before exit to avoid polluting later operations.
        /// This method does not update model parameters.
        /// </summary>
        public float ValidateSequential(
            MultimodalInput[] inputs,
            ModelTarget[][] targets,
            double[] timestamps,
            double timeUnitsPerPosition = 1.0,
            int maxNewsMemory = 100,
            int maxPriceMemory = 200)
        {
            if (inputs == null) throw new ArgumentNullException(nameof(inputs));
            if (targets == null) throw new ArgumentNullException(nameof(targets));
            if (timestamps == null) throw new ArgumentNullException(nameof(timestamps));

            if (inputs.Length != targets.Length || inputs.Length != timestamps.Length)
                throw new ArgumentException("inputs, targets, and timestamps must all have the same length.");

            float total = 0f;
            int count = 0;

            int[] ordered = Enumerable.Range(0, inputs.Length)
                                      .OrderBy(i => timestamps[i])
                                      .ToArray();

            _model.ClearAllMemory();

            try
            {
                foreach (int idx in ordered)
                {
                    var input = inputs[idx];
                    var sampleTargets = targets[idx];

                    if (input == null || input.PriceSequence == null || sampleTargets == null)
                        continue;

                    int sl = input.PriceSequence.GetLength(0);

                    if (sl < 2 || sampleTargets.Length < sl)
                        continue;

                    try
                    {
                        var priceInp = SliceRows(input.PriceSequence, 0, sl - 1);

                        var wrappedInput = new MultimodalInput
                        {
                            PredictionTimestamp = input.PredictionTimestamp,
                            PriceSequence = priceInp,
                            GlobalFeatures = input.GlobalFeatures,
                            NewsStories = input.NewsStories
                        };

                        var pred = PredictWithCurrentMemoryNoCommit(
                            wrappedInput,
                            timestamps[idx],
                            timeUnitsPerPosition);

                        var tgt = sampleTargets[sl - 1];

                        total += ComputeSequentialValLoss(pred, tgt);
                        count++;

                        CommitObservedSampleToMemory(
                            input,
                            timestamps[idx],
                            timeUnitsPerPosition,
                            maxNewsMemory,
                            maxPriceMemory);
                    }
                    catch
                    {
                    }
                }

                return count > 0 ? total / count : 0f;
            }
            finally
            {
                _model.ClearAllMemory();
            }
        }
        /// <summary>
        /// Evaluates the model on a batch of samples using a validation objective aligned with the
        /// current training objective.
        /// In addition to regression, range, quality, direction, and mid-direction terms, this method
        /// also includes binary cross-entropy for classification heads and the close-direction
        /// consistency penalty via <see cref="ComputeAlignedValLoss"/>.
        /// This method does not update model parameters.
        /// </summary>
        public float ValidateAligned(MultimodalInput[] inputs, ModelTarget[][] targets)
        {
            float total = 0f;
            int count = 0;
            int minSplit = _config.PriceContext.MinHistoryLength + _config.PriceContext.MinCurrentLength + 1;

            for (int i = 0; i < inputs.Length; i++)
            {
                int sl = inputs[i].PriceSequence.GetLength(0);
                if (sl < 2)
                {
                    continue;
                }

                try
                {
                    float[,] inp, tgtReg, tgtRange, tgtQuality, tgtDir, tgtMid;
                    float[] prevClose;

                    if (_config.PriceContext.Enabled && sl >= minSplit)
                    {
                        int sp = (sl / 2);
                        var hist = SliceRows(inputs[i].PriceSequence, 0, sp);
                        inp = SliceRows(inputs[i].PriceSequence, sp, sl - 1);
                        int csl = inp.GetLength(0);

                        BuildTargetArrays(targets[i], sp + 1, csl, out tgtReg, out tgtRange, out tgtQuality, out tgtDir, out tgtMid);
                        prevClose = BuildPreviousCloseArray(targets[i], sp + 1, csl);

                        var priceCtxH = _model.EncodePriceHistory(hist);
                        var priceCtxT = Enumerable.Range(0, sp).Select(t => -(float)(sp - t)).ToArray();

                        var wrappedInput = new MultimodalInput
                        {
                            PredictionTimestamp = inputs[i].PredictionTimestamp,
                            PriceSequence = inp,
                            GlobalFeatures = inputs[i].GlobalFeatures,
                            NewsStories = AdjustStoryTimes(inputs[i].NewsStories, sp)
                        };

                        var cache = new MmtacForwardCache(_config.Text.NumLayers, _config.Price.NumLayers);
                        var (reg, range, quality, dir, midDir, _) =
                            _model.ForwardWithPriceContextAndCache(wrappedInput, priceCtxH, priceCtxT, cache, isTraining: false);

                        total += ComputeAlignedValLoss(reg, range, quality, dir, midDir, tgtReg, tgtRange, tgtQuality, tgtDir, tgtMid, prevClose);
                        count += csl;
                    }
                    else
                    {
                        inp = SliceRows(inputs[i].PriceSequence, 0, sl - 1);
                        int eff = inp.GetLength(0);

                        BuildTargetArrays(targets[i], 1, eff, out tgtReg, out tgtRange, out tgtQuality, out tgtDir, out tgtMid);
                        prevClose = BuildPreviousCloseArray(targets[i], 1, eff);

                        var wrappedInput = new MultimodalInput
                        {
                            PredictionTimestamp = inputs[i].PredictionTimestamp,
                            PriceSequence = inp,
                            GlobalFeatures = inputs[i].GlobalFeatures,
                            NewsStories = inputs[i].NewsStories
                        };

                        var cache = new MmtacForwardCache(_config.Text.NumLayers, _config.Price.NumLayers);
                        var (reg, range, quality, dir, midDir, _) = _model.ForwardWithCache(wrappedInput, cache, isTraining: false);

                        total += ComputeAlignedValLoss(reg, range, quality, dir, midDir, tgtReg, tgtRange, tgtQuality, tgtDir, tgtMid, prevClose);
                        count += eff;
                    }
                }
                catch
                {
                }
            }

            return count > 0 ? total / count : 0f;
        }

        /// <summary>
        /// Evaluates the model sequentially in chronological order using rolling memory and a validation
        /// objective aligned with the current training objective.
        /// For each sample, the method predicts the final step from prior rows, compares the prediction
        /// to the final target using <see cref="ComputeAlignedSequentialValLoss"/>, and clears model
        /// memory before exit to prevent state leakage.
        /// This method does not update model parameters.
        /// </summary>
        public float ValidateSequentialAligned(
            MultimodalInput[] inputs,
            ModelTarget[][] targets,
            double[] timestamps,
            double timeUnitsPerPosition = 1.0,
            int maxNewsMemory = 100,
            int maxPriceMemory = 200)
        {
            if (inputs == null) throw new ArgumentNullException(nameof(inputs));
            if (targets == null) throw new ArgumentNullException(nameof(targets));
            if (timestamps == null) throw new ArgumentNullException(nameof(timestamps));

            if (inputs.Length != targets.Length || inputs.Length != timestamps.Length)
                throw new ArgumentException("inputs, targets, and timestamps must all have the same length.");

            float total = 0f;
            int count = 0;

            int[] ordered = Enumerable.Range(0, inputs.Length)
                                      .OrderBy(i => timestamps[i])
                                      .ToArray();

            _model.ClearAllMemory();

            try
            {
                foreach (int idx in ordered)
                {
                    var input = inputs[idx];
                    var sampleTargets = targets[idx];

                    if (input == null || input.PriceSequence == null || sampleTargets == null)
                        continue;

                    int sl = input.PriceSequence.GetLength(0);

                    if (sl < 2 || sampleTargets.Length < sl)
                        continue;

                    try
                    {
                        var priceInp = SliceRows(input.PriceSequence, 0, sl - 1);

                        var wrappedInput = new MultimodalInput
                        {
                            PredictionTimestamp = input.PredictionTimestamp,
                            PriceSequence = priceInp,
                            GlobalFeatures = input.GlobalFeatures,
                            NewsStories = input.NewsStories
                        };

                        var pred = PredictWithCurrentMemoryNoCommit(
                            wrappedInput,
                            timestamps[idx],
                            timeUnitsPerPosition);

                        var tgt = sampleTargets[sl - 1];
                        float prevClose = sampleTargets[sl - 2].Close;

                        total += ComputeAlignedSequentialValLoss(pred, tgt, prevClose);
                        count++;

                        CommitObservedSampleToMemory(
                            input,
                            timestamps[idx],
                            timeUnitsPerPosition,
                            maxNewsMemory,
                            maxPriceMemory);
                    }
                    catch
                    {
                    }
                }

                return count > 0 ? total / count : 0f;
            }
            finally
            {
                _model.ClearAllMemory();
            }
        }
        #region Validation helpers
        private float ComputeValLoss(float[,] reg, float[,] range, float[,] quality, float[,] dir, float[,] midDir, float[,] tgtReg, float[,] tgtRange, float[,] tgtQuality, float[,] tgtDir, float[,] tgtMid)
        {
            int sequenceLength = reg.GetLength(0);
            int regressionDim = MmtacOutputConfig.RegressionOutputCount;

            float totalLoss = 0f;

            for (int t = 0; t < sequenceLength; t++)
            {
                // Regression loss (High, Low, Close)
                for (int j = 0; j < regressionDim; j++)
                {
                    float diff = reg[t, j] - tgtReg[t, j];
                    totalLoss += diff * diff;
                }

                // Range loss
                {
                    float diff = range[t, 0] - tgtRange[t, 0];
                    totalLoss += _config.Output.RangeLossWeight * diff * diff;
                }

                // Quality loss
                {
                    float diff = quality[t, 0] - tgtQuality[t, 0];
                    totalLoss += _config.Output.QualityLossWeight * diff * diff;
                }

                // Direction loss
                {
                    float diff = dir[t, 0] - tgtDir[t, 0];
                    totalLoss += _config.Output.DirectionLossWeight * diff * diff;
                }

                // Mid-direction loss
                {
                    float diff = midDir[t, 0] - tgtMid[t, 0];
                    totalLoss += _config.Output.MidDirectionLossWeight * diff * diff;
                }
            }

            return totalLoss;
        }

        private float ComputeSequentialValLoss(ModelPrediction pred, ModelTarget tgt)
        {
            float totalLoss = 0f;

            // Regression loss (High, Low, Close)
            {
                float diff = pred.High - tgt.High;
                totalLoss += diff * diff;
            }
            {
                float diff = pred.Low - tgt.Low;
                totalLoss += diff * diff;
            }
            {
                float diff = pred.Close - tgt.Close;
                totalLoss += diff * diff;
            }

            // Range loss
            {
                float diff = pred.Range - tgt.Range;
                totalLoss += _config.Output.RangeLossWeight * diff * diff;
            }

            // Quality loss
            {
                float diff = pred.Quality - tgt.Quality;
                totalLoss += _config.Output.QualityLossWeight * diff * diff;
            }

            // Direction loss
            {
                float diff = pred.DirectionProb - tgt.Direction;
                totalLoss += _config.Output.DirectionLossWeight * diff * diff;
            }

            // Mid-direction loss
            {
                float diff = pred.MidWindowDirectionProb - tgt.MidWindowDirection;
                totalLoss += _config.Output.MidDirectionLossWeight * diff * diff;
            }

            // Intentionally ignore confidence here, to stay consistent with the
            // existing ComputeValLoss(...) used by Validate().
            return totalLoss;
        }
        private static float BinaryCrossEntropy(float p, float y)
        {
            float pc = Math.Clamp(p, 1e-7f, 1f - 1e-7f);
            return -(y * MathF.Log(pc) + (1f - y) * MathF.Log(1f - pc));
        }
        private float ComputeCloseDirectionConsistencyLoss(float[,] reg, float[,] tgtDir, float[] prevClose)
        {
            if (prevClose == null || _config.Output.CloseDirectionConsistencyWeight <= 0f)
            {
                return 0f;
            }

            int sl = reg.GetLength(0);
            float margin = _config.Output.CloseDirectionConsistencyMargin;
            float loss = 0f;

            for (int t = 0; t < sl; t++)
            {
                float sign = tgtDir[t, 0] >= 0.5f ? 1f : -1f;
                float z = sign * (reg[t, 2] - prevClose[t] - sign * margin);

                if (z > 20f)
                {
                    loss += MathF.Exp(-z);
                }
                else if (z < -20f)
                {
                    loss += -z;
                }
                else
                {
                    loss += MathF.Log(1f + MathF.Exp(-z));
                }
            }

            return loss / sl;
        }
        private float ComputeAlignedValLoss(float[,] reg, float[,] range, float[,] quality, float[,] dir, float[,] midDir, float[,] tgtReg, float[,] tgtRange, float[,] tgtQuality, float[,] tgtDir, float[,] tgtMid, float[] prevClose)
        {
            int sl = reg.GetLength(0);
            int rDim = MmtacOutputConfig.RegressionOutputCount;

            float mseLoss = 0f;
            float rangeLoss = 0f;
            float qualityLoss = 0f;
            float dirLoss = 0f;
            float midDirLoss = 0f;

            for (int t = 0; t < sl; t++)
            {
                for (int j = 0; j < rDim; j++)
                {
                    float diff = reg[t, j] - tgtReg[t, j];
                    mseLoss += diff * diff;
                }

                {
                    float diff = range[t, 0] - tgtRange[t, 0];
                    rangeLoss += diff * diff;
                }

                {
                    float diff = quality[t, 0] - tgtQuality[t, 0];
                    qualityLoss += diff * diff;
                }

                dirLoss += BinaryCrossEntropy(dir[t, 0], tgtDir[t, 0]);
                midDirLoss += BinaryCrossEntropy(midDir[t, 0], tgtMid[t, 0]);
            }

            mseLoss /= sl;
            rangeLoss /= sl;
            qualityLoss /= sl;
            dirLoss /= sl;
            midDirLoss /= sl;

            float closeDirConsistencyLoss = ComputeCloseDirectionConsistencyLoss(reg, tgtDir, prevClose);

            return mseLoss
                + _config.Output.CloseDirectionConsistencyWeight * closeDirConsistencyLoss
                + _config.Output.RangeLossWeight * rangeLoss
                + _config.Output.QualityLossWeight * qualityLoss
                + _config.Output.DirectionLossWeight * dirLoss
                + _config.Output.MidDirectionLossWeight * midDirLoss;
        }

        private float ComputeAlignedSequentialValLoss(ModelPrediction pred, ModelTarget tgt, float prevClose)
        {
            float mseLoss = 0f;

            {
                float diff = pred.High - tgt.High;
                mseLoss += diff * diff;
            }
            {
                float diff = pred.Low - tgt.Low;
                mseLoss += diff * diff;
            }
            {
                float diff = pred.Close - tgt.Close;
                mseLoss += diff * diff;
            }

            float rangeLoss = (pred.Range - tgt.Range) * (pred.Range - tgt.Range);
            float qualityLoss = (pred.Quality - tgt.Quality) * (pred.Quality - tgt.Quality);
            float dirLoss = BinaryCrossEntropy(pred.DirectionProb, tgt.Direction);
            float midDirLoss = BinaryCrossEntropy(pred.MidWindowDirectionProb, tgt.MidWindowDirection);

            float closeDirConsistencyLoss = 0f;
            if (_config.Output.CloseDirectionConsistencyWeight > 0f)
            {
                float sign = tgt.Direction >= 0.5f ? 1f : -1f;
                float z = sign * (pred.Close - prevClose - sign * _config.Output.CloseDirectionConsistencyMargin);

                if (z > 20f)
                    closeDirConsistencyLoss = MathF.Exp(-z);
                else if (z < -20f)
                    closeDirConsistencyLoss = -z;
                else
                    closeDirConsistencyLoss = MathF.Log(1f + MathF.Exp(-z));
            }

            return mseLoss
                + _config.Output.CloseDirectionConsistencyWeight * closeDirConsistencyLoss
                + _config.Output.RangeLossWeight * rangeLoss
                + _config.Output.QualityLossWeight * qualityLoss
                + _config.Output.DirectionLossWeight * dirLoss
                + _config.Output.MidDirectionLossWeight * midDirLoss;
        }

        #endregion
        #endregion
    }
}