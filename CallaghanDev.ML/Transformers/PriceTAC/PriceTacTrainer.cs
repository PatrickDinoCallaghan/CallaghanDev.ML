using CallaghanDev.ML.AccelerationManagers;
using CallaghanDev.ML.Transformers.Configuration;
using CallaghanDev.ML.Transformers.MultiTypeTransformer;
using CallaghanDev.ML.Transformers.TACAMT;
using System;
using System.Collections.Generic;
using System.Linq;

namespace CallaghanDev.ML.Transformers.PriceTAC
{
    /// <summary>
    /// Trainer for PriceTacModel.  The loss and backward path mirror MMTAC's price
    /// heads and learned time-decay price context, without language-side terms.
    /// </summary>
    public sealed class PriceTacTrainer
    {
        private readonly PriceTacModel _model;
        private readonly PriceTacConfig _config;
        private readonly TrainingConfig _trainConfig;
        private readonly PriceTacGradients _gradients;
        private readonly IAccelerationManager _accel;
        private readonly Random _random;
        private readonly Random _dropoutRng;
        private readonly RotaryPositionEmbedding _rotaryPositionEmbedding;
        private readonly PriceTacForwardCache _trainingForwardCache;
        private readonly List<List<float[,]>> _priceFFNWeightGrads;
        private readonly List<List<float[]>> _priceFFNBiasGrads;

        private float[,] _trainTargetRegScratch;
        private float[,] _trainTargetRangeScratch;
        private float[,] _trainTargetQualityScratch;
        private float[,] _trainTargetDirScratch;
        private float[,] _trainTargetMidScratch;
        private float[] _trainPrevCloseScratch;
        private float[] _trainConfScratch;
        private float[] _trainPriceContextTimesScratch;
        private readonly PriceTacInput _priceContextTrainingInput;

        public PriceTacTrainer(PriceTacModel model, TrainingConfig trainConfig)
        {
            _model = model ?? throw new ArgumentNullException(nameof(model));
            _config = model.Config;
            _trainConfig = trainConfig ?? throw new ArgumentNullException(nameof(trainConfig));
            _gradients = new PriceTacGradients(_config);
            _accel = model.AccelerationManager;

            if (_trainConfig.RandomSeed.HasValue)
            {
                int seed = _trainConfig.RandomSeed.Value;
                _random = new Random(seed);
                _dropoutRng = new Random(unchecked(seed ^ 0x5F3759DF));
            }
            else
            {
                _random = new Random();
                _dropoutRng = new Random(_random.Next());
            }

            _rotaryPositionEmbedding = new RotaryPositionEmbedding(_accel);
            _trainingForwardCache = new PriceTacForwardCache(_config.Price.NumLayers);
            _priceContextTrainingInput = new PriceTacInput();

            _priceFFNWeightGrads = new List<List<float[,]>>();
            _priceFFNBiasGrads = new List<List<float[]>>();

            for (int i = 0; i < _config.Price.NumLayers; i++)
            {
                var (w, b) = model.PriceBlocks[i].FeedForwardNetwork.CreateGradientStorage();
                _priceFFNWeightGrads.Add(w);
                _priceFFNBiasGrads.Add(b);
            }
        }

        public void Train(PriceTacInput[] inputs, PriceTacTarget[][] targets, float[][] confTargets = null)
        {
            _trainConfig.Validate();

            if (inputs == null)
                throw new ArgumentNullException(nameof(inputs));
            if (targets == null)
                throw new ArgumentNullException(nameof(targets));
            if (inputs.Length != targets.Length)
                throw new ArgumentException("inputs and targets must have the same length.");
            if (confTargets != null && confTargets.Length != inputs.Length)
                throw new ArgumentException("confTargets must be null or have the same length as inputs.");

            for (int i = 0; i < inputs.Length; i++)
            {
                if (inputs[i]?.PriceSequence == null)
                    continue;

                if (targets[i] == null)
                    throw new ArgumentException($"targets[{i}] is null.");

                int sl = inputs[i].PriceSequence.GetLength(0);

                if (targets[i].Length < sl)
                    throw new ArgumentException($"targets[{i}].Length ({targets[i].Length}) must be at least the input sequence length ({sl}).");

                if (confTargets != null && confTargets[i] != null && confTargets[i].Length < sl)
                    throw new ArgumentException($"confTargets[{i}].Length ({confTargets[i].Length}) must be at least the input sequence length ({sl}).");
            }

            int n = inputs.Length;
            for (int ep = 0; ep < _trainConfig.Epochs; ep++)
            {
                float lr = ComputeLR(ep);

                if (_trainConfig.Verbose)
                    Console.WriteLine($"\n=== Epoch {ep + 1}/{_trainConfig.Epochs} ===");

                var shuffled = CreateSequentialIndices(n);
                ShuffleInPlace(shuffled);

                float epochLoss = 0f;
                int batchCount = 0;

                for (int i = 0; i < shuffled.Length; i += _trainConfig.BatchSize)
                {
                    int batchSize = Math.Min(_trainConfig.BatchSize, shuffled.Length - i);
                    float bl = TrainBatch(shuffled, i, batchSize, inputs, targets, confTargets, lr);

                    epochLoss += bl;
                    batchCount++;

                    if (_trainConfig.Verbose && batchCount % 10 == 0)
                        Console.WriteLine($"  Batch {batchCount}: Loss = {bl:F6}");
                }

                if (_trainConfig.Verbose)
                    Console.WriteLine($"  Epoch {ep + 1} Avg Loss: {(batchCount > 0 ? epochLoss / batchCount : 0f):F6}");
            }
        }

        public void TrainSequential(PriceTacInput[] inputs, PriceTacTarget[][] targets, double[] timestamps, double timeUnitsPerPosition = 1.0, int maxPriceMemory = 200, float[][] confTargets = null)
        {
            _trainConfig.Validate();

            if (inputs == null)
                throw new ArgumentNullException(nameof(inputs));
            if (targets == null)
                throw new ArgumentNullException(nameof(targets));
            if (timestamps == null)
                throw new ArgumentNullException(nameof(timestamps));
            if (inputs.Length != targets.Length || inputs.Length != timestamps.Length)
                throw new ArgumentException("inputs, targets, and timestamps must all have the same length.");
            if (timeUnitsPerPosition == 0.0)
                throw new ArgumentOutOfRangeException(nameof(timeUnitsPerPosition), "Must be non-zero.");

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

                int[] ordered = CreateSequentialIndices(n);
                Array.Sort(ordered, (a, b) =>
                {
                    int cmp = timestamps[a].CompareTo(timestamps[b]);
                    return cmp != 0 ? cmp : a.CompareTo(b);
                });

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
                        var priceInp = _accel.SliceRows(inp.PriceSequence, 0, sl - 1);
                        var cache = new PriceTacForwardCache(_config.Price.NumLayers);

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

                        int storedPriceCount = _model.PriceMemory.Count;
                        foreach (var e in _model.PriceMemory)
                        {
                            var v = new float[embDim];
                            for (int d = 0; d < embDim; d++)
                                v[d] = e.HiddenState[d];

                            ctxH.Add(v);
                            ctxT.Add(-(float)((currentTs - e.AbsoluteTimestamp) * invTime));
                            ctxTypes.Add(1);
                        }

                        _model.BuildManualContext(ctxH, ctxT, ctxTypes, embDim, out var combinedHidden, out var combinedTimes);

                        cache.NumGlobalContext = globalToken != null ? 1 : 0;
                        cache.NumPriceContext = storedPriceCount;
                        cache.ContextHidden = combinedHidden;
                        cache.ContextTimes = combinedTimes;

                        var ph = _model.ForwardPriceDecoderWithCache(priceInp, 0, priceInp.GetLength(0), combinedHidden, combinedTimes, cache, true, _dropoutRng);
                        cache.PriceFinalHidden = ph;

                        var (reg, range, quality, dir, midDir, conf) = _model.ProjectToOutputs(ph, cache);

                        int tsl = sl - 1;
                        BuildTargetArraysForTraining(targets[idx], 1, tsl, out var tgtReg, out var tgtRange, out var tgtQuality, out var tgtDir, out var tgtMid);
                        var prevClose = BuildPreviousCloseArrayForTraining(targets[idx], 1, tsl);
                        float[] ct = SliceOptionalForTraining(confTargets?[idx], 1, tsl);

                        ZeroAllGradients();
                        float loss = BackwardPass(reg, range, quality, dir, midDir, conf, tgtReg, tgtRange, tgtQuality, tgtDir, tgtMid, prevClose, ct, cache);

                        if (float.IsFinite(loss))
                        {
                            if (_trainConfig.UseGradientClipping)
                                ClipGradients(_trainConfig.GradientClipThreshold);

                            UpdateAllParameters(lr);
                            epochLoss += loss;
                            validCount++;

                            CommitObservedSampleToMemory(inp, currentTs, timeUnitsPerPosition, maxPriceMemory);
                        }

                        if (_trainConfig.Verbose && validCount % 50 == 0)
                            Console.WriteLine($"  Sample {validCount}: Loss = {loss:F6}");
                    }
                    catch (Exception ex)
                    {
                        ZeroAllGradients();
                        if (_trainConfig.Verbose)
                            Console.WriteLine($"  WARNING: {ex.Message}");
                    }
                }

                if (_trainConfig.Verbose)
                    Console.WriteLine($"  Epoch {ep + 1} Avg Loss: {(validCount > 0 ? epochLoss / validCount : 0f):F6}");
            }
        }

        private float TrainBatch(int[] batchIndices, PriceTacInput[] allInputs, PriceTacTarget[][] allTargets, float[][] allConf, float lr)
        {
            if (batchIndices == null)
                return 0f;

            return TrainBatch(batchIndices, 0, batchIndices.Length, allInputs, allTargets, allConf, lr);
        }

        private float TrainBatch(int[] batchIndices, int batchStart, int batchCount, PriceTacInput[] allInputs, PriceTacTarget[][] allTargets, float[][] allConf, float lr)
        {
            if (batchIndices == null || batchCount <= 0)
                return 0f;

            if (batchStart < 0 || batchStart + batchCount > batchIndices.Length)
                throw new ArgumentOutOfRangeException(nameof(batchStart));

            ZeroAllGradients();

            if (BatchContainsInvalidSample(batchIndices, batchStart, batchCount, allInputs, allTargets, allConf))
            {
                ZeroAllGradients();
                return 0f;
            }

            float totalLoss = 0f;
            int validCount = 0;
            bool discardBatch = false;
            int minSplit = _config.PriceContext.MinHistoryLength + _config.PriceContext.MinCurrentLength + 1;

            for (int bi = 0; bi < batchCount; bi++)
            {
                int idx = batchIndices[batchStart + bi];
                var input = allInputs[idx];

                if (input?.PriceSequence == null)
                    continue;

                int sl = input.PriceSequence.GetLength(0);
                if (sl < 2)
                    continue;

                try
                {
                    float loss = (_config.PriceContext.Enabled && sl >= minSplit)
                        ? TrainWithPriceContext(idx, allInputs, allTargets, allConf)
                        : TrainSimple(idx, allInputs, allTargets, allConf);

                    if (!float.IsFinite(loss))
                    {
                        discardBatch = true;
                        if (_trainConfig.Verbose)
                            Console.WriteLine($"  WARNING: non-finite sample loss at index {idx}: {loss}; discarding batch gradients.");
                        break;
                    }

                    totalLoss += loss;
                    validCount++;
                }
                catch (Exception ex)
                {
                    discardBatch = true;
                    if (_trainConfig.Verbose)
                        Console.WriteLine($"  WARNING: {ex.Message}; discarding batch gradients.");
                    break;
                }
            }

            if (discardBatch || validCount == 0)
            {
                ZeroAllGradients();
                return 0f;
            }

            ScaleAllGradients(1.0f / validCount);

            float gradNorm = ComputeGradientNorm();
            if (!float.IsFinite(gradNorm))
            {
                if (_trainConfig.Verbose)
                    Console.WriteLine($"  WARNING: non-finite batch gradient norm: {gradNorm}; discarding batch gradients.");

                ZeroAllGradients();
                return 0f;
            }

            if (_trainConfig.UseGradientClipping)
                ClipGradients(_trainConfig.GradientClipThreshold, gradNorm);

            UpdateAllParameters(lr);
            return totalLoss / validCount;
        }

        private bool BatchContainsInvalidSample(int[] batchIndices, int batchStart, int batchCount, PriceTacInput[] allInputs, PriceTacTarget[][] allTargets, float[][] allConf)
        {
            int expectedPriceFeatureDim = _config.Price.InputFeatureDim;

            if (batchIndices == null || batchStart < 0 || batchCount < 0 || batchStart + batchCount > batchIndices.Length)
                return true;

            for (int bi = 0; bi < batchCount; bi++)
            {
                int idx = batchIndices[batchStart + bi];
                if ((uint)idx >= (uint)allInputs.Length)
                    return true;

                var input = allInputs[idx];
                if (input?.PriceSequence == null)
                    continue;

                if ((uint)idx >= (uint)allTargets.Length)
                    return true;

                var target = allTargets[idx];
                if (target == null)
                    return true;

                int sl = input.PriceSequence.GetLength(0);
                if (sl < 2)
                    continue;

                if (input.PriceSequence.GetLength(1) != expectedPriceFeatureDim)
                    return true;

                if (target.Length < sl)
                    return true;

                if (allConf != null)
                {
                    if ((uint)idx >= (uint)allConf.Length)
                        return true;
                    if (allConf[idx] != null && allConf[idx].Length < sl)
                        return true;
                }
            }

            return false;
        }

        private float TrainSimple(int idx, PriceTacInput[] allInputs, PriceTacTarget[][] allTargets, float[][] allConf)
        {
            var inp = allInputs[idx];
            int sl = inp.PriceSequence.GetLength(0);
            int eff = sl - 1;

            BuildTargetArraysForTraining(allTargets[idx], 1, eff, out var tgtReg, out var tgtRange, out var tgtQuality, out var tgtDir, out var tgtMid);
            var prevClose = BuildPreviousCloseArrayForTraining(allTargets[idx], 1, eff);
            float[] ct = SliceOptionalForTraining(allConf?[idx], 1, eff);

            var cache = _trainingForwardCache;
            var (reg, range, quality, dir, midDir, conf) = _model.ForwardWithCache(inp, 0, eff, cache, isTraining: true, dropoutRng: _dropoutRng);

            return BackwardPass(reg, range, quality, dir, midDir, conf, tgtReg, tgtRange, tgtQuality, tgtDir, tgtMid, prevClose, ct, cache);
        }

        private float TrainWithPriceContext(int idx, PriceTacInput[] allInputs, PriceTacTarget[][] allTargets, float[][] allConf)
        {
            var inp = allInputs[idx];
            int sl = inp.PriceSequence.GetLength(0);
            int minH = _config.PriceContext.MinHistoryLength;
            int maxH = sl - _config.PriceContext.MinCurrentLength - 1;
            int sp = minH + _random.Next(maxH - minH + 1);

            int csl = sl - 1 - sp;
            if (csl < 2)
                return 0f;

            BuildTargetArraysForTraining(allTargets[idx], sp + 1, csl, out var tgtReg, out var tgtRange, out var tgtQuality, out var tgtDir, out var tgtMid);
            var prevClose = BuildPreviousCloseArrayForTraining(allTargets[idx], sp + 1, csl);
            float[] ct = SliceOptionalForTraining(allConf?[idx], sp + 1, csl);

            var priceCtxH = _model.EncodePriceHistory(inp.PriceSequence, 0, sp);
            var priceCtxT = BuildPriceContextTimesForTraining(sp);

            _priceContextTrainingInput.PredictionTimestamp = inp.PredictionTimestamp;
            _priceContextTrainingInput.PriceSequence = inp.PriceSequence;
            _priceContextTrainingInput.GlobalFeatures = inp.GlobalFeatures;

            var cache = _trainingForwardCache;
            var (reg, range, quality, dir, midDir, conf) = _model.ForwardWithPriceContextSliceAndCache(
                _priceContextTrainingInput,
                sp,
                csl,
                priceCtxH,
                priceCtxT,
                cache,
                isTraining: true,
                dropoutRng: _dropoutRng);

            return BackwardPass(reg, range, quality, dir, midDir, conf, tgtReg, tgtRange, tgtQuality, tgtDir, tgtMid, prevClose, ct, cache);
        }

        private void BuildTargetArraysForTraining(PriceTacTarget[] targets, int offset, int count, out float[,] tgtReg, out float[,] tgtRange, out float[,] tgtQuality, out float[,] tgtDir, out float[,] tgtMid)
        {
            const int rDim = PriceTacOutputConfig.RegressionOutputCount;

            tgtReg = EnsureMatrix(ref _trainTargetRegScratch, count, rDim);
            tgtRange = EnsureMatrix(ref _trainTargetRangeScratch, count, 1);
            tgtQuality = EnsureMatrix(ref _trainTargetQualityScratch, count, 1);
            tgtDir = EnsureMatrix(ref _trainTargetDirScratch, count, 1);
            tgtMid = EnsureMatrix(ref _trainTargetMidScratch, count, 1);

            for (int t = 0; t < count; t++)
            {
                var tg = targets[offset + t];
                tgtReg[t, 0] = tg.High;
                tgtReg[t, 1] = tg.Low;
                tgtReg[t, 2] = tg.Close;
                tgtRange[t, 0] = tg.Range;
                tgtQuality[t, 0] = tg.Quality;
                tgtDir[t, 0] = tg.Direction;
                tgtMid[t, 0] = tg.MidWindowDirection;
            }
        }

        private float[] BuildPreviousCloseArrayForTraining(PriceTacTarget[] targets, int offset, int count)
        {
            if (targets == null)
                throw new ArgumentNullException(nameof(targets));
            if (offset <= 0)
                throw new ArgumentOutOfRangeException(nameof(offset), "offset must be >= 1 so a previous close exists.");
            if (offset + count > targets.Length)
                throw new ArgumentException("offset + count exceeds targets length.");

            var prevClose = EnsureVector(ref _trainPrevCloseScratch, count);
            for (int t = 0; t < count; t++)
                prevClose[t] = targets[offset + t - 1].Close;

            return prevClose;
        }

        private float[] SliceOptionalForTraining(float[] source, int start, int count)
        {
            if (source == null)
                return null;
            if (count <= 0)
                return Array.Empty<float>();
            if (start < 0 || start + count > source.Length)
                throw new ArgumentOutOfRangeException(nameof(start), $"Invalid slice: start={start}, count={count}, length={source.Length}.");

            var result = EnsureVector(ref _trainConfScratch, count);
            Array.Copy(source, start, result, 0, count);
            return result;
        }

        private float[] BuildPriceContextTimesForTraining(int count)
        {
            var result = EnsureVector(ref _trainPriceContextTimesScratch, count);
            for (int t = 0; t < count; t++)
                result[t] = -(float)(count - t);

            return result;
        }

        private static float[,] EnsureMatrix(ref float[,] matrix, int rows, int cols)
        {
            if (matrix == null || matrix.GetLength(0) != rows || matrix.GetLength(1) != cols)
                matrix = new float[rows, cols];
            return matrix;
        }

        private static float[] EnsureVector(ref float[] vector, int length)
        {
            if (vector == null || vector.Length != length)
                vector = new float[length];
            return vector;
        }

        private float BackwardPass(float[,] reg, float[,] range, float[,] quality, float[,] dir, float[,] midDir, float[,] conf, float[,] tgtReg, float[,] tgtRange, float[,] tgtQuality, float[,] tgtDir, float[,] tgtMid, float[] prevClose, float[] confTgt, PriceTacForwardCache cache)
        {
            if (cache == null)
                throw new ArgumentNullException(nameof(cache));
            if (cache.PriceFinalHidden == null)
                throw new ArgumentException("cache.PriceFinalHidden is required for backward pass.", nameof(cache));
            if (cache.RegressionLogits == null)
                throw new ArgumentException("cache.RegressionLogits is required. Call ProjectToOutputs(..., cache) before BackwardPass.", nameof(cache));
            if (cache.RangeLogits == null)
                throw new ArgumentException("cache.RangeLogits is required. Call ProjectToOutputs(..., cache) before BackwardPass.", nameof(cache));

            var outputBackprop = _accel.BackpropMmtacOutputHeads(
                reg,
                range,
                quality,
                dir,
                midDir,
                conf,
                tgtReg,
                tgtRange,
                tgtQuality,
                tgtDir,
                tgtMid,
                prevClose,
                confTgt,
                cache.PriceFinalHidden,
                cache.RegressionLogits,
                cache.RangeLogits,
                _model.RegressionProjection,
                _model.RangeProjection,
                _model.QualityProjection,
                _model.DirectionProjection,
                _model.MidDirectionProjection,
                _model.ConfidenceProjection,
                _gradients.RegressionProjectionGrad,
                _gradients.RegressionBiasGrad,
                _gradients.RangeProjectionGrad,
                _gradients.RangeBiasGrad,
                _gradients.QualityProjectionGrad,
                _gradients.QualityBiasGrad,
                _gradients.DirectionProjectionGrad,
                _gradients.DirectionBiasGrad,
                _gradients.MidDirectionProjectionGrad,
                _gradients.MidDirectionBiasGrad,
                _gradients.ConfidenceProjectionGrad,
                _gradients.ConfidenceBiasGrad,
                _config.Output.RangeLossWeight,
                _config.Output.QualityLossWeight,
                _config.Output.DirectionLossWeight,
                _config.Output.MidDirectionLossWeight,
                _config.Output.CloseDirectionConsistencyWeight,
                _config.Output.CloseDirectionConsistencyMargin,
                _config.Output.UseConfidenceHead ? MathF.Max(0f, _trainConfig.ConfidenceLossWeight) : 0f,
                _config.Output.UseConfidenceHead);

            float outputLoss = outputBackprop.loss;
            var dHidden = outputBackprop.dHidden;

            float[] dGlobalHidden = BackpropPriceDecoder(dHidden, cache);

            if (_config.Global.GlobalFeatureDim > 0 && cache.GlobalTokenEmbedded != null && cache.GlobalRawInput != null && dGlobalHidden != null && _gradients.GlobalFeatureProjectionGrad != null)
            {
                _accel.AccumulateGlobalProjectionGradients(dGlobalHidden, cache.GlobalRawInput, _gradients.GlobalFeatureProjectionGrad, _gradients.GlobalFeatureBiasGrad);
            }

            return outputLoss;
        }

        private float[] BackpropPriceDecoder(float[,] dOut, PriceTacForwardCache cache)
        {
            int ed = _config.Price.EmbeddingDim;
            int nh = _config.Price.NumHeads;
            int hd = ed / nh;
            float scale = 1.0f / MathF.Sqrt(hd);

            int numGlobal = cache.NumGlobalContext;
            int numPriceCtx = cache.NumPriceContext;
            int totalContext = cache.ContextHidden?.GetLength(0) ?? 0;
            bool hasContext = totalContext > 0;
            int priceOffset = numGlobal;
            float[] dGlobalHidden = numGlobal > 0 ? new float[ed] : null;

            var dX = dOut;

            for (int layer = _config.Price.NumLayers - 1; layer >= 0; layer--)
            {
                var block = _model.PriceBlocks[layer];
                var bc = cache.PriceBlockCaches[layer];
                var bg = _gradients.PriceBlockGrads[layer];
                int seqLen = dX.GetLength(0);

                var (dFfnRes, dGammaFFN, dBetaFFN) = _accel.LayerNormBackward(
                    dX,
                    bc.LNFFNCache.Normalized,
                    block.LNFFNGamma,
                    bc.LNFFNCache.Input,
                    bc.LNFFNCache.Mean,
                    bc.LNFFNCache.Variance);
                _accel.VectorAccumulate(bg.LNFFNGrads.GammaGrad, dGammaFFN);
                _accel.VectorAccumulate(bg.LNFFNGrads.BetaGrad, dBetaFFN);

                var dFfnIn = block.FeedForwardNetwork.ComputeInputGradientBatch(
                    bc.NormedCross,
                    dFfnRes,
                    _priceFFNWeightGrads[layer],
                    _priceFFNBiasGrads[layer]);
                _accel.MatrixAddInPlace(dFfnIn, dFfnRes);

                var (dCrossRes, dGammaCross, dBetaCross) = _accel.LayerNormBackward(
                    dFfnIn,
                    bc.LNCrossCache.Normalized,
                    block.LnCrossGamma,
                    bc.LNCrossCache.Input,
                    bc.LNCrossCache.Mean,
                    bc.LNCrossCache.Variance);
                _accel.VectorAccumulate(bg.LNCrossGrads.GammaGrad, dGammaCross);
                _accel.VectorAccumulate(bg.LNCrossGrads.BetaGrad, dBetaCross);

                float[,] dSelfIn;

                if (hasContext && bc.CrossQ != null && totalContext > 0)
                {
                    var crossGrads = bg.CrossAttnGrads;
                    var dCrossCombined = new float[seqLen, ed];
                    _accel.BackpropLinearProjection(
                        bc.CrossAttnOutput,
                        dCrossRes,
                        block.CrossAttention.WO,
                        crossGrads.WO_Grad,
                        crossGrads.BiasO_Grad,
                        dCrossCombined);

                    var (dQ, dK, dV, dDecayBias) = BackpropTimeDecayedAttn(
                        bc.CrossQ,
                        bc.CrossK,
                        bc.CrossV,
                        dCrossCombined,
                        bc.CrossAttentionWeights,
                        bc.TimeDiffs,
                        block);

                    if (bc.DecayCache != null && dDecayBias != null)
                    {
                        var (decayParamGrads, dQueryEmb, dKeyEmb) = block.DecayNetwork.Backward(dDecayBias, bc.DecayCache);
                        AccumulateDecayGrads(bg.DecayGrads, decayParamGrads);

                        if (dKeyEmb != null)
                        {
                            _accel.AccumulateMmtacContextGradients(
                                dKeyEmb,
                                null,
                                _gradients.ContextTypeEmbeddingGrad,
                                null,
                                dGlobalHidden,
                                numGlobal,
                                0,
                                0,
                                0,
                                numPriceCtx,
                                totalContext,
                                priceOffset);
                        }

                        if (dQueryEmb != null)
                            _accel.MatrixAddInPlace(dCrossRes, dQueryEmb);
                    }

                    var dFromQ = new float[seqLen, ed];
                    _accel.BackpropLinearProjection(
                        bc.NormedSelf,
                        dQ,
                        block.CrossAttention.WQ,
                        crossGrads.WQ_Grad,
                        crossGrads.BiasQ_Grad,
                        dFromQ);

                    var dCtxFromKV = _accel.BackpropKV(
                        cache.ContextHidden,
                        dK,
                        dV,
                        block.CrossAttention.WK,
                        block.CrossAttention.WV,
                        crossGrads.WK_Grad,
                        crossGrads.BiasK_Grad,
                        crossGrads.WV_Grad,
                        crossGrads.BiasV_Grad);

                    _accel.AccumulateMmtacContextGradients(
                        dCtxFromKV,
                        null,
                        _gradients.ContextTypeEmbeddingGrad,
                        null,
                        dGlobalHidden,
                        numGlobal,
                        0,
                        0,
                        0,
                        numPriceCtx,
                        totalContext,
                        priceOffset);

                    _accel.MatrixAddInPlace(dFromQ, dCrossRes);
                    dSelfIn = dFromQ;
                }
                else
                {
                    dSelfIn = dCrossRes;
                }

                var (dSelfRes, dGammaSelf, dBetaSelf) = _accel.LayerNormBackward(
                    dSelfIn,
                    bc.LNSelfCache.Normalized,
                    block.LNSelfGamma,
                    bc.LNSelfCache.Input,
                    bc.LNSelfCache.Mean,
                    bc.LNSelfCache.Variance);
                _accel.VectorAccumulate(bg.LNSelfGrads.GammaGrad, dGammaSelf);
                _accel.VectorAccumulate(bg.LNSelfGrads.BetaGrad, dBetaSelf);

                var selfGrads = bg.SelfAttnGrads;
                var dSelfCombined = new float[seqLen, ed];
                _accel.BackpropLinearProjection(
                    bc.SelfAttnOutput,
                    dSelfRes,
                    block.SelfAttention.WO,
                    selfGrads.WO_Grad,
                    selfGrads.BiasO_Grad,
                    dSelfCombined);

                var (dQSelf, dKSelf, dVSelf) = _accel.MultiHeadAttentionBackward(
                    bc.SelfQ,
                    bc.SelfK,
                    bc.SelfV,
                    dSelfCombined,
                    nh,
                    scale,
                    _config.Price.UseDecoderOnly);

                _rotaryPositionEmbedding.ApplyBackwardInPlace(dQSelf, dKSelf, nh);

                var dBlockIn = _accel.BackpropQKV(
                    bc.BlockInput,
                    dQSelf,
                    dKSelf,
                    dVSelf,
                    block.SelfAttention.WQ,
                    block.SelfAttention.WK,
                    block.SelfAttention.WV,
                    selfGrads.WQ_Grad,
                    selfGrads.BiasQ_Grad,
                    selfGrads.WK_Grad,
                    selfGrads.BiasK_Grad,
                    selfGrads.WV_Grad,
                    selfGrads.BiasV_Grad);
                _accel.MatrixAddInPlace(dBlockIn, dSelfRes);
                dX = dBlockIn;
            }

            BackpropInputProjectionFromCache(dX, cache);
            return dGlobalHidden;
        }

        private void BackpropInputProjectionFromCache(float[,] dX, PriceTacForwardCache cache)
        {
            if (dX == null)
                throw new ArgumentNullException(nameof(dX));
            if (cache?.PriceContinuousInput == null)
                throw new ArgumentException("cache.PriceContinuousInput is required for input projection backward.", nameof(cache));

            int seqLen = dX.GetLength(0);
            int embeddingDim = _config.Price.EmbeddingDim;
            int inputFeatureDim = _config.Price.InputFeatureDim;
            int rowStart = cache.PriceContinuousInputRowStart;
            int rowCount = cache.PriceContinuousInputRowCount > 0 ? cache.PriceContinuousInputRowCount : seqLen;
            var input = cache.PriceContinuousInput;

            if (rowCount != seqLen)
                throw new InvalidOperationException($"Cached price input row count ({rowCount}) does not match dX rows ({seqLen}).");
            if (rowStart < 0 || rowStart + seqLen > input.GetLength(0))
                throw new ArgumentOutOfRangeException(nameof(cache.PriceContinuousInputRowStart), $"Invalid cached price row slice: start={rowStart}, count={seqLen}, rows={input.GetLength(0)}.");
            if (input.GetLength(1) != inputFeatureDim)
                throw new InvalidOperationException($"Cached price input feature dim ({input.GetLength(1)}) does not match config ({inputFeatureDim}).");

            _accel.BackpropInputProjection(
                dX,
                input,
                rowStart,
                _gradients.PriceInputProjectionGrad,
                _gradients.PriceInputProjectionBiasGrad,
                seqLen,
                embeddingDim,
                inputFeatureDim);
        }

        private (float[,] dQ, float[,] dK, float[,] dV, float[,,] dDecayBias) BackpropTimeDecayedAttn(float[,] Q, float[,] K, float[,] V, float[,] dOutput, float[][,] attnW, float[,] timeDiffs, TacamtBlock block)
        {
            return _accel.BackpropTimeDecayedAttention(Q, K, V, dOutput, attnW, timeDiffs, _config.Price.EmbeddingDim, _config.Price.NumHeads);
        }

        private void CommitObservedSampleToMemory(PriceTacInput inp, double currentTs, double timeUnitsPerPosition, int maxPriceMemory)
        {
            if (inp == null || inp.PriceSequence == null)
                return;
            if (timeUnitsPerPosition == 0.0)
                throw new ArgumentOutOfRangeException(nameof(timeUnitsPerPosition), "Must be non-zero.");

            int embDim = _config.Price.EmbeddingDim;
            float invTime = (float)(1.0 / timeUnitsPerPosition);

            var postCtxH = new List<float[]>();
            var postCtxT = new List<float>();
            var postCtxTypes = new List<int>();

            int numGlobal = 0;
            if (_config.Global.GlobalFeatureDim > 0 && inp.GlobalFeatures != null)
            {
                var postGlobalToken = _model.EmbedGlobalFeatures(inp.GlobalFeatures);
                postCtxH.Add(postGlobalToken);
                postCtxT.Add(0f);
                postCtxTypes.Add(2);
                numGlobal = 1;
            }

            foreach (var e in _model.PriceMemory)
            {
                if (e?.HiddenState == null)
                    continue;

                var v = new float[embDim];
                int copyDim = Math.Min(embDim, e.HiddenState.Length);
                for (int d = 0; d < copyDim; d++)
                    v[d] = e.HiddenState[d];

                postCtxH.Add(v);
                postCtxT.Add(-(float)((currentTs - e.AbsoluteTimestamp) * invTime));
                postCtxTypes.Add(1);
            }

            _model.BuildManualContext(postCtxH, postCtxT, postCtxTypes, embDim, out var postCombinedHidden, out var postCombinedTimes);

            var phUpdated = _model.ForwardPriceDecoderForMemoryBuild(inp.PriceSequence, postCombinedHidden, postCombinedTimes, numGlobal);
            int appendedPriceEntries = phUpdated.GetLength(0);

            for (int t = 0; t < appendedPriceEntries; t++)
            {
                var pv = new float[embDim];
                for (int d = 0; d < embDim; d++)
                    pv[d] = phUpdated[t, d];

                _model.PriceMemory.Add(new PriceMemoryEntry
                {
                    HiddenState = pv,
                    AbsoluteTimestamp = currentTs + t * timeUnitsPerPosition
                });
            }

            _model.LastPriceTimestamp = currentTs + Math.Max(0, inp.PriceSequence.GetLength(0) - 1) * timeUnitsPerPosition;
            _model.PricePruneMemoryAfterAppend(maxPriceMemory, appendedPriceEntries);
        }

        private PriceTacPrediction PredictWithCurrentMemoryNoCommit(PriceTacInput input, double currentAbsoluteTimestamp, double timeUnitsPerPosition = 1.0)
        {
            if (input == null)
                throw new ArgumentNullException(nameof(input));
            if (input.PriceSequence == null)
                throw new ArgumentNullException(nameof(input.PriceSequence));
            if (timeUnitsPerPosition == 0.0)
                throw new ArgumentOutOfRangeException(nameof(timeUnitsPerPosition), "Must be non-zero.");

            int embDim = _config.Price.EmbeddingDim;
            var ctxH = new List<float[]>();
            var ctxT = new List<float>();
            var ctxTypes = new List<int>();
            int numGlobal = 0;

            if (_config.Global.GlobalFeatureDim > 0 && input.GlobalFeatures != null)
            {
                var globalToken = _model.EmbedGlobalFeatures(input.GlobalFeatures);
                ctxH.Add(globalToken);
                ctxT.Add(0f);
                ctxTypes.Add(2);
                numGlobal = 1;
            }

            foreach (var e in _model.PriceMemory)
            {
                float relTime = -(float)((currentAbsoluteTimestamp - e.AbsoluteTimestamp) / timeUnitsPerPosition);
                var v = new float[embDim];
                for (int d = 0; d < embDim; d++)
                    v[d] = e.HiddenState[d];

                ctxH.Add(v);
                ctxT.Add(relTime);
                ctxTypes.Add(1);
            }

            _model.BuildManualContext(ctxH, ctxT, ctxTypes, embDim, out var contextHidden, out var contextTimes);

            var priceHidden = _model.ForwardPriceDecoderForMemoryBuild(input.PriceSequence, contextHidden, contextTimes, numGlobal);
            var (reg, range, quality, dir, midDir, conf) = _model.ProjectToOutputs(priceHidden);
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

        #region Gradient utilities

        private void AccumulateDecayGrads(ContentAwareDecayGradients tgt, ContentAwareDecayGradients src)
        {
            if (tgt == null || src == null)
                return;

            _accel.VectorAccumulate(tgt.LogBaseDecayRateGrad, src.LogBaseDecayRateGrad);
            _accel.VectorAccumulate(tgt.B2Grad, src.B2Grad);
            _accel.Matrix3DAddInPlace(tgt.QueryProjectionGrad, src.QueryProjectionGrad);
            _accel.Matrix3DAddInPlace(tgt.KeyProjectionGrad, src.KeyProjectionGrad);
            _accel.Matrix3DAddInPlace(tgt.TimeProjGrad, src.TimeProjGrad);
            _accel.Matrix3DAddInPlace(tgt.MemAttnOutputWGrad, src.MemAttnOutputWGrad);
            _accel.Matrix3DAddInPlace(tgt.W1Grad, src.W1Grad);
            _accel.MatrixAddInPlace(tgt.QueryProjectionBiasGrad, src.QueryProjectionBiasGrad);
            _accel.MatrixAddInPlace(tgt.KeyProjectionBiasGrad, src.KeyProjectionBiasGrad);
            _accel.MatrixAddInPlace(tgt.TimeLogFreqGrad, src.TimeLogFreqGrad);
            _accel.MatrixAddInPlace(tgt.TimeProjBiasGrad, src.TimeProjBiasGrad);
            _accel.MatrixAddInPlace(tgt.MemAttnOutputBGrad, src.MemAttnOutputBGrad);
            _accel.MatrixAddInPlace(tgt.B1Grad, src.B1Grad);
            _accel.MatrixAddInPlace(tgt.W2Grad, src.W2Grad);
        }

        private void ZeroAllGradients()
        {
            _gradients.Zero();

            for (int i = 0; i < _config.Price.NumLayers; i++)
            {
                foreach (var w in _priceFFNWeightGrads[i])
                    _accel.ZeroMatrix(w);
                foreach (var b in _priceFFNBiasGrads[i])
                    _accel.ZeroVector(b);
            }
        }

        private void ScaleAllGradients(float scale)
        {
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
                ScaleAttnGrads(bg.SelfAttnGrads, scale);
                ScaleAttnGrads(bg.CrossAttnGrads, scale);
                ScaleLNGrads(bg.LNSelfGrads, scale);
                ScaleLNGrads(bg.LNCrossGrads, scale);
                ScaleLNGrads(bg.LNFFNGrads, scale);
                ScaleDecayGrads(bg.DecayGrads, scale);
                foreach (var w in _priceFFNWeightGrads[i]) ScaleMatrix(w, scale);
                foreach (var b in _priceFFNBiasGrads[i]) ScaleVector(b, scale);
            }

            ScaleMatrix(_gradients.RegressionProjectionGrad, scale);
            ScaleVector(_gradients.RegressionBiasGrad, scale);
            ScaleMatrix(_gradients.RangeProjectionGrad, scale);
            ScaleVector(_gradients.RangeBiasGrad, scale);
            ScaleMatrix(_gradients.QualityProjectionGrad, scale);
            ScaleVector(_gradients.QualityBiasGrad, scale);
            ScaleMatrix(_gradients.DirectionProjectionGrad, scale);
            ScaleVector(_gradients.DirectionBiasGrad, scale);
            ScaleMatrix(_gradients.MidDirectionProjectionGrad, scale);
            ScaleVector(_gradients.MidDirectionBiasGrad, scale);

            if (_gradients.ConfidenceProjectionGrad != null)
            {
                ScaleMatrix(_gradients.ConfidenceProjectionGrad, scale);
                ScaleVector(_gradients.ConfidenceBiasGrad, scale);
            }

            ScaleMatrix(_gradients.ContextTypeEmbeddingGrad, scale);
        }

        private void ClipGradients(float threshold, float? precomputedNorm = null)
        {
            if (threshold <= 0f)
                return;

            float norm = precomputedNorm ?? ComputeGradientNorm();
            if (norm > threshold && norm > 0f)
                ScaleAllGradients(threshold / norm);
        }

        private static int[] CreateSequentialIndices(int count)
        {
            var indices = new int[count];
            for (int i = 0; i < count; i++)
                indices[i] = i;
            return indices;
        }

        private void ShuffleInPlace(int[] data)
        {
            for (int i = data.Length - 1; i > 0; i--)
            {
                int j = _random.Next(i + 1);
                (data[i], data[j]) = (data[j], data[i]);
            }
        }

        private float ComputeGradientNorm()
        {
            float sum = 0f;

            sum += _accel.MatrixSquaredNorm(_gradients.PriceInputProjectionGrad);
            sum += _accel.VectorSquaredNorm(_gradients.PriceInputProjectionBiasGrad);

            if (_gradients.GlobalFeatureProjectionGrad != null)
            {
                sum += _accel.MatrixSquaredNorm(_gradients.GlobalFeatureProjectionGrad);
                sum += _accel.VectorSquaredNorm(_gradients.GlobalFeatureBiasGrad);
            }

            for (int i = 0; i < _config.Price.NumLayers; i++)
            {
                var bg = _gradients.PriceBlockGrads[i];
                sum += AttnGradNorm(bg.SelfAttnGrads);
                sum += AttnGradNorm(bg.CrossAttnGrads);
                sum += LNGradNorm(bg.LNSelfGrads);
                sum += LNGradNorm(bg.LNCrossGrads);
                sum += LNGradNorm(bg.LNFFNGrads);
                sum += DecayGradNorm(bg.DecayGrads);
                sum += FFNGradNorm(i);
            }

            sum += _accel.MatrixSquaredNorm(_gradients.RegressionProjectionGrad) + _accel.VectorSquaredNorm(_gradients.RegressionBiasGrad);
            sum += _accel.MatrixSquaredNorm(_gradients.RangeProjectionGrad) + _accel.VectorSquaredNorm(_gradients.RangeBiasGrad);
            sum += _accel.MatrixSquaredNorm(_gradients.QualityProjectionGrad) + _accel.VectorSquaredNorm(_gradients.QualityBiasGrad);
            sum += _accel.MatrixSquaredNorm(_gradients.DirectionProjectionGrad) + _accel.VectorSquaredNorm(_gradients.DirectionBiasGrad);
            sum += _accel.MatrixSquaredNorm(_gradients.MidDirectionProjectionGrad) + _accel.VectorSquaredNorm(_gradients.MidDirectionBiasGrad);

            if (_gradients.ConfidenceProjectionGrad != null)
                sum += _accel.MatrixSquaredNorm(_gradients.ConfidenceProjectionGrad) + _accel.VectorSquaredNorm(_gradients.ConfidenceBiasGrad);

            sum += _accel.MatrixSquaredNorm(_gradients.ContextTypeEmbeddingGrad);
            return MathF.Sqrt(sum);
        }

        private float AttnGradNorm(AttentionGradients ag)
        {
            return _accel.MatrixSquaredNorm(ag.WQ_Grad)
                + _accel.MatrixSquaredNorm(ag.WK_Grad)
                + _accel.MatrixSquaredNorm(ag.WV_Grad)
                + _accel.MatrixSquaredNorm(ag.WO_Grad)
                + _accel.VectorSquaredNorm(ag.BiasQ_Grad)
                + _accel.VectorSquaredNorm(ag.BiasK_Grad)
                + _accel.VectorSquaredNorm(ag.BiasV_Grad)
                + _accel.VectorSquaredNorm(ag.BiasO_Grad);
        }

        private float LNGradNorm(LayerNormGradients lg)
        {
            return _accel.VectorSquaredNorm(lg.GammaGrad) + _accel.VectorSquaredNorm(lg.BetaGrad);
        }

        private float FFNGradNorm(int layer)
        {
            var weightGrads = _priceFFNWeightGrads[layer];
            var biasGrads = _priceFFNBiasGrads[layer];

            float sum = 0f;
            foreach (var w in weightGrads)
                sum += _accel.MatrixSquaredNorm(w);
            foreach (var b in biasGrads)
                sum += _accel.VectorSquaredNorm(b);
            return sum;
        }

        private void ScaleDecayGrads(ContentAwareDecayGradients g, float scale)
        {
            if (g == null)
                return;

            _accel.Matrix3DScaleInPlace(g.QueryProjectionGrad, scale);
            ScaleMatrix(g.QueryProjectionBiasGrad, scale);
            _accel.Matrix3DScaleInPlace(g.KeyProjectionGrad, scale);
            ScaleMatrix(g.KeyProjectionBiasGrad, scale);
            ScaleMatrix(g.TimeLogFreqGrad, scale);
            _accel.Matrix3DScaleInPlace(g.TimeProjGrad, scale);
            ScaleMatrix(g.TimeProjBiasGrad, scale);
            _accel.Matrix3DScaleInPlace(g.MemAttnOutputWGrad, scale);
            ScaleMatrix(g.MemAttnOutputBGrad, scale);
            _accel.Matrix3DScaleInPlace(g.W1Grad, scale);
            ScaleMatrix(g.B1Grad, scale);
            ScaleMatrix(g.W2Grad, scale);
            ScaleVector(g.B2Grad, scale);
            ScaleVector(g.LogBaseDecayRateGrad, scale);
        }

        private float DecayGradNorm(ContentAwareDecayGradients g)
        {
            if (g == null)
                return 0f;

            return _accel.MatrixSquaredNorm3D(g.QueryProjectionGrad)
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
                + _accel.MatrixSquaredNorm(g.W2Grad)
                + _accel.VectorSquaredNorm(g.B2Grad)
                + _accel.VectorSquaredNorm(g.LogBaseDecayRateGrad);
        }

        private void UpdateAllParameters(float lr)
        {
            // Keep lr=0 as a true no-op, matching the original training guard.
            if (lr == 0f)
                return;

            _accel.MatrixUpdate(_model.PriceInputProjection, _gradients.PriceInputProjectionGrad, lr);
            _accel.VectorUpdate(_model.PriceInputProjectionBias, _gradients.PriceInputProjectionBiasGrad, lr);

            if (_model.GlobalFeatureProjection != null && _gradients.GlobalFeatureProjectionGrad != null)
            {
                _accel.MatrixUpdate(_model.GlobalFeatureProjection, _gradients.GlobalFeatureProjectionGrad, lr);
                _accel.VectorUpdate(_model.GlobalFeatureBias, _gradients.GlobalFeatureBiasGrad, lr);
            }

            for (int i = 0; i < _config.Price.NumLayers; i++)
            {
                var b = _model.PriceBlocks[i];
                var bg = _gradients.PriceBlockGrads[i];

                UpdateAttn(b.SelfAttention, bg.SelfAttnGrads, lr);
                UpdateAttn(b.CrossAttention, bg.CrossAttnGrads, lr);

                if (bg.DecayGrads != null)
                    UpdateDecayNetwork(b.DecayNetwork, bg.DecayGrads, lr);

                _accel.VectorUpdate(b.LNSelfGamma, bg.LNSelfGrads.GammaGrad, lr);
                _accel.VectorUpdate(b.LNSelfBeta, bg.LNSelfGrads.BetaGrad, lr);
                _accel.VectorUpdate(b.LnCrossGamma, bg.LNCrossGrads.GammaGrad, lr);
                _accel.VectorUpdate(b.LnCrossBeta, bg.LNCrossGrads.BetaGrad, lr);
                _accel.VectorUpdate(b.LNFFNGamma, bg.LNFFNGrads.GammaGrad, lr);
                _accel.VectorUpdate(b.LNFFNBeta, bg.LNFFNGrads.BetaGrad, lr);

                b.FeedForwardNetwork.ApplyExternalGradients(_priceFFNWeightGrads[i], _priceFFNBiasGrads[i], lr);
            }

            _accel.MatrixUpdate(_model.RegressionProjection, _gradients.RegressionProjectionGrad, lr);
            _accel.VectorUpdate(_model.RegressionBias, _gradients.RegressionBiasGrad, lr);
            _accel.MatrixUpdate(_model.RangeProjection, _gradients.RangeProjectionGrad, lr);
            _accel.VectorUpdate(_model.RangeBias, _gradients.RangeBiasGrad, lr);
            _accel.MatrixUpdate(_model.QualityProjection, _gradients.QualityProjectionGrad, lr);
            _accel.VectorUpdate(_model.QualityBias, _gradients.QualityBiasGrad, lr);
            _accel.MatrixUpdate(_model.DirectionProjection, _gradients.DirectionProjectionGrad, lr);
            _accel.VectorUpdate(_model.DirectionBias, _gradients.DirectionBiasGrad, lr);
            _accel.MatrixUpdate(_model.MidDirectionProjection, _gradients.MidDirectionProjectionGrad, lr);
            _accel.VectorUpdate(_model.MidDirectionBias, _gradients.MidDirectionBiasGrad, lr);

            if (_config.Output.UseConfidenceHead && _model.ConfidenceProjection != null)
            {
                _accel.MatrixUpdate(_model.ConfidenceProjection, _gradients.ConfidenceProjectionGrad, lr);
                _accel.VectorUpdate(_model.ConfidenceBias, _gradients.ConfidenceBiasGrad, lr);
            }

            _accel.MatrixUpdate(_model.ContextTypeEmbedding, _gradients.ContextTypeEmbeddingGrad, lr);
        }

        private void UpdateAttn(MultiHeadAttention attn, AttentionGradients ag, float lr)
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

        private void UpdateDecayNetwork(ContentAwareDecayNetwork net, ContentAwareDecayGradients g, float lr)
        {
            if (net == null || g == null)
                return;

            float minLogRate = net.GetMinLogBaseDecayRate();
            float maxLogRate = net.GetMaxLogBaseDecayRate();

            _accel.VectorUpdateClamped(net.LogBaseDecayRate, g.LogBaseDecayRateGrad, lr, minLogRate, maxLogRate);
            _accel.VectorUpdate(net.B2, g.B2Grad, lr);
            _accel.Matrix3DUpdate(net.QueryProjection, g.QueryProjectionGrad, lr);
            _accel.Matrix3DUpdate(net.KeyProjection, g.KeyProjectionGrad, lr);
            _accel.Matrix3DUpdate(net.TimeProj, g.TimeProjGrad, lr);
            _accel.Matrix3DUpdate(net.MemAttnOutputW, g.MemAttnOutputWGrad, lr);
            _accel.Matrix3DUpdate(net.W1, g.W1Grad, lr);
            _accel.MatrixUpdate(net.QueryProjectionBias, g.QueryProjectionBiasGrad, lr);
            _accel.MatrixUpdate(net.KeyProjectionBias, g.KeyProjectionBiasGrad, lr);
            _accel.MatrixUpdate(net.TimeLogFreq, g.TimeLogFreqGrad, lr);
            _accel.MatrixUpdate(net.TimeProjBias, g.TimeProjBiasGrad, lr);
            _accel.MatrixUpdate(net.MemAttnOutputB, g.MemAttnOutputBGrad, lr);
            _accel.MatrixUpdate(net.B1, g.B1Grad, lr);
            _accel.MatrixUpdate(net.W2, g.W2Grad, lr);
        }

        private float ComputeLR(int epoch)
        {
            float baseLR = _trainConfig.LearningRate;
            if (_trainConfig.WarmupEpochs > 0 && epoch < _trainConfig.WarmupEpochs)
                return baseLR * (epoch + 1f) / _trainConfig.WarmupEpochs;
            if (_trainConfig.UseLearningRateDecay)
                return baseLR * MathF.Pow(_trainConfig.LearningRateDecay, epoch - _trainConfig.WarmupEpochs);
            return baseLR;
        }

        private void ScaleMatrix(float[,] m, float s) => _accel.MatrixScaleInPlace(m, s);
        private void ScaleVector(float[] v, float s) => _accel.VectorScaleInPlace(v, s);

        private void ScaleAttnGrads(AttentionGradients ag, float s)
        {
            ScaleMatrix(ag.WQ_Grad, s);
            ScaleMatrix(ag.WK_Grad, s);
            ScaleMatrix(ag.WV_Grad, s);
            ScaleMatrix(ag.WO_Grad, s);
            ScaleVector(ag.BiasQ_Grad, s);
            ScaleVector(ag.BiasK_Grad, s);
            ScaleVector(ag.BiasV_Grad, s);
            ScaleVector(ag.BiasO_Grad, s);
        }

        private void ScaleLNGrads(LayerNormGradients lg, float s)
        {
            ScaleVector(lg.GammaGrad, s);
            ScaleVector(lg.BetaGrad, s);
        }

        private static void BuildTargetArrays(PriceTacTarget[] targets, int offset, int count, out float[,] tgtReg, out float[,] tgtRange, out float[,] tgtQuality, out float[,] tgtDir, out float[,] tgtMid)
        {
            int rDim = PriceTacOutputConfig.RegressionOutputCount;
            tgtReg = new float[count, rDim];
            tgtRange = new float[count, 1];
            tgtQuality = new float[count, 1];
            tgtDir = new float[count, 1];
            tgtMid = new float[count, 1];

            for (int t = 0; t < count; t++)
            {
                var tg = targets[offset + t];
                tgtReg[t, 0] = tg.High;
                tgtReg[t, 1] = tg.Low;
                tgtReg[t, 2] = tg.Close;
                tgtRange[t, 0] = tg.Range;
                tgtQuality[t, 0] = tg.Quality;
                tgtDir[t, 0] = tg.Direction;
                tgtMid[t, 0] = tg.MidWindowDirection;
            }
        }

        private static float[] BuildPreviousCloseArray(PriceTacTarget[] targets, int offset, int count)
        {
            if (targets == null)
                throw new ArgumentNullException(nameof(targets));
            if (offset <= 0)
                throw new ArgumentOutOfRangeException(nameof(offset), "offset must be >= 1 so a previous close exists.");
            if (offset + count > targets.Length)
                throw new ArgumentException("offset + count exceeds targets length.");

            var prevClose = new float[count];
            for (int t = 0; t < count; t++)
                prevClose[t] = targets[offset + t - 1].Close;

            return prevClose;
        }

        #endregion

        #region Validation

        public float Validate(PriceTacInput[] inputs, PriceTacTarget[][] targets)
        {
            if (inputs == null)
                throw new ArgumentNullException(nameof(inputs));
            if (targets == null)
                throw new ArgumentNullException(nameof(targets));
            if (inputs.Length != targets.Length)
                throw new ArgumentException("inputs and targets must have the same length.");

            float total = 0f;
            int count = 0;
            int minSplit = _config.PriceContext.MinHistoryLength + _config.PriceContext.MinCurrentLength + 1;

            for (int i = 0; i < inputs.Length; i++)
            {
                if (inputs[i]?.PriceSequence == null || targets[i] == null)
                    continue;

                int sl = inputs[i].PriceSequence.GetLength(0);
                if (sl < 2 || targets[i].Length < sl)
                    continue;

                try
                {
                    if (_config.PriceContext.Enabled && sl >= minSplit)
                    {
                        int sp = sl / 2;
                        int csl = sl - 1 - sp;

                        BuildTargetArrays(targets[i], sp + 1, csl, out var tgtReg, out var tgtRange, out var tgtQuality, out var tgtDir, out var tgtMid);

                        var priceCtxH = _model.EncodePriceHistory(inputs[i].PriceSequence, 0, sp);
                        var priceCtxT = new float[sp];
                        for (int t = 0; t < sp; t++)
                            priceCtxT[t] = -(float)(sp - t);

                        var cache = new PriceTacForwardCache(_config.Price.NumLayers);
                        var (reg, range, quality, dir, midDir, _) = _model.ForwardWithPriceContextSliceAndCache(
                            inputs[i],
                            sp,
                            csl,
                            priceCtxH,
                            priceCtxT,
                            cache,
                            isTraining: false);

                        total += ComputeValLoss(reg, range, quality, dir, midDir, tgtReg, tgtRange, tgtQuality, tgtDir, tgtMid);
                        count += csl;
                    }
                    else
                    {
                        int eff = sl - 1;
                        BuildTargetArrays(targets[i], 1, eff, out var tgtReg, out var tgtRange, out var tgtQuality, out var tgtDir, out var tgtMid);

                        var cache = new PriceTacForwardCache(_config.Price.NumLayers);
                        var (reg, range, quality, dir, midDir, _) = _model.ForwardWithCache(inputs[i], 0, eff, cache, isTraining: false);

                        total += ComputeValLoss(reg, range, quality, dir, midDir, tgtReg, tgtRange, tgtQuality, tgtDir, tgtMid);
                        count += eff;
                    }
                }
                catch
                {
                    // Keep validation robust against malformed samples.
                }
            }

            return count > 0 ? total / count : 0f;
        }

        public float ValidateSequential(PriceTacInput[] inputs, PriceTacTarget[][] targets, double[] timestamps, double timeUnitsPerPosition = 1.0, int maxPriceMemory = 200)
        {
            if (inputs == null)
                throw new ArgumentNullException(nameof(inputs));
            if (targets == null)
                throw new ArgumentNullException(nameof(targets));
            if (timestamps == null)
                throw new ArgumentNullException(nameof(timestamps));
            if (inputs.Length != targets.Length || inputs.Length != timestamps.Length)
                throw new ArgumentException("inputs, targets, and timestamps must all have the same length.");
            if (timeUnitsPerPosition == 0.0)
                throw new ArgumentOutOfRangeException(nameof(timeUnitsPerPosition), "Must be non-zero.");

            float total = 0f;
            int count = 0;
            int[] ordered = Enumerable.Range(0, inputs.Length).OrderBy(i => timestamps[i]).ToArray();

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
                        var priceInp = _accel.SliceRows(input.PriceSequence, 0, sl - 1);
                        var wrappedInput = new PriceTacInput
                        {
                            PredictionTimestamp = input.PredictionTimestamp,
                            PriceSequence = priceInp,
                            GlobalFeatures = input.GlobalFeatures
                        };

                        var pred = PredictWithCurrentMemoryNoCommit(wrappedInput, timestamps[idx], timeUnitsPerPosition);
                        var tgt = sampleTargets[sl - 1];

                        total += ComputeSequentialValLoss(pred, tgt);
                        count++;

                        CommitObservedSampleToMemory(input, timestamps[idx], timeUnitsPerPosition, maxPriceMemory);
                    }
                    catch
                    {
                        // Keep validation robust against malformed samples.
                    }
                }

                return count > 0 ? total / count : 0f;
            }
            finally
            {
                _model.ClearAllMemory();
            }
        }

        public float ValidateAligned(PriceTacInput[] inputs, PriceTacTarget[][] targets)
        {
            if (inputs == null)
                throw new ArgumentNullException(nameof(inputs));
            if (targets == null)
                throw new ArgumentNullException(nameof(targets));
            if (inputs.Length != targets.Length)
                throw new ArgumentException("inputs and targets must have the same length.");

            float total = 0f;
            int count = 0;
            int minSplit = _config.PriceContext.MinHistoryLength + _config.PriceContext.MinCurrentLength + 1;

            for (int i = 0; i < inputs.Length; i++)
            {
                if (inputs[i]?.PriceSequence == null || targets[i] == null)
                    continue;

                int sl = inputs[i].PriceSequence.GetLength(0);
                if (sl < 2 || targets[i].Length < sl)
                    continue;

                try
                {
                    if (_config.PriceContext.Enabled && sl >= minSplit)
                    {
                        int sp = sl / 2;
                        int csl = sl - 1 - sp;

                        BuildTargetArrays(targets[i], sp + 1, csl, out var tgtReg, out var tgtRange, out var tgtQuality, out var tgtDir, out var tgtMid);
                        var prevClose = BuildPreviousCloseArray(targets[i], sp + 1, csl);

                        var priceCtxH = _model.EncodePriceHistory(inputs[i].PriceSequence, 0, sp);
                        var priceCtxT = new float[sp];
                        for (int t = 0; t < sp; t++)
                            priceCtxT[t] = -(float)(sp - t);

                        var cache = new PriceTacForwardCache(_config.Price.NumLayers);
                        var (reg, range, quality, dir, midDir, _) = _model.ForwardWithPriceContextSliceAndCache(
                            inputs[i],
                            sp,
                            csl,
                            priceCtxH,
                            priceCtxT,
                            cache,
                            isTraining: false);

                        total += ComputeAlignedValLoss(reg, range, quality, dir, midDir, tgtReg, tgtRange, tgtQuality, tgtDir, tgtMid, prevClose) * csl;
                        count += csl;
                    }
                    else
                    {
                        int eff = sl - 1;

                        BuildTargetArrays(targets[i], 1, eff, out var tgtReg, out var tgtRange, out var tgtQuality, out var tgtDir, out var tgtMid);
                        var prevClose = BuildPreviousCloseArray(targets[i], 1, eff);

                        var cache = new PriceTacForwardCache(_config.Price.NumLayers);
                        var (reg, range, quality, dir, midDir, _) = _model.ForwardWithCache(inputs[i], 0, eff, cache, isTraining: false);

                        total += ComputeAlignedValLoss(reg, range, quality, dir, midDir, tgtReg, tgtRange, tgtQuality, tgtDir, tgtMid, prevClose) * eff;
                        count += eff;
                    }
                }
                catch
                {
                    // Keep validation robust against malformed samples.
                }
            }

            return count > 0 ? total / count : 0f;
        }

        public float ValidateSequentialAligned(PriceTacInput[] inputs, PriceTacTarget[][] targets, double[] timestamps, double timeUnitsPerPosition = 1.0, int maxPriceMemory = 200)
        {
            if (inputs == null)
                throw new ArgumentNullException(nameof(inputs));
            if (targets == null)
                throw new ArgumentNullException(nameof(targets));
            if (timestamps == null)
                throw new ArgumentNullException(nameof(timestamps));
            if (inputs.Length != targets.Length || inputs.Length != timestamps.Length)
                throw new ArgumentException("inputs, targets, and timestamps must all have the same length.");
            if (timeUnitsPerPosition == 0.0)
                throw new ArgumentOutOfRangeException(nameof(timeUnitsPerPosition), "Must be non-zero.");

            float total = 0f;
            int count = 0;
            int[] ordered = Enumerable.Range(0, inputs.Length).OrderBy(i => timestamps[i]).ToArray();

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
                        var priceInp = _accel.SliceRows(input.PriceSequence, 0, sl - 1);
                        var wrappedInput = new PriceTacInput
                        {
                            PredictionTimestamp = input.PredictionTimestamp,
                            PriceSequence = priceInp,
                            GlobalFeatures = input.GlobalFeatures
                        };

                        var pred = PredictWithCurrentMemoryNoCommit(wrappedInput, timestamps[idx], timeUnitsPerPosition);
                        var tgt = sampleTargets[sl - 1];
                        float prevClose = sampleTargets[sl - 2].Close;

                        total += ComputeAlignedSequentialValLoss(pred, tgt, prevClose);
                        count++;

                        CommitObservedSampleToMemory(input, timestamps[idx], timeUnitsPerPosition, maxPriceMemory);
                    }
                    catch
                    {
                        // Keep validation robust against malformed individual samples.
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
            int regressionDim = PriceTacOutputConfig.RegressionOutputCount;

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

        private float ComputeSequentialValLoss(PriceTacPrediction pred, PriceTacTarget tgt)
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
            int rDim = PriceTacOutputConfig.RegressionOutputCount;

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

            mseLoss /= sl * rDim;
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

        private float ComputeAlignedSequentialValLoss(PriceTacPrediction pred, PriceTacTarget tgt, float prevClose)
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

            // True MSE over the three regression outputs: High, Low, Close.
            mseLoss /= 3f;

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
                {
                    closeDirConsistencyLoss = MathF.Exp(-z);
                }
                else if (z < -20f)
                {
                    closeDirConsistencyLoss = -z;
                }
                else
                {
                    closeDirConsistencyLoss = MathF.Log(1f + MathF.Exp(-z));
                }
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

        public void TrainSequentialFromWindowEndTimestamps(PriceTacInput[] inputs, PriceTacTarget[][] targets, double[] windowEndTimestamps, double timeUnitsPerPosition = 1.0, int maxPriceMemory = 200, float[][] confTargets = null)
        {
            var windowStartTimestamps = ConvertWindowEndTimestampsToWindowStart(inputs, windowEndTimestamps, timeUnitsPerPosition);

            TrainSequential(inputs, targets, windowStartTimestamps, timeUnitsPerPosition, maxPriceMemory, confTargets);
        }

        public float ValidateSequentialFromWindowEndTimestamps(PriceTacInput[] inputs, PriceTacTarget[][] targets, double[] windowEndTimestamps, double timeUnitsPerPosition = 1.0, int maxPriceMemory = 200)
        {
            var windowStartTimestamps = ConvertWindowEndTimestampsToWindowStart(
                inputs,
                windowEndTimestamps,
                timeUnitsPerPosition);

            return ValidateSequential(
                inputs,
                targets,
                windowStartTimestamps,
                timeUnitsPerPosition,
                maxPriceMemory);
        }

        public float ValidateSequentialAlignedFromWindowEndTimestamps(PriceTacInput[] inputs, PriceTacTarget[][] targets, double[] windowEndTimestamps, double timeUnitsPerPosition = 1.0, int maxPriceMemory = 200)
        {
            var windowStartTimestamps = ConvertWindowEndTimestampsToWindowStart(
                inputs,
                windowEndTimestamps,
                timeUnitsPerPosition);

            return ValidateSequentialAligned(
                inputs,
                targets,
                windowStartTimestamps,
                timeUnitsPerPosition,
                maxPriceMemory);
        }

        private static double[] ConvertWindowEndTimestampsToWindowStart(PriceTacInput[] inputs, double[] windowEndTimestamps, double timeUnitsPerPosition)
        {
            if (inputs == null)
                throw new ArgumentNullException(nameof(inputs));
            if (windowEndTimestamps == null)
                throw new ArgumentNullException(nameof(windowEndTimestamps));
            if (inputs.Length != windowEndTimestamps.Length)
                throw new ArgumentException("inputs and windowEndTimestamps must have the same length.");
            if (timeUnitsPerPosition == 0.0)
                throw new ArgumentOutOfRangeException(nameof(timeUnitsPerPosition), "Must be non-zero.");

            var converted = new double[windowEndTimestamps.Length];

            for (int i = 0; i < converted.Length; i++)
            {
                int seqLen = inputs[i]?.PriceSequence?.GetLength(0) ?? 1;
                converted[i] = PriceTacModel.WindowStartTimestampFromWindowEnd(
                    windowEndTimestamps[i],
                    seqLen,
                    timeUnitsPerPosition);
            }

            return converted;
        }
    }
}
