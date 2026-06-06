using CallaghanDev.ML.AccelerationManagers;
using CallaghanDev.ML.Transformers.Cache;
using CallaghanDev.ML.Transformers.Configuration;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;

namespace CallaghanDev.ML.Transformers.MultiTypeTransformer
{
    public class TransformerTrainer
    {
        private readonly LanguageModel _model;
        private readonly TransformerConfig _modelConfig;
        private readonly TrainingConfig _trainConfig;
        private readonly TransformerGradients _gradients;
        private readonly IAccelerationManager _accel;
        private readonly Random _random;
        private readonly RotaryPositionEmbedding _rotaryPositionEmbedding;
        private readonly List<List<float[,]>> _ffnWeightGrads;
        private readonly List<List<float[]>> _ffnBiasGrads;

        public TransformerTrainer(LanguageModel model, TrainingConfig trainConfig)
        {
            _model = model;
            _modelConfig = model.Config;
            _trainConfig = trainConfig;
            _gradients = new TransformerGradients(_modelConfig);
            _accel = model.AccelerationManager;
            _random = new Random();

            _ffnWeightGrads = new List<List<float[,]>>();
            _ffnBiasGrads = new List<List<float[]>>();
            _rotaryPositionEmbedding = new RotaryPositionEmbedding(_accel);

            for (int i = 0; i < _modelConfig.NumLayers; i++)
            {
                var (wGrads, bGrads) = _model.Blocks[i].FeedForwardNetwork.CreateGradientStorage();
                _ffnWeightGrads.Add(wGrads);
                _ffnBiasGrads.Add(bGrads);
            }
        }
        public void Train(int[][] sequences, int[][] validationSequences = null)
        {
            if (!_modelConfig.Data.UsesDiscreteTokens)
            {
                throw new InvalidOperationException($"Use TrainContinuous() for {_modelConfig.Data.DataType}.");
            }
            if (sequences == null)
            {
                throw new ArgumentNullException(nameof(sequences));
            }
            if (_trainConfig.BatchSize <= 0)
            {
                throw new ArgumentOutOfRangeException(nameof(_trainConfig.BatchSize), "BatchSize must be positive.");
            }
            if (_trainConfig.Epochs < 0)
            {
                throw new ArgumentOutOfRangeException(nameof(_trainConfig.Epochs), "Epochs must be non-negative.");
            }

            var validSequences = sequences.Where(seq => seq != null && seq.Length >= 2).ToArray();

            if (validSequences.Length == 0)
            {
                throw new ArgumentException("No valid sequences to train on. All sequences must contain at least 2 tokens.", nameof(sequences));
            }

            if (validSequences.Length < sequences.Length && _trainConfig.Verbose)
            {
                Console.WriteLine($"Filtered out {sequences.Length - validSequences.Length} sequences that were too short.");
            }

            float currentLR = _trainConfig.LearningRate;

            for (int epoch = 0; epoch < _trainConfig.Epochs; epoch++)
            {
                if (_trainConfig.Verbose)
                {
                    Console.WriteLine($"\n=== Epoch {epoch + 1}/{_trainConfig.Epochs} ===");
                }

                var shuffled = ShuffleArray(validSequences);
                float epochLoss = 0f;
                int numBatches = 0;

                for (int i = 0; i < shuffled.Length; i += _trainConfig.BatchSize)
                {
                    int batchSize = Math.Min(_trainConfig.BatchSize, shuffled.Length - i);
                    if (batchSize == 0) continue;

                    float batchLoss = TrainBatchDiscrete(shuffled, i, batchSize, currentLR);
                    epochLoss += batchLoss;
                    numBatches++;

                    bool shouldValidate = validationSequences != null
                                       && _trainConfig.ValidationInterval > 0
                                       && numBatches % _trainConfig.ValidationInterval == 0;

                    if (shouldValidate)
                    {
                        float valLoss = Validate(validationSequences);
                        if (_trainConfig.Verbose)
                        {
                            Console.WriteLine($"Batch {numBatches}: Train Loss = {batchLoss:F4}, Val Loss = {valLoss:F4}");
                        }
                    }
                    else if (_trainConfig.Verbose && numBatches % 10 == 0)
                    {
                        Console.WriteLine($"Batch {numBatches}: Loss = {batchLoss:F4}");
                    }
                }

                if (_trainConfig.Verbose)
                {
                    Console.WriteLine($"Epoch {epoch + 1} Average Loss: {(numBatches > 0 ? epochLoss / numBatches : 0):F4}");
                }

                if (_trainConfig.UseLearningRateDecay)
                {
                    currentLR *= _trainConfig.LearningRateDecay;
                    if (_trainConfig.Verbose)
                    {
                        Console.WriteLine($"Learning rate: {currentLR:F6}");
                    }
                }
            }
        }

        public void TrainContinuous(float[][,] inputs, float[][,] regressionTargets = null, int[][] classTargets = null, float[][,] valInputs = null, float[][,] valRegressionTargets = null, int[][] valClassTargets = null)
        {
            if (_modelConfig.Data.UsesDiscreteTokens)
                throw new InvalidOperationException($"Use Train(int[][]) for {_modelConfig.Data.DataType}.");
            if (inputs == null)
                throw new ArgumentNullException(nameof(inputs));
            if (_trainConfig.BatchSize <= 0)
                throw new ArgumentOutOfRangeException(nameof(_trainConfig.BatchSize), "BatchSize must be positive.");
            if (_trainConfig.Epochs < 0)
                throw new ArgumentOutOfRangeException(nameof(_trainConfig.Epochs), "Epochs must be non-negative.");

            if (_modelConfig.Data.DataType == TransformerDataType.TimeSeriesRegression)
            {
                if (regressionTargets == null)
                    throw new ArgumentException("regressionTargets is required for TimeSeriesRegression.", nameof(regressionTargets));
                if (regressionTargets.Length != inputs.Length)
                    throw new ArgumentException("regressionTargets length must match inputs length.", nameof(regressionTargets));
            }

            if (_modelConfig.Data.DataType == TransformerDataType.TimeSeriesClassification)
            {
                if (classTargets == null)
                    throw new ArgumentException("classTargets is required for TimeSeriesClassification.", nameof(classTargets));
                if (classTargets.Length != inputs.Length)
                    throw new ArgumentException("classTargets length must match inputs length.", nameof(classTargets));
            }

            var validIndices = new List<int>();
            for (int i = 0; i < inputs.Length; i++)
            {
                if (inputs[i] == null || inputs[i].GetLength(0) < 2)
                    continue;

                if (inputs[i].GetLength(1) != _modelConfig.InputFeatureDim)
                    throw new ArgumentException($"Input sequence {i} has feature dimension {inputs[i].GetLength(1)}, expected {_modelConfig.InputFeatureDim}.", nameof(inputs));

                validIndices.Add(i);
            }

            if (validIndices.Count == 0)
                throw new ArgumentException("No valid sequences. All must have at least 2 timesteps.", nameof(inputs));

            float currentLR = _trainConfig.LearningRate;

            for (int epoch = 0; epoch < _trainConfig.Epochs; epoch++)
            {
                if (_trainConfig.Verbose)
                    Console.WriteLine($"\n=== Epoch {epoch + 1}/{_trainConfig.Epochs} ===");

                var shuffled = validIndices.ToArray();
                ShuffleInPlace(shuffled);
                float epochLoss = 0f;
                int numBatches = 0;

                for (int i = 0; i < shuffled.Length; i += _trainConfig.BatchSize)
                {
                    int batchSize = Math.Min(_trainConfig.BatchSize, shuffled.Length - i);
                    if (batchSize == 0) continue;

                    float batchLoss = TrainBatchContinuous(shuffled, i, batchSize, inputs, regressionTargets, classTargets, currentLR);
                    epochLoss += batchLoss;
                    numBatches++;

                    if (_trainConfig.Verbose && numBatches % 10 == 0)
                        Console.WriteLine($"Batch {numBatches}: Loss = {batchLoss:F4}");
                }

                if (_trainConfig.Verbose)
                    Console.WriteLine($"Epoch {epoch + 1} Average Loss: {(numBatches > 0 ? epochLoss / numBatches : 0):F4}");

                if (_trainConfig.UseLearningRateDecay)
                {
                    currentLR *= _trainConfig.LearningRateDecay;
                    if (_trainConfig.Verbose)
                        Console.WriteLine($"Learning rate: {currentLR:F6}");
                }
            }
        }

        private float TrainBatchDiscrete(int[][] batch, float learningRate)
        {
            if (batch == null)
                throw new ArgumentNullException(nameof(batch));

            return TrainBatchDiscrete(batch, 0, batch.Length, learningRate);
        }

        private float TrainBatchDiscrete(int[][] batch, int start, int count, float learningRate)
        {
            if (batch == null)
                throw new ArgumentNullException(nameof(batch));
            if (start < 0 || count < 0 || start + count > batch.Length)
                throw new ArgumentOutOfRangeException(nameof(start));

            ZeroAllGradients();
            float totalLoss = 0f;
            int validCount = 0;
            var buckets = new Dictionary<int, List<int>>();

            for (int batchOffset = 0; batchOffset < count; batchOffset++)
            {
                int sourceIndex = start + batchOffset;
                var sequence = batch[sourceIndex];
                if (sequence == null || sequence.Length < 2 || !TokensAreValid(sequence))
                {
                    continue;
                }

                int inputLen = sequence.Length - 1;
                if (!buckets.TryGetValue(inputLen, out var bucket))
                {
                    bucket = new List<int>();
                    buckets[inputLen] = bucket;
                }
                bucket.Add(sourceIndex);
            }

            foreach (var pair in buckets)
            {
                int inputLen = pair.Key;
                var offsets = pair.Value;
                if (offsets.Count == 0)
                {
                    continue;
                }

                try
                {
                    var cache = new ForwardCache(_modelConfig.NumLayers);
                    float[,] logits;
                    int[] targets;

                    if (offsets.Count == 1)
                    {
                        var sequence = batch[offsets[0]];
                        logits = ForwardWithCacheDiscrete(sequence, 0, inputLen, cache);
                        targets = sequence;
                        int effectiveSingle = Math.Min(logits.GetLength(0), inputLen);
                        if (effectiveSingle <= 0) continue;

                        var (loss, dLogits) = _accel.CrossEntropyLossAndGradient(logits, targets, 1, effectiveSingle);
                        if (!IsFinite(loss) || !MatrixAllFinite(dLogits))
                            continue;

                        BackpropFromOutput(dLogits, cache);
                        totalLoss += loss;
                        validCount++;
                    }
                    else
                    {
                        logits = ForwardWithCacheDiscreteBatch(batch, offsets, inputLen, cache);
                        targets = BuildDiscreteFlatTargets(batch, offsets, inputLen);
                        int effectiveLen = Math.Min(logits.GetLength(0), targets.Length);
                        if (effectiveLen <= 0) continue;

                        var (loss, dLogits) = _accel.CrossEntropyLossAndGradient(logits, targets, effectiveLen);
                        if (!IsFinite(loss) || !MatrixAllFinite(dLogits))
                            continue;

                        // CrossEntropyLossAndGradient averages over all rows. The historical trainer
                        // accumulated per-sample average gradients and divided by valid sample count
                        // at the end, so multiply by the bucket sample count before the final batch scale.
                        ScaleMatrixLocal(dLogits, offsets.Count);
                        BackpropFromOutput(dLogits, cache);
                        totalLoss += loss * offsets.Count;
                        validCount += offsets.Count;
                    }
                }
                catch (Exception ex)
                {
                    // Match the old all-or-nothing safety behavior for a failed bucket.
                    ZeroAllGradients();
                    totalLoss = 0f;
                    validCount = 0;

                    if (_trainConfig.Verbose)
                        Console.WriteLine($"ERROR during training: {ex.Message}");
                }
            }

            if (validCount == 0)
                return 0f;

            ScaleGradients(1.0f / validCount);

            float gradNorm = ComputeGradientNorm();
            if (!IsFinite(gradNorm))
            {
                ZeroAllGradients();
                return totalLoss / validCount;
            }

            if (_trainConfig.UseGradientClipping)
                ClipGradients(_trainConfig.GradientClipThreshold, gradNorm);

            UpdateParameters(learningRate);
            return totalLoss / validCount;
        }

        private float TrainBatchContinuous(int[] batchIndices, float[][,] allInputs, float[][,] allRegTargets, int[][] allClassTargets, float learningRate)
        {
            if (batchIndices == null)
                throw new ArgumentNullException(nameof(batchIndices));

            return TrainBatchContinuous(batchIndices, 0, batchIndices.Length, allInputs, allRegTargets, allClassTargets, learningRate);
        }

        private float TrainBatchContinuous(int[] batchIndices, int start, int count, float[][,] allInputs, float[][,] allRegTargets, int[][] allClassTargets, float learningRate)
        {
            if (batchIndices == null)
                throw new ArgumentNullException(nameof(batchIndices));
            if (allInputs == null)
                throw new ArgumentNullException(nameof(allInputs));
            if (start < 0 || count < 0 || start + count > batchIndices.Length)
                throw new ArgumentOutOfRangeException(nameof(start));

            ZeroAllGradients();
            float totalLoss = 0f;
            int validCount = 0;
            var buckets = new Dictionary<int, List<int>>();

            for (int batchOffset = 0; batchOffset < count; batchOffset++)
            {
                int idx = batchIndices[start + batchOffset];
                if ((uint)idx >= (uint)allInputs.Length)
                    continue;

                var inputSeq = allInputs[idx];
                if (inputSeq == null) continue;

                int seqLen = inputSeq.GetLength(0);
                if (seqLen < 2 || inputSeq.GetLength(1) != _modelConfig.InputFeatureDim)
                    continue;

                if (!MatrixAllFinite(inputSeq))
                    continue;

                if (_modelConfig.Data.DataType == TransformerDataType.TimeSeriesRegression)
                {
                    if (!RegressionTargetIsValid(allRegTargets, idx, seqLen))
                        continue;
                }
                else
                {
                    if (!ClassTargetIsValid(allClassTargets, idx, seqLen))
                        continue;
                }

                int inputLen = seqLen - 1;
                if (!buckets.TryGetValue(inputLen, out var bucket))
                {
                    bucket = new List<int>();
                    buckets[inputLen] = bucket;
                }
                bucket.Add(idx);
            }

            foreach (var pair in buckets)
            {
                int inputLen = pair.Key;
                var sourceIndices = pair.Value;
                if (sourceIndices.Count == 0)
                {
                    continue;
                }

                try
                {
                    var cache = new ForwardCache(_modelConfig.NumLayers);
                    var output = sourceIndices.Count == 1
                        ? ForwardWithCacheContinuous(allInputs[sourceIndices[0]], 0, inputLen, cache)
                        : ForwardWithCacheContinuousBatch(sourceIndices, 0, sourceIndices.Count, allInputs, inputLen, cache);

                    float loss;
                    float[,] dOutput;
                    int effectiveLen = Math.Min(output.GetLength(0), sourceIndices.Count * inputLen);
                    if (effectiveLen <= 0)
                        continue;

                    if (_modelConfig.Data.DataType == TransformerDataType.TimeSeriesRegression)
                    {
                        if (sourceIndices.Count == 1)
                        {
                            var result = _accel.MSELossAndGradient(output, allRegTargets[sourceIndices[0]], 1, Math.Min(output.GetLength(0), inputLen));
                            loss = result.loss;
                            dOutput = result.dOutput;
                        }
                        else
                        {
                            var targets = BuildContinuousRegressionFlatTargets(sourceIndices, 0, sourceIndices.Count, allRegTargets, inputLen);
                            var result = _accel.MSELossAndGradient(output, targets, effectiveLen);
                            loss = result.loss;
                            dOutput = result.dOutput;
                        }
                    }
                    else
                    {
                        if (sourceIndices.Count == 1)
                        {
                            var result = _accel.CrossEntropyLossAndGradient(output, allClassTargets[sourceIndices[0]], 1, Math.Min(output.GetLength(0), inputLen));
                            loss = result.loss;
                            dOutput = result.dLogits;
                        }
                        else
                        {
                            var targets = BuildContinuousClassFlatTargets(sourceIndices, 0, sourceIndices.Count, allClassTargets, inputLen);
                            var result = _accel.CrossEntropyLossAndGradient(output, targets, effectiveLen);
                            loss = result.loss;
                            dOutput = result.dLogits;
                        }
                    }

                    if (!IsFinite(loss) || !MatrixAllFinite(dOutput))
                        continue;

                    if (sourceIndices.Count > 1)
                    {
                        // Loss helpers average over all flattened rows. The historical trainer
                        // averaged each sample first, accumulated those sample gradients, then
                        // divided by valid sample count at the end. For equal-length buckets,
                        // multiplying the flattened gradient by the sample count preserves that scale.
                        ScaleMatrixLocal(dOutput, sourceIndices.Count);
                    }

                    BackpropFromOutput(dOutput, cache);
                    totalLoss += loss * sourceIndices.Count;
                    validCount += sourceIndices.Count;
                }
                catch (Exception ex)
                {
                    ZeroAllGradients();
                    totalLoss = 0f;
                    validCount = 0;

                    if (_trainConfig.Verbose)
                        Console.WriteLine($"ERROR during continuous training: {ex.Message}");
                }
            }

            if (validCount == 0)
                return 0f;

            ScaleGradients(1.0f / validCount);

            float gradNorm = ComputeGradientNorm();
            if (!IsFinite(gradNorm))
            {
                ZeroAllGradients();
                return totalLoss / validCount;
            }

            if (_trainConfig.UseGradientClipping)
                ClipGradients(_trainConfig.GradientClipThreshold, gradNorm);

            UpdateParameters(learningRate);
            return totalLoss / validCount;
        }
        private float[,] ForwardWithCacheDiscrete(int[] tokenIds, ForwardCache cache)
        {
            return ForwardWithCacheDiscrete(tokenIds, 0, tokenIds.Length, cache);
        }

        private float[,] ForwardWithCacheDiscrete(int[] tokenIds, int tokenStart, int tokenCount, ForwardCache cache)
        {
            if (tokenIds == null)
                throw new ArgumentNullException(nameof(tokenIds));
            if (cache == null)
                throw new ArgumentNullException(nameof(cache));
            if (tokenStart < 0 || tokenCount < 0 || tokenStart + tokenCount > tokenIds.Length)
                throw new ArgumentOutOfRangeException(nameof(tokenStart));

            var embedded = _accel.EmbedTokenIds(tokenIds, tokenStart, tokenCount, _model.TokenEmbedding, _modelConfig.EmbeddingDim);

            cache.EmbeddedInput = embedded;
            cache.TokenIds = tokenIds;
            cache.TokenStart = tokenStart;
            cache.TokenCount = tokenCount;
            cache.ContinuousInput = null;
            cache.ContinuousInputRowStart = 0;
            cache.ContinuousInputRowCount = 0;

            return ForwardFromEmbeddingWithCache(embedded, tokenCount, cache);
        }

        private float[,] ForwardWithCacheContinuous(float[,] inputSequence, ForwardCache cache)
        {
            return ForwardWithCacheContinuous(inputSequence, 0, inputSequence.GetLength(0), cache);
        }

        private float[,] ForwardWithCacheContinuous(float[,] inputSequence, int rowStart, int rowCount, ForwardCache cache)
        {
            if (inputSequence == null)
                throw new ArgumentNullException(nameof(inputSequence));
            if (cache == null)
                throw new ArgumentNullException(nameof(cache));
            if (rowStart < 0 || rowCount < 0 || rowStart + rowCount > inputSequence.GetLength(0))
                throw new ArgumentOutOfRangeException(nameof(rowStart));

            var embedded = _accel.BatchDotProductAddBias(_model.InputProjection, inputSequence, rowStart, rowCount, _model.InputProjectionBias);

            cache.EmbeddedInput = embedded;
            cache.TokenIds = null;
            cache.TokenStart = 0;
            cache.TokenCount = 0;
            cache.ContinuousInput = inputSequence;
            cache.ContinuousInputRowStart = rowStart;
            cache.ContinuousInputRowCount = rowCount;

            return ForwardFromEmbeddingWithCache(embedded, rowCount, cache);
        }
        private float[,] ForwardFromEmbeddingWithCache(float[,] embedded, int seqLen, ForwardCache cache, int segmentLength = 0, int segmentCount = 1)
        {
            bool selfCausal = _modelConfig.UseDecoderOnly;

            // Important:
            // Do not allocate a causal bool[,] mask when the accelerator can handle
            // causal attention internally.
            bool[,] selfMask = null;

            var x = embedded;

            for (int layer = 0; layer < _modelConfig.NumLayers; layer++)
            {
                cache.LayerInputs.Add(x);

                var block = _model.Blocks[layer];

                var attnCache = cache.AttentionCaches[layer];
                attnCache.Input = x;

                var attnOutput = segmentCount > 1
                    ? SegmentedAttentionForwardWithCache(block.Attention, x, segmentLength, segmentCount, selfCausal, attnCache)
                    : AttentionForwardWithCache(block.Attention, x, selfMask, selfCausal, attnCache);

                var ln1Cache = cache.LN1Caches[layer];
                var (normed1, ln1Means, ln1Vars, ln1Normalized, attnResidual) = _accel.ResidualLayerNormForward(x, attnOutput, block.LN1Gamma, block.LN1Beta);

                ln1Cache.Input = attnResidual;
                ln1Cache.Mean = ln1Means;
                ln1Cache.Variance = ln1Vars;
                ln1Cache.Normalized = ln1Normalized;

                var ffOutput = _accel.FFNForwardBatch(normed1, seqLen, _modelConfig.EmbeddingDim, block.FeedForwardNetwork.ForwardPassOnly);

                cache.FFNInputs.Add(normed1);
                cache.FFNOutputs.Add(ffOutput);

                var ln2Cache = cache.LN2Caches[layer];
                var (normed2, ln2Means, ln2Vars, ln2Normalized, ffResidual) = _accel.ResidualLayerNormForward(normed1, ffOutput, block.LN2Gamma, block.LN2Beta);

                ln2Cache.Input = ffResidual;
                ln2Cache.Mean = ln2Means;
                ln2Cache.Variance = ln2Vars;
                ln2Cache.Normalized = ln2Normalized;

                x = normed2;
            }

            cache.FinalHiddenStates = x;
            return ProjectToOutput(x);
        }
        private float[,] ForwardWithCacheDiscreteBatch(IReadOnlyList<int[]> sequences, IReadOnlyList<int> offsets, int inputLen, ForwardCache cache)
        {
            if (sequences == null) throw new ArgumentNullException(nameof(sequences));
            if (offsets == null) throw new ArgumentNullException(nameof(offsets));
            if (cache == null) throw new ArgumentNullException(nameof(cache));
            if (inputLen <= 0) throw new ArgumentOutOfRangeException(nameof(inputLen));

            int batchSize = offsets.Count;
            int totalRows = batchSize * inputLen;
            var flatTokenIds = new int[totalRows];

            for (int b = 0; b < batchSize; b++)
            {
                var sequence = sequences[offsets[b]];
                for (int t = 0; t < inputLen; t++)
                {
                    flatTokenIds[(b * inputLen) + t] = sequence[t];
                }
            }

            var embedded = _accel.EmbedTokenIds(flatTokenIds, _model.TokenEmbedding, _modelConfig.EmbeddingDim);

            cache.EmbeddedInput = embedded;
            cache.TokenIds = flatTokenIds;
            cache.TokenStart = 0;
            cache.TokenCount = totalRows;
            cache.ContinuousInput = null;
            cache.ContinuousInputRowStart = 0;
            cache.ContinuousInputRowCount = 0;

            return ForwardFromEmbeddingWithCache(embedded, totalRows, cache, inputLen, batchSize);
        }

        private float[,] ForwardWithCacheContinuousBatch(IReadOnlyList<int> sourceIndices, int start, int count, float[][,] allInputs, int inputLen, ForwardCache cache)
        {
            if (sourceIndices == null) throw new ArgumentNullException(nameof(sourceIndices));
            if (allInputs == null) throw new ArgumentNullException(nameof(allInputs));
            if (cache == null) throw new ArgumentNullException(nameof(cache));
            if (inputLen <= 0) throw new ArgumentOutOfRangeException(nameof(inputLen));

            int totalRows = count * inputLen;
            var flatInput = new float[totalRows, _modelConfig.InputFeatureDim];

            for (int b = 0; b < count; b++)
            {
                var source = allInputs[sourceIndices[start + b]];
                int dstBase = b * inputLen;
                for (int t = 0; t < inputLen; t++)
                {
                    for (int f = 0; f < _modelConfig.InputFeatureDim; f++)
                    {
                        flatInput[dstBase + t, f] = source[t, f];
                    }
                }
            }

            var embedded = _accel.BatchDotProductAddBias(_model.InputProjection, flatInput, _model.InputProjectionBias);

            cache.EmbeddedInput = embedded;
            cache.TokenIds = null;
            cache.TokenStart = 0;
            cache.TokenCount = 0;
            cache.ContinuousInput = flatInput;
            cache.ContinuousInputRowStart = 0;
            cache.ContinuousInputRowCount = totalRows;

            return ForwardFromEmbeddingWithCache(embedded, totalRows, cache, inputLen, count);
        }

        private int[] BuildDiscreteFlatTargets(IReadOnlyList<int[]> sequences, IReadOnlyList<int> offsets, int inputLen)
        {
            int batchSize = offsets.Count;
            var targets = new int[batchSize * inputLen];
            for (int b = 0; b < batchSize; b++)
            {
                var sequence = sequences[offsets[b]];
                int dstBase = b * inputLen;
                for (int t = 0; t < inputLen; t++)
                {
                    targets[dstBase + t] = sequence[t + 1];
                }
            }
            return targets;
        }

        private int[] BuildContinuousClassFlatTargets(IReadOnlyList<int> sourceIndices, int start, int count, int[][] allClassTargets, int inputLen)
        {
            var targets = new int[count * inputLen];
            for (int b = 0; b < count; b++)
            {
                var source = allClassTargets[sourceIndices[start + b]];
                int dstBase = b * inputLen;
                for (int t = 0; t < inputLen; t++)
                {
                    targets[dstBase + t] = source[t + 1];
                }
            }
            return targets;
        }

        private float[,] BuildContinuousRegressionFlatTargets(IReadOnlyList<int> sourceIndices, int start, int count, float[][,] allRegTargets, int inputLen)
        {
            int outputDim = _modelConfig.EffectiveOutputDim;
            var targets = new float[count * inputLen, outputDim];
            for (int b = 0; b < count; b++)
            {
                var source = allRegTargets[sourceIndices[start + b]];
                int dstBase = b * inputLen;
                for (int t = 0; t < inputLen; t++)
                {
                    for (int o = 0; o < outputDim; o++)
                    {
                        targets[dstBase + t, o] = source[t + 1, o];
                    }
                }
            }
            return targets;
        }

        private void ScaleMatrixLocal(float[,] matrix, float scale)
        {
            int rows = matrix.GetLength(0);
            int cols = matrix.GetLength(1);
            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < cols; j++)
                {
                    matrix[i, j] *= scale;
                }
            }
        }

        private float[,] SegmentedAttentionForwardWithCache(MultiHeadAttention attention, float[,] input, int segmentLength, int segmentCount, bool causal, AttentionCache cache)
        {
            int totalRows = input.GetLength(0);
            int embeddingDim = _modelConfig.EmbeddingDim;
            int numHeads = _modelConfig.NumHeads;
            int headDim = embeddingDim / numHeads;
            if (segmentLength <= 0 || segmentCount <= 0 || totalRows != segmentLength * segmentCount)
            {
                throw new ArgumentException("Invalid segmented attention shape.");
            }

            var (Q, K, V) = _accel.ProjectQKV(input, attention.WQ, attention.BiasQ, attention.WK, attention.BiasK, attention.WV, attention.BiasV);
            ApplySegmentedRotaryInPlace(Q, K, numHeads, segmentLength, inverse: false);

            var concatenated = new float[totalRows, embeddingDim];
            float scale = 1.0f / MathF.Sqrt(headDim);

            Parallel.For(0, segmentCount * numHeads, work =>
            {
                int segment = work / numHeads;
                int head = work - segment * numHeads;
                int rowBase = segment * segmentLength;
                int offset = head * headDim;
                var scores = new float[segmentLength];

                for (int qi = 0; qi < segmentLength; qi++)
                {
                    int qRow = rowBase + qi;
                    int visibleKeys = causal ? qi + 1 : segmentLength;
                    float maxScore = float.NegativeInfinity;

                    for (int kj = 0; kj < visibleKeys; kj++)
                    {
                        int kRow = rowBase + kj;
                        float dot = 0f;
                        for (int d = 0; d < headDim; d++)
                        {
                            int col = offset + d;
                            dot += Q[qRow, col] * K[kRow, col];
                        }

                        float score = dot * scale;
                        scores[kj] = score;
                        if (score > maxScore)
                        {
                            maxScore = score;
                        }
                    }

                    float sumExp = 0f;
                    for (int kj = 0; kj < visibleKeys; kj++)
                    {
                        float exp = MathF.Exp(scores[kj] - maxScore);
                        scores[kj] = exp;
                        sumExp += exp;
                    }

                    float invSum = sumExp > 0f ? 1.0f / sumExp : 0f;
                    for (int kj = 0; kj < visibleKeys; kj++)
                    {
                        float weight = scores[kj] * invSum;
                        if (weight == 0f)
                        {
                            continue;
                        }

                        int kRow = rowBase + kj;
                        for (int d = 0; d < headDim; d++)
                        {
                            int col = offset + d;
                            concatenated[qRow, col] += weight * V[kRow, col];
                        }
                    }
                }
            });

            cache.Q = Q;
            cache.K = K;
            cache.V = V;
            cache.AttentionOutput = concatenated;
            cache.IsSegmented = true;
            cache.SegmentLength = segmentLength;
            cache.SegmentCount = segmentCount;
            cache.Causal = causal;

            return _accel.ProjectOutputBatch(concatenated, attention.WO, attention.BiasO, totalRows, embeddingDim);
        }

        private static void ApplySegmentedRotaryInPlace(float[,] q, float[,] k, int numHeads, int segmentLength, bool inverse)
        {
            int rows = q.GetLength(0);
            int embeddingDim = q.GetLength(1);
            int headDim = embeddingDim / numHeads;
            const float baseTheta = 10000f;

            for (int row = 0; row < rows; row++)
            {
                int pos = row % segmentLength;
                for (int head = 0; head < numHeads; head++)
                {
                    int offset = head * headDim;
                    for (int pair = 0; pair < headDim / 2; pair++)
                    {
                        int even = offset + pair * 2;
                        int odd = even + 1;
                        float theta = pos / MathF.Pow(baseTheta, (2.0f * pair) / headDim);
                        float cos = MathF.Cos(theta);
                        float sin = MathF.Sin(theta);
                        if (inverse)
                        {
                            sin = -sin;
                        }

                        float q0 = q[row, even];
                        float q1 = q[row, odd];
                        q[row, even] = (q0 * cos) - (q1 * sin);
                        q[row, odd] = (q0 * sin) + (q1 * cos);

                        float k0 = k[row, even];
                        float k1 = k[row, odd];
                        k[row, even] = (k0 * cos) - (k1 * sin);
                        k[row, odd] = (k0 * sin) + (k1 * cos);
                    }
                }
            }
        }

        /*
        private float BackwardPassCrossEntropy(float[,] logits, int[] targets, ForwardCache cache)
        {
            int effectiveLen = Math.Min(logits.GetLength(0), targets.Length);
            var (loss, dLogits) = _accel.CrossEntropyLossAndGradient(logits, targets, effectiveLen);

            if (!IsFinite(loss) || !MatrixAllFinite(dLogits))
                throw new InvalidOperationException("Cross-entropy produced a non-finite loss or gradient.");

            BackpropFromOutput(dLogits, cache);
            return loss;
        }

        private float BackwardPassMSE(float[,] predictions, float[,] targets, ForwardCache cache)
        {
            int effectiveLen = Math.Min(predictions.GetLength(0), targets.GetLength(0));
            var (loss, dOutput) = _accel.MSELossAndGradient(predictions, targets, effectiveLen);

            if (!IsFinite(loss) || !MatrixAllFinite(dOutput))
                throw new InvalidOperationException("MSE produced a non-finite loss or gradient.");

            BackpropFromOutput(dOutput, cache);
            return loss;
        }*/
        private void BackpropFromOutput(float[,] dOutput, ForwardCache cache)
        {
            var dX = _accel.BackpropOutputProjection(
                dOutput, cache.FinalHiddenStates, _model.OutputProjection,
                _gradients.OutputProjectionGrad, _gradients.OutputBiasGrad,
                dOutput.GetLength(0), _modelConfig.EffectiveOutputDim, _modelConfig.EmbeddingDim);

            for (int layer = _modelConfig.NumLayers - 1; layer >= 0; layer--)
                dX = BackpropBlock(layer, dX, cache);

            BackpropInput(dX, cache);
        }

        private void BackpropInput(float[,] dX, ForwardCache cache)
        {
            if (_modelConfig.Data.UsesDiscreteTokens)
            {
                _gradients.MarkTokenRows(cache.TokenIds, cache.TokenStart, dX.GetLength(0));
                _accel.AccumulateTokenEmbeddingGrad(_gradients.TokenEmbeddingGrad, dX, cache.TokenIds, cache.TokenStart, dX.GetLength(0), dX.GetLength(1));
            }
            else
                _accel.BackpropInputProjection(dX, cache.ContinuousInput, cache.ContinuousInputRowStart, _gradients.InputProjectionGrad, _gradients.InputProjectionBiasGrad, dX.GetLength(0), _modelConfig.EmbeddingDim, _modelConfig.InputFeatureDim);
        }

        private float[,] BackpropBlock(int layerIdx, float[,] dOut, ForwardCache cache)
        {
            var block = _model.Blocks[layerIdx];

            var ln2Cache = cache.LN2Caches[layerIdx];
            var (dFFResidual, dLN2Gamma, dLN2Beta) = _accel.LayerNormBackward(dOut, ln2Cache.Normalized, block.LN2Gamma, ln2Cache.Input, ln2Cache.Mean, ln2Cache.Variance);
            var ln2Grads = _gradients.LN2Grads[layerIdx];
            _accel.VectorAccumulate(ln2Grads.GammaGrad, dLN2Gamma);
            _accel.VectorAccumulate(ln2Grads.BetaGrad, dLN2Beta);

            var dNormed1_from_ffn = BackpropFFN(layerIdx, dFFResidual, cache);
            _accel.MatrixAddInPlace(dNormed1_from_ffn, dFFResidual);
            var dNormed1 = dNormed1_from_ffn;

            var ln1Cache = cache.LN1Caches[layerIdx];
            var (dAttnResidual, dLN1Gamma, dLN1Beta) = _accel.LayerNormBackward(dNormed1, ln1Cache.Normalized, block.LN1Gamma, ln1Cache.Input, ln1Cache.Mean, ln1Cache.Variance);
            var ln1Grads = _gradients.LN1Grads[layerIdx];
            _accel.VectorAccumulate(ln1Grads.GammaGrad, dLN1Gamma);
            _accel.VectorAccumulate(ln1Grads.BetaGrad, dLN1Beta);

            var dX_from_attn = BackpropAttention(layerIdx, dAttnResidual, cache.AttentionCaches[layerIdx]);
            _accel.MatrixAddInPlace(dX_from_attn, dAttnResidual);
            return dX_from_attn;
        }

        private float[,] BackpropFFN(int layerIdx, float[,] dFFOutput, ForwardCache cache)
        {
            var block = _model.Blocks[layerIdx];
            return block.FeedForwardNetwork.ComputeInputGradientBatch(
                cache.FFNInputs[layerIdx],
                dFFOutput,
                _ffnWeightGrads[layerIdx],
                _ffnBiasGrads[layerIdx]);
        }

        private float[,] BackpropAttention(int layerIdx, float[,] dOut, AttentionCache cache)
        {
            var attention = _model.Blocks[layerIdx].Attention;
            var grads = _gradients.AttentionGrads[layerIdx];

            int seqLen = dOut.GetLength(0);
            int embeddingDim = _modelConfig.EmbeddingDim;
            int numHeads = _modelConfig.NumHeads;
            int headDim = embeddingDim / numHeads;

            var dConcatenated = new float[seqLen, embeddingDim];

            _accel.BackpropLinearProjection(cache.AttentionOutput, dOut, attention.WO, grads.WO_Grad, grads.BiasO_Grad, dConcatenated);

            float scale = 1.0f / MathF.Sqrt(headDim);

            (float[,] dQFull, float[,] dKFull, float[,] dVFull) attentionBack;

            if (cache.IsSegmented)
            {
                attentionBack = SegmentedMultiHeadAttentionBackward(
                    cache.Q,
                    cache.K,
                    cache.V,
                    dConcatenated,
                    numHeads,
                    scale,
                    cache.SegmentLength,
                    cache.SegmentCount,
                    cache.Causal);
            }
            else if (_modelConfig.UseDecoderOnly)
            {
                // Avoid allocating CreateCausalMask(seqLen) during every backward pass.
                attentionBack = _accel.MultiHeadAttentionBackward(cache.Q, cache.K, cache.V, dConcatenated, numHeads, scale, useDecoderMask: true);
            }
            else
            {
                attentionBack = _accel.MultiHeadAttentionBackward(cache.Q, cache.K, cache.V, dConcatenated, numHeads, scale, (bool[,])null);
            }

            var dQFull = attentionBack.dQFull;
            var dKFull = attentionBack.dKFull;
            var dVFull = attentionBack.dVFull;

            if (cache.IsSegmented)
            {
                ApplySegmentedRotaryInPlace(dQFull, dKFull, numHeads, cache.SegmentLength, inverse: true);
            }
            else
            {
                _rotaryPositionEmbedding.ApplyBackwardInPlace(dQFull, dKFull, numHeads);
            }

            return _accel.BackpropQKV(cache.Input, dQFull, dKFull, dVFull,  attention.WQ, attention.WK,  attention.WV, grads.WQ_Grad, grads.BiasQ_Grad, grads.WK_Grad, grads.BiasK_Grad, grads.WV_Grad, grads.BiasV_Grad);
        }
        private (float[,] dQFull, float[,] dKFull, float[,] dVFull) SegmentedMultiHeadAttentionBackward(
            float[,] Q,
            float[,] K,
            float[,] V,
            float[,] dConcatenated,
            int numHeads,
            float scale,
            int segmentLength,
            int segmentCount,
            bool causal)
        {
            int totalRows = Q.GetLength(0);
            int embeddingDim = Q.GetLength(1);
            int headDim = embeddingDim / numHeads;
            if (segmentLength <= 0 || segmentCount <= 0 || totalRows != segmentLength * segmentCount)
            {
                throw new ArgumentException("Invalid segmented attention backward shape.");
            }

            var dQFull = new float[totalRows, embeddingDim];
            var dKFull = new float[totalRows, embeddingDim];
            var dVFull = new float[totalRows, embeddingDim];

            Parallel.For(0, segmentCount * numHeads, work =>
            {
                int segment = work / numHeads;
                int head = work - segment * numHeads;
                int rowBase = segment * segmentLength;
                int offset = head * headDim;
                var weights = new float[segmentLength];
                var dAttn = new float[segmentLength];
                var qRow = new float[headDim];
                var doutRow = new float[headDim];
                var dqRow = new float[headDim];

                for (int qi = 0; qi < segmentLength; qi++)
                {
                    Array.Clear(weights, 0, weights.Length);
                    Array.Clear(dAttn, 0, dAttn.Length);
                    Array.Clear(dqRow, 0, dqRow.Length);

                    int qIndex = rowBase + qi;
                    int visibleKeys = causal ? qi + 1 : segmentLength;

                    for (int d = 0; d < headDim; d++)
                    {
                        int col = offset + d;
                        qRow[d] = Q[qIndex, col];
                        doutRow[d] = dConcatenated[qIndex, col];
                    }

                    float maxScore = float.NegativeInfinity;
                    for (int kj = 0; kj < visibleKeys; kj++)
                    {
                        int keyIndex = rowBase + kj;
                        float dot = 0f;
                        for (int d = 0; d < headDim; d++)
                        {
                            dot += qRow[d] * K[keyIndex, offset + d];
                        }

                        float score = dot * scale;
                        weights[kj] = score;
                        if (score > maxScore)
                        {
                            maxScore = score;
                        }
                    }

                    float sumExp = 0f;
                    for (int kj = 0; kj < visibleKeys; kj++)
                    {
                        float w = MathF.Exp(weights[kj] - maxScore);
                        weights[kj] = w;
                        sumExp += w;
                    }

                    if (sumExp <= 0f)
                    {
                        continue;
                    }

                    float invSum = 1.0f / sumExp;
                    for (int kj = 0; kj < visibleKeys; kj++)
                    {
                        weights[kj] *= invSum;
                    }

                    float rowDot = 0f;
                    for (int kj = 0; kj < visibleKeys; kj++)
                    {
                        int keyIndex = rowBase + kj;
                        float w = weights[kj];
                        float dAttnJ = 0f;

                        for (int d = 0; d < headDim; d++)
                        {
                            int col = offset + d;
                            float dout = doutRow[d];
                            dVFull[keyIndex, col] += w * dout;
                            dAttnJ += dout * V[keyIndex, col];
                        }

                        dAttn[kj] = dAttnJ;
                        rowDot += w * dAttnJ;
                    }

                    for (int kj = 0; kj < visibleKeys; kj++)
                    {
                        int keyIndex = rowBase + kj;
                        float dDot = weights[kj] * (dAttn[kj] - rowDot) * scale;
                        if (dDot == 0f)
                        {
                            continue;
                        }

                        for (int d = 0; d < headDim; d++)
                        {
                            int col = offset + d;
                            dqRow[d] += dDot * K[keyIndex, col];
                            dKFull[keyIndex, col] += dDot * qRow[d];
                        }
                    }

                    for (int d = 0; d < headDim; d++)
                    {
                        dQFull[qIndex, offset + d] += dqRow[d];
                    }
                }
            });

            return (dQFull, dKFull, dVFull);
        }

        private float[,] AttentionForwardWithCache(MultiHeadAttention attention, float[,] input, bool[,] mask, bool causal, AttentionCache cache)
        {
            int seqLen = input.GetLength(0);
            int embeddingDim = _modelConfig.EmbeddingDim;
            int numHeads = _modelConfig.NumHeads;
            var (Q, K, V) = _accel.ProjectQKV(input, attention.WQ, attention.BiasQ, attention.WK, attention.BiasK, attention.WV, attention.BiasV);

            _rotaryPositionEmbedding.ApplyInPlace(Q, K, numHeads);

            cache.Q = Q;
            cache.K = K;
            cache.V = V;

            var concatenated = _accel.ScaledDotProductAttention(Q, K, V, numHeads, mask, causal: causal && mask == null);

            cache.AttentionOutput = concatenated;

            return _accel.ProjectOutputBatch( concatenated, attention.WO, attention.BiasO, seqLen, embeddingDim);
        }
        public float Validate(int[][] validationSequences)
        {
            if (validationSequences == null)
            {
                throw new ArgumentNullException(nameof(validationSequences));
            }

            float totalLoss = 0.0f;
            int totalTokens = 0;

            foreach (var sequence in validationSequences)
            {
                if (sequence == null || sequence.Length < 2)
                {
                    continue;
                }

                if (!TokensAreValid(sequence))
                {
                    continue;
                }

                int inputLen = sequence.Length - 1;
                var logits = _model.ForwardTokenSlice(sequence, 0, inputLen);
                int effectiveLen = Math.Min(logits.GetLength(0), inputLen);

                if (effectiveLen <= 0)
                {
                    continue;
                }

                var result = _accel.CrossEntropyLossAndGradient(logits, sequence, 1, effectiveLen);

                if (!IsFinite(result.loss))
                {
                    continue;
                }

                totalLoss += result.loss * effectiveLen;
                totalTokens += effectiveLen;
            }

            return totalTokens > 0 ? totalLoss / totalTokens : 0.0f;
        }
        public float ValidateContinuous(float[][,] inputs, float[][,] regressionTargets = null, int[][] classTargets = null)
        {
            if (inputs == null)
            {
                throw new ArgumentNullException(nameof(inputs));
            }

            float totalLoss = 0.0f;
            int totalItems = 0;

            for (int idx = 0; idx < inputs.Length; idx++)
            {
                var inputSeq = inputs[idx];

                if (inputSeq == null)
                {
                    continue;
                }

                int seqLen = inputSeq.GetLength(0);

                if (seqLen < 2)
                {
                    continue;
                }

                int inputLen = seqLen - 1;
                var output = _model.ForwardContinuousSlice(inputSeq, 0, inputLen);
                int effectiveLen = output.GetLength(0);

                if (_modelConfig.Data.DataType == TransformerDataType.TimeSeriesRegression)
                {
                    if (regressionTargets == null || regressionTargets[idx] == null)
                    {
                        continue;
                    }

                    effectiveLen = Math.Min(effectiveLen, inputLen);

                    if (effectiveLen <= 0)
                    {
                        continue;
                    }

                    var result = _accel.MSELossAndGradient(output, regressionTargets[idx], 1, effectiveLen);

                    if (!IsFinite(result.loss))
                    {
                        continue;
                    }

                    // MSELossAndGradient returns average over effectiveLen * outputDim.
                    totalLoss += result.loss * effectiveLen;
                    totalItems += effectiveLen;
                }
                else
                {
                    if (classTargets == null || classTargets[idx] == null)
                    {
                        continue;
                    }

                    effectiveLen = Math.Min(effectiveLen, inputLen);

                    if (effectiveLen <= 0)
                    {
                        continue;
                    }

                    var result = _accel.CrossEntropyLossAndGradient(output, classTargets[idx], 1, effectiveLen);

                    if (!IsFinite(result.loss))
                    {
                        continue;
                    }

                    totalLoss += result.loss * effectiveLen;
                    totalItems += effectiveLen;
                }
            }

            return totalItems > 0 ? totalLoss / totalItems : 0.0f;
        }
        private void UpdateParameters(float learningRate)
        {
            if (_modelConfig.Data.UsesDiscreteTokens)
                _gradients.UpdateTokenEmbedding(_model.TokenEmbedding, learningRate);
            else
            {
                _accel.MatrixUpdate(_model.InputProjection, _gradients.InputProjectionGrad, learningRate);
                _accel.VectorUpdate(_model.InputProjectionBias, _gradients.InputProjectionBiasGrad, learningRate);
            }

            for (int i = 0; i < _modelConfig.NumLayers; i++)
            {
                var block = _model.Blocks[i];
                var attnGrad = _gradients.AttentionGrads[i];
                var ln1Grad = _gradients.LN1Grads[i];
                var ln2Grad = _gradients.LN2Grads[i];

                _accel.MatrixUpdate(block.Attention.WQ, attnGrad.WQ_Grad, learningRate);
                _accel.MatrixUpdate(block.Attention.WK, attnGrad.WK_Grad, learningRate);
                _accel.MatrixUpdate(block.Attention.WV, attnGrad.WV_Grad, learningRate);
                _accel.MatrixUpdate(block.Attention.WO, attnGrad.WO_Grad, learningRate);
                _accel.VectorUpdate(block.Attention.BiasQ, attnGrad.BiasQ_Grad, learningRate);
                _accel.VectorUpdate(block.Attention.BiasK, attnGrad.BiasK_Grad, learningRate);
                _accel.VectorUpdate(block.Attention.BiasV, attnGrad.BiasV_Grad, learningRate);
                _accel.VectorUpdate(block.Attention.BiasO, attnGrad.BiasO_Grad, learningRate);

                _accel.VectorUpdate(block.LN1Gamma, ln1Grad.GammaGrad, learningRate);
                _accel.VectorUpdate(block.LN1Beta, ln1Grad.BetaGrad, learningRate);
                _accel.VectorUpdate(block.LN2Gamma, ln2Grad.GammaGrad, learningRate);
                _accel.VectorUpdate(block.LN2Beta, ln2Grad.BetaGrad, learningRate);

                block.FeedForwardNetwork.ApplyExternalGradients(_ffnWeightGrads[i], _ffnBiasGrads[i], learningRate);
            }

            _accel.MatrixUpdate(_model.OutputProjection, _gradients.OutputProjectionGrad, learningRate);
            _accel.VectorUpdate(_model.OutputBias, _gradients.OutputBiasGrad, learningRate);
        }
        private void ClipGradients(float threshold)
        {
            ClipGradients(threshold, ComputeGradientNorm());
        }

        private void ClipGradients(float threshold, float totalNorm)
        {
            if (threshold <= 0f || float.IsNaN(threshold) || float.IsInfinity(threshold))
                throw new ArgumentOutOfRangeException(nameof(threshold), "Gradient clip threshold must be finite and positive.");

            if (IsFinite(totalNorm) && totalNorm > threshold)
                ScaleGradients(threshold / totalNorm);
        }
        private float ComputeGradientNorm()
        {
            float sum = 0;

            if (_modelConfig.Data.UsesDiscreteTokens)
                sum += _gradients.TokenEmbeddingSquaredNorm();
            else
            {
                sum += _accel.MatrixSquaredNorm(_gradients.InputProjectionGrad);
                sum += _accel.VectorSquaredNorm(_gradients.InputProjectionBiasGrad);
            }

            sum += _accel.MatrixSquaredNorm(_gradients.OutputProjectionGrad);
            sum += _accel.VectorSquaredNorm(_gradients.OutputBiasGrad);

            foreach (var g in _gradients.AttentionGrads)
            {
                sum += _accel.MatrixSquaredNorm(g.WQ_Grad) + _accel.MatrixSquaredNorm(g.WK_Grad)
                     + _accel.MatrixSquaredNorm(g.WV_Grad) + _accel.MatrixSquaredNorm(g.WO_Grad);
                sum += _accel.VectorSquaredNorm(g.BiasQ_Grad) + _accel.VectorSquaredNorm(g.BiasK_Grad)
                     + _accel.VectorSquaredNorm(g.BiasV_Grad) + _accel.VectorSquaredNorm(g.BiasO_Grad);
            }

            foreach (var g in _gradients.LN1Grads) { sum += _accel.VectorSquaredNorm(g.GammaGrad) + _accel.VectorSquaredNorm(g.BetaGrad); }
            foreach (var g in _gradients.LN2Grads) { sum += _accel.VectorSquaredNorm(g.GammaGrad) + _accel.VectorSquaredNorm(g.BetaGrad); }

            for (int layer = 0; layer < _modelConfig.NumLayers; layer++)
            {
                foreach (var wg in _ffnWeightGrads[layer]) sum += _accel.MatrixSquaredNorm(wg);
                foreach (var bg in _ffnBiasGrads[layer]) sum += _accel.VectorSquaredNorm(bg);
            }

            return MathF.Sqrt(sum);
        }

        private void ScaleGradients(float scale)
        {
            if (_modelConfig.Data.UsesDiscreteTokens)
                _gradients.ScaleTokenEmbeddingGrad(scale);
            else
            {
                _accel.MatrixScaleInPlace(_gradients.InputProjectionGrad, scale);
                _accel.VectorScaleInPlace(_gradients.InputProjectionBiasGrad, scale);
            }

            _accel.MatrixScaleInPlace(_gradients.OutputProjectionGrad, scale);
            _accel.VectorScaleInPlace(_gradients.OutputBiasGrad, scale);

            foreach (var g in _gradients.AttentionGrads)
            {
                _accel.MatrixScaleInPlace(g.WQ_Grad, scale); _accel.MatrixScaleInPlace(g.WK_Grad, scale);
                _accel.MatrixScaleInPlace(g.WV_Grad, scale); _accel.MatrixScaleInPlace(g.WO_Grad, scale);
                _accel.VectorScaleInPlace(g.BiasQ_Grad, scale); _accel.VectorScaleInPlace(g.BiasK_Grad, scale);
                _accel.VectorScaleInPlace(g.BiasV_Grad, scale); _accel.VectorScaleInPlace(g.BiasO_Grad, scale);
            }

            foreach (var g in _gradients.LN1Grads) { _accel.VectorScaleInPlace(g.GammaGrad, scale); _accel.VectorScaleInPlace(g.BetaGrad, scale); }
            foreach (var g in _gradients.LN2Grads) { _accel.VectorScaleInPlace(g.GammaGrad, scale); _accel.VectorScaleInPlace(g.BetaGrad, scale); }

            for (int layer = 0; layer < _modelConfig.NumLayers; layer++)
            {
                foreach (var wg in _ffnWeightGrads[layer]) _accel.MatrixScaleInPlace(wg, scale);
                foreach (var bg in _ffnBiasGrads[layer]) _accel.VectorScaleInPlace(bg, scale);
            }
        }

        private void ZeroAllGradients()
        {
            _gradients.Zero();

            for (int layer = 0; layer < _modelConfig.NumLayers; layer++)
            {
                foreach (var wg in _ffnWeightGrads[layer])
                {
                    _accel.ZeroMatrix(wg);
                }

                foreach (var bg in _ffnBiasGrads[layer])
                {
                    _accel.ZeroVector(bg);
                }
            }
        }

        private float[,] ProjectToOutput(float[,] hidden)
        {
            return _accel.ProjectOutputBatch(hidden, _model.OutputProjection, _model.OutputBias, hidden.GetLength(0), _modelConfig.EffectiveOutputDim);
        }
        private T[] ShuffleArray<T>(T[] data)
        {
            if (data == null)
            {
                throw new ArgumentNullException(nameof(data));
            }

            var result = new T[data.Length];
            Array.Copy(data, result, data.Length);
            ShuffleInPlace(result);
            return result;
        }

        private void ShuffleInPlace<T>(T[] data)
        {
            if (data == null)
            {
                throw new ArgumentNullException(nameof(data));
            }

            for (int i = data.Length - 1; i > 0; i--)
            {
                int j = _random.Next(i + 1);
                (data[i], data[j]) = (data[j], data[i]);
            }
        }

        private static T[] CopyBatch<T>(T[] source, int start, int count)
        {
            if (source == null)
            {
                throw new ArgumentNullException(nameof(source));
            }
            if (start < 0 || count < 0 || start + count > source.Length)
            {
                throw new ArgumentOutOfRangeException(nameof(start));
            }

            var result = new T[count];
            Array.Copy(source, start, result, 0, count);
            return result;
        }

        private static T[] SliceArray<T>(T[] source, int start, int count)
        {
            if (source == null)
            {
                throw new ArgumentNullException(nameof(source));
            }
            if (start < 0 || count < 0 || start + count > source.Length)
            {
                throw new ArgumentOutOfRangeException(nameof(start));
            }

            var result = new T[count];
            Array.Copy(source, start, result, 0, count);
            return result;
        }


        private bool TokensAreValid(int[] tokens)
        {
            if (tokens == null || tokens.Length == 0)
                return false;

            int vocabSize = _modelConfig.VocabSize;
            for (int i = 0; i < tokens.Length; i++)
                if ((uint)tokens[i] >= (uint)vocabSize)
                    return false;

            return true;
        }

        private bool RegressionTargetIsValid(float[][,] targets, int idx, int seqLen)
        {
            return targets != null
                && idx >= 0
                && idx < targets.Length
                && targets[idx] != null
                && targets[idx].GetLength(0) >= seqLen
                && targets[idx].GetLength(1) == _modelConfig.OutputDim
                && MatrixAllFinite(targets[idx]);
        }

        private bool ClassTargetIsValid(int[][] targets, int idx, int seqLen)
        {
            if (targets == null || idx < 0 || idx >= targets.Length || targets[idx] == null || targets[idx].Length < seqLen)
                return false;

            int classCount = _modelConfig.OutputDim;
            for (int i = 1; i < seqLen; i++)
                if ((uint)targets[idx][i] >= (uint)classCount)
                    return false;

            return true;
        }

        private static bool IsFinite(float value)
        {
            return !float.IsNaN(value) && !float.IsInfinity(value);
        }


        private static bool MatrixAllFinite(float[,] matrix)
        {
            if (matrix == null)
                return false;

            int rows = matrix.GetLength(0);
            int cols = matrix.GetLength(1);

            for (int i = 0; i < rows; i++)
                for (int j = 0; j < cols; j++)
                    if (!IsFinite(matrix[i, j]))
                        return false;

            return true;
        }
    }
}