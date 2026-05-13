using CallaghanDev.ML.AccelerationManagers;
using CallaghanDev.ML.Transformers.Cache;
using CallaghanDev.ML.Transformers.Configuration;
using System;
using System.Collections.Generic;
using System.Linq;

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
            _rotaryPositionEmbedding = new RotaryPositionEmbedding(model.Config.Runtime);

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
                    var batch = shuffled.Skip(i).Take(batchSize).ToArray();
                    if (batch.Length == 0) continue;

                    float batchLoss = TrainBatchDiscrete(batch, currentLR);
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

                var shuffled = validIndices.OrderBy(_ => _random.Next()).ToArray();
                float epochLoss = 0f;
                int numBatches = 0;

                for (int i = 0; i < shuffled.Length; i += _trainConfig.BatchSize)
                {
                    int batchSize = Math.Min(_trainConfig.BatchSize, shuffled.Length - i);
                    var batchIndices = shuffled.Skip(i).Take(batchSize).ToArray();
                    if (batchIndices.Length == 0) continue;

                    float batchLoss = TrainBatchContinuous(batchIndices, inputs, regressionTargets, classTargets, currentLR);
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
            ZeroAllGradients();
            float totalLoss = 0f;
            int validCount = 0;

            foreach (var sequence in batch)
            {
                if (sequence == null || sequence.Length < 2)
                {
                    continue;
                }

                var input = sequence.Take(sequence.Length - 1).ToArray();
                var target = sequence.Skip(1).ToArray();

                if (!TokensAreValid(input) || !TokensAreValid(target))
                {
                    continue;
                }

                try
                {
                    var cache = new ForwardCache(_modelConfig.NumLayers);
                    var logits = ForwardWithCacheDiscrete(input, cache);
                    int effectiveLen = Math.Min(logits.GetLength(0), target.Length);
                    if (effectiveLen <= 0) continue;

                    var (loss, dLogits) = _accel.CrossEntropyLossAndGradient(logits, target, effectiveLen);
                    if (!IsFinite(loss) || !MatrixAllFinite(dLogits))
                        continue;

                    BackpropFromOutput(dLogits, cache);
                    totalLoss += loss;
                    validCount++;
                }
                catch (Exception ex)
                {
                    // Avoid applying a partially accumulated batch if a sample failed mid-backprop.
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
                ClipGradients(_trainConfig.GradientClipThreshold);

            UpdateParameters(learningRate);
            return totalLoss / validCount;
        }

        private float TrainBatchContinuous(int[] batchIndices, float[][,] allInputs, float[][,] allRegTargets, int[][] allClassTargets, float learningRate)
        {
            ZeroAllGradients();
            float totalLoss = 0f;
            int validCount = 0;

            foreach (int idx in batchIndices)
            {
                var inputSeq = allInputs[idx];
                if (inputSeq == null) continue;

                int seqLen = inputSeq.GetLength(0);
                if (seqLen < 2 || inputSeq.GetLength(1) != _modelConfig.InputFeatureDim)
                    continue;

                if (!MatrixAllFinite(inputSeq))
                    continue;

                try
                {
                    var inputSlice = _accel.SliceRows(inputSeq, 0, seqLen - 1);
                    var cache = new ForwardCache(_modelConfig.NumLayers);
                    var output = ForwardWithCacheContinuous(inputSlice, cache);
                    float loss;
                    float[,] dOutput;

                    if (_modelConfig.Data.DataType == TransformerDataType.TimeSeriesRegression)
                    {
                        if (!RegressionTargetIsValid(allRegTargets, idx, seqLen))
                            continue;

                        var targetSlice = _accel.SliceRows(allRegTargets[idx], 1, seqLen);
                        int effectiveLen = Math.Min(output.GetLength(0), targetSlice.GetLength(0));
                        var result = _accel.MSELossAndGradient(output, targetSlice, effectiveLen);
                        loss = result.loss;
                        dOutput = result.dOutput;
                    }
                    else
                    {
                        if (!ClassTargetIsValid(allClassTargets, idx, seqLen))
                            continue;

                        var targetSlice = allClassTargets[idx].Skip(1).Take(seqLen - 1).ToArray();
                        int effectiveLen = Math.Min(output.GetLength(0), targetSlice.Length);
                        var result = _accel.CrossEntropyLossAndGradient(output, targetSlice, effectiveLen);
                        loss = result.loss;
                        dOutput = result.dLogits;
                    }

                    if (!IsFinite(loss) || !MatrixAllFinite(dOutput))
                        continue;

                    BackpropFromOutput(dOutput, cache);
                    totalLoss += loss;
                    validCount++;
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
                ClipGradients(_trainConfig.GradientClipThreshold);

            UpdateParameters(learningRate);
            return totalLoss / validCount;
        }
        private float[,] ForwardWithCacheDiscrete(int[] tokenIds, ForwardCache cache)
        {
            int seqLen = tokenIds.Length;

            var embedded = _accel.EmbedTokenIds(tokenIds, _model.TokenEmbedding, _modelConfig.EmbeddingDim);

            cache.EmbeddedInput = embedded;
            cache.TokenIds = tokenIds;
            cache.ContinuousInput = null;

            return ForwardFromEmbeddingWithCache(embedded, seqLen, cache);
        }

        private float[,] ForwardWithCacheContinuous(float[,] inputSequence, ForwardCache cache)
        {
            int seqLen = inputSequence.GetLength(0);

            var embedded = _accel.ProjectOutputBatch(inputSequence, _model.InputProjection, _model.InputProjectionBias, seqLen, _modelConfig.EmbeddingDim);

            cache.EmbeddedInput = embedded;
            cache.TokenIds = null;
            cache.ContinuousInput = inputSequence;

            return ForwardFromEmbeddingWithCache(embedded, seqLen, cache);
        }
        private float[,] ForwardFromEmbeddingWithCache(float[,] embedded, int seqLen, ForwardCache cache)
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

                var attnOutput = AttentionForwardWithCache(block.Attention, x, selfMask, selfCausal, attnCache);

                var attnResidual = _accel.MatrixAdd(x, attnOutput);

                var ln1Cache = cache.LN1Caches[layer];
                var (normed1, ln1Means, ln1Vars, ln1Normalized) = _accel.LayerNormForward(attnResidual, block.LN1Gamma, block.LN1Beta);

                ln1Cache.Input = attnResidual;
                ln1Cache.Mean = ln1Means;
                ln1Cache.Variance = ln1Vars;
                ln1Cache.Normalized = ln1Normalized;

                // Cache rows for FFN backprop.
                // This avoids repeated ExtractRow calls in the forward path.
                var ffnInputRows = CopyRowsToJagged(normed1);

                var ffOutput = _accel.FFNForwardBatch(normed1, seqLen, _modelConfig.EmbeddingDim, block.FeedForwardNetwork.ForwardPassOnly);

                cache.FFNInputs.Add(ffnInputRows);
                cache.FFNOutputs.Add(ffOutput);

                var ffResidual = _accel.MatrixAdd(normed1, ffOutput);

                var ln2Cache = cache.LN2Caches[layer];
                var (normed2, ln2Means, ln2Vars, ln2Normalized) = _accel.LayerNormForward(ffResidual, block.LN2Gamma, block.LN2Beta);

                ln2Cache.Input = ffResidual;
                ln2Cache.Mean = ln2Means;
                ln2Cache.Variance = ln2Vars;
                ln2Cache.Normalized = ln2Normalized;

                x = normed2;
            }

            cache.FinalHiddenStates = x;
            return ProjectToOutput(x);
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
                _accel.AccumulateTokenEmbeddingGrad(_gradients.TokenEmbeddingGrad, dX, cache.TokenIds, dX.GetLength(0), dX.GetLength(1));
            else
                _accel.BackpropInputProjection(dX, cache.ContinuousInput, _gradients.InputProjectionGrad, _gradients.InputProjectionBiasGrad, dX.GetLength(0), _modelConfig.EmbeddingDim, _modelConfig.InputFeatureDim);
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
            var dNormed1 = _accel.MatrixAdd(dFFResidual, dNormed1_from_ffn);

            var ln1Cache = cache.LN1Caches[layerIdx];
            var (dAttnResidual, dLN1Gamma, dLN1Beta) = _accel.LayerNormBackward(dNormed1, ln1Cache.Normalized, block.LN1Gamma, ln1Cache.Input, ln1Cache.Mean, ln1Cache.Variance);
            var ln1Grads = _gradients.LN1Grads[layerIdx];
            _accel.VectorAccumulate(ln1Grads.GammaGrad, dLN1Gamma);
            _accel.VectorAccumulate(ln1Grads.BetaGrad, dLN1Beta);

            var dX_from_attn = BackpropAttention(layerIdx, dAttnResidual, cache.AttentionCaches[layerIdx]);
            return _accel.MatrixAdd(dAttnResidual, dX_from_attn);
        }

        private float[,] BackpropFFN(int layerIdx, float[,] dFFOutput, ForwardCache cache)
        {
            var block = _model.Blocks[layerIdx];
            int seqLen = dFFOutput.GetLength(0);
            int embeddingDim = _modelConfig.EmbeddingDim;
            var dNormed1 = new float[seqLen, embeddingDim];

            for (int i = 0; i < seqLen; i++)
            {
                var dOutRow = _accel.ExtractRow(dFFOutput, i, embeddingDim);
                var inputRow = cache.FFNInputs[layerIdx][i];
                block.FeedForwardNetwork.ForwardPassOnly(inputRow);
                var dInputRow = block.FeedForwardNetwork.ComputeInputGradient(dOutRow, _ffnWeightGrads[layerIdx], _ffnBiasGrads[layerIdx]);
                _accel.SetRow(dNormed1, i, dInputRow, embeddingDim);
            }
            return dNormed1;
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

            if (_modelConfig.UseDecoderOnly)
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

            _rotaryPositionEmbedding.ApplyBackwardInPlace(dQFull, dKFull, numHeads);

            return _accel.BackpropQKV(cache.Input, dQFull, dKFull, dVFull,  attention.WQ, attention.WK,  attention.WV, grads.WQ_Grad, grads.BiasQ_Grad, grads.WK_Grad, grads.BiasK_Grad, grads.WV_Grad, grads.BiasV_Grad);
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

            var concatenated = _accel.ScaledDotProductAttention(Q, K, V, numHeads, mask, causal: _modelConfig.UseDecoderOnly && mask == null);

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

                var input = sequence.Take(sequence.Length - 1).ToArray();
                var target = sequence.Skip(1).ToArray();

                if (!TokensAreValid(input) || !TokensAreValid(target))
                {
                    continue;
                }

                var logits = _model.Forward(input);
                int effectiveLen = Math.Min(logits.GetLength(0), target.Length);

                if (effectiveLen <= 0)
                {
                    continue;
                }

                var result = _accel.CrossEntropyLossAndGradient(logits, target, effectiveLen);

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

                var inputSlice = _accel.SliceRows(inputSeq, 0, seqLen - 1);
                var output = _model.Forward(inputSlice);
                int effectiveLen = output.GetLength(0);

                if (_modelConfig.Data.DataType == TransformerDataType.TimeSeriesRegression)
                {
                    if (regressionTargets == null || regressionTargets[idx] == null)
                    {
                        continue;
                    }

                    var targetSlice = _accel.SliceRows(regressionTargets[idx], 1, seqLen);
                    effectiveLen = Math.Min(effectiveLen, targetSlice.GetLength(0));

                    if (effectiveLen <= 0)
                    {
                        continue;
                    }

                    var result = _accel.MSELossAndGradient(output,targetSlice, effectiveLen);

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

                    var targetSlice = classTargets[idx].Skip(1).Take(seqLen - 1).ToArray();

                    effectiveLen = Math.Min(effectiveLen, targetSlice.Length);

                    if (effectiveLen <= 0)
                    {
                        continue;
                    }

                    var result = _accel.CrossEntropyLossAndGradient(output, targetSlice, effectiveLen);

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
                _accel.MatrixUpdate(_model.TokenEmbedding, _gradients.TokenEmbeddingGrad, learningRate);
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
            if (threshold <= 0f || float.IsNaN(threshold) || float.IsInfinity(threshold))
                throw new ArgumentOutOfRangeException(nameof(threshold), "Gradient clip threshold must be finite and positive.");

            float totalNorm = ComputeGradientNorm();
            if (IsFinite(totalNorm) && totalNorm > threshold)
                ScaleGradients(threshold / totalNorm);
        }
        private float ComputeGradientNorm()
        {
            float sum = 0;

            if (_modelConfig.Data.UsesDiscreteTokens)
                sum += _accel.MatrixSquaredNorm(_gradients.TokenEmbeddingGrad);
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
                _accel.MatrixScaleInPlace(_gradients.TokenEmbeddingGrad, scale);
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
            return Enumerable.Range(0, data.Length).OrderBy(x => _random.Next()).Select(i => data[i]).ToArray();
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

        private static float[][] CopyRowsToJagged(float[,] matrix)
        {
            if (matrix == null)
            {
                throw new ArgumentNullException(nameof(matrix));
            }

            int rows = matrix.GetLength(0);
            int cols = matrix.GetLength(1);
            int bytesPerRow = cols * sizeof(float);

            var result = new float[rows][];

            for (int i = 0; i < rows; i++)
            {
                var row = new float[cols];
                Buffer.BlockCopy(matrix, i * bytesPerRow, row, 0, bytesPerRow);
                result[i] = row;
            }

            return result;
        }
    }
}