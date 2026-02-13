using CallaghanDev.ML.AccelerationManagers;
using CallaghanDev.ML.Transformers.Cache;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace CallaghanDev.ML.Transformers
{
    public class TransformerTrainer
    {
        private readonly LanguageModel _model;
        private readonly TransformerConfig _modelConfig;
        private readonly TrainingConfig _trainConfig;
        private readonly TransformerGradients _gradients;
        private readonly IAccelerationManager _accel;
        private readonly Random _random;

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
            for (int i = 0; i < _modelConfig.NumLayers; i++)
            {
                var (wGrads, bGrads) = _model.Blocks[i].FeedForwardNetwork.CreateGradientStorage();
                _ffnWeightGrads.Add(wGrads);
                _ffnBiasGrads.Add(bGrads);
            }
        }

        public void Train(int[][] sequences, int[][] validationSequences = null)
        {
            if (!_modelConfig.UsesDiscreteTokens) // For text and symbolicSequence
            {
                throw new InvalidOperationException($"Use TrainContinuous() for {_modelConfig.DataType}.");
            }

            var validSequences = sequences.Where(seq => seq != null && seq.Length >= 2).ToArray();

            if (validSequences.Length == 0)
            {
                throw new ArgumentException("No valid sequences to train on. All sequences must contain no less than 2 tokens.");
            }

            if (validSequences.Length < sequences.Length && _trainConfig.Verbose)
            {
                Console.WriteLine($"Filtered out {sequences.Length - validSequences.Length} sequences that were too short (I need at least 2 tokens for this)");
            }

            float currentLR = _trainConfig.LearningRate;

            for (int epoch = 0; epoch < _trainConfig.Epochs; epoch++)
            {
                if (_trainConfig.Verbose)
                {
                    Console.WriteLine($"\n=== Epoch {epoch + 1}/{_trainConfig.Epochs} ===");
                }

                var shuffled = ShuffleArray(validSequences);
                float epochLoss = 0;
                int numBatches = 0;

                for (int i = 0; i < shuffled.Length; i += _trainConfig.BatchSize)
                {
                    int batchSize = Math.Min(_trainConfig.BatchSize, shuffled.Length - i);
                    var batch = shuffled.Skip(i).Take(batchSize).ToArray();

                    if (batch.Length == 0)
                    {
                        continue;
                    }

                    float batchLoss = TrainBatchDiscrete(batch, currentLR);
                    epochLoss += batchLoss;
                    numBatches++;

                    if (validationSequences != null && numBatches % _trainConfig.ValidationInterval == 0)
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

                float avgLoss = numBatches > 0 ? epochLoss / numBatches : 0;

                if (_trainConfig.Verbose)
                {
                    Console.WriteLine($"Epoch {epoch + 1} Average Loss: {avgLoss:F4}");
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
            if (_modelConfig.UsesDiscreteTokens)
            {
                throw new InvalidOperationException($"Use Train(int[][]) for {_modelConfig.DataType}.");
            }

            if (_modelConfig.DataType == TransformerDataType.TimeSeriesRegression && regressionTargets == null)
            {
                throw new ArgumentException("regressionTargets is required for TimeSeriesRegression.");
            }
            if (_modelConfig.DataType == TransformerDataType.TimeSeriesClassification && classTargets == null)
            {
                throw new ArgumentException("classTargets is required for TimeSeriesClassification.");
            }

            int totalSamples = inputs.Length;

            // Filter out sequences that are too short (need at least 2 timesteps)
            var validIndices = new List<int>();
            for (int i = 0; i < totalSamples; i++)
            {
                if (inputs[i] != null && inputs[i].GetLength(0) >= 2)
                {
                    validIndices.Add(i);
                }
            }

            if (validIndices.Count == 0)
            {
                throw new ArgumentException("No valid sequences. All must no less than 2 timesteps.");
            }

            float currentLR = _trainConfig.LearningRate;

            for (int epoch = 0; epoch < _trainConfig.Epochs; epoch++)
            {
                if (_trainConfig.Verbose)
                {
                    Console.WriteLine($"\n=== Epoch {epoch + 1}/{_trainConfig.Epochs} ===");
                }

                var shuffled = validIndices.OrderBy(_ => _random.Next()).ToArray();
                float epochLoss = 0;
                int numBatches = 0;

                for (int i = 0; i < shuffled.Length; i += _trainConfig.BatchSize)
                {
                    int batchSize = Math.Min(_trainConfig.BatchSize, shuffled.Length - i);
                    var batchIndices = shuffled.Skip(i).Take(batchSize).ToArray();

                    if (batchIndices.Length == 0)
                    {
                        continue;
                    }

                    float batchLoss = TrainBatchContinuous(batchIndices, inputs, regressionTargets, classTargets, currentLR);
                    epochLoss += batchLoss;
                    numBatches++;

                    if (_trainConfig.Verbose && numBatches % 10 == 0)
                    {
                        Console.WriteLine($"Batch {numBatches}: Loss = {batchLoss:F4}");
                    }
                }

                float avgLoss = numBatches > 0 ? epochLoss / numBatches : 0;

                if (_trainConfig.Verbose)
                {
                    Console.WriteLine($"Epoch {epoch + 1} Average Loss: {avgLoss:F4}");
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

        private float TrainBatchDiscrete(int[][] batch, float learningRate)
        {
            ZeroAllGradients();
            float totalLoss = 0;
            int validCount = 0;

            foreach (var sequence in batch)
            {
                if (sequence == null || sequence.Length < 2)
                {
                    continue;
                }

                var input = sequence.Take(sequence.Length - 1).ToArray();
                var target = sequence.Skip(1).ToArray();

                if (input.Length == 0 || target.Length == 0)
                {
                    continue;
                }
                if (input.Length != target.Length)
                {
                    continue;
                }

                bool hasInvalidTokens = false;
                int vocabSize = _modelConfig.VocabSize;

                foreach (int token in input.Concat(target))
                {
                    if (token < 0 || token >= vocabSize) { hasInvalidTokens = true; break; }
                }

                if (hasInvalidTokens)
                {
                    continue;
                }

                try
                {
                    var cache = new ForwardCache(_modelConfig.NumLayers);
                    var logits = ForwardWithCacheDiscrete(input, cache);

                    if (logits.GetLength(0) != input.Length)
                    {
                        continue;
                    }

                    float loss = BackwardPassCrossEntropy(logits, target, cache);

                    if (float.IsNaN(loss) || float.IsInfinity(loss))
                    {
                        continue;
                    }

                    totalLoss += loss;
                    validCount++;
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"ERROR during training: {ex.Message}");
                    continue;
                }
            }

            if (validCount == 0)
            {
                return 0f;
            }

            float invCount = 1.0f / validCount;
            ScaleGradients(invCount);

            if (_trainConfig.UseGradientClipping)
            {
                ClipGradients(_trainConfig.GradientClipThreshold);
            }

            UpdateParameters(learningRate);
            return totalLoss / validCount;
        }

        private float TrainBatchContinuous(int[] batchIndices, float[][,] allInputs, float[][,] allRegTargets, int[][] allClassTargets, float learningRate)
        {
            ZeroAllGradients();
            float totalLoss = 0;
            int validCount = 0;

            foreach (int idx in batchIndices)
            {
                var inputSeq = allInputs[idx];
                int seqLen = inputSeq.GetLength(0);

                if (seqLen < 2)
                {
                    continue;
                }

                var inputSlice = SliceRows(inputSeq, 0, seqLen - 1);


                try
                {
                    var cache = new ForwardCache(_modelConfig.NumLayers);
                    var output = ForwardWithCacheContinuous(inputSlice, cache);

                    float loss;

                    if (_modelConfig.DataType == TransformerDataType.TimeSeriesRegression)
                    {
                        var targetSlice = SliceRows(allRegTargets[idx], 1, seqLen);
                        loss = BackwardPassMSE(output, targetSlice, cache);
                    }
                    else
                    {
                        var targetSlice = allClassTargets[idx].Skip(1).Take(seqLen - 1).ToArray();
                        loss = BackwardPassCrossEntropy(output, targetSlice, cache);
                    }

                    if (float.IsNaN(loss) || float.IsInfinity(loss)) continue;

                    totalLoss += loss;
                    validCount++;
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"ERROR during continuous training: {ex.Message}");
                    continue;
                }
            }

            if (validCount == 0)
            {
                return 0f;
            }

            float invCount = 1.0f / validCount;
            ScaleGradients(invCount);

            if (_trainConfig.UseGradientClipping)
            {
                ClipGradients(_trainConfig.GradientClipThreshold);
            }

            UpdateParameters(learningRate);

            return totalLoss / validCount;
        }

        private float[,] ForwardWithCacheDiscrete(int[] tokenIds, ForwardCache cache)
        {
            int seqLen = tokenIds.Length;

            var embedded = new float[seqLen, _modelConfig.EmbeddingDim];

            for (int i = 0; i < seqLen; i++)
            {
                for (int j = 0; j < _modelConfig.EmbeddingDim; j++)
                {
                    embedded[i, j] = _model.TokenEmbedding[tokenIds[i], j] + _model.PositionalEncoding[i, j];
                }
            }

            cache.EmbeddedInput = embedded;
            cache.TokenIds = tokenIds;
            cache.ContinuousInput = null;

            return ForwardFromEmbeddingWithCache(embedded, seqLen, cache);
        }

        private float[,] ForwardWithCacheContinuous(float[,] inputSequence, ForwardCache cache)
        {
            int seqLen = inputSequence.GetLength(0);

            var projected = _accel.BatchDotProduct(_model.InputProjection, inputSequence);

            var embedded = new float[seqLen, _modelConfig.EmbeddingDim];

            for (int i = 0; i < seqLen; i++)
            {
                for (int j = 0; j < _modelConfig.EmbeddingDim; j++)
                {
                    embedded[i, j] = projected[i, j] + _model.InputProjectionBias[j] + _model.PositionalEncoding[i, j];
                }
            }

            cache.EmbeddedInput = embedded;
            cache.TokenIds = null;
            cache.ContinuousInput = inputSequence;

            return ForwardFromEmbeddingWithCache(embedded, seqLen, cache);
        }

        private float[,] ForwardFromEmbeddingWithCache(float[,] embedded, int seqLen, ForwardCache cache)
        {
            bool[,] mask = null;

            if (_modelConfig.UseDecoderOnly)
            {
                mask = CreateCausalMask(seqLen);
            }

            var x = embedded;

            for (int layer = 0; layer < _modelConfig.NumLayers; layer++)
            {
                cache.LayerInputs.Add(x);

                var block = _model.Blocks[layer];

                var attnCache = cache.AttentionCaches[layer];
                attnCache.Input = x;
                var attnOutput = AttentionForwardWithCache(block.Attention, x, mask, attnCache);

                var attnResidual = _accel.MatrixAdd(x, attnOutput);

                var ln1Cache = cache.LN1Caches[layer];
                var (normed1, ln1Means, ln1Vars, ln1Normalized) = _accel.LayerNormForward(attnResidual, block.LN1Gamma, block.LN1Beta);
                ln1Cache.Input = attnResidual;
                ln1Cache.Mean = ln1Means;
                ln1Cache.Variance = ln1Vars;
                ln1Cache.Normalized = ln1Normalized;

                var ffnInputRows = new float[seqLen][];
                var ffOutput = new float[seqLen, _modelConfig.EmbeddingDim];


                for (int i = 0; i < seqLen; i++)
                {
                    var inputRow = new float[_modelConfig.EmbeddingDim];
                    for (int j = 0; j < _modelConfig.EmbeddingDim; j++)
                    {
                        inputRow[j] = normed1[i, j];
                    }
                    ffnInputRows[i] = inputRow;
                    var outputRow = block.FeedForwardNetwork.ForwardPassOnly(inputRow);
                    for (int j = 0; j < _modelConfig.EmbeddingDim; j++)
                    {
                        ffOutput[i, j] = outputRow[j];
                    }
                }

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

        private float BackwardPassCrossEntropy(float[,] logits, int[] targets, ForwardCache cache)
        {
            int seqLen = logits.GetLength(0);
            int outputDim = logits.GetLength(1);

            float loss = 0;
            var dLogits = new float[seqLen, outputDim];

            int effectiveLen = Math.Min(seqLen, targets.Length);

            for (int i = 0; i < effectiveLen; i++)
            {
                float max = float.NegativeInfinity;
                for (int j = 0; j < outputDim; j++)
                {
                    max = Math.Max(max, logits[i, j]);
                }

                float sum = 0;
                var probs = new float[outputDim];
                for (int j = 0; j < outputDim; j++)
                {
                    probs[j] = MathF.Exp(logits[i, j] - max);
                    sum += probs[j];
                }
                for (int j = 0; j < outputDim; j++)
                {
                    probs[j] /= sum;
                }

                int targetToken = targets[i];
                loss -= MathF.Log(probs[targetToken] + 1e-10f);

                for (int j = 0; j < outputDim; j++)
                {
                    dLogits[i, j] = probs[j];
                    if (j == targetToken)
                    {
                        dLogits[i, j] -= 1.0f;
                    }
                }
            }

            loss  = loss/ effectiveLen;

            float invLen = 1.0f / effectiveLen;
            for (int i = 0; i < effectiveLen; i++)
            {
                for (int j = 0; j < outputDim; j++)
                {
                    dLogits[i, j] *= invLen;
                }
            }

            BackpropFromOutput(dLogits, cache);

            return loss;
        }

        private float BackwardPassMSE(float[,] predictions, float[,] targets, ForwardCache cache)
        {
            int seqLen = predictions.GetLength(0);
            int outputDim = predictions.GetLength(1);

            float loss = 0;
            var dOutput = new float[seqLen, outputDim];

            int effectiveLen = Math.Min(seqLen, targets.GetLength(0));

            for (int i = 0; i < effectiveLen; i++)
            {
                for (int j = 0; j < outputDim; j++)
                {
                    float diff = predictions[i, j] - targets[i, j];

                    loss += diff * diff;

                    dOutput[i, j] = 2.0f * diff;
                }
            }

            loss /= (effectiveLen * outputDim);

            float invLen = 1.0f / (effectiveLen * outputDim);
            for (int i = 0; i < effectiveLen; i++)
            {
                for (int j = 0; j < outputDim; j++)
                {
                    dOutput[i, j] *= invLen;
                }
            }

            BackpropFromOutput(dOutput, cache);

            return loss;
        }


        private void BackpropFromOutput(float[,] dOutput, ForwardCache cache)
        {
            BackpropOutputLayer(dOutput, cache.FinalHiddenStates);
            var dX = ComputeOutputGradient(dOutput);

            for (int layer = _modelConfig.NumLayers - 1; layer >= 0; layer--)
            {
                dX = BackpropBlock(layer, dX, cache);
            }

            BackpropInput(dX, cache);
        }

        private void BackpropInput(float[,] dX, ForwardCache cache)
        {
            if (_modelConfig.UsesDiscreteTokens)
            {
                BackpropEmbeddings(dX, cache.EmbeddedInput, cache.TokenIds);
            }
            else
            {
                BackpropInputProjection(dX, cache.ContinuousInput);
            }
        }

        private void BackpropEmbeddings(float[,] dX, float[,] embedded, int[] tokenIds)
        {
            int seqLen = dX.GetLength(0);
            int embeddingDim = dX.GetLength(1);

            for (int i = 0; i < seqLen; i++)
            {
                int tokenId = tokenIds[i];

                for (int j = 0; j < embeddingDim; j++)
                {
                    _gradients.TokenEmbeddingGrad[tokenId, j] += dX[i, j];
                }
            }
        }

        private void BackpropInputProjection(float[,] dX, float[,] continuousInput)
        {
            int seqLen = dX.GetLength(0);
            int embeddingDim = _modelConfig.EmbeddingDim;
            int inputFeatureDim = _modelConfig.InputFeatureDim;

            for (int i = 0; i < seqLen; i++)
            {
                for (int e = 0; e < embeddingDim; e++)
                {
                    float dVal = dX[i, e];

                    for (int f = 0; f < inputFeatureDim; f++)
                    {
                        _gradients.InputProjectionGrad[e, f] += dVal * continuousInput[i, f];
                    }

                    _gradients.InputProjectionBiasGrad[e] += dVal;
                }
            }
        }

        private void BackpropOutputLayer(float[,] dLogits, float[,] input)
        {
            int seqLen = dLogits.GetLength(0);
            int outputDim = _modelConfig.EffectiveOutputDim;

            for (int i = 0; i < seqLen; i++)
            {
                for (int v = 0; v < outputDim; v++)
                {
                    for (int e = 0; e < _modelConfig.EmbeddingDim; e++)
                    {
                        _gradients.OutputProjectionGrad[v, e] += input[i, e] * dLogits[i, v];
                    }
                }

                for (int v = 0; v < outputDim; v++)
                {
                    _gradients.OutputBiasGrad[v] += dLogits[i, v];
                }
            }
        }

        private float[,] ComputeOutputGradient(float[,] dLogits)
        {
            int seqLen = dLogits.GetLength(0);
            int outputDim = dLogits.GetLength(1);
            var dX = new float[seqLen, _modelConfig.EmbeddingDim];

            for (int i = 0; i < seqLen; i++)
            {
                for (int e = 0; e < _modelConfig.EmbeddingDim; e++)
                {
                    float grad = 0;
                    for (int v = 0; v < outputDim; v++)
                    {
                        grad += dLogits[i, v] * _model.OutputProjection[v, e];
                    }
                    dX[i, e] = grad;
                }
            }

            return dX;
        }
        private float[,] BackpropBlock(int layerIdx, float[,] dOut, ForwardCache cache)
        {
            var block = _model.Blocks[layerIdx];

            var ln2Cache = cache.LN2Caches[layerIdx];
            var (dFFResidual, dLN2Gamma, dLN2Beta) = _accel.LayerNormBackward(dOut, ln2Cache.Normalized, block.LN2Gamma, ln2Cache.Input, ln2Cache.Mean, ln2Cache.Variance);

            var ln2Grads = _gradients.LN2Grads[layerIdx];

            for (int j = 0; j < dLN2Gamma.Length; j++)
            {
                ln2Grads.GammaGrad[j] += dLN2Gamma[j];
                ln2Grads.BetaGrad[j] += dLN2Beta[j];
            }

            var dNormed1_from_ffn = BackpropFFN(layerIdx, dFFResidual, cache);
            var dNormed1 = _accel.MatrixAdd(dFFResidual, dNormed1_from_ffn);

            var ln1Cache = cache.LN1Caches[layerIdx];

            var (dAttnResidual, dLN1Gamma, dLN1Beta) = _accel.LayerNormBackward(dNormed1, ln1Cache.Normalized, block.LN1Gamma,ln1Cache.Input, ln1Cache.Mean, ln1Cache.Variance);

            var ln1Grads = _gradients.LN1Grads[layerIdx];

            for (int j = 0; j < dLN1Gamma.Length; j++)
            {
                ln1Grads.GammaGrad[j] += dLN1Gamma[j];
                ln1Grads.BetaGrad[j] += dLN1Beta[j];
            }

            var dX_from_attn = BackpropAttention(layerIdx, dAttnResidual, cache.AttentionCaches[layerIdx]);

            var dX = _accel.MatrixAdd(dAttnResidual, dX_from_attn);

            return dX;
        }

        private float[,] BackpropFFN(int layerIdx, float[,] dFFOutput, ForwardCache cache)
        {
            var block = _model.Blocks[layerIdx];
            int seqLen = dFFOutput.GetLength(0);
            int embeddingDim = _modelConfig.EmbeddingDim;

            var dNormed1 = new float[seqLen, embeddingDim];

            for (int i = 0; i < seqLen; i++)
            {
                var dOutRow = new float[embeddingDim];

                for (int j = 0; j < embeddingDim; j++)
                {
                    dOutRow[j] = dFFOutput[i, j];
                }

                var inputRow = cache.FFNInputs[layerIdx][i];
                block.FeedForwardNetwork.ForwardPassOnly(inputRow);

                var dInputRow = block.FeedForwardNetwork.ComputeInputGradient(dOutRow, _ffnWeightGrads[layerIdx], _ffnBiasGrads[layerIdx]);

                for (int j = 0; j < embeddingDim; j++)
                {
                    dNormed1[i, j] = dInputRow[j];
                }
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

            var (dQ_full, dK_full, dV_full) = _accel.MultiHeadAttentionBackward( cache.Q, cache.K, cache.V, dConcatenated, numHeads, scale, _modelConfig.UseDecoderOnly);

            var dInput = new float[seqLen, embeddingDim];

            _accel.BackpropLinearProjection(cache.Input, dQ_full, attention.WQ, grads.WQ_Grad, grads.BiasQ_Grad, dInput);
            _accel.BackpropLinearProjection(cache.Input, dK_full, attention.WK, grads.WK_Grad, grads.BiasK_Grad, dInput);
            _accel.BackpropLinearProjection(cache.Input, dV_full, attention.WV, grads.WV_Grad, grads.BiasV_Grad, dInput);

            return dInput;
        }

        private float[,] AttentionForwardWithCache(MultiHeadAttention attention, float[,] input, bool[,] mask, AttentionCache cache)
        {
            int seqLen = input.GetLength(0);
            int embeddingDim = _modelConfig.EmbeddingDim;
            int numHeads = _modelConfig.NumHeads;
            int headDim = embeddingDim / numHeads;

            var Q_noBias = _accel.BatchDotProduct(attention.WQ, input);
            var K_noBias = _accel.BatchDotProduct(attention.WK, input);
            var V_noBias = _accel.BatchDotProduct(attention.WV, input);

            var Q = AddBiasToMatrix(Q_noBias, attention.BiasQ);
            var K = AddBiasToMatrix(K_noBias, attention.BiasK);
            var V = AddBiasToMatrix(V_noBias, attention.BiasV);

            cache.Q = Q;
            cache.K = K;
            cache.V = V;

            float scale = 1.0f / MathF.Sqrt(headDim);

            var concatenated = _accel.MultiHeadAttentionForward(Q, K, V, numHeads, scale, mask);

            cache.AttentionOutput = concatenated;

            var output_noBias = _accel.BatchDotProduct(attention.WO, concatenated);
            var output = AddBiasToMatrix(output_noBias, attention.BiasO);

            return output;
        }

        public float Validate(int[][] validationSequences)
        {
            float totalLoss = 0;
            int count = 0;
            int outputDim = _modelConfig.EffectiveOutputDim;

            foreach (var sequence in validationSequences)
            {
                if (sequence.Length < 2)
                {
                    continue;
                }

                var input = sequence.Take(sequence.Length - 1).ToArray();
                var target = sequence.Skip(1).ToArray();

                var logits = _model.Forward(input);

                for (int i = 0; i < Math.Min(logits.GetLength(0), target.Length); i++)
                {
                    float max = float.NegativeInfinity;
                    for (int j = 0; j < outputDim; j++)
                    {
                        max = Math.Max(max, logits[i, j]);
                    }

                    float sum = 0;
                    for (int j = 0; j < outputDim; j++)
                    {
                        sum += MathF.Exp(logits[i, j] - max);
                    }

                    float prob = MathF.Exp(logits[i, target[i]] - max) / sum;

                    totalLoss -= MathF.Log(prob + 1e-10f);

                    count++;
                }
            }

            return count > 0 ? totalLoss / count : 0;
        }

        public float ValidateContinuous(float[][,] inputs, float[][,] regressionTargets = null, int[][] classTargets = null)
        {
            float totalLoss = 0;
            int count = 0;

            for (int idx = 0; idx < inputs.Length; idx++)
            {
                var inputSeq = inputs[idx];
                int seqLen = inputSeq.GetLength(0);

                if (seqLen < 2)
                {
                    continue;
                }

                var inputSlice = SliceRows(inputSeq, 0, seqLen - 1);
                var output = _model.Forward(inputSlice);

                int effectiveLen = output.GetLength(0);

                if (_modelConfig.DataType == TransformerDataType.TimeSeriesRegression)
                {
                    var targetSlice = SliceRows(regressionTargets[idx], 1, seqLen);
                    int outputDim = _modelConfig.OutputDim;

                    for (int i = 0; i < effectiveLen; i++)
                    {
                        for (int j = 0; j < outputDim; j++)
                        {
                            float diff = output[i, j] - targetSlice[i, j];
                            totalLoss += diff * diff;
                        }
                        count++;
                    }
                }
                else // For timeSeries classification
                {
                    int outputDim = _modelConfig.OutputDim;
                    var targetSlice = classTargets[idx].Skip(1).Take(seqLen - 1).ToArray();

                    for (int i = 0; i < Math.Min(effectiveLen, targetSlice.Length); i++)
                    {
                        float max = float.NegativeInfinity;

                        for (int j = 0; j < outputDim; j++)
                        {
                            max = Math.Max(max, output[i, j]);
                        }

                        float sum = 0;
                        for (int j = 0; j < outputDim; j++)
                        {
                            sum += MathF.Exp(output[i, j] - max);
                        }

                        float prob = MathF.Exp(output[i, targetSlice[i]] - max) / sum;
                        totalLoss -= MathF.Log(prob + 1e-10f);
                        count++;
                    }
                }
            }

            return count > 0 ? totalLoss / count : 0;
        }

        private void UpdateParameters(float learningRate)
        {
            if (_modelConfig.UsesDiscreteTokens)
            {
                _accel.MatrixUpdate(_model.TokenEmbedding, _gradients.TokenEmbeddingGrad, learningRate);
            }
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
            float totalNorm = ComputeGradientNorm();
            if (totalNorm > threshold)
            {
                float scale = threshold / totalNorm;
                ScaleGradients(scale);
            }
        }

        private float ComputeGradientNorm()
        {
            float sum = 0;

            if (_modelConfig.UsesDiscreteTokens)
            {
                sum += _accel.MatrixSquaredNorm(_gradients.TokenEmbeddingGrad);
            }
            else
            {
                sum += _accel.MatrixSquaredNorm(_gradients.InputProjectionGrad);
                sum += VectorNorm(_gradients.InputProjectionBiasGrad);
            }

            sum += _accel.MatrixSquaredNorm(_gradients.OutputProjectionGrad);
            sum += VectorNorm(_gradients.OutputBiasGrad);

            foreach (var g in _gradients.AttentionGrads)
            {
                sum += _accel.MatrixSquaredNorm(g.WQ_Grad);
                sum += _accel.MatrixSquaredNorm(g.WK_Grad);
                sum += _accel.MatrixSquaredNorm(g.WV_Grad);
                sum += _accel.MatrixSquaredNorm(g.WO_Grad);
                sum += VectorNorm(g.BiasQ_Grad);
                sum += VectorNorm(g.BiasK_Grad);
                sum += VectorNorm(g.BiasV_Grad);
                sum += VectorNorm(g.BiasO_Grad);
            }

            foreach (var g in _gradients.LN1Grads)
            {
                sum += VectorNorm(g.GammaGrad);
                sum += VectorNorm(g.BetaGrad);
            }
            foreach (var g in _gradients.LN2Grads)
            {
                sum += VectorNorm(g.GammaGrad);
                sum += VectorNorm(g.BetaGrad);
            }

            for (int layer = 0; layer < _modelConfig.NumLayers; layer++)
            {
                foreach (var wGrad in _ffnWeightGrads[layer])
                {
                    sum += _accel.MatrixSquaredNorm(wGrad);
                }
                foreach (var bGrad in _ffnBiasGrads[layer])
                {
                    sum += VectorNorm(bGrad);
                }
            }

            return MathF.Sqrt(sum);
        }

        private void ScaleGradients(float scale)
        {
            if (_modelConfig.UsesDiscreteTokens)
            {
                _accel.MatrixScaleInPlace(_gradients.TokenEmbeddingGrad, scale);
            }
            else
            {
                _accel.MatrixScaleInPlace(_gradients.InputProjectionGrad, scale);
                _accel.VectorScaleInPlace(_gradients.InputProjectionBiasGrad, scale);
            }

            _accel.MatrixScaleInPlace(_gradients.OutputProjectionGrad, scale);
            _accel.VectorScaleInPlace(_gradients.OutputBiasGrad, scale);

            foreach (var g in _gradients.AttentionGrads)
            {
                _accel.MatrixScaleInPlace(g.WQ_Grad, scale);
                _accel.MatrixScaleInPlace(g.WK_Grad, scale);
                _accel.MatrixScaleInPlace(g.WV_Grad, scale);
                _accel.MatrixScaleInPlace(g.WO_Grad, scale);
                _accel.VectorScaleInPlace(g.BiasQ_Grad, scale);
                _accel.VectorScaleInPlace(g.BiasK_Grad, scale);
                _accel.VectorScaleInPlace(g.BiasV_Grad, scale);
                _accel.VectorScaleInPlace(g.BiasO_Grad, scale);
            }


            foreach (var g in _gradients.LN1Grads)
            {
                _accel.VectorScaleInPlace(g.GammaGrad, scale);
                _accel.VectorScaleInPlace(g.BetaGrad, scale);
            }
            foreach (var g in _gradients.LN2Grads)
            {
                _accel.VectorScaleInPlace(g.GammaGrad, scale);
                _accel.VectorScaleInPlace(g.BetaGrad, scale);
            }

            for (int layer = 0; layer < _modelConfig.NumLayers; layer++)
            {
                foreach (var wGrad in _ffnWeightGrads[layer])
                {
                    _accel.MatrixScaleInPlace(wGrad, scale);
                }
                foreach (var bGrad in _ffnBiasGrads[layer])
                {
                    _accel.VectorScaleInPlace(bGrad, scale);
                }
            }
        }

        private void ZeroAllGradients()
        {
            _gradients.Zero();

            for (int layer = 0; layer < _modelConfig.NumLayers; layer++)
            {
                foreach (var wGrad in _ffnWeightGrads[layer])
                {
                    _accel.ZeroMatrix(wGrad);
                }
                foreach (var bGrad in _ffnBiasGrads[layer])
                {
                    Array.Clear(bGrad, 0, bGrad.Length);
                }
            }
        }

        private float[,] ProjectToOutput(float[,] hidden)
        {
            var logits = _accel.BatchDotProduct(_model.OutputProjection, hidden);

            return AddBiasToMatrix(logits, _model.OutputBias);
        }

        private float[,] AddBiasToMatrix(float[,] matrix, float[] bias)
        {
            int rows = matrix.GetLength(0);
            int cols = matrix.GetLength(1);

            var result = new float[rows, cols];

            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < cols; j++)
                {
                    result[i, j] = matrix[i, j] + bias[j];
                }
            }
            return result;
        }

        private bool[,] CreateCausalMask(int seqLen)
        {
            var mask = new bool[seqLen, seqLen];

            for (int i = 0; i < seqLen; i++)
            {
                for (int j = 0; j <= i; j++)
                {
                    mask[i, j] = true;
                }
            }

            return mask;
        }

        private float VectorNorm(float[] vector)
        {
            float sum = 0;

            for (int i = 0; i < vector.Length; i++)
            {
                sum += vector[i] * vector[i];
            }

            return sum;
        }

        private T[] ShuffleArray<T>(T[] data)
        {
            var indices = Enumerable.Range(0, data.Length).OrderBy(x => _random.Next()).ToArray();

            return indices.Select(i => data[i]).ToArray();
        }

        private float[,] SliceRows(float[,] matrix, int startRow, int endRow)
        {
            int cols = matrix.GetLength(1);

            int numRows = endRow - startRow;
            var result = new float[numRows, cols];

            for (int i = 0; i < numRows; i++)
            {
                for (int j = 0; j < cols; j++)
                {
                    result[i, j] = matrix[startRow + i, j];
                }
            }

            return result;
        }
    }
}