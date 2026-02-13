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

        // Neuralnetwork gradient storage needs one per layer, created once and zeroed each batch.
        private readonly List<List<float[,]>> _ffnWeightGrads;
        private readonly List<List<float[]>> _ffnBiasGrads;

        public TransformerTrainer(LanguageModel model, TrainingConfig trainConfig)
        {
            _model = model;
            _modelConfig = model.Config;
            _trainConfig = trainConfig;
            _gradients = new TransformerGradients(_modelConfig.NumLayers, _modelConfig.EmbeddingDim, _modelConfig.VocabSize);
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

        private float[,] LayerNormForwardWithCache(float[,] input, float[] gamma, float[] beta, LayerNormCache cache)
        {
            int batchSize = input.GetLength(0);
            int features = input.GetLength(1);

            cache.Input = input;
            cache.Mean = new float[batchSize];
            cache.Variance = new float[batchSize];

            var normalized = new float[batchSize, features];
            const float epsilon = 1e-5f;

            for (int i = 0; i < batchSize; i++)
            {
                float mean = 0.0f;
                for (int j = 0; j < features; j++)
                {
                    mean += input[i, j];
                }
                mean /= features;
                cache.Mean[i] = mean;

                float variance = 0.0f;
                for (int j = 0; j < features; j++)
                {
                    float diff = input[i, j] - mean;
                    variance += diff * diff;
                }
                variance /= features;
                cache.Variance[i] = variance;

                float stdDev = MathF.Sqrt(variance + epsilon);
                for (int j = 0; j < features; j++)
                {
                    normalized[i, j] = (input[i, j] - mean) / stdDev;
                }
            }

            cache.Normalized = normalized;

            var output = new float[batchSize, features];
            for (int i = 0; i < batchSize; i++)
            {
                for (int j = 0; j < features; j++)
                {
                    output[i, j] = gamma[j] * normalized[i, j] + beta[j];
                }
            }

            return output;
        }

        private float BackwardPass(float[,] logits, int[] targets, ForwardCache cache)
        {
            try
            {
                int seqLen = logits.GetLength(0);
                int vocabSize = logits.GetLength(1);

                float loss = 0;
                var dLogits = new float[seqLen, vocabSize];

                for (int i = 0; i < Math.Min(seqLen, targets.Length); i++)
                {
                    float max = float.NegativeInfinity;
                    for (int j = 0; j < vocabSize; j++)
                    {
                        max = Math.Max(max, logits[i, j]);
                    }

                    float sum = 0;
                    var probs = new float[vocabSize];
                    for (int j = 0; j < vocabSize; j++)
                    {
                        probs[j] = MathF.Exp(logits[i, j] - max);
                        sum += probs[j];
                    }

                    for (int j = 0; j < vocabSize; j++)
                    {
                        probs[j] /= sum;
                    }

                    int targetToken = targets[i];
                    loss -= MathF.Log(probs[targetToken] + 1e-10f);

                    for (int j = 0; j < vocabSize; j++)
                    {
                        dLogits[i, j] = probs[j];
                        if (j == targetToken)
                        {
                            dLogits[i, j] -= 1.0f;
                        }
                    }
                }

                int effectiveLen = Math.Min(seqLen, targets.Length);
                loss /= effectiveLen;

                float invLen = 1.0f / effectiveLen;
                for (int i = 0; i < effectiveLen; i++)
                {
                    for (int j = 0; j < vocabSize; j++)
                    {
                        dLogits[i, j] *= invLen;
                    }
                }

                BackpropOutputLayer(dLogits, cache.FinalHiddenStates);
                var dX = ComputeOutputGradient(dLogits);

                for (int layer = _modelConfig.NumLayers - 1; layer >= 0; layer--)
                {
                    dX = BackpropBlock(layer, dX, cache);
                }

                BackpropEmbeddings(dX, cache.EmbeddedInput, cache.TokenIds);

                return loss;
            }
            catch (Exception ex)
            {
                Console.WriteLine($"!!! EXCEPTION IN BackwardPass !!!");
                Console.WriteLine($"Message: {ex.Message}");
                Console.WriteLine($"Stack Trace:\n{ex.StackTrace}");
                throw;
            }
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

        private float[,] BackpropLayerNorm(float[,] dOut, LayerNormCache cache, float[] gamma, LayerNormGradients grads)
        {
            int batchSize = dOut.GetLength(0);
            int features = dOut.GetLength(1);
            const float epsilon = 1e-5f;

            var dInput = new float[batchSize, features];

            for (int i = 0; i < batchSize; i++)
            {
                float invStd = 1.0f / MathF.Sqrt(cache.Variance[i] + epsilon);

                for (int j = 0; j < features; j++)
                {
                    grads.GammaGrad[j] += dOut[i, j] * cache.Normalized[i, j];
                    grads.BetaGrad[j] += dOut[i, j];
                }

                var dNorm = new float[features];
                for (int j = 0; j < features; j++)
                {
                    dNorm[j] = dOut[i, j] * gamma[j];
                }

                float dVar = 0;
                float invStdCubed = invStd * invStd * invStd;
                for (int j = 0; j < features; j++)
                {
                    float xMinusMean = cache.Input[i, j] - cache.Mean[i];
                    dVar += dNorm[j] * xMinusMean * (-0.5f) * invStdCubed;
                }

                float dMean = 0;
                for (int j = 0; j < features; j++)
                {
                    dMean += dNorm[j] * (-invStd);
                }

                float invN = 1.0f / features;
                for (int j = 0; j < features; j++)
                {
                    float xMinusMean = cache.Input[i, j] - cache.Mean[i];
                    dInput[i, j] = dNorm[j] * invStd  + dVar * 2.0f * xMinusMean * invN + dMean * invN;
                }
            }

            return dInput;
        }

        private void BackpropOutputLayer(float[,] dLogits, float[,] input)
        {
            int seqLen = dLogits.GetLength(0);

            for (int i = 0; i < seqLen; i++)
            {
                for (int v = 0; v < _modelConfig.VocabSize; v++)
                {
                    for (int e = 0; e < _modelConfig.EmbeddingDim; e++)
                    {
                        _gradients.OutputProjectionGrad[v, e] += input[i, e] * dLogits[i, v];
                    }
                }

                for (int v = 0; v < _modelConfig.VocabSize; v++)
                {
                    _gradients.OutputBiasGrad[v] += dLogits[i, v];
                }
            }
        }

        private float[,] ComputeOutputGradient(float[,] dLogits)
        {
            int seqLen = dLogits.GetLength(0);
            int vocabSize = dLogits.GetLength(1);
            var dX = new float[seqLen, _modelConfig.EmbeddingDim];

            for (int i = 0; i < seqLen; i++)
            {
                for (int e = 0; e < _modelConfig.EmbeddingDim; e++)
                {
                    float grad = 0;
                    for (int v = 0; v < vocabSize; v++)
                    {
                        grad += dLogits[i, v] * _model.OutputProjection[v, e];
                    }
                    dX[i, e] = grad;
                }
            }

            return dX;
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

        private void BackpropLinearProjection(float[,] input, float[,] dOutput, float[,] weights, float[,] weightGrad, float[] biasGrad, float[,] dInput, int seqLen, int embeddingDim)
        {
            for (int i = 0; i < seqLen; i++)
            {
                for (int j = 0; j < embeddingDim; j++)
                {
                    float dOutVal = dOutput[i, j];

                    for (int k = 0; k < embeddingDim; k++)
                    {
                        weightGrad[k, j] += input[i, k] * dOutVal;
                    }

                    biasGrad[j] += dOutVal;
                }

                for (int k = 0; k < embeddingDim; k++)
                {
                    float sum = 0;
                    for (int j = 0; j < embeddingDim; j++)
                    {
                        sum += dOutput[i, j] * weights[k, j];
                    }
                    dInput[i, k] += sum;
                }
            }
        }

        private float[,] ExtractHead(float[,] matrix, int startIdx, int headDim, int seqLen)
        {
            var result = new float[seqLen, headDim];
            for (int i = 0; i < seqLen; i++)
            {
                for (int j = 0; j < headDim; j++)
                {
                    result[i, j] = matrix[i, startIdx + j];
                }
            }
            return result;
        }


        private void UpdateMatrix(float[,] weights, float[,] gradients, float lr)
        {
            int rows = weights.GetLength(0);
            int cols = weights.GetLength(1);
            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < cols; j++)
                {
                    weights[i, j] -= lr * gradients[i, j];
                }
            }
        }

        private void UpdateVector(float[] weights, float[] gradients, float lr)
        {
            for (int i = 0; i < weights.Length; i++)
            {
                weights[i] -= lr * gradients[i];
            }
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


        private float MatrixNorm(float[,] matrix)
        {
            float sum = 0;
            for (int i = 0; i < matrix.GetLength(0); i++)
            {
                for (int j = 0; j < matrix.GetLength(1); j++)
                {
                    sum += matrix[i, j] * matrix[i, j];
                }
            }
            return sum;
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


        private void ScaleMatrix(float[,] matrix, float scale)
        {
            for (int i = 0; i < matrix.GetLength(0); i++)
            {
                for (int j = 0; j < matrix.GetLength(1); j++)
                {
                    matrix[i, j] *= scale;
                }
            }
        }

        private void ScaleVector(float[] vector, float scale)
        {
            for (int i = 0; i < vector.Length; i++)
            {
                vector[i] *= scale;
            }
        }


        private void ZeroMatrix(float[,] matrix)
        {
            int rows = matrix.GetLength(0);
            int cols = matrix.GetLength(1);
            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < cols; j++)
                {
                    matrix[i, j] = 0;
                }
            }
        }

        public void Train(int[][] sequences, int[][] validationSequences = null)
        {
            var validSequences = sequences.Where(seq => seq != null && seq.Length >= 2).ToArray();

            if (validSequences.Length == 0)
            {
                throw new ArgumentException("No valid sequences to train on. All sequences must have at least 2 tokens.");
            }

            if (validSequences.Length < sequences.Length && _trainConfig.Verbose)
            {
                Console.WriteLine($"Filtered out {sequences.Length - validSequences.Length} sequences that were too short (need at least 2 tokens)");
            }

            float currentLR = _trainConfig.LearningRate;

            for (int epoch = 0; epoch < _trainConfig.Epochs; epoch++)
            {
                if (_trainConfig.Verbose)
                {
                    Console.WriteLine($"\n=== Epoch {epoch + 1}/{_trainConfig.Epochs} ===");
                }

                var shuffled = ShuffleData(validSequences);

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

                    float batchLoss = TrainBatch(batch, currentLR);
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

        private float TrainBatch(int[][] batch, float learningRate)
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
                    Console.WriteLine($"WARNING: Input length ({input.Length}) != Target length ({target.Length}). Skipping sequence.");
                    continue;
                }

                bool hasInvalidTokens = false;
                int vocabSize = _modelConfig.VocabSize;

                foreach (int token in input)
                {
                    if (token < 0 || token >= vocabSize)
                    {
                        Console.WriteLine($"WARNING: Invalid input token {token} (vocab size: {vocabSize}). Skipping sequence.");
                        hasInvalidTokens = true;
                        break;
                    }
                }

                if (!hasInvalidTokens)
                {
                    foreach (int token in target)
                    {
                        if (token < 0 || token >= vocabSize)
                        {
                            Console.WriteLine($"WARNING: Invalid target token {token} (vocab size: {vocabSize}). Skipping sequence.");
                            hasInvalidTokens = true;
                            break;
                        }
                    }
                }

                if (hasInvalidTokens)
                {
                    continue;
                }

                try
                {
                    var cache = new ForwardCache(_modelConfig.NumLayers);
                    var logits = ForwardWithCache(input, cache);

                    int logitsSeqLen = logits.GetLength(0);
                    int logitsVocabSize = logits.GetLength(1);

                    if (logitsSeqLen != input.Length)
                    {
                        Console.WriteLine($"ERROR: Logits sequence length ({logitsSeqLen}) != Input length ({input.Length})");
                        continue;
                    }

                    if (logitsVocabSize != vocabSize)
                    {
                        Console.WriteLine($"ERROR: Logits vocab size ({logitsVocabSize}) != Expected vocab size ({vocabSize})");
                        continue;
                    }

                    float loss = BackwardPass(logits, target, cache);

                    if (float.IsNaN(loss) || float.IsInfinity(loss))
                    {
                        Console.WriteLine($"WARNING: Invalid loss value ({loss}). Skipping sequence.");
                        continue;
                    }

                    totalLoss += loss;
                    validCount++;
                }
                catch (IndexOutOfRangeException ex)
                {
                    Console.WriteLine($"ERROR: Index out of range during training: {ex.Message}");
                    Console.WriteLine($"  Sequence length: {sequence.Length}");
                    Console.WriteLine($"  Input length: {input.Length}");
                    Console.WriteLine($"  Target length: {target.Length}");
                    continue;
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"ERROR during training: {ex.Message}");
                    continue;
                }
            }

            if (validCount == 0)
            {
                Console.WriteLine("WARNING: No valid sequences in batch. Returning zero loss.");
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


        public float Validate(int[][] validationSequences)
        {
            float totalLoss = 0;
            int count = 0;

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
                    for (int j = 0; j < _modelConfig.VocabSize; j++)
                    {
                        max = Math.Max(max, logits[i, j]);
                    }

                    float sum = 0;
                    for (int j = 0; j < _modelConfig.VocabSize; j++)
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


        private int[][] ShuffleData(int[][] data)
        {
            var indices = Enumerable.Range(0, data.Length).OrderBy(x => _random.Next()).ToArray();
            return indices.Select(i => data[i]).ToArray();
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

        private float[,] ForwardWithCache(int[] tokenIds, ForwardCache cache)
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

                // Use LayerNormForward instead of manual LayerNormForwardWithCache
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

                // Use LayerNormForward instead of manual LayerNormForwardWithCache
                var ln2Cache = cache.LN2Caches[layer];
                var (normed2, ln2Means, ln2Vars, ln2Normalized) = _accel.LayerNormForward(ffResidual, block.LN2Gamma, block.LN2Beta);
                ln2Cache.Input = ffResidual;
                ln2Cache.Mean = ln2Means;
                ln2Cache.Variance = ln2Vars;
                ln2Cache.Normalized = ln2Normalized;
                x = normed2;
            }

            cache.FinalHiddenStates = x;

            return ProjectToVocab(x);
        }

        private float[,] AttentionForwardWithCache(MultiHeadAttention attention, float[,] input, bool[,] mask, AttentionCache cache)
        {
            int seqLen = input.GetLength(0);
            int embeddingDim = _modelConfig.EmbeddingDim;
            int numHeads = _modelConfig.NumHeads;
            int headDim = embeddingDim / numHeads;

            // Use BatchDotProduct + bias addition instead of row-by-row MatMulWithBias
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

            // Use MultiHeadAttentionForward instead of manual per-head loop
            var concatenated = _accel.MultiHeadAttentionForward(Q, K, V, numHeads, scale, mask);

            cache.AttentionOutput = concatenated;

            // Use BatchDotProduct for output projection
            var output_noBias = _accel.BatchDotProduct(attention.WO, concatenated);
            var output = AddBiasToMatrix(output_noBias, attention.BiasO);

            cache.AttentionScores = _accel.MatrixScale(_accel.MatrixMultiplyTranspose(Q, K), 1.0f / MathF.Sqrt(embeddingDim));
            cache.AttentionWeights = _accel.Softmax(cache.AttentionScores, mask);

            return output;
        }

        private float[,] BackpropBlock(int layerIdx, float[,] dOut, ForwardCache cache)
        {
            var block = _model.Blocks[layerIdx];

            // Use LayerNormBackward instead of manual BackpropLayerNorm
            var ln2Cache = cache.LN2Caches[layerIdx];
            var (dFFResidual, dLN2Gamma, dLN2Beta) = _accel.LayerNormBackward(
                dOut, ln2Cache.Normalized, block.LN2Gamma,
                ln2Cache.Input, ln2Cache.Mean, ln2Cache.Variance);
            // Accumulate LN gradients
            var ln2Grads = _gradients.LN2Grads[layerIdx];
            for (int j = 0; j < dLN2Gamma.Length; j++)
            {
                ln2Grads.GammaGrad[j] += dLN2Gamma[j];
                ln2Grads.BetaGrad[j] += dLN2Beta[j];
            }

            var dNormed1_from_ffn = BackpropFFN(layerIdx, dFFResidual, cache);

            var dNormed1 = _accel.MatrixAdd(dFFResidual, dNormed1_from_ffn);

            // Use LayerNormBackward instead of manual BackpropLayerNorm
            var ln1Cache = cache.LN1Caches[layerIdx];
            var (dAttnResidual, dLN1Gamma, dLN1Beta) = _accel.LayerNormBackward(
                dNormed1, ln1Cache.Normalized, block.LN1Gamma,
                ln1Cache.Input, ln1Cache.Mean, ln1Cache.Variance);
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

        private float[,] BackpropAttention(int layerIdx, float[,] dOut, AttentionCache cache)
        {
            var attention = _model.Blocks[layerIdx].Attention;
            var grads = _gradients.AttentionGrads[layerIdx];

            int seqLen = dOut.GetLength(0);
            int embeddingDim = _modelConfig.EmbeddingDim;
            int numHeads = _modelConfig.NumHeads;
            int headDim = embeddingDim / numHeads;

            // Backprop through output projection using _accel
            var dConcatenated = new float[seqLen, embeddingDim];
            _accel.BackpropLinearProjection(cache.AttentionOutput, dOut, attention.WO,
                grads.WO_Grad, grads.BiasO_Grad, dConcatenated);

            float scale = 1.0f / MathF.Sqrt(headDim);

            // Use MultiHeadAttentionBackward instead of manual per-head loop
            var (dQ_full, dK_full, dV_full) = _accel.MultiHeadAttentionBackward(
                cache.Q, cache.K, cache.V, dConcatenated,
                numHeads, scale, _modelConfig.UseDecoderOnly);

            // Backprop through Q, K, V projections using _accel
            var dInput = new float[seqLen, embeddingDim];
            _accel.BackpropLinearProjection(cache.Input, dQ_full, attention.WQ,
                grads.WQ_Grad, grads.BiasQ_Grad, dInput);
            _accel.BackpropLinearProjection(cache.Input, dK_full, attention.WK,
                grads.WK_Grad, grads.BiasK_Grad, dInput);
            _accel.BackpropLinearProjection(cache.Input, dV_full, attention.WV,
                grads.WV_Grad, grads.BiasV_Grad, dInput);

            return dInput;
        }

        private void UpdateParameters(float learningRate)
        {
            _accel.MatrixUpdate(_model.TokenEmbedding, _gradients.TokenEmbeddingGrad, learningRate);

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

        private float ComputeGradientNorm()
        {
            float sum = 0;

            sum += _accel.MatrixSquaredNorm(_gradients.TokenEmbeddingGrad);
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
            _accel.MatrixScaleInPlace(_gradients.TokenEmbeddingGrad, scale);
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

        private float[,] MatMulWithBias(float[,] input, float[,] weights, float[] bias)
        {
            var result = _accel.BatchDotProduct(weights, input);
            return AddBiasToMatrix(result, bias);
        }

        private float[,] ProjectToVocab(float[,] hidden)
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
    }
}