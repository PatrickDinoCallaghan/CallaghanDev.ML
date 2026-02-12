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

                var ln1Cache = cache.LN1Caches[layer];
                var normed1 = LayerNormForwardWithCache(attnResidual, block.LN1Gamma, block.LN1Beta, ln1Cache);

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
                x = LayerNormForwardWithCache(ffResidual, block.LN2Gamma, block.LN2Beta, ln2Cache);
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

            var Q = MatMulWithBias(input, attention.WQ, attention.BiasQ);
            var K = MatMulWithBias(input, attention.WK, attention.BiasK);
            var V = MatMulWithBias(input, attention.WV, attention.BiasV);

            cache.Q = Q;
            cache.K = K;
            cache.V = V;

            var concatenated = new float[seqLen, embeddingDim];
            float scale = 1.0f / MathF.Sqrt(headDim);

            for (int head = 0; head < numHeads; head++)
            {
                int startIdx = head * headDim;

                var Q_head = ExtractHead(Q, startIdx, headDim, seqLen);
                var K_head = ExtractHead(K, startIdx, headDim, seqLen);
                var V_head = ExtractHead(V, startIdx, headDim, seqLen);

                var scores = _accel.MatrixMultiplyTranspose(Q_head, K_head);
                var scaledScores = _accel.MatrixScale(scores, scale);

                var attnWeights = _accel.Softmax(scaledScores, mask);

                var headOutput = _accel.MatrixMultiply(attnWeights, V_head);

                for (int i = 0; i < seqLen; i++)
                {
                    for (int j = 0; j < headDim; j++)
                    {
                        concatenated[i, startIdx + j] = headOutput[i, j];
                    }
                }
            }

            cache.AttentionOutput = concatenated;

            var output = MatMulWithBias(concatenated, attention.WO, attention.BiasO);

            cache.AttentionScores = _accel.MatrixScale(_accel.MatrixMultiplyTranspose(Q, K), 1.0f / MathF.Sqrt(embeddingDim));
            cache.AttentionWeights = _accel.Softmax(cache.AttentionScores, mask);

            return output;
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

        private float[,] BackpropBlock(int layerIdx, float[,] dOut, ForwardCache cache)
        {
            var block = _model.Blocks[layerIdx];
            int seqLen = dOut.GetLength(0);
            int embeddingDim = _modelConfig.EmbeddingDim;

            var dFFResidual = BackpropLayerNorm(dOut, cache.LN2Caches[layerIdx], block.LN2Gamma, _gradients.LN2Grads[layerIdx]);

            var dNormed1_from_ffn = BackpropFFN(layerIdx, dFFResidual, cache);

            var dNormed1 = _accel.MatrixAdd(dFFResidual, dNormed1_from_ffn);

            var dAttnResidual = BackpropLayerNorm(dNormed1, cache.LN1Caches[layerIdx], block.LN1Gamma, _gradients.LN1Grads[layerIdx]);

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
                    dInput[i, j] = dNorm[j] * invStd
                                 + dVar * 2.0f * xMinusMean * invN
                                 + dMean * invN;
                }
            }

            return dInput;
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

            for (int i = 0; i < seqLen; i++)
            {
                for (int j = 0; j < embeddingDim; j++)
                {
                    for (int k = 0; k < embeddingDim; k++)
                    {
                        grads.WO_Grad[k, j] += cache.AttentionOutput[i, k] * dOut[i, j];
                    }
                    grads.BiasO_Grad[j] += dOut[i, j];

                    float sum = 0;
                    for (int k = 0; k < embeddingDim; k++)
                    {
                        sum += dOut[i, k] * attention.WO[j, k];
                    }
                    dConcatenated[i, j] = sum;
                }
            }

            var dQ_full = new float[seqLen, embeddingDim];
            var dK_full = new float[seqLen, embeddingDim];
            var dV_full = new float[seqLen, embeddingDim];

            float scale = 1.0f / MathF.Sqrt(headDim);

            for (int head = 0; head < numHeads; head++)
            {
                int startIdx = head * headDim;

                var Q_head = ExtractHead(cache.Q, startIdx, headDim, seqLen);
                var K_head = ExtractHead(cache.K, startIdx, headDim, seqLen);
                var V_head = ExtractHead(cache.V, startIdx, headDim, seqLen);
                var dHeadOutput = ExtractHead(dConcatenated, startIdx, headDim, seqLen);

                var scores = _accel.MatrixMultiplyTranspose(Q_head, K_head);
                var scaledScores = _accel.MatrixScale(scores, scale);

                var attnWeights = new float[seqLen, seqLen];
                for (int i = 0; i < seqLen; i++)
                {
                    float max = float.NegativeInfinity;
                    for (int j = 0; j < seqLen; j++)
                    {
                        if (_modelConfig.UseDecoderOnly && j > i)
                        {
                            continue;
                        }
                        max = Math.Max(max, scaledScores[i, j]);
                    }
                    float expSum = 0;
                    for (int j = 0; j < seqLen; j++)
                    {
                        if (_modelConfig.UseDecoderOnly && j > i)
                        {
                            attnWeights[i, j] = 0;
                            continue;
                        }
                        attnWeights[i, j] = MathF.Exp(scaledScores[i, j] - max);
                        expSum += attnWeights[i, j];
                    }
                    for (int j = 0; j < seqLen; j++)
                    {
                        attnWeights[i, j] /= (expSum + 1e-10f);
                    }
                }

                var dAttnWeights = new float[seqLen, seqLen];
                var dV_head = new float[seqLen, headDim];

                for (int i = 0; i < seqLen; i++)
                {
                    for (int j = 0; j < seqLen; j++)
                    {
                        float sum = 0;
                        for (int k = 0; k < headDim; k++)
                        {
                            sum += dHeadOutput[i, k] * V_head[j, k];
                        }
                        dAttnWeights[i, j] = sum;
                    }
                }

                for (int i = 0; i < seqLen; i++)
                {
                    for (int k = 0; k < headDim; k++)
                    {
                        float sum = 0;
                        for (int j = 0; j < seqLen; j++)
                        {
                            sum += attnWeights[j, i] * dHeadOutput[j, k];
                        }
                        dV_head[i, k] = sum;
                    }
                }

                var dScaledScores = new float[seqLen, seqLen];
                for (int i = 0; i < seqLen; i++)
                {
                    float dot = 0;
                    for (int j = 0; j < seqLen; j++)
                    {
                        dot += attnWeights[i, j] * dAttnWeights[i, j];
                    }

                    for (int j = 0; j < seqLen; j++)
                    {
                        dScaledScores[i, j] = attnWeights[i, j] * (dAttnWeights[i, j] - dot);
                        if (_modelConfig.UseDecoderOnly && j > i)
                        {
                            dScaledScores[i, j] = 0;
                        }
                    }
                }

                var dScores = new float[seqLen, seqLen];
                for (int i = 0; i < seqLen; i++)
                {
                    for (int j = 0; j < seqLen; j++)
                    {
                        dScores[i, j] = dScaledScores[i, j] * scale;
                    }
                }

                var dQ_head = new float[seqLen, headDim];
                var dK_head = new float[seqLen, headDim];

                for (int i = 0; i < seqLen; i++)
                {
                    for (int k = 0; k < headDim; k++)
                    {
                        float sum = 0;
                        for (int j = 0; j < seqLen; j++)
                        {
                            sum += dScores[i, j] * K_head[j, k];
                        }
                        dQ_head[i, k] = sum;
                    }
                }

                for (int j = 0; j < seqLen; j++)
                {
                    for (int k = 0; k < headDim; k++)
                    {
                        float sum = 0;
                        for (int i = 0; i < seqLen; i++)
                        {
                            sum += dScores[i, j] * Q_head[i, k];
                        }
                        dK_head[j, k] = sum;
                    }
                }

                for (int i = 0; i < seqLen; i++)
                {
                    for (int j = 0; j < headDim; j++)
                    {
                        dQ_full[i, startIdx + j] += dQ_head[i, j];
                        dK_full[i, startIdx + j] += dK_head[i, j];
                        dV_full[i, startIdx + j] += dV_head[i, j];
                    }
                }
            }

            var dInput = new float[seqLen, embeddingDim];

            BackpropLinearProjection(cache.Input, dQ_full, attention.WQ, grads.WQ_Grad, grads.BiasQ_Grad, dInput, seqLen, embeddingDim);

            BackpropLinearProjection(cache.Input, dK_full, attention.WK, grads.WK_Grad, grads.BiasK_Grad, dInput, seqLen, embeddingDim);

            BackpropLinearProjection(cache.Input, dV_full, attention.WV, grads.WV_Grad, grads.BiasV_Grad, dInput, seqLen, embeddingDim);

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

        private void UpdateParameters(float learningRate)
        {
            UpdateMatrix(_model.TokenEmbedding, _gradients.TokenEmbeddingGrad, learningRate);

            for (int i = 0; i < _modelConfig.NumLayers; i++)
            {
                var block = _model.Blocks[i];
                var attnGrad = _gradients.AttentionGrads[i];
                var ln1Grad = _gradients.LN1Grads[i];
                var ln2Grad = _gradients.LN2Grads[i];

                UpdateMatrix(block.Attention.WQ, attnGrad.WQ_Grad, learningRate);
                UpdateMatrix(block.Attention.WK, attnGrad.WK_Grad, learningRate);
                UpdateMatrix(block.Attention.WV, attnGrad.WV_Grad, learningRate);
                UpdateMatrix(block.Attention.WO, attnGrad.WO_Grad, learningRate);
                UpdateVector(block.Attention.BiasQ, attnGrad.BiasQ_Grad, learningRate);
                UpdateVector(block.Attention.BiasK, attnGrad.BiasK_Grad, learningRate);
                UpdateVector(block.Attention.BiasV, attnGrad.BiasV_Grad, learningRate);
                UpdateVector(block.Attention.BiasO, attnGrad.BiasO_Grad, learningRate);

                UpdateVector(block.LN1Gamma, ln1Grad.GammaGrad, learningRate);
                UpdateVector(block.LN1Beta, ln1Grad.BetaGrad, learningRate);
                UpdateVector(block.LN2Gamma, ln2Grad.GammaGrad, learningRate);
                UpdateVector(block.LN2Beta, ln2Grad.BetaGrad, learningRate);

                block.FeedForwardNetwork.ApplyExternalGradients(_ffnWeightGrads[i], _ffnBiasGrads[i], learningRate);
            }

            UpdateMatrix(_model.OutputProjection, _gradients.OutputProjectionGrad, learningRate);
            UpdateVector(_model.OutputBias, _gradients.OutputBiasGrad, learningRate);
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

        private float ComputeGradientNorm()
        {
            float sum = 0;

            sum += MatrixNorm(_gradients.TokenEmbeddingGrad);
            sum += MatrixNorm(_gradients.OutputProjectionGrad);
            sum += VectorNorm(_gradients.OutputBiasGrad);

            foreach (var g in _gradients.AttentionGrads)
            {
                sum += MatrixNorm(g.WQ_Grad);
                sum += MatrixNorm(g.WK_Grad);
                sum += MatrixNorm(g.WV_Grad);
                sum += MatrixNorm(g.WO_Grad);
                sum += VectorNorm(g.BiasQ_Grad);
                sum += VectorNorm(g.BiasK_Grad);
                sum += VectorNorm(g.BiasV_Grad);
                sum += VectorNorm(g.BiasO_Grad);
            }

            for (int layer = 0; layer < _modelConfig.NumLayers; layer++)
            {
                foreach (var wGrad in _ffnWeightGrads[layer])
                {
                    sum += MatrixNorm(wGrad);
                }
                foreach (var bGrad in _ffnBiasGrads[layer])
                {
                    sum += VectorNorm(bGrad);
                }
            }

            return MathF.Sqrt(sum);
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

        private void ScaleGradients(float scale)
        {
            ScaleMatrix(_gradients.TokenEmbeddingGrad, scale);
            ScaleMatrix(_gradients.OutputProjectionGrad, scale);
            ScaleVector(_gradients.OutputBiasGrad, scale);

            foreach (var g in _gradients.AttentionGrads)
            {
                ScaleMatrix(g.WQ_Grad, scale);
                ScaleMatrix(g.WK_Grad, scale);
                ScaleMatrix(g.WV_Grad, scale);
                ScaleMatrix(g.WO_Grad, scale);
                ScaleVector(g.BiasQ_Grad, scale);
                ScaleVector(g.BiasK_Grad, scale);
                ScaleVector(g.BiasV_Grad, scale);
                ScaleVector(g.BiasO_Grad, scale);
            }

            for (int layer = 0; layer < _modelConfig.NumLayers; layer++)
            {
                foreach (var wGrad in _ffnWeightGrads[layer])
                {
                    ScaleMatrix(wGrad, scale);
                }
                foreach (var bGrad in _ffnBiasGrads[layer])
                {
                    ScaleVector(bGrad, scale);
                }
            }
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
        private void ZeroAllGradients()
        {
            _gradients.Zero();

            for (int layer = 0; layer < _modelConfig.NumLayers; layer++)
            {
                foreach (var wGrad in _ffnWeightGrads[layer])
                {
                    ZeroMatrix(wGrad);
                }
                foreach (var bGrad in _ffnBiasGrads[layer])
                {
                    Array.Clear(bGrad, 0, bGrad.Length);
                }
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

        private float[,] MatMulWithBias(float[,] input, float[,] weights, float[] bias)
        {
            int seqLen = input.GetLength(0);
            int outputDim = weights.GetLength(1);
            var result = new float[seqLen, outputDim];

            for (int i = 0; i < seqLen; i++)
            {
                var inputRow = new float[input.GetLength(1)];
                for (int k = 0; k < input.GetLength(1); k++)
                {
                    inputRow[k] = input[i, k];
                }

                var outputRow = _accel.CalculateDotProduct(weights, inputRow);

                for (int j = 0; j < outputDim; j++)
                {
                    result[i, j] = outputRow[j] + bias[j];
                }
            }
            return result;
        }

        private float[,] ProjectToVocab(float[,] hidden)
        {
            int seqLen = hidden.GetLength(0);
            var logits = new float[seqLen, _modelConfig.VocabSize];

            for (int i = 0; i < seqLen; i++)
            {
                var inputRow = new float[_modelConfig.EmbeddingDim];
                for (int k = 0; k < _modelConfig.EmbeddingDim; k++)
                {
                    inputRow[k] = hidden[i, k];
                }

                var outputRow = _accel.CalculateDotProduct(_model.OutputProjection, inputRow);

                for (int j = 0; j < _modelConfig.VocabSize; j++)
                {
                    logits[i, j] = outputRow[j] + _model.OutputBias[j];
                }
            }
            return logits;
        }
    }
}