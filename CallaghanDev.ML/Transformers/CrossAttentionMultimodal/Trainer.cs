using CallaghanDev.ML.AccelerationManagers;
using CallaghanDev.ML.Transformers.Cache;
using ILGPU.IR.Values;
using System;
using System.Collections.Generic;
using System.Linq;

namespace CallaghanDev.ML.Transformers.CrossAttentionMultimodal
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

        public Trainer(Model model, MultimodalTrainingConfig trainConfig)
        {
            _model = model;
            _config = model.Config;
            _trainConfig = trainConfig;
            _gradients = new Gradients(_config);
            _accel = model.AccelerationManager;
            _random = new Random();

            // Allocate FFN gradient storage
            _textFFNWeightGrads = new List<List<float[,]>>();
            _textFFNBiasGrads = new List<List<float[]>>();
            for (int i = 0; i < _config.TextNumLayers; i++)
            {
                var (wGrads, bGrads) = model.TextBlocks[i].FeedForwardNetwork.CreateGradientStorage();
                _textFFNWeightGrads.Add(wGrads);
                _textFFNBiasGrads.Add(bGrads);
            }

            _priceFFNWeightGrads = new List<List<float[,]>>();
            _priceFFNBiasGrads = new List<List<float[]>>();
            for (int i = 0; i < _config.PriceNumLayers; i++)
            {
                var (wGrads, bGrads) = model.PriceBlocks[i].FeedForwardNetwork.CreateGradientStorage();
                _priceFFNWeightGrads.Add(wGrads);
                _priceFFNBiasGrads.Add(bGrads);
            }
        }

        public void Train(int[][] textSequences, float[][,] priceInputs, float[][,] priceTargets, float[][] confidenceTargets = null)
        {
            int totalSamples = textSequences.Length;

            if (priceInputs.Length != totalSamples || priceTargets.Length != totalSamples)
            {
                throw new ArgumentException("All input arrays must have the same number of samples.");
            }

            float currentLR = _trainConfig.LearningRate;

            for (int epoch = 0; epoch < _trainConfig.Epochs; epoch++)
            {
                if (_trainConfig.Verbose)
                {
                    Console.WriteLine($"\n=== Epoch {epoch + 1}/{_trainConfig.Epochs} ===");
                }

                var shuffled = Enumerable.Range(0, totalSamples).OrderBy(_ => _random.Next()).ToArray();
                float epochLoss = 0;
                int numBatches = 0;

                for (int i = 0; i < shuffled.Length; i += _trainConfig.BatchSize)
                {
                    int batchSize = Math.Min(_trainConfig.BatchSize, shuffled.Length - i);
                    var batchIndices = shuffled.Skip(i).Take(batchSize).ToArray();

                    float batchLoss = TrainBatch(batchIndices, textSequences, priceInputs, priceTargets, confidenceTargets, currentLR);
                    epochLoss += batchLoss;
                    numBatches++;

                    if (_trainConfig.Verbose && numBatches % 10 == 0)
                    {
                        Console.WriteLine($"  Batch {numBatches}: Loss = {batchLoss:F6}");
                    }
                }

                float avgLoss = numBatches > 0 ? epochLoss / numBatches : 0;
                if (_trainConfig.Verbose)
                {
                    Console.WriteLine($"  Epoch {epoch + 1} Average Loss: {avgLoss:F6}");
                }

                if (_trainConfig.UseLearningRateDecay)
                {
                    currentLR *= _trainConfig.LearningRateDecay;
                    if (_trainConfig.Verbose)
                    {
                        Console.WriteLine($"  Learning rate: {currentLR:F8}");
                    }
                }
            }
        }

        public float Validate(int[][] textSequences, float[][,] priceInputs, float[][,] priceTargets, float[][] confidenceTargets = null)
        {
            float totalLoss = 0;
            int count = 0;

            for (int idx = 0; idx < textSequences.Length; idx++)
            {
                var priceSeq = priceInputs[idx];
                int seqLen = priceSeq.GetLength(0);
                if (seqLen < 2) continue;

                var inputSlice = SliceRows(priceSeq, 0, seqLen - 1);
                var targetSlice = SliceRows(priceTargets[idx], 1, seqLen);

                var (predictions, confidence) = _model.Forward(textSequences[idx], inputSlice);

                int effectiveLen = predictions.GetLength(0);
                for (int t = 0; t < effectiveLen; t++)
                {
                    for (int j = 0; j < _config.OutputDim; j++)
                    {
                        float diff = predictions[t, j] - targetSlice[t, j];
                        totalLoss += diff * diff;
                    }
                    count++;
                }
            }

            return count > 0 ? totalLoss / (count * _config.OutputDim) : 0;
        }

        private float TrainBatch(int[] batchIndices, int[][] allText, float[][,] allPriceInputs, float[][,] allPriceTargets, float[][] allConfTargets, float learningRate)
        {
            ZeroAllGradients();
            float totalLoss = 0;
            int validCount = 0;

            foreach (int idx in batchIndices)
            {
                var textTokens = allText[idx];
                var priceSeq = allPriceInputs[idx];
                int seqLen = priceSeq.GetLength(0);
                if (seqLen < 2) continue;

                // Slice: input[0..N-2], target[1..N-1]
                var inputSlice = SliceRows(priceSeq, 0, seqLen - 1);
                var targetSlice = SliceRows(allPriceTargets[idx], 1, seqLen);

                float[] confTargetSlice = null;
                if (allConfTargets != null && allConfTargets[idx] != null)
                {
                    confTargetSlice = allConfTargets[idx].Skip(1).Take(seqLen - 1).ToArray();
                }

                try
                {
                    var cache = new MultimodalForwardCache(_config.TextNumLayers, _config.PriceNumLayers);
                    var (predictions, confidence) = _model.ForwardWithCache(textTokens, inputSlice, cache);

                    if (!IsFinite(predictions) || !IsFinite(confidence))
                    {
                        continue;
                    }

                    float loss = BackwardPass(predictions, confidence, targetSlice, confTargetSlice, cache);

                    if (float.IsNaN(loss) || float.IsInfinity(loss))
                    {
                        continue;
                    }
                    if (!float.IsFinite(loss))
                    {
                        ZeroAllGradients();
                        return 0f;
                    }

                    totalLoss += loss;

                    validCount++;
                }
                catch (Exception ex)
                {
                    ZeroAllGradients();
                    if (_trainConfig.Verbose)
                    {
                        Console.WriteLine($"  WARNING: Training error on sample {idx}: {ex.Message}");
                    }
                    //Dont train this batch at all
                    return 0f; 
                }
            }

            if (validCount == 0) return 0f;

            float invCount = 1.0f / validCount;
            ScaleAllGradients(invCount);

            if (_trainConfig.UseGradientClipping)
                ClipGradients(_trainConfig.GradientClipThreshold);

            UpdateAllParameters(learningRate);
            return totalLoss / validCount;
        }
        private bool IsFinite(float[,] matrix)
        {
            int rows = matrix.GetLength(0), cols = matrix.GetLength(1);
            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < cols; j++)
                {

                    if (!float.IsFinite(matrix[i, j])) 
                    {
                        return false;
                    }
                }
            }
            return true;
        }

        private float BackwardPass(float[,] predictions, float[,] confidence, float[,] targets, float[] confTargets, MultimodalForwardCache cache)
        {
            int seqLen = predictions.GetLength(0);
            int outputDim = _config.OutputDim;
            int embDim = _config.PriceEmbeddingDim;

            // Compute MSE loss and gradient.
            float mseLoss = 0;
            var dPredictions = new float[seqLen, outputDim];

            for (int t = 0; t < seqLen; t++)
            {
                for (int j = 0; j < outputDim; j++)
                {
                    float diff = predictions[t, j] - targets[t, j];
                    mseLoss += diff * diff;
                    dPredictions[t, j] = 2.0f * diff / (seqLen * outputDim);
                }
            }
            mseLoss /= (seqLen * outputDim);

            // Compute confidence loss (BCE) and gradient
            float confLoss = 0;
            /* var dHidden = new float[seqLen, embDim]; // gradient to accumulate into hidden states

              for (int t = 0; t < seqLen; t++)
              {
                  // Gradient to output projection weights and bias
                  for (int v = 0; v < outputDim; v++)
                  {
                      for (int e = 0; e < embDim; e++)
                          _gradients.OutputProjectionGrad[v, e] += cache.PriceFinalHidden[t, e] * dPredictions[t, v];
                      _gradients.OutputBiasGrad[v] += dPredictions[t, v];
                  }

                  // Gradient to hidden states
                  for (int e = 0; e < embDim; e++)
                  {
                      float grad = 0;
                      for (int v = 0; v < outputDim; v++)
                          grad += dPredictions[t, v] * _model.OutputProjection[v, e];
                      dHidden[t, e] += grad;
                  }
              }*/

            // Backprop through output projection: dHidden += dPredictions at OutputProjection
            var dHidden = _accel.BackpropOutputProjection(dPredictions, cache.PriceFinalHidden, _model.OutputProjection, _gradients.OutputProjectionGrad, _gradients.OutputBiasGrad, seqLen, outputDim, embDim);
 
            if (_config.UseConfidenceHead && confidence != null)
            {
                for (int t = 0; t < seqLen; t++)
                {
                    float pred = confidence[t, 0];
                    float target;

                    if (confTargets != null)
                    {
                        target = confTargets[t];
                    }
                    else
                    {
                        // Auto-generate confidence target: low error => high confidence
                        float errMag = 0;
                        for (int j = 0; j < outputDim; j++)
                        {
                            float diff = predictions[t, j] - targets[t, j];
                            errMag += diff * diff;
                        }
                        errMag = MathF.Sqrt(errMag / outputDim);
                        target = MathF.Exp(-errMag * 5.0f); // exponential decay
                    }

                    // BCE loss: -[y*log(p) + (1-y)*log(1-p)]
                    float clampedPred = Math.Clamp(pred, 1e-7f, 1.0f - 1e-7f);
                    confLoss -= target * MathF.Log(clampedPred) + (1 - target) * MathF.Log(1 - clampedPred);

                    // BCE gradient through sigmoid: dL/dlogit = pred - target
                    float dLogit = (pred - target) * _trainConfig.ConfidenceLossWeight / seqLen;

                    for (int e = 0; e < embDim; e++)
                    {
                        _gradients.ConfidenceProjectionGrad[0, e] += dLogit * cache.PriceFinalHidden[t, e];
                        dHidden[t, e] += dLogit * _model.ConfidenceProjection[0, e];
                    }
                    _gradients.ConfidenceBiasGrad[0] += dLogit;
                }
                confLoss /= seqLen;
            }

            float totalLoss = mseLoss + _trainConfig.ConfidenceLossWeight * confLoss;

            bool hasText = cache.TextFinalHidden != null;
            var dTextHidden = BackpropPriceDecoder(dHidden, cache, hasText);

            // ---- Backprop through text encoder (unless frozen or no text) ----
            if (!_config.FreezeTextEncoder && hasText && dTextHidden != null)
            {
                BackpropTextEncoder(dTextHidden, cache);
            }

            return totalLoss;
        }

        private float[,] BackpropPriceDecoder(float[,] dOut, MultimodalForwardCache cache, bool hasText)
        {
            int embDim = _config.PriceEmbeddingDim;
            int numHeads = _config.PriceNumHeads;
            int headDim = embDim / numHeads;
            float scale = 1.0f / MathF.Sqrt(headDim);

            // Accumulate gradient w.r.t. text hidden states across all layers
            float[,] dTextHiddenTotal = null;
            int textSeqLen = 0;
            if (hasText)
            {
                textSeqLen = cache.TextFinalHidden.GetLength(0);
                dTextHiddenTotal = new float[textSeqLen, embDim];
            }

            var dX = dOut;

            for (int layer = _config.PriceNumLayers - 1; layer >= 0; layer--)
            {
                var block = _model.PriceBlocks[layer];
                var blockCache = cache.PriceBlockCaches[layer];
                var blockGrads = _gradients.PriceBlockGrads[layer];

                // ================================================
                // Backprop through FFN LayerNorm
                // ================================================
                var (dFFResidual, dLNFFNGamma, dLNFFNBeta) = _accel.LayerNormBackward(dX, blockCache.LNFFNCache.Normalized, block.LNFFNGamma, blockCache.LNFFNCache.Input, blockCache.LNFFNCache.Mean, blockCache.LNFFNCache.Variance);
                /*
                for (int j = 0; j < embDim; j++)
                {
                    blockGrads.LNFFNGrads.GammaGrad[j] += dLNFFNGamma[j];
                    blockGrads.LNFFNGrads.BetaGrad[j] += dLNFFNBeta[j];
                }*/

                 _accel.AccumulateVectorGradients(blockGrads.LNFFNGrads.GammaGrad, dLNFFNGamma);
                  _accel.AccumulateVectorGradients(blockGrads.LNFFNGrads.BetaGrad, dLNFFNBeta);

                // Backprop through FFN (residual split)
                /*int seqLen = dFFResidual.GetLength(0);
                var dNormedCross_from_ffn = new float[seqLen, embDim];

                for (int i = 0; i < seqLen; i++)
                {
                    var dOutRow = new float[embDim];

                    for (int j = 0; j < embDim; j++)
                    {
                        dOutRow[j] = dFFResidual[i, j];
                    }

                    block.FeedForwardNetwork.ForwardPassOnly(blockCache.FFNInputRows[i]);
                    var dInputRow = block.FeedForwardNetwork.ComputeInputGradient(dOutRow, _priceFFNWeightGrads[layer], _priceFFNBiasGrads[layer]);

                    for (int j = 0; j < embDim; j++)
                    {
                        dNormedCross_from_ffn[i, j] = dInputRow[j];
                    }
                }*/
                int seqLen = dFFResidual.GetLength(0);
                var dNormedCross_from_ffn = new float[seqLen, embDim];
                for (int i = 0; i < seqLen; i++)
                {
                    var dOutRow = _accel.ExtractRow(dFFResidual, i, embDim);
                    block.FeedForwardNetwork.ForwardPassOnly(blockCache.FFNInputRows[i]);
                    var dInputRow = block.FeedForwardNetwork.ComputeInputGradient(dOutRow, _priceFFNWeightGrads[layer], _priceFFNBiasGrads[layer]);
                    _accel.SetRow(dNormedCross_from_ffn, i, dInputRow, embDim);
                }


                var dNormedCross = _accel.MatrixAdd(dFFResidual, dNormedCross_from_ffn);

                // ================================================
                // 2. Backprop through Cross-Attention LayerNorm
                // ================================================
                var (dCrossResidual, dLNCrossGamma, dLNCrossBeta) = _accel.LayerNormBackward(
                    dNormedCross, blockCache.LNCrossCache.Normalized, block.LNCrossGamma,
                    blockCache.LNCrossCache.Input, blockCache.LNCrossCache.Mean, blockCache.LNCrossCache.Variance);


                for (int j = 0; j < embDim; j++)
                {
                    blockGrads.LNCrossGrads.GammaGrad[j] += dLNCrossGamma[j];
                    blockGrads.LNCrossGrads.BetaGrad[j] += dLNCrossBeta[j];
                }

                float[,] dNormedSelf;

                if (hasText && blockCache.CrossQ != null)
                {
                    // Cross attention was active: full backprop
                    var crossAttnGrads = blockGrads.CrossAttnGrads;

                    // Backprop through WO projection
                    var dCrossConcatenated = new float[seqLen, embDim];

                    _accel.BackpropLinearProjection(blockCache.CrossAttnOutput, dCrossResidual, block.CrossAttention.WO, crossAttnGrads.WO_Grad, crossAttnGrads.BiasO_Grad, dCrossConcatenated);

                    var (dCrossQ, dCrossK, dCrossV) = _accel.MultiHeadAttentionBackward(blockCache.CrossQ, blockCache.CrossK, blockCache.CrossV, dCrossConcatenated, numHeads, scale, false);

                    var dNormedSelf_from_cross = new float[seqLen, embDim];

                    _accel.BackpropLinearProjection(blockCache.NormedSelf, dCrossQ, block.CrossAttention.WQ, crossAttnGrads.WQ_Grad, crossAttnGrads.BiasQ_Grad, dNormedSelf_from_cross);

                    // Text hidden gradients from cross-attn K and V projections
                    var dTextFromK = new float[textSeqLen, embDim];
                    _accel.BackpropLinearProjection(cache.TextFinalHidden, dCrossK, block.CrossAttention.WK, crossAttnGrads.WK_Grad, crossAttnGrads.BiasK_Grad, dTextFromK);

                    var dTextFromV = new float[textSeqLen, embDim];
                    _accel.BackpropLinearProjection(cache.TextFinalHidden, dCrossV, block.CrossAttention.WV, crossAttnGrads.WV_Grad, crossAttnGrads.BiasV_Grad, dTextFromV);

                    /* for (int i = 0; i < textSeqLen; i++)
                         for (int j = 0; j < embDim; j++)
                             dTextHiddenTotal[i, j] += dTextFromK[i, j] + dTextFromV[i, j];*/

                    // Accumulate text gradients
                    _accel.MatrixAccumulate(dTextHiddenTotal, dTextFromK);
                    _accel.MatrixAccumulate(dTextHiddenTotal, dTextFromV);
                    dNormedSelf = _accel.MatrixAdd(dCrossResidual, dNormedSelf_from_cross);
                }
                else
                {
                    // Cross-attention was skipped: gradient passes straight through 
                    dNormedSelf = dCrossResidual;
                }

                // ================================================
                // Backprop through Self-Attention LayerNorm
                // ================================================
                var (dSelfResidual, dLNSelfGamma, dLNSelfBeta) = _accel.LayerNormBackward(dNormedSelf, blockCache.LNSelfCache.Normalized, block.LNSelfGamma, blockCache.LNSelfCache.Input, blockCache.LNSelfCache.Mean, blockCache.LNSelfCache.Variance);
                
                for (int j = 0; j < embDim; j++)
                {
                    blockGrads.LNSelfGrads.GammaGrad[j] += dLNSelfGamma[j];
                    blockGrads.LNSelfGrads.BetaGrad[j] += dLNSelfBeta[j];
                }

                // Backprop through self-attention
                var selfAttnGrads = blockGrads.SelfAttnGrads;
                var dSelfConcatenated = new float[seqLen, embDim];

                _accel.BackpropLinearProjection(blockCache.SelfAttnOutput, dSelfResidual, block.SelfAttention.WO, selfAttnGrads.WO_Grad, selfAttnGrads.BiasO_Grad, dSelfConcatenated);

                var (dSelfQ, dSelfK, dSelfV) = _accel.MultiHeadAttentionBackward(
                    blockCache.SelfQ, blockCache.SelfK, blockCache.SelfV,
                    dSelfConcatenated, numHeads, scale, _config.PriceUseDecoderOnly);

                var dBlockInput = new float[seqLen, embDim];
                _accel.BackpropLinearProjection(blockCache.BlockInput, dSelfQ,
                    block.SelfAttention.WQ, selfAttnGrads.WQ_Grad, selfAttnGrads.BiasQ_Grad, dBlockInput);
                _accel.BackpropLinearProjection(blockCache.BlockInput, dSelfK,
                    block.SelfAttention.WK, selfAttnGrads.WK_Grad, selfAttnGrads.BiasK_Grad, dBlockInput);
                _accel.BackpropLinearProjection(blockCache.BlockInput, dSelfV,
                    block.SelfAttention.WV, selfAttnGrads.WV_Grad, selfAttnGrads.BiasV_Grad, dBlockInput);

                // Residual connection: dX_prev = dSelfResidual + dBlockInput
                dX = _accel.MatrixAdd(dSelfResidual, dBlockInput);
            }

            // Backprop through price input projection
            BackpropPriceInputProjection(dX, cache.PriceContinuousInput);

            return dTextHiddenTotal;
        }
        private void BackpropPriceInputProjection(float[,] dX, float[,] continuousInput)
        {
            int seqLen = dX.GetLength(0);
            /*int embDim = _config.PriceEmbeddingDim;
            int inputDim = _config.PriceInputFeatureDim;

            for (int i = 0; i < seqLen; i++)
            {
                for (int e = 0; e < embDim; e++)
                {
                    float dVal = dX[i, e];
                    for (int f = 0; f < inputDim; f++)
                        _gradients.PriceInputProjectionGrad[e, f] += dVal * continuousInput[i, f];
                    _gradients.PriceInputProjectionBiasGrad[e] += dVal;
                }
            }*/


            _accel.BackpropInputProjection(dX, continuousInput,
            _gradients.PriceInputProjectionGrad, _gradients.PriceInputProjectionBiasGrad, seqLen, _config.PriceEmbeddingDim, _config.PriceInputFeatureDim);
        }

        private void BackpropTextEncoder(float[,] dTextHidden, MultimodalForwardCache cache)
        {
            int embDim = _config.TextEmbeddingDim;
            int numHeads = _config.TextNumHeads;
            int headDim = embDim / numHeads;

            var dX = dTextHidden;

            for (int layer = _config.TextNumLayers - 1; layer >= 0; layer--)
            {
                var block = _model.TextBlocks[layer];
                var textAttnGrads = _gradients.TextAttnGrads[layer];

                // LayerNorm 2 backward
                var ln2Cache = cache.TextLN2Caches[layer];
                var (dFFResidual, dLN2Gamma, dLN2Beta) = _accel.LayerNormBackward(dX, ln2Cache.Normalized, block.LN2Gamma, ln2Cache.Input, ln2Cache.Mean, ln2Cache.Variance);
                var ln2Grads = _gradients.TextLN2Grads[layer];
                /*
                for (int j = 0; j < dLN2Gamma.Length; j++)
                {
                    ln2Grads.GammaGrad[j] += dLN2Gamma[j];
                    ln2Grads.BetaGrad[j] += dLN2Beta[j];
                }*/
                _accel.AccumulateVectorGradients(ln2Grads.GammaGrad, dLN2Gamma);
                _accel.AccumulateVectorGradients(ln2Grads.BetaGrad, dLN2Beta);


                // FFN backward
                int seqLen = dFFResidual.GetLength(0);
                var dNormed1_from_ffn = new float[seqLen, embDim];
                /*
                for (int i = 0; i < seqLen; i++)
                {
                    var dOutRow = new float[embDim];

                    for (int j = 0; j < embDim; j++)
                    {
                        dOutRow[j] = dFFResidual[i, j];
                    }

                    block.FeedForwardNetwork.ForwardPassOnly(cache.TextFFNInputs[layer][i]);
                    var dInputRow = block.FeedForwardNetwork.ComputeInputGradient(dOutRow, _textFFNWeightGrads[layer], _textFFNBiasGrads[layer]);

                    for (int j = 0; j < embDim; j++)
                    {
                        dNormed1_from_ffn[i, j] = dInputRow[j];
                    }
                }*/


                for (int i = 0; i < seqLen; i++)
                {
                    var dOutRow = _accel.ExtractRow(dFFResidual, i, embDim);
                    block.FeedForwardNetwork.ForwardPassOnly(cache.TextFFNInputs[layer][i]);
                    var dInputRow = block.FeedForwardNetwork.ComputeInputGradient(dOutRow, _textFFNWeightGrads[layer], _textFFNBiasGrads[layer]);
                    _accel.SetRow(dNormed1_from_ffn, i, dInputRow, embDim);
                }
                var dNormed1 = _accel.MatrixAdd(dFFResidual, dNormed1_from_ffn);


                // LayerNorm 1 backward
                var ln1Cache = cache.TextLN1Caches[layer];
                var (dAttnResidual, dLN1Gamma, dLN1Beta) = _accel.LayerNormBackward(dNormed1, ln1Cache.Normalized, block.LN1Gamma, ln1Cache.Input, ln1Cache.Mean, ln1Cache.Variance);
                var ln1Grads = _gradients.TextLN1Grads[layer];

                /*
                for (int j = 0; j < dLN1Gamma.Length; j++)
                {
                    ln1Grads.GammaGrad[j] += dLN1Gamma[j];
                    ln1Grads.BetaGrad[j] += dLN1Beta[j];
                }*/

                _accel.AccumulateVectorGradients(ln1Grads.GammaGrad, dLN1Gamma);
                _accel.AccumulateVectorGradients(ln1Grads.BetaGrad, dLN1Beta);

                // Attention backward
                var attnCache = cache.TextAttentionCaches[layer];
                float scale = 1.0f / MathF.Sqrt(headDim);

                var dConcatenated = new float[seqLen, embDim];

                _accel.BackpropLinearProjection(attnCache.AttentionOutput, dAttnResidual, block.Attention.WO, textAttnGrads.WO_Grad, textAttnGrads.BiasO_Grad, dConcatenated);

                var (dQ, dK, dV) = _accel.MultiHeadAttentionBackward(attnCache.Q, attnCache.K, attnCache.V, dConcatenated, numHeads, scale, _config.TextUseDecoderOnly);

                var dInput = new float[seqLen, embDim];

                _accel.BackpropLinearProjection(attnCache.Input, dQ, block.Attention.WQ, textAttnGrads.WQ_Grad, textAttnGrads.BiasQ_Grad, dInput);
                _accel.BackpropLinearProjection(attnCache.Input, dK, block.Attention.WK, textAttnGrads.WK_Grad, textAttnGrads.BiasK_Grad, dInput);
                _accel.BackpropLinearProjection(attnCache.Input, dV, block.Attention.WV, textAttnGrads.WV_Grad, textAttnGrads.BiasV_Grad, dInput);

                dX = _accel.MatrixAdd(dAttnResidual, dInput);
            }
            int textSeqLen = dX.GetLength(0);
            /*
            for (int i = 0; i < textSeqLen; i++)
            {
                int tokenId = cache.TextTokenIds[i];
                for (int j = 0; j < embDim; j++)
                    _gradients.TextEmbeddingGrad[tokenId, j] += dX[i, j];
            }
            */
            // Backprop into text token embeddings
            _accel.AccumulateTokenEmbeddingGrad(_gradients.TextEmbeddingGrad, dX, cache.TextTokenIds, textSeqLen, embDim);
        }

        private void UpdateAllParameters(float lr)
        {
            // --- Text Encoder ---
            if (!_config.FreezeTextEncoder)
            {
                _accel.MatrixUpdate(_model.TextTokenEmbedding, _gradients.TextEmbeddingGrad, lr);

                for (int i = 0; i < _config.TextNumLayers; i++)
                {
                    var block = _model.TextBlocks[i];
                    var attnGrad = _gradients.TextAttnGrads[i];
                    var ln1Grad = _gradients.TextLN1Grads[i];
                    var ln2Grad = _gradients.TextLN2Grads[i];

                    _accel.MatrixUpdate(block.Attention.WQ, attnGrad.WQ_Grad, lr);
                    _accel.MatrixUpdate(block.Attention.WK, attnGrad.WK_Grad, lr);
                    _accel.MatrixUpdate(block.Attention.WV, attnGrad.WV_Grad, lr);
                    _accel.MatrixUpdate(block.Attention.WO, attnGrad.WO_Grad, lr);
                    _accel.VectorUpdate(block.Attention.BiasQ, attnGrad.BiasQ_Grad, lr);
                    _accel.VectorUpdate(block.Attention.BiasK, attnGrad.BiasK_Grad, lr);
                    _accel.VectorUpdate(block.Attention.BiasV, attnGrad.BiasV_Grad, lr);
                    _accel.VectorUpdate(block.Attention.BiasO, attnGrad.BiasO_Grad, lr);

                    _accel.VectorUpdate(block.LN1Gamma, ln1Grad.GammaGrad, lr);
                    _accel.VectorUpdate(block.LN1Beta, ln1Grad.BetaGrad, lr);
                    _accel.VectorUpdate(block.LN2Gamma, ln2Grad.GammaGrad, lr);
                    _accel.VectorUpdate(block.LN2Beta, ln2Grad.BetaGrad, lr);

                    block.FeedForwardNetwork.ApplyExternalGradients(_textFFNWeightGrads[i], _textFFNBiasGrads[i], lr);
                }
            }

            // --- Price Decoder ---
            _accel.MatrixUpdate(_model.PriceInputProjection, _gradients.PriceInputProjectionGrad, lr);
            _accel.VectorUpdate(_model.PriceInputProjectionBias, _gradients.PriceInputProjectionBiasGrad, lr);

            for (int i = 0; i < _config.PriceNumLayers; i++)
            {
                var block = _model.PriceBlocks[i];
                var blockGrads = _gradients.PriceBlockGrads[i];

                // Self-attention
                var saGrads = blockGrads.SelfAttnGrads;
                _accel.MatrixUpdate(block.SelfAttention.WQ, saGrads.WQ_Grad, lr);
                _accel.MatrixUpdate(block.SelfAttention.WK, saGrads.WK_Grad, lr);
                _accel.MatrixUpdate(block.SelfAttention.WV, saGrads.WV_Grad, lr);
                _accel.MatrixUpdate(block.SelfAttention.WO, saGrads.WO_Grad, lr);
                _accel.VectorUpdate(block.SelfAttention.BiasQ, saGrads.BiasQ_Grad, lr);
                _accel.VectorUpdate(block.SelfAttention.BiasK, saGrads.BiasK_Grad, lr);
                _accel.VectorUpdate(block.SelfAttention.BiasV, saGrads.BiasV_Grad, lr);
                _accel.VectorUpdate(block.SelfAttention.BiasO, saGrads.BiasO_Grad, lr);

                // Cross-attention
                var caGrads = blockGrads.CrossAttnGrads;
                _accel.MatrixUpdate(block.CrossAttention.WQ, caGrads.WQ_Grad, lr);
                _accel.MatrixUpdate(block.CrossAttention.WK, caGrads.WK_Grad, lr);
                _accel.MatrixUpdate(block.CrossAttention.WV, caGrads.WV_Grad, lr);
                _accel.MatrixUpdate(block.CrossAttention.WO, caGrads.WO_Grad, lr);
                _accel.VectorUpdate(block.CrossAttention.BiasQ, caGrads.BiasQ_Grad, lr);
                _accel.VectorUpdate(block.CrossAttention.BiasK, caGrads.BiasK_Grad, lr);
                _accel.VectorUpdate(block.CrossAttention.BiasV, caGrads.BiasV_Grad, lr);
                _accel.VectorUpdate(block.CrossAttention.BiasO, caGrads.BiasO_Grad, lr);

                // LayerNorms
                _accel.VectorUpdate(block.LNSelfGamma, blockGrads.LNSelfGrads.GammaGrad, lr);
                _accel.VectorUpdate(block.LNSelfBeta, blockGrads.LNSelfGrads.BetaGrad, lr);
                _accel.VectorUpdate(block.LNCrossGamma, blockGrads.LNCrossGrads.GammaGrad, lr);
                _accel.VectorUpdate(block.LNCrossBeta, blockGrads.LNCrossGrads.BetaGrad, lr);
                _accel.VectorUpdate(block.LNFFNGamma, blockGrads.LNFFNGrads.GammaGrad, lr);
                _accel.VectorUpdate(block.LNFFNBeta, blockGrads.LNFFNGrads.BetaGrad, lr);

                // FFN
                block.FeedForwardNetwork.ApplyExternalGradients(_priceFFNWeightGrads[i], _priceFFNBiasGrads[i], lr);
            }

            // --- Output Head ---
            _accel.MatrixUpdate(_model.OutputProjection, _gradients.OutputProjectionGrad, lr);
            _accel.VectorUpdate(_model.OutputBias, _gradients.OutputBiasGrad, lr);

            if (_config.UseConfidenceHead)
            {
                _accel.MatrixUpdate(_model.ConfidenceProjection, _gradients.ConfidenceProjectionGrad, lr);
                _accel.VectorUpdate(_model.ConfidenceBias, _gradients.ConfidenceBiasGrad, lr);
            }
        }


        private void ZeroAllGradients()
        {
            _gradients.Zero();

            for (int i = 0; i < _config.TextNumLayers; i++)
            {
                foreach (var wg in _textFFNWeightGrads[i]) _accel.ZeroMatrix(wg);
                foreach (var bg in _textFFNBiasGrads[i]) Array.Clear(bg, 0, bg.Length);
            }
            for (int i = 0; i < _config.PriceNumLayers; i++)
            {
                foreach (var wg in _priceFFNWeightGrads[i]) _accel.ZeroMatrix(wg);
                foreach (var bg in _priceFFNBiasGrads[i]) Array.Clear(bg, 0, bg.Length);
            }
        }

        private void ClipGradients(float threshold)
        {
            float norm = ComputeGradientNorm();
            if (norm > threshold)
                ScaleAllGradients(threshold / norm);
        }

        private float ComputeGradientNorm()
        {
            float sum = 0;

            if (!_config.FreezeTextEncoder)
            {
                sum += _accel.MatrixSquaredNorm(_gradients.TextEmbeddingGrad);
                foreach (var g in _gradients.TextAttnGrads)
                {
                    sum += _accel.MatrixSquaredNorm(g.WQ_Grad) + _accel.MatrixSquaredNorm(g.WK_Grad)
                         + _accel.MatrixSquaredNorm(g.WV_Grad) + _accel.MatrixSquaredNorm(g.WO_Grad);
                    sum += VectorNorm(g.BiasQ_Grad) + VectorNorm(g.BiasK_Grad)
                         + VectorNorm(g.BiasV_Grad) + VectorNorm(g.BiasO_Grad);
                }
                foreach (var g in _gradients.TextLN1Grads) { sum += VectorNorm(g.GammaGrad) + VectorNorm(g.BetaGrad); }
                foreach (var g in _gradients.TextLN2Grads) { sum += VectorNorm(g.GammaGrad) + VectorNorm(g.BetaGrad); }
                for (int i = 0; i < _config.TextNumLayers; i++)
                {
                    foreach (var wg in _textFFNWeightGrads[i]) sum += _accel.MatrixSquaredNorm(wg);
                    foreach (var bg in _textFFNBiasGrads[i]) sum += VectorNorm(bg);
                }
            }

            sum += _accel.MatrixSquaredNorm(_gradients.PriceInputProjectionGrad);
            sum += VectorNorm(_gradients.PriceInputProjectionBiasGrad);

            foreach (var bg in _gradients.PriceBlockGrads)
            {
                foreach (var g in new[] { bg.SelfAttnGrads, bg.CrossAttnGrads })
                {
                    sum += _accel.MatrixSquaredNorm(g.WQ_Grad) + _accel.MatrixSquaredNorm(g.WK_Grad)
                         + _accel.MatrixSquaredNorm(g.WV_Grad) + _accel.MatrixSquaredNorm(g.WO_Grad);
                    sum += VectorNorm(g.BiasQ_Grad) + VectorNorm(g.BiasK_Grad)
                         + VectorNorm(g.BiasV_Grad) + VectorNorm(g.BiasO_Grad);
                }
                sum += VectorNorm(bg.LNSelfGrads.GammaGrad) + VectorNorm(bg.LNSelfGrads.BetaGrad);
                sum += VectorNorm(bg.LNCrossGrads.GammaGrad) + VectorNorm(bg.LNCrossGrads.BetaGrad);
                sum += VectorNorm(bg.LNFFNGrads.GammaGrad) + VectorNorm(bg.LNFFNGrads.BetaGrad);
            }
            for (int i = 0; i < _config.PriceNumLayers; i++)
            {
                foreach (var wg in _priceFFNWeightGrads[i]) sum += _accel.MatrixSquaredNorm(wg);
                foreach (var bg in _priceFFNBiasGrads[i]) sum += VectorNorm(bg);
            }

            sum += _accel.MatrixSquaredNorm(_gradients.OutputProjectionGrad);
            sum += VectorNorm(_gradients.OutputBiasGrad);

            if (_config.UseConfidenceHead)
            {
                sum += _accel.MatrixSquaredNorm(_gradients.ConfidenceProjectionGrad);
                sum += VectorNorm(_gradients.ConfidenceBiasGrad);
            }

            return MathF.Sqrt(sum);
        }

        private void ScaleAllGradients(float scale)
        {
            if (!_config.FreezeTextEncoder)
            {
                _accel.MatrixScaleInPlace(_gradients.TextEmbeddingGrad, scale);
                foreach (var g in _gradients.TextAttnGrads)
                {
                    _accel.MatrixScaleInPlace(g.WQ_Grad, scale); _accel.MatrixScaleInPlace(g.WK_Grad, scale);
                    _accel.MatrixScaleInPlace(g.WV_Grad, scale); _accel.MatrixScaleInPlace(g.WO_Grad, scale);
                    _accel.VectorScaleInPlace(g.BiasQ_Grad, scale); _accel.VectorScaleInPlace(g.BiasK_Grad, scale);
                    _accel.VectorScaleInPlace(g.BiasV_Grad, scale); _accel.VectorScaleInPlace(g.BiasO_Grad, scale);
                }
                foreach (var g in _gradients.TextLN1Grads) { _accel.VectorScaleInPlace(g.GammaGrad, scale); _accel.VectorScaleInPlace(g.BetaGrad, scale); }
                foreach (var g in _gradients.TextLN2Grads) { _accel.VectorScaleInPlace(g.GammaGrad, scale); _accel.VectorScaleInPlace(g.BetaGrad, scale); }
                for (int i = 0; i < _config.TextNumLayers; i++)
                {
                    foreach (var wg in _textFFNWeightGrads[i]) _accel.MatrixScaleInPlace(wg, scale);
                    foreach (var bg in _textFFNBiasGrads[i]) _accel.VectorScaleInPlace(bg, scale);
                }
            }

            _accel.MatrixScaleInPlace(_gradients.PriceInputProjectionGrad, scale);
            _accel.VectorScaleInPlace(_gradients.PriceInputProjectionBiasGrad, scale);

            foreach (var bg in _gradients.PriceBlockGrads)
            {
                foreach (var g in new[] { bg.SelfAttnGrads, bg.CrossAttnGrads })
                {
                    _accel.MatrixScaleInPlace(g.WQ_Grad, scale); _accel.MatrixScaleInPlace(g.WK_Grad, scale);
                    _accel.MatrixScaleInPlace(g.WV_Grad, scale); _accel.MatrixScaleInPlace(g.WO_Grad, scale);
                    _accel.VectorScaleInPlace(g.BiasQ_Grad, scale); _accel.VectorScaleInPlace(g.BiasK_Grad, scale);
                    _accel.VectorScaleInPlace(g.BiasV_Grad, scale); _accel.VectorScaleInPlace(g.BiasO_Grad, scale);
                }
                _accel.VectorScaleInPlace(bg.LNSelfGrads.GammaGrad, scale); _accel.VectorScaleInPlace(bg.LNSelfGrads.BetaGrad, scale);
                _accel.VectorScaleInPlace(bg.LNCrossGrads.GammaGrad, scale); _accel.VectorScaleInPlace(bg.LNCrossGrads.BetaGrad, scale);
                _accel.VectorScaleInPlace(bg.LNFFNGrads.GammaGrad, scale); _accel.VectorScaleInPlace(bg.LNFFNGrads.BetaGrad, scale);
            }
            for (int i = 0; i < _config.PriceNumLayers; i++)
            {
                foreach (var wg in _priceFFNWeightGrads[i]) _accel.MatrixScaleInPlace(wg, scale);
                foreach (var bg in _priceFFNBiasGrads[i]) _accel.VectorScaleInPlace(bg, scale);
            }

            _accel.MatrixScaleInPlace(_gradients.OutputProjectionGrad, scale);
            _accel.VectorScaleInPlace(_gradients.OutputBiasGrad, scale);

            if (_config.UseConfidenceHead)
            {
                _accel.MatrixScaleInPlace(_gradients.ConfidenceProjectionGrad, scale);
                _accel.VectorScaleInPlace(_gradients.ConfidenceBiasGrad, scale);
            }
        }


        private float VectorNorm(float[] v)
        {
            /*
            float sum = 0;
            for (int i = 0; i < v.Length; i++) sum += v[i] * v[i];
            return sum;*/

            return _accel.VectorSquaredNorm(v);
        }

        private float[,] SliceRows(float[,] matrix, int startRow, int endRow)
        {
            /*
            int cols = matrix.GetLength(1);
            int numRows = endRow - startRow;
            var result = new float[numRows, cols];
            for (int i = 0; i < numRows; i++)
                for (int j = 0; j < cols; j++)
                    result[i, j] = matrix[startRow + i, j];
            return result;*/

            return _accel.SliceRows(matrix, startRow, endRow);
        }

    }
}
