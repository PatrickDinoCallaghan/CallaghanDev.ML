using CallaghanDev.ML.AccelerationManagers;
using CallaghanDev.ML.Transformers.Cache;
using CallaghanDev.ML.Transformers.Configuration;
using ILGPU.IR.Values;
using System;
using System.Collections.Generic;
using System.Linq;

namespace CallaghanDev.ML.Transformers.CrossAttentionMultimodal
{
    public class Trainer
    {
        private readonly Model _model;
        private readonly MultimodalTransformerConfig _config;
        private readonly  TrainingConfig _trainConfig;
        private readonly Gradients _gradients;
        private readonly IAccelerationManager _accel;
        private readonly Random _random;

        private readonly List<List<float[,]>> _textFFNWeightGrads;
        private readonly List<List<float[]>> _textFFNBiasGrads;

        private readonly List<List<float[,]>> _priceFFNWeightGrads;
        private readonly List<List<float[]>> _priceFFNBiasGrads;

        public Trainer(Model model,  TrainingConfig trainConfig)
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
            for (int i = 0; i < _config.Text.NumLayers; i++)
            {
                var (wGrads, bGrads) = model.TextBlocks[i].FeedForwardNetwork.CreateGradientStorage();
                _textFFNWeightGrads.Add(wGrads);
                _textFFNBiasGrads.Add(bGrads);
            }

            _priceFFNWeightGrads = new List<List<float[,]>>();
            _priceFFNBiasGrads = new List<List<float[]>>();
            for (int i = 0; i < _config.Price.NumLayers; i++)
            {
                var (wGrads, bGrads) = model.PriceBlocks[i].FeedForwardNetwork.CreateGradientStorage();
                _priceFFNWeightGrads.Add(wGrads);
                _priceFFNBiasGrads.Add(bGrads);
            }
        }

        public void Train(int[][] textSequences, float[][,] priceInputs, float[][,] priceTargets, float[][] confidenceTargets = null)
        {
            int totalSamples = ValidateDatasetAndGetCount(textSequences, priceInputs, priceTargets, confidenceTargets);

            if (textSequences == null)
                textSequences = new int[totalSamples][];

            if (_trainConfig.BatchSize <= 0)
                throw new ArgumentOutOfRangeException(nameof(_trainConfig.BatchSize), "BatchSize must be greater than zero.");

            if (_trainConfig.Epochs <= 0 || totalSamples == 0)
                return;

            float currentLR = _trainConfig.LearningRate;

            for (int epoch = 0; epoch < _trainConfig.Epochs; epoch++)
            {
                if (_trainConfig.Verbose)
                    Console.WriteLine($"\n=== Epoch {epoch + 1}/{_trainConfig.Epochs} ===");

                var shuffled = Enumerable.Range(0, totalSamples).OrderBy(_ => _random.Next()).ToArray();
                float epochLoss = 0f;
                int numBatches = 0;

                for (int i = 0; i < shuffled.Length; i += _trainConfig.BatchSize)
                {
                    int batchSize = Math.Min(_trainConfig.BatchSize, shuffled.Length - i);
                    var batchIndices = shuffled.Skip(i).Take(batchSize).ToArray();

                    float batchLoss = TrainBatch(batchIndices, textSequences, priceInputs, priceTargets, confidenceTargets, currentLR);
                    epochLoss += batchLoss;
                    numBatches++;

                    if (_trainConfig.Verbose && numBatches % 10 == 0)
                        Console.WriteLine($"  Batch {numBatches}: Loss = {batchLoss:F6}");
                }

                if (_trainConfig.Verbose)
                {
                    float avgLoss = numBatches > 0 ? epochLoss / numBatches : 0f;
                    Console.WriteLine($"  Epoch {epoch + 1} Average Loss: {avgLoss:F6}");
                }

                if (_trainConfig.UseLearningRateDecay)
                {
                    currentLR *= _trainConfig.LearningRateDecay;

                    if (_trainConfig.Verbose)
                        Console.WriteLine($"  Learning rate: {currentLR:F8}");
                }
            }
        }
        public float Validate(int[][] textSequences, float[][,] priceInputs, float[][,] priceTargets, float[][] confidenceTargets = null)
        {
            int totalSamples = ValidateDatasetAndGetCount(textSequences, priceInputs, priceTargets, confidenceTargets);

            if (textSequences == null)
                textSequences = new int[totalSamples][];

            float totalMse = 0f;
            int timestepCount = 0;

            float totalConfidenceLoss = 0f;
            int confidenceCount = 0;

            for (int idx = 0; idx < totalSamples; idx++)
            {
                var priceSeq = priceInputs[idx];
                int seqLen = priceSeq.GetLength(0);
                if (seqLen < 2)
                    continue;

                var inputSlice = SliceRows(priceSeq, 0, seqLen - 1);
                var targetSlice = SliceRows(priceTargets[idx], 1, seqLen);

                var (predictions, confidence) = _model.Forward(textSequences[idx], inputSlice);

                int effectiveLen = predictions.GetLength(0);
                for (int t = 0; t < effectiveLen; t++)
                {
                    for (int j = 0; j < _config.Output.OutputDim; j++)
                    {
                        float diff = predictions[t, j] - targetSlice[t, j];
                        totalMse += diff * diff;
                    }

                    timestepCount++;
                }

                // Only include confidence validation when explicit confidence labels are provided.
                if (_config.Output.UseConfidenceHead && confidenceTargets != null && confidenceTargets[idx] != null && confidence != null)
                {
                    for (int t = 0; t < effectiveLen; t++)
                    {
                        float target = Math.Clamp(confidenceTargets[idx][t + 1], 0f, 1f);
                        float pred = Math.Clamp(confidence[t, 0], 1e-7f, 1.0f - 1e-7f);
                        totalConfidenceLoss -= target * MathF.Log(pred) + (1f - target) * MathF.Log(1f - pred);
                        confidenceCount++;
                    }
                }
            }

            if (timestepCount == 0)
                return 0f;

            float mse = totalMse / (timestepCount * _config.Output.OutputDim);

            if (confidenceCount > 0)
                mse += _trainConfig.ConfidenceLossWeight * (totalConfidenceLoss / confidenceCount);

            return mse;
        }
        private float TrainBatch(
       int[] batchIndices,
       int[][] allText,
       float[][,] allPriceInputs,
       float[][,] allPriceTargets,
       float[][] allConfTargets,
       float learningRate)
        {
            ZeroAllGradients();

            float totalLoss = 0f;
            int validCount = 0;

            foreach (int idx in batchIndices)
            {
                var textTokens = allText == null ? null : allText[idx];
                var priceSeq = allPriceInputs[idx];
                int seqLen = priceSeq.GetLength(0);

                if (seqLen < 2)
                    continue;

                var inputSlice = SliceRows(priceSeq, 0, seqLen - 1);
                var targetSlice = SliceRows(allPriceTargets[idx], 1, seqLen);

                float[] confTargetSlice = null;
                if (allConfTargets != null && allConfTargets[idx] != null)
                    confTargetSlice = SliceConfidenceTargets(allConfTargets[idx], 1, seqLen);

                try
                {
                    var cache = new MultimodalForwardCache(_config.Text.NumLayers, _config.Price.NumLayers);
                    var (predictions, confidence) = _model.ForwardWithCache(textTokens, inputSlice, cache);

                    if (!IsFinite(predictions) || (confidence != null && !IsFinite(confidence)))
                        throw new InvalidOperationException($"Non-finite forward output for sample {idx}.");

                    float loss = BackwardPass(predictions, confidence, targetSlice, confTargetSlice, cache);

                    if (!float.IsFinite(loss))
                        throw new InvalidOperationException($"Non-finite loss for sample {idx}: {loss}.");

                    totalLoss += loss;
                    validCount++;
                }
                catch (Exception ex)
                {
                    ZeroAllGradients();
                    throw new InvalidOperationException($"Training failed on sample index {idx}.", ex);
                }
            }

            if (validCount == 0)
                return 0f;

            ScaleAllGradients(1.0f / validCount);

            float gradNorm = ComputeGradientNorm();
            if (!float.IsFinite(gradNorm))
            {
                ZeroAllGradients();
                throw new InvalidOperationException($"Non-finite gradient norm before update: {gradNorm}.");
            }

            if (_trainConfig.UseGradientClipping && _trainConfig.GradientClipThreshold > 0f)
                ClipGradients(_trainConfig.GradientClipThreshold);

            UpdateAllParameters(learningRate);
            return totalLoss / validCount;
        }
        private bool IsFinite(float[,] matrix)
        {
            if (matrix == null)
                return true;
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
            if (predictions == null)
                throw new ArgumentNullException(nameof(predictions));
            if (targets == null)
                throw new ArgumentNullException(nameof(targets));
            if (cache == null)
                throw new ArgumentNullException(nameof(cache));

            int seqLen = predictions.GetLength(0);
            int outputDim = _config.Output.OutputDim;
            int embDim = _config.Price.EmbeddingDim;

            if (targets.GetLength(0) < seqLen)
                throw new ArgumentException($"Targets have {targets.GetLength(0)} rows but predictions have {seqLen} rows.", nameof(targets));
            if (targets.GetLength(1) < outputDim)
                throw new ArgumentException($"Targets have {targets.GetLength(1)} columns but output dim is {outputDim}.", nameof(targets));

            float mseLoss = 0f;
            var dPredictions = new float[seqLen, outputDim];
            float mseGradScale = 2.0f / (seqLen * outputDim);

            for (int t = 0; t < seqLen; t++)
            {
                for (int j = 0; j < outputDim; j++)
                {
                    float diff = predictions[t, j] - targets[t, j];
                    mseLoss += diff * diff;
                    dPredictions[t, j] = diff * mseGradScale;
                }
            }

            mseLoss /= (seqLen * outputDim);

            var dHidden = _accel.BackpropOutputProjection(
                dPredictions,
                cache.PriceFinalHidden,
                _model.OutputProjection,
                _gradients.OutputProjectionGrad,
                _gradients.OutputBiasGrad,
                seqLen,
                outputDim,
                embDim);

            float confLoss = 0f;

            if (_config.Output.UseConfidenceHead && _trainConfig.ConfidenceLossWeight > 0f)
            {
                if (confidence == null)
                    throw new InvalidOperationException("Confidence head is enabled but confidence output is null.");
                if (confidence.GetLength(0) < seqLen)
                    throw new ArgumentException($"Confidence has {confidence.GetLength(0)} rows but predictions have {seqLen} rows.", nameof(confidence));
                if (confTargets != null && confTargets.Length < seqLen)
                    throw new ArgumentException($"Confidence targets length {confTargets.Length} is shorter than sequence length {seqLen}.", nameof(confTargets));

                float confidenceGradScale = _trainConfig.ConfidenceLossWeight / seqLen;

                for (int t = 0; t < seqLen; t++)
                {
                    float rawPred = confidence[t, 0];
                    float clampedPred = Math.Clamp(rawPred, 1e-7f, 1.0f - 1e-7f);
                    float target;

                    if (confTargets != null)
                    {
                        target = Math.Clamp(confTargets[t], 0f, 1f);
                    }
                    else
                    {
                        float errMag = 0f;
                        for (int j = 0; j < outputDim; j++)
                        {
                            float diff = predictions[t, j] - targets[t, j];
                            errMag += diff * diff;
                        }

                        errMag = MathF.Sqrt(errMag / outputDim);
                        target = MathF.Exp(-errMag * 5.0f);
                    }

                    confLoss -= target * MathF.Log(clampedPred) + (1f - target) * MathF.Log(1f - clampedPred);

                    // BCE-with-sigmoid derivative: sigmoid(logit) - target.
                    // Critical fix: multiply by ConfidenceLossWeight, since totalLoss does.
                    float dLogit = (rawPred - target) * confidenceGradScale;

                    for (int e = 0; e < embDim; e++)
                    {
                        _gradients.ConfidenceProjectionGrad[0, e] += dLogit * cache.PriceFinalHidden[t, e];
                        dHidden[t, e] += dLogit * _model.ConfidenceProjection[0, e];
                    }

                    _gradients.ConfidenceBiasGrad[0] += dLogit;
                }

                confLoss /= seqLen;
            }

            float totalLoss = mseLoss + (_config.Output.UseConfidenceHead ? _trainConfig.ConfidenceLossWeight * confLoss : 0f);

            bool hasText = cache.TextFinalHidden != null;
            var dTextHidden = BackpropPriceDecoder(dHidden, cache, hasText);

            if (!_config.Text.Freeze && hasText && dTextHidden != null)
                BackpropTextEncoder(dTextHidden, cache);

            return totalLoss;
        }
        private float[,] BackpropPriceDecoder(float[,] dOut, MultimodalForwardCache cache, bool hasText)
        {
            int embDim = _config.Price.EmbeddingDim;
            int numHeads = _config.Price.NumHeads;
            int headDim = embDim / numHeads;
            float scale = 1.0f / MathF.Sqrt(headDim);

            // Accumulate gradient w.r.t. text hidden states across all layers
            float[,] dTextHiddenTotal = hasText
     ? new float[cache.TextFinalHidden.GetLength(0), embDim]
     : null;
            int textSeqLen = 0;
            if (hasText)
            {
                textSeqLen = cache.TextFinalHidden.GetLength(0);
                dTextHiddenTotal = new float[textSeqLen, embDim];
            }

            var dX = dOut;

            for (int layer = _config.Price.NumLayers - 1; layer >= 0; layer--)
            {
                var block = _model.PriceBlocks[layer];
                var blockCache = cache.PriceBlockCaches[layer];
                var blockGrads = _gradients.PriceBlockGrads[layer];

                // ================================================
                // Backprop through FFN LayerNorm
                // ================================================
                var (dFFResidual, dLNFFNGamma, dLNFFNBeta) = _accel.LayerNormBackward(dX, blockCache.LNFFNCache.Normalized, block.LNFFNGamma, blockCache.LNFFNCache.Input, blockCache.LNFFNCache.Mean, blockCache.LNFFNCache.Variance);
        
                 _accel.AccumulateVectorGradients(blockGrads.LNFFNGrads.GammaGrad, dLNFFNGamma);
                _accel.AccumulateVectorGradients(blockGrads.LNFFNGrads.BetaGrad, dLNFFNBeta);
 
                int seqLen = dFFResidual.GetLength(0);
                var dNormedCross_from_ffn = new float[seqLen, embDim];
                for (int i = 0; i < seqLen; i++)
                {
                    var dOutRow = _accel.ExtractRow(dFFResidual, i, embDim);
                    var ffnInput = blockCache.FFNInputRows[i];
                    if (ffnInput == null)
                        throw new Exception($"FFNInputRows[{i}] is null (layer {layer})");

                    block.FeedForwardNetwork.ForwardPassOnly(ffnInput);
                    var dInputRow = block.FeedForwardNetwork.ComputeInputGradient(dOutRow, _priceFFNWeightGrads[layer], _priceFFNBiasGrads[layer]);
                    _accel.SetRow(dNormedCross_from_ffn, i, dInputRow, embDim);
                }


                var dNormedCross = _accel.MatrixAdd(dFFResidual, dNormedCross_from_ffn);

                // ================================================
                // Backprop through Cross-Attention LayerNorm
                // ================================================
                var (dCrossResidual, dLNCrossGamma, dLNCrossBeta) = _accel.LayerNormBackward(dNormedCross, blockCache.LNCrossCache.Normalized, block.LNCrossGamma, blockCache.LNCrossCache.Input, blockCache.LNCrossCache.Mean, blockCache.LNCrossCache.Variance);


                for (int j = 0; j < embDim; j++)
                {
                    blockGrads.LNCrossGrads.GammaGrad[j] += dLNCrossGamma[j];
                    blockGrads.LNCrossGrads.BetaGrad[j] += dLNCrossBeta[j];
                }

                float[,] dNormedSelf;
                if (hasText)
                {
                    // Cross attention was active: full backprop
                    var crossAttnGrads = blockGrads.CrossAttnGrads;

                    // Backprop through WO projection
                    var dCrossConcatenated = new float[seqLen, embDim];

                    _accel.BackpropLinearProjection(blockCache.CrossAttnOutput, dCrossResidual, block.CrossAttention.WO, crossAttnGrads.WO_Grad, crossAttnGrads.BiasO_Grad, dCrossConcatenated);

                    var (dCrossQ, dCrossK, dCrossV) = _accel.MultiHeadAttentionBackward(
            blockCache.CrossQ, blockCache.CrossK, blockCache.CrossV,
            dCrossConcatenated, numHeads, scale, false);

                    RotaryPositionEmbedding.ApplyBackwardInPlace(dCrossQ, dCrossK, numHeads);

                    var dNormedSelf_from_cross = new float[seqLen, embDim];

                    _accel.BackpropLinearProjection(blockCache.NormedSelf, dCrossQ, block.CrossAttention.WQ, crossAttnGrads.WQ_Grad, crossAttnGrads.BiasQ_Grad, dNormedSelf_from_cross);

                    // Text hidden gradients from cross-attn K and V projections
                    var dTextFromK = new float[textSeqLen, embDim];
                    _accel.BackpropLinearProjection(cache.TextFinalHidden, dCrossK, block.CrossAttention.WK, crossAttnGrads.WK_Grad, crossAttnGrads.BiasK_Grad, dTextFromK);

                    var dTextFromV = new float[textSeqLen, embDim];
                    _accel.BackpropLinearProjection(cache.TextFinalHidden, dCrossV, block.CrossAttention.WV, crossAttnGrads.WV_Grad, crossAttnGrads.BiasV_Grad, dTextFromV);


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
               dSelfConcatenated, numHeads, scale, _config.Price.UseDecoderOnly);

                RotaryPositionEmbedding.ApplyBackwardInPlace(dSelfQ, dSelfK, numHeads);

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

            _accel.BackpropInputProjection(dX, continuousInput,
            _gradients.PriceInputProjectionGrad, _gradients.PriceInputProjectionBiasGrad, seqLen, _config.Price.EmbeddingDim, _config.Price.InputFeatureDim);
        }

        private void BackpropTextEncoder(float[,] dTextHidden, MultimodalForwardCache cache)
        {
            int embDim = _config.Text.EmbeddingDim;
            int numHeads = _config.Text.NumHeads;
            int headDim = embDim / numHeads;

            var dX = dTextHidden;

            for (int layer = _config.Text.NumLayers - 1; layer >= 0; layer--)
            {
                var block = _model.TextBlocks[layer];
                var textAttnGrads = _gradients.TextAttnGrads[layer];

                // LayerNorm 2 backward
                var ln2Cache = cache.TextLN2Caches[layer];
                var (dFFResidual, dLN2Gamma, dLN2Beta) = _accel.LayerNormBackward(dX, ln2Cache.Normalized, block.LN2Gamma, ln2Cache.Input, ln2Cache.Mean, ln2Cache.Variance);
                var ln2Grads = _gradients.TextLN2Grads[layer];
     

                _accel.AccumulateVectorGradients(ln2Grads.GammaGrad, dLN2Gamma);
                _accel.AccumulateVectorGradients(ln2Grads.BetaGrad, dLN2Beta);


                // FFN backward
                int seqLen = dFFResidual.GetLength(0);
                var dNormed1_from_ffn = new float[seqLen, embDim];
         


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

                _accel.AccumulateVectorGradients(ln1Grads.GammaGrad, dLN1Gamma);
                _accel.AccumulateVectorGradients(ln1Grads.BetaGrad, dLN1Beta);

                // Attention backward
                var attnCache = cache.TextAttentionCaches[layer];
                float scale = 1.0f / MathF.Sqrt(headDim);

                var dConcatenated = new float[seqLen, embDim];

                _accel.BackpropLinearProjection(attnCache.AttentionOutput, dAttnResidual, block.Attention.WO, textAttnGrads.WO_Grad, textAttnGrads.BiasO_Grad, dConcatenated);

                var (dQ, dK, dV) = _accel.MultiHeadAttentionBackward(
                   attnCache.Q, attnCache.K, attnCache.V,
                   dConcatenated, numHeads, scale, _config.Text.UseDecoderOnly);

                RotaryPositionEmbedding.ApplyBackwardInPlace(dQ, dK, numHeads);

                var dInput = new float[seqLen, embDim];

                _accel.BackpropLinearProjection(attnCache.Input, dQ, block.Attention.WQ, textAttnGrads.WQ_Grad, textAttnGrads.BiasQ_Grad, dInput);
                _accel.BackpropLinearProjection(attnCache.Input, dK, block.Attention.WK, textAttnGrads.WK_Grad, textAttnGrads.BiasK_Grad, dInput);
                _accel.BackpropLinearProjection(attnCache.Input, dV, block.Attention.WV, textAttnGrads.WV_Grad, textAttnGrads.BiasV_Grad, dInput);

                dX = _accel.MatrixAdd(dAttnResidual, dInput);
            }
            int textSeqLen = dX.GetLength(0);
     
            _accel.AccumulateTokenEmbeddingGrad(_gradients.TextEmbeddingGrad, dX, cache.TextTokenIds, textSeqLen, embDim);
        }

        private void UpdateAllParameters(float lr)
        {
            // --- Text Encoder ---
            if (!_config.Text.Freeze)
            {
                _accel.MatrixUpdate(_model.TextTokenEmbedding, _gradients.TextEmbeddingGrad, lr);

                for (int i = 0; i < _config.Text.NumLayers; i++)
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

            for (int i = 0; i < _config.Price.NumLayers; i++)
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

            if (_config.Output.UseConfidenceHead)
            {
                _accel.MatrixUpdate(_model.ConfidenceProjection, _gradients.ConfidenceProjectionGrad, lr);
                _accel.VectorUpdate(_model.ConfidenceBias, _gradients.ConfidenceBiasGrad, lr);
            }
        }


        private void ZeroAllGradients()
        {
            _gradients.Zero();

            for (int i = 0; i < _config.Text.NumLayers; i++)
            {
                foreach (var wg in _textFFNWeightGrads[i]) _accel.ZeroMatrix(wg);
                foreach (var bg in _textFFNBiasGrads[i]) Array.Clear(bg, 0, bg.Length);
            }
            for (int i = 0; i < _config.Price.NumLayers; i++)
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

            if (!_config.Text.Freeze)
            {
                sum += _accel.MatrixSquaredNorm(_gradients.TextEmbeddingGrad);
                foreach (var g in _gradients.TextAttnGrads)
                {
                    sum += _accel.MatrixSquaredNorm(g.WQ_Grad) + _accel.MatrixSquaredNorm(g.WK_Grad)
                         + _accel.MatrixSquaredNorm(g.WV_Grad) + _accel.MatrixSquaredNorm(g.WO_Grad);
                    sum += VectorNorm(g.BiasQ_Grad) + VectorNorm(g.BiasK_Grad)
                         + VectorNorm(g.BiasV_Grad) + VectorNorm(g.BiasO_Grad);
                }
                foreach (var g in _gradients.TextLN1Grads)
                {
                    sum += VectorNorm(g.GammaGrad) + VectorNorm(g.BetaGrad); 
                }
                foreach (var g in _gradients.TextLN2Grads) 
                { 
                    sum += VectorNorm(g.GammaGrad) + VectorNorm(g.BetaGrad);
                }
                for (int i = 0; i < _config.Text.NumLayers; i++)
                {
                    foreach (var wg in _textFFNWeightGrads[i])
                    {
                        sum += _accel.MatrixSquaredNorm(wg);
                    }
                    foreach (var bg in _textFFNBiasGrads[i])
                    {
                        sum += VectorNorm(bg);
                    }
                }
            }

            sum += _accel.MatrixSquaredNorm(_gradients.PriceInputProjectionGrad);
            sum += VectorNorm(_gradients.PriceInputProjectionBiasGrad);

            foreach (var bg in _gradients.PriceBlockGrads)
            {
                foreach (var g in new[] { bg.SelfAttnGrads, bg.CrossAttnGrads })
                {
                    sum += _accel.MatrixSquaredNorm(g.WQ_Grad) + _accel.MatrixSquaredNorm(g.WK_Grad) + _accel.MatrixSquaredNorm(g.WV_Grad) + _accel.MatrixSquaredNorm(g.WO_Grad);
                    sum += VectorNorm(g.BiasQ_Grad) + VectorNorm(g.BiasK_Grad)  + VectorNorm(g.BiasV_Grad) + VectorNorm(g.BiasO_Grad);
                }
                sum += VectorNorm(bg.LNSelfGrads.GammaGrad) + VectorNorm(bg.LNSelfGrads.BetaGrad);
                sum += VectorNorm(bg.LNCrossGrads.GammaGrad) + VectorNorm(bg.LNCrossGrads.BetaGrad);
                sum += VectorNorm(bg.LNFFNGrads.GammaGrad) + VectorNorm(bg.LNFFNGrads.BetaGrad);
            }
            for (int i = 0; i < _config.Price.NumLayers; i++)
            {
                foreach (var wg in _priceFFNWeightGrads[i])
                {
                    sum += _accel.MatrixSquaredNorm(wg);
                }

                foreach (var bg in _priceFFNBiasGrads[i])
                {
                    sum += VectorNorm(bg);
                }
            }

            sum += _accel.MatrixSquaredNorm(_gradients.OutputProjectionGrad);
            sum += VectorNorm(_gradients.OutputBiasGrad);

            if (_config.Output.UseConfidenceHead)
            {
                sum += _accel.MatrixSquaredNorm(_gradients.ConfidenceProjectionGrad);
                sum += VectorNorm(_gradients.ConfidenceBiasGrad);
            }

            return MathF.Sqrt(sum);
        }

        private void ScaleAllGradients(float scale)
        {
            if (!_config.Text.Freeze)
            {
                _accel.MatrixScaleInPlace(_gradients.TextEmbeddingGrad, scale);

                foreach (var g in _gradients.TextAttnGrads)
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
                foreach (var g in _gradients.TextLN1Grads)
                {
                    _accel.VectorScaleInPlace(g.GammaGrad, scale);
                    _accel.VectorScaleInPlace(g.BetaGrad, scale); 
                }
                foreach (var g in _gradients.TextLN2Grads)
                { 
                    _accel.VectorScaleInPlace(g.GammaGrad, scale);
                    _accel.VectorScaleInPlace(g.BetaGrad, scale);
                }
                for (int i = 0; i < _config.Text.NumLayers; i++)
                {
                    foreach (var wg in _textFFNWeightGrads[i])
                    {
                        _accel.MatrixScaleInPlace(wg, scale);
                    }
                    foreach (var bg in _textFFNBiasGrads[i])
                    {
                        _accel.VectorScaleInPlace(bg, scale);
                    }
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
            for (int i = 0; i < _config.Price.NumLayers; i++)
            {
                foreach (var wg in _priceFFNWeightGrads[i]) _accel.MatrixScaleInPlace(wg, scale);
                foreach (var bg in _priceFFNBiasGrads[i]) _accel.VectorScaleInPlace(bg, scale);
            }

            _accel.MatrixScaleInPlace(_gradients.OutputProjectionGrad, scale);
            _accel.VectorScaleInPlace(_gradients.OutputBiasGrad, scale);

            if (_config.Output.UseConfidenceHead)
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


        private int ValidateDatasetAndGetCount(
    int[][] textSequences,
    float[][,] priceInputs,
    float[][,] priceTargets,
    float[][] confidenceTargets)
        {
            if (priceInputs == null)
                throw new ArgumentNullException(nameof(priceInputs));
            if (priceTargets == null)
                throw new ArgumentNullException(nameof(priceTargets));

            int totalSamples = priceInputs.Length;

            if (priceTargets.Length != totalSamples)
                throw new ArgumentException("priceInputs and priceTargets must contain the same number of samples.", nameof(priceTargets));

            if (textSequences != null && textSequences.Length != totalSamples)
                throw new ArgumentException("textSequences must be null or contain the same number of samples as priceInputs.", nameof(textSequences));

            if (confidenceTargets != null && confidenceTargets.Length != totalSamples)
                throw new ArgumentException("confidenceTargets must be null or contain the same number of samples as priceInputs.", nameof(confidenceTargets));

            for (int i = 0; i < totalSamples; i++)
            {
                ValidateSampleOrThrow(
                    i,
                    textSequences == null ? null : textSequences[i],
                    priceInputs[i],
                    priceTargets[i],
                    confidenceTargets == null ? null : confidenceTargets[i]);
            }

            return totalSamples;
        }

        private void ValidateSampleOrThrow(int sampleIndex, int[] textTokens, float[,] priceInput, float[,] priceTarget, float[] confidenceTarget)
        {
            if (priceInput == null)
                throw new ArgumentNullException(nameof(priceInput), $"priceInputs[{sampleIndex}] is null.");
            if (priceTarget == null)
                throw new ArgumentNullException(nameof(priceTarget), $"priceTargets[{sampleIndex}] is null.");

            int priceRows = priceInput.GetLength(0);
            int priceFeatures = priceInput.GetLength(1);

            if (priceFeatures != _config.Price.InputFeatureDim)
            {
                throw new ArgumentException(
                    $"priceInputs[{sampleIndex}] has feature dim {priceFeatures}; expected {_config.Price.InputFeatureDim}.",
                    nameof(priceInput));
            }

            // Training slices price input to [0..N-2], so the actual model input length is N - 1.
            if (priceRows > 1 && priceRows - 1 > _config.Price.MaxSequenceLength)
            {
                throw new ArgumentException(
                    $"priceInputs[{sampleIndex}] produces model input length {priceRows - 1}; max is {_config.Price.MaxSequenceLength}.",
                    nameof(priceInput));
            }

            if (priceTarget.GetLength(0) < priceRows)
            {
                throw new ArgumentException(
                    $"priceTargets[{sampleIndex}] must have at least {priceRows} rows for next-step slicing.",
                    nameof(priceTarget));
            }

            if (priceTarget.GetLength(1) < _config.Output.OutputDim)
            {
                throw new ArgumentException(
                    $"priceTargets[{sampleIndex}] has output dim {priceTarget.GetLength(1)}; expected at least {_config.Output.OutputDim}.",
                    nameof(priceTarget));
            }

            if (confidenceTarget != null && confidenceTarget.Length < priceRows)
            {
                throw new ArgumentException(
                    $"confidenceTargets[{sampleIndex}] must have at least {priceRows} values for next-step slicing.",
                    nameof(confidenceTarget));
            }

            if (textTokens != null)
            {
                if (textTokens.Length > _config.Text.MaxSequenceLength)
                {
                    throw new ArgumentException(
                        $"textSequences[{sampleIndex}] length {textTokens.Length} exceeds max {_config.Text.MaxSequenceLength}.",
                        nameof(textTokens));
                }

                for (int i = 0; i < textTokens.Length; i++)
                {
                    int tokenId = textTokens[i];
                    if ((uint)tokenId >= (uint)_config.Text.VocabSize)
                    {
                        throw new ArgumentOutOfRangeException(
                            nameof(textTokens),
                            $"textSequences[{sampleIndex}][{i}] token id {tokenId} is outside vocab size {_config.Text.VocabSize}.");
                    }
                }
            }
        }

        private static float[] SliceConfidenceTargets(float[] source, int startRow, int endRow)
        {
            if (source == null)
                return null;

            int count = endRow - startRow;
            if (count < 0)
                throw new ArgumentException("endRow must be greater than or equal to startRow.");
            if (startRow < 0 || endRow > source.Length)
                throw new ArgumentOutOfRangeException(nameof(source), $"Cannot slice confidence targets [{startRow}, {endRow}) from length {source.Length}.");

            var result = new float[count];
            Array.Copy(source, startRow, result, 0, count);
            return result;
        }
    }
}
