using CallaghanDev.ML.AccelerationManagers;
using CallaghanDev.ML.Enums;
using CallaghanDev.ML.Transformers.Cache;
using CallaghanDev.ML.Transformers.Configuration;
using CallaghanDev.ML.Transformers.MultiTypeTransformer;


namespace CallaghanDev.ML.Transformers.CrossAttentionMultimodal
{
    /// <summary>
    /// Cross-Attention Multimodal Transformer for combining text (news/sentiment)
    /// with time series (price data) to predict OHLCV + confidence.
    /// </summary>
    public class Model
    {
        private readonly MultimodalTransformerConfig _config;
        private readonly Random _random;
        private readonly IAccelerationManager _accel;

        public MultimodalTransformerConfig Config => _config;
        public IAccelerationManager AccelerationManager => _accel;
        public float[,] TextTokenEmbedding { get; set; }
        public MultiTypeTransformer.TransformerBlock[] TextBlocks { get; set; }
        public float[,] PriceInputProjection { get; set; }
        public float[] PriceInputProjectionBias { get; set; }
        public TransformerBlock[] PriceBlocks { get; set; }
        public float[,] OutputProjection { get; set; }
        public float[] OutputBias { get; set; }
        public float[,] ConfidenceProjection { get; set; }
        public float[] ConfidenceBias { get; set; }

        public Model(MultimodalTransformerConfig config, Random random = null)
        {
            config.Validate();
            _config = config;
            _random = random ?? new Random();

            if (_config.Runtime.AccelerationType == AccelerationType.GPU || _config.Runtime.AccelerationType == AccelerationType.CUDA)
            {
                _accel = new AccelerationGPU(_config.Runtime.AccelerationType, _config.Runtime.AccelerationDeviceId);
            }
            else if (_config.Runtime.AccelerationType == AccelerationType.CPU)
            {
                _accel = new AccelerationCPU();
            }
            else if (_config.Runtime.AccelerationType == AccelerationType.MultiThreadCPU)
            {
                _accel = new AccelerationMutliThreadCPU();
            }
            else
            {
                throw new Exception("Unsupported AccelerationType");
            }

            InitTextEncoder();
            InitPriceDecoder();
            InitOutputHead();
        }

        private void InitTextEncoder()
        {
            TextTokenEmbedding = new float[_config.Text.VocabSize, _config.Text.EmbeddingDim];
            float std = MathF.Sqrt(1.0f / _config.Text.EmbeddingDim);
            for (int i = 0; i < _config.Text.VocabSize; i++)
                for (int j = 0; j < _config.Text.EmbeddingDim; j++)
                    TextTokenEmbedding[i, j] = SampleGaussian() * std;
             
            // No throwaway TransformerConfig needed — pass parameters directly.
            TextBlocks = new MultiTypeTransformer.TransformerBlock[_config.Text.NumLayers];
            for (int i = 0; i < _config.Text.NumLayers; i++)
            {
                TextBlocks[i] = new MultiTypeTransformer.TransformerBlock(
                    embeddingDim: _config.Text.EmbeddingDim,
                    numHeads: _config.Text.NumHeads,
                    feedForwardDim: _config.Text.FeedForwardDim,
                    ffnActivationType: _config.Runtime.FFNActivationType,
                    costFunction: CostFunctionType.mse,
                    activationDistribution: ActivationDistribution.Normal,
                    l2RegulationLamda: _config.Regularization.L2RegulationLamda,
                    gradientClippingThreshold: _config.Regularization.GradientClippingThreshold,
                    accelerationType: _config.Runtime.AccelerationType,
                    accelerationDeviceId: _config.Runtime.AccelerationDeviceId,
                    accel: _accel,
                    random: _random);
            }
        }


        private void InitPriceDecoder()
        {
            PriceInputProjection = new float[_config.Price.EmbeddingDim, _config.Price.InputFeatureDim];
            PriceInputProjectionBias = new float[_config.Price.EmbeddingDim];

            float std = MathF.Sqrt(2.0f / (_config.Price.InputFeatureDim + _config.Price.EmbeddingDim));

            for (int i = 0; i < _config.Price.EmbeddingDim; i++)
            {
                for (int j = 0; j < _config.Price.InputFeatureDim; j++)
                {
                    PriceInputProjection[i, j] = SampleGaussian() * std;
                }
            }
             
            PriceBlocks = new TransformerBlock[_config.Price.NumLayers];

            for (int i = 0; i < _config.Price.NumLayers; i++)
            {
                PriceBlocks[i] = new TransformerBlock(_config.Price.EmbeddingDim, _config.Price.NumHeads, _config.Price.FeedForwardDim, _config.Runtime.FFNActivationType, _accel, _random, _config.Runtime.AccelerationType, _config.Runtime.AccelerationDeviceId, _config.Regularization.L2RegulationLamda);
            }
        }

        private void InitOutputHead()
        {
            int embDim = _config.Price.EmbeddingDim;

            OutputProjection = new float[_config.Output.OutputDim, embDim];
            OutputBias = new float[_config.Output.OutputDim];

            float std = MathF.Sqrt(2.0f / (embDim + _config.Output.OutputDim));

            for (int i = 0; i < _config.Output.OutputDim; i++)
            {
                for (int j = 0; j < embDim; j++)
                {
                    OutputProjection[i, j] = SampleGaussian() * std;
                }
            }

            if (_config.Output.UseConfidenceHead)
            {
                ConfidenceProjection = new float[1, embDim];
                ConfidenceBias = new float[1];
                float cStd = MathF.Sqrt(2.0f / (embDim + 1));

                for (int j = 0; j < embDim; j++)
                {
                    ConfidenceProjection[0, j] = SampleGaussian() * cStd;
                }
            }
        }




        private float[,] ForwardTextEncoder(int[] textTokenIds)
        {
            int seqLen = textTokenIds.Length;
            var embedded = EmbedTextTokens(textTokenIds, seqLen);

            bool[,] mask = null;
            if (_config.Text.UseDecoderOnly)
                mask = _accel.CreateCausalMask(seqLen);

            var x = embedded;
            foreach (var block in TextBlocks)
                x = block.Forward(x, mask);

            return x;
        }

        private float[,] ForwardTextEncoderWithCache(int[] textTokenIds, MultimodalForwardCache cache)
        {
            int seqLen = textTokenIds.Length;
            var embedded = EmbedTextTokens(textTokenIds, seqLen);

            cache.TextEmbedded = embedded;
            cache.TextTokenIds = textTokenIds;

            bool[,] mask = null;
            if (_config.Text.UseDecoderOnly)
                mask = _accel.CreateCausalMask(seqLen);

            var x = embedded;
            for (int layer = 0; layer < _config.Text.NumLayers; layer++)
            {
                cache.TextLayerInputs.Add(x);
                var block = TextBlocks[layer];

                var attnCache = cache.TextAttentionCaches[layer];
                attnCache.Input = x;
                var attnOutput = AttentionForwardWithCache(block.Attention, x, x, x, mask, attnCache);

                var attnResidual = _accel.MatrixAdd(x, attnOutput);

                var ln1Cache = cache.TextLN1Caches[layer];
                var (normed1, ln1Means, ln1Vars, ln1Normalized) = _accel.LayerNormForward(attnResidual, block.LN1Gamma, block.LN1Beta);
                ln1Cache.Input = attnResidual;
                ln1Cache.Mean = ln1Means;
                ln1Cache.Variance = ln1Vars;
                ln1Cache.Normalized = ln1Normalized;

                var ffnInputRows = new float[seqLen][];
                var ffOutput = new float[seqLen, _config.Text.EmbeddingDim];

                for (int i = 0; i < seqLen; i++)
                {
                    var inputRow = _accel.ExtractRow(normed1, i, _config.Text.EmbeddingDim);
                    ffnInputRows[i] = inputRow;
                    var outputRow = block.FeedForwardNetwork.ForwardPassOnly(inputRow);
                    _accel.SetRow(ffOutput, i, outputRow, _config.Text.EmbeddingDim);
                }

                cache.TextFFNInputs.Add(ffnInputRows);
                cache.TextFFNOutputs.Add(ffOutput);

                var ffResidual = _accel.MatrixAdd(normed1, ffOutput);

                var ln2Cache = cache.TextLN2Caches[layer];
                var (normed2, ln2Means, ln2Vars, ln2Normalized) = _accel.LayerNormForward(ffResidual, block.LN2Gamma, block.LN2Beta);
                ln2Cache.Input = ffResidual;
                ln2Cache.Mean = ln2Means;
                ln2Cache.Variance = ln2Vars;
                ln2Cache.Normalized = ln2Normalized;

                x = normed2;
            }

            cache.TextFinalHidden = x;
            return x;
        }


        private (float[,] predictions, float[,] confidence) ProjectToOutput(float[,] hidden)
        {
            var predRaw = _accel.BatchDotProduct(OutputProjection, hidden);
            var predictions = _accel.MatrixAddBias(predRaw, OutputBias);

            float[,] confidence = null;
            if (_config.Output.UseConfidenceHead)
            {
                var confRaw = _accel.BatchDotProduct(ConfidenceProjection, hidden);
                confidence = _accel.MatrixAddBias(confRaw, ConfidenceBias);
                _accel.SigmoidInPlace(confidence);
            }

            return (predictions, confidence);
        }

        private float[,] EmbedPriceSequence(float[,] priceSequence, int seqLen)
        {
            var projected = _accel.BatchDotProduct(PriceInputProjection, priceSequence);
            return _accel.MatrixAddBias(projected, PriceInputProjectionBias);
        }
        private float[,] EmbedTextTokens(int[] textTokenIds, int seqLen)
        {
            var embedded = new float[seqLen, _config.Text.EmbeddingDim];

            for (int i = 0; i < seqLen; i++)
            {
                int tokenId = textTokenIds[i];
                for (int j = 0; j < _config.Text.EmbeddingDim; j++)
                    embedded[i, j] = TextTokenEmbedding[tokenId, j];
            }

            return embedded;
        }

        private float[,] ComputeProjection(float[,] input, float[,] weight, float[] bias)
        {
            var projected = _accel.BatchDotProduct(weight, input);
            return _accel.MatrixAddBias(projected, bias);
        }

        private float[,] AttentionForwardWithCache(MultiHeadAttention attention, float[,] qSource, float[,] kvSource_K, float[,] kvSource_V, bool[,] mask, AttentionCache cache)
        {
            int embDim = _config.Text.EmbeddingDim;
            int numHeads = _config.Text.NumHeads;
            int headDim = embDim / numHeads;

            var Q = ComputeProjection(qSource, attention.WQ, attention.BiasQ);
            var K = ComputeProjection(kvSource_K, attention.WK, attention.BiasK);
            var V = ComputeProjection(kvSource_V, attention.WV, attention.BiasV);

            RotaryPositionEmbedding.ApplyInPlace(Q, K, numHeads);

            cache.Q = Q;
            cache.K = K;
            cache.V = V;

            float scale = 1.0f / MathF.Sqrt(headDim);
            var concatenated = _accel.MultiHeadAttentionForward(Q, K, V, numHeads, scale, mask);

            cache.AttentionOutput = concatenated;

            var output = ComputeProjection(concatenated, attention.WO, attention.BiasO);
            return output;
        }
        private float SampleGaussian()
        {
            float u1 = 1.0f - _random.NextSingle();
            float u2 = 1.0f - _random.NextSingle();
            return MathF.Sqrt(-2.0f * MathF.Log(u1)) * MathF.Cos(2.0f * MathF.PI * u2);
        }

        private static float Sigmoid(float x)
        {
            if (x >= 0)
            {
                float ex = MathF.Exp(-x);
                return 1.0f / (1.0f + ex);
            }
            else
            {
                float ex = MathF.Exp(x);
                return ex / (1.0f + ex);
            }
        }

        public (float[,] predictions, float[,] confidence) Forward(int[] textTokenIds, float[,] priceSequence)
        {
            // 1. Encode text (if available)
            float[,] textHidden = null;
            if (textTokenIds != null && textTokenIds.Length > 0)
            {
                textHidden = ForwardTextEncoder(textTokenIds);
            }

            // 2. Encode price with cross-attention to text (or self-attention only if text is null)
            var priceHidden = ForwardPriceDecoder(priceSequence, textHidden);

            // 3. Project to output
            return ProjectToOutput(priceHidden);
        }

        /// <summary>
        /// Forward with caching for training (backprop needs intermediate values).
        /// textTokenIds can be null when no text is available.
        /// </summary>
        public (float[,] predictions, float[,] confidence) ForwardWithCache(int[] textTokenIds, float[,] priceSequence, MultimodalForwardCache cache)
        {
            float[,] textHidden = null;

            if (textTokenIds != null && textTokenIds.Length > 0)
            {
                textHidden = ForwardTextEncoderWithCache(textTokenIds, cache);
            }
            else
            {
                cache.TextFinalHidden = null;
                cache.TextTokenIds = null;
            }

            // Price decoder with cross-attention (or self-attention only) and cache
            var priceHidden = ForwardPriceDecoderWithCache(priceSequence, textHidden, cache);

            // Output
            cache.PriceFinalHidden = priceHidden;
            return ProjectToOutput(priceHidden);
        }

        /// <summary>
        /// Predict for the last timestep only (inference convenience method).
        /// textTokenIds can be null when no text is available.
        /// </summary>
        public (float[] prediction, float confidence) PredictNext(int[] textTokenIds, float[,] priceSequence)
        {
            var (predictions, confidenceMatrix) = Forward(textTokenIds, priceSequence);
            int lastPos = predictions.GetLength(0) - 1;

            var prediction = new float[_config.Output.OutputDim];
            for (int j = 0; j < _config.Output.OutputDim; j++)
            {
                prediction[j] = predictions[lastPos, j];

            }
            float confidence = _config.Output.UseConfidenceHead ? confidenceMatrix[lastPos, 0] : 1.0f;
            return (prediction, confidence);
        }

        // Replace ForwardPriceDecoder:

        private float[,] ForwardPriceDecoder(float[,] priceSequence, float[,] textHidden)
        {
            int seqLen = priceSequence.GetLength(0);
            var embedded = EmbedPriceSequence(priceSequence, seqLen);

            bool[,] mask = null;
            if (_config.Price.UseDecoderOnly)
            {
                //mask = CreateCausalMask(seqLen);
                mask = _accel.CreateCausalMask(seqLen);
            }

            var x = embedded;
            foreach (var block in PriceBlocks)
            {
                x = block.Forward(x, textHidden, mask, _accel);
            }

            return x;
        }

        private float[,] ForwardPriceDecoderWithCache(float[,] priceSequence, float[,] textHidden, MultimodalForwardCache cache)
        {
            int seqLen = priceSequence.GetLength(0);
            int embDim = _config.Price.EmbeddingDim;
            int numHeads = _config.Price.NumHeads;
            int headDim = embDim / numHeads;
            float scale = 1.0f / MathF.Sqrt(headDim);

            var embedded = EmbedPriceSequence(priceSequence, seqLen);
            cache.PriceEmbedded = embedded;
            cache.PriceContinuousInput = priceSequence;

            bool[,] selfMask = null;
            if (_config.Price.UseDecoderOnly)
            {
                //selfMask = CreateCausalMask(seqLen);
                selfMask = _accel.CreateCausalMask(seqLen);
            }

            var x = embedded;

            for (int layer = 0; layer < _config.Price.NumLayers; layer++)
            {
                var block = PriceBlocks[layer];
                var blockCache = cache.PriceBlockCaches[layer];
                blockCache.BlockInput = x;

                // Self-Attention
                var selfQ = ComputeProjection(x, block.SelfAttention.WQ, block.SelfAttention.BiasQ);
                var selfK = ComputeProjection(x, block.SelfAttention.WK, block.SelfAttention.BiasK);
                var selfV = ComputeProjection(x, block.SelfAttention.WV, block.SelfAttention.BiasV);

                RotaryPositionEmbedding.ApplyInPlace(selfQ, selfK, numHeads);

                blockCache.SelfQ = selfQ;
                blockCache.SelfK = selfK;
                blockCache.SelfV = selfV;

                var selfAttnOut = _accel.MultiHeadAttentionForward(selfQ, selfK, selfV, numHeads, scale, selfMask);
                blockCache.SelfAttnOutput = selfAttnOut;
                var selfProjected = ComputeProjection(selfAttnOut, block.SelfAttention.WO, block.SelfAttention.BiasO);

                var selfResidual = _accel.MatrixAdd(x, selfProjected);
                blockCache.SelfResidualInput = selfResidual;
                var (normedSelf, selfMeans, selfVars, selfNormed) = _accel.LayerNormForward(selfResidual, block.LNSelfGamma, block.LNSelfBeta);
                blockCache.LNSelfCache.Input = selfResidual;
                blockCache.LNSelfCache.Mean = selfMeans;
                blockCache.LNSelfCache.Variance = selfVars;
                blockCache.LNSelfCache.Normalized = selfNormed;
                blockCache.NormedSelf = normedSelf;

                // === Cross-Attention (skip if no text) ===
                float[,] normedCross;
                if (textHidden != null)
                {
                    var crossQ = ComputeProjection(normedSelf, block.CrossAttention.WQ, block.CrossAttention.BiasQ);
                    var crossK = ComputeProjection(textHidden, block.CrossAttention.WK, block.CrossAttention.BiasK);
                    var crossV = ComputeProjection(textHidden, block.CrossAttention.WV, block.CrossAttention.BiasV);
                    RotaryPositionEmbedding.ApplyInPlace(crossQ, crossK, numHeads);
                    blockCache.CrossQ = crossQ;
                    blockCache.CrossK = crossK;
                    blockCache.CrossV = crossV;

                    var crossAttnOut = _accel.MultiHeadAttentionForward(crossQ, crossK, crossV, numHeads, scale, null);
                    blockCache.CrossAttnOutput = crossAttnOut;
                    var crossProjected = ComputeProjection(crossAttnOut, block.CrossAttention.WO, block.CrossAttention.BiasO);

                    var crossResidual = _accel.MatrixAdd(normedSelf, crossProjected);
                    blockCache.CrossResidualInput = crossResidual;
                    var (normedCrossResult, crossMeans, crossVars, crossNormedVals) = _accel.LayerNormForward(crossResidual, block.LNCrossGamma, block.LNCrossBeta);
                    blockCache.LNCrossCache.Input = crossResidual;
                    blockCache.LNCrossCache.Mean = crossMeans;
                    blockCache.LNCrossCache.Variance = crossVars;
                    blockCache.LNCrossCache.Normalized = crossNormedVals;
                    blockCache.NormedCross = normedCrossResult;
                    normedCross = normedCrossResult;
                }
                else
                {
                    // No text: skip cross-attention, just pass through cross-attn LN
                    blockCache.CrossQ = null;
                    blockCache.CrossK = null;
                    blockCache.CrossV = null;
                    blockCache.CrossAttnOutput = null;

                    var (normedCrossResult, crossMeans, crossVars, crossNormedVals) = _accel.LayerNormForward(normedSelf, block.LNCrossGamma, block.LNCrossBeta);
                    blockCache.LNCrossCache.Input = normedSelf;
                    blockCache.LNCrossCache.Mean = crossMeans;
                    blockCache.LNCrossCache.Variance = crossVars;
                    blockCache.LNCrossCache.Normalized = crossNormedVals;
                    blockCache.NormedCross = normedCrossResult;
                    normedCross = normedCrossResult;
                }

                // === FFN ===
                var ffnInputRows = new float[seqLen][];
                var ffOutput = new float[seqLen, embDim];

                for (int i = 0; i < seqLen; i++)
                {
                    var inputRow = _accel.ExtractRow(normedCross, i, _config.Text.EmbeddingDim);

                    ffnInputRows[i] = inputRow;

                    var outputRow = block.FeedForwardNetwork.ForwardPassOnly(inputRow);

                    _accel.SetRow(ffOutput, i, outputRow, _config.Text.EmbeddingDim);
                } 

                blockCache.FFNInputRows = ffnInputRows;
                blockCache.FFNOutput = ffOutput;

                var ffResidual = _accel.MatrixAdd(normedCross, ffOutput);

                blockCache.FFNResidualInput = ffResidual;

                var (normedFF, ffMeans, ffVars, ffNormedVals) = _accel.LayerNormForward(ffResidual, block.LNFFNGamma, block.LNFFNBeta);

                blockCache.LNFFNCache.Input = ffResidual;
                blockCache.LNFFNCache.Mean = ffMeans;
                blockCache.LNFFNCache.Variance = ffVars;
                blockCache.LNFFNCache.Normalized = ffNormedVals;

                x = normedFF;
            }

            return x;
        }

        #region Save Load


        public void Save(string directory)
        {
            if (!System.IO.Directory.Exists(directory))
            {
                System.IO.Directory.CreateDirectory(directory);
            }


            var configPath = System.IO.Path.Combine(directory, "config.json");
            var configDict = new Dictionary<string, object>
            {
                ["TextVocabSize"] = _config.Text.VocabSize,
                ["TextMaxSequenceLength"] = _config.Text.MaxSequenceLength,
                ["TextEmbeddingDim"] = _config.Text.EmbeddingDim,
                ["TextNumHeads"] = _config.Text.NumHeads,
                ["TextNumLayers"] = _config.Text.NumLayers,
                ["TextFeedForwardDim"] = _config.Text.FeedForwardDim,
                ["TextUseDecoderOnly"] = _config.Text.UseDecoderOnly,
                ["FreezeTextEncoder"] = _config.Text.Freeze,
                ["PriceInputFeatureDim"] = _config.Price.InputFeatureDim,
                ["PriceMaxSequenceLength"] = _config.Price.MaxSequenceLength,
                ["PriceEmbeddingDim"] = _config.Price.EmbeddingDim,
                ["PriceNumHeads"] = _config.Price.NumHeads,
                ["PriceNumLayers"] = _config.Price.NumLayers,
                ["PriceFeedForwardDim"] = _config.Price.FeedForwardDim,
                ["PriceUseDecoderOnly"] = _config.Price.UseDecoderOnly,
                ["OutputDim"] = _config.Output.OutputDim,
                ["UseConfidenceHead"] = _config.Output.UseConfidenceHead,
                ["FFNActivationType"] = (int)_config.Runtime.FFNActivationType,
                ["AccelerationType"] = (int)_config.Runtime.AccelerationType,
                ["AccelerationDeviceId"] = _config.Runtime.AccelerationDeviceId,
                ["L2RegulationLamda"] = _config.Regularization.L2RegulationLamda,
                ["GradientClippingThreshold"] = _config.Regularization.GradientClippingThreshold
            };


            var configJson = System.Text.Json.JsonSerializer.Serialize(configDict, new System.Text.Json.JsonSerializerOptions { WriteIndented = true });
            System.IO.File.WriteAllText(configPath, configJson);

            var weightsPath = System.IO.Path.Combine(directory, "weights.bin");
            using (var stream = new System.IO.FileStream(weightsPath, System.IO.FileMode.Create))
            using (var writer = new System.IO.BinaryWriter(stream))
            {

                WriteMatrix(writer, TextTokenEmbedding);

                for (int layer = 0; layer < _config.Text.NumLayers; layer++)
                {
                    var block = TextBlocks[layer];
                    WriteAttention(writer, block.Attention);
                    WriteVector(writer, block.LN1Gamma);
                    WriteVector(writer, block.LN1Beta);
                    WriteVector(writer, block.LN2Gamma);
                    WriteVector(writer, block.LN2Beta);
                }

                WriteMatrix(writer, PriceInputProjection);
                WriteVector(writer, PriceInputProjectionBias);

                for (int layer = 0; layer < _config.Price.NumLayers; layer++)
                {
                    var block = PriceBlocks[layer];

                    WriteAttention(writer, block.SelfAttention);
                    WriteVector(writer, block.LNSelfGamma);
                    WriteVector(writer, block.LNSelfBeta);

                    WriteAttention(writer, block.CrossAttention);
                    WriteVector(writer, block.LNCrossGamma);
                    WriteVector(writer, block.LNCrossBeta);

                    WriteVector(writer, block.LNFFNGamma);
                    WriteVector(writer, block.LNFFNBeta);
                }


                WriteMatrix(writer, OutputProjection);
                WriteVector(writer, OutputBias);

                if (_config.Output.UseConfidenceHead)
                {
                    WriteMatrix(writer, ConfidenceProjection);
                    WriteVector(writer, ConfidenceBias);
                }
            }

            for (int layer = 0; layer < _config.Text.NumLayers; layer++)
            {
                var ffnDir = System.IO.Path.Combine(directory, $"text_ffn_{layer}");
                TextBlocks[layer].FeedForwardNetwork.Save(ffnDir);
            }
            for (int layer = 0; layer < _config.Price.NumLayers; layer++)
            {
                var ffnDir = System.IO.Path.Combine(directory, $"price_ffn_{layer}");
                PriceBlocks[layer].FeedForwardNetwork.Save(ffnDir);
            }
        }

        public static Model Load(string directory)
        {
            var configPath = System.IO.Path.Combine(directory, "config.json");
            var configJson = System.IO.File.ReadAllText(configPath);
            var dict = System.Text.Json.JsonSerializer.Deserialize<Dictionary<string, System.Text.Json.JsonElement>>(configJson);

            var config = new MultimodalTransformerConfig
            {
                Text = new TextEncoderConfig
                {
                    VocabSize = dict["TextVocabSize"].GetInt32(),
                    MaxSequenceLength = dict["TextMaxSequenceLength"].GetInt32(),
                    EmbeddingDim = dict["TextEmbeddingDim"].GetInt32(),
                    NumHeads = dict["TextNumHeads"].GetInt32(),
                    NumLayers = dict["TextNumLayers"].GetInt32(),
                    FeedForwardDim = dict["TextFeedForwardDim"].GetInt32(),
                    UseDecoderOnly = dict["TextUseDecoderOnly"].GetBoolean(),
                    Freeze = dict["FreezeTextEncoder"].GetBoolean()
                },

                Price = new PriceDecoderConfig
                {
                    InputFeatureDim = dict["PriceInputFeatureDim"].GetInt32(),
                    MaxSequenceLength = dict["PriceMaxSequenceLength"].GetInt32(),
                    EmbeddingDim = dict["PriceEmbeddingDim"].GetInt32(),
                    NumHeads = dict["PriceNumHeads"].GetInt32(),
                    NumLayers = dict["PriceNumLayers"].GetInt32(),
                    FeedForwardDim = dict["PriceFeedForwardDim"].GetInt32(),
                    UseDecoderOnly = dict["PriceUseDecoderOnly"].GetBoolean()
                },

                Output = new OutputHeadConfig
                {
                    OutputDim = dict["OutputDim"].GetInt32(),
                    UseConfidenceHead = dict["UseConfidenceHead"].GetBoolean()
                },

                Runtime = new RuntimeConfig
                {
                    FFNActivationType = (ActivationType)dict["FFNActivationType"].GetInt32(),
                    AccelerationType = (AccelerationType)dict["AccelerationType"].GetInt32(),
                    AccelerationDeviceId = dict["AccelerationDeviceId"].GetInt32()
                },

                Regularization = new RegularizationConfig
                {
                    L2RegulationLamda = dict["L2RegulationLamda"].GetSingle(),
                    GradientClippingThreshold = dict["GradientClippingThreshold"].GetSingle()
                }
            };
            
            config.Validate();

            var model = new Model(config);

            var weightsPath = System.IO.Path.Combine(directory, "weights.bin");
            using (var stream = new System.IO.FileStream(weightsPath, System.IO.FileMode.Open))
            using (var reader = new System.IO.BinaryReader(stream))
            {
                ReadMatrixInto(reader, model.TextTokenEmbedding);

                for (int layer = 0; layer < config.Text.NumLayers; layer++)
                {
                    var block = model.TextBlocks[layer];
                    ReadAttentionInto(reader, block.Attention);
                    ReadVectorInto(reader, block.LN1Gamma);
                    ReadVectorInto(reader, block.LN1Beta);
                    ReadVectorInto(reader, block.LN2Gamma);
                    ReadVectorInto(reader, block.LN2Beta);
                }

                ReadMatrixInto(reader, model.PriceInputProjection);
                ReadVectorInto(reader, model.PriceInputProjectionBias);

                for (int layer = 0; layer < config.Price.NumLayers; layer++)
                {
                    var block = model.PriceBlocks[layer];
                    ReadAttentionInto(reader, block.SelfAttention);
                    ReadVectorInto(reader, block.LNSelfGamma);
                    ReadVectorInto(reader, block.LNSelfBeta);
                    ReadAttentionInto(reader, block.CrossAttention);
                    ReadVectorInto(reader, block.LNCrossGamma);
                    ReadVectorInto(reader, block.LNCrossBeta);
                    ReadVectorInto(reader, block.LNFFNGamma);
                    ReadVectorInto(reader, block.LNFFNBeta);
                }

                ReadMatrixInto(reader, model.OutputProjection);
                ReadVectorInto(reader, model.OutputBias);

                if (config.Output.UseConfidenceHead)
                {
                    ReadMatrixInto(reader, model.ConfidenceProjection);
                    ReadVectorInto(reader, model.ConfidenceBias);
                }
            }
            for (int layer = 0; layer < config.Text.NumLayers; layer++)
            {
                var ffnDir = System.IO.Path.Combine(directory, $"text_ffn_{layer}");
                var loadedFfn = NeuralNetwork.Load(ffnDir, config.Runtime.AccelerationType);
                model.TextBlocks[layer].ReplaceFeedForwardNetwork(loadedFfn);
            }

            for (int layer = 0; layer < config.Price.NumLayers; layer++)
            {
                var ffnDir = System.IO.Path.Combine(directory, $"price_ffn_{layer}");
                model.PriceBlocks[layer].FeedForwardNetwork = NeuralNetwork.Load(ffnDir, config.Runtime.AccelerationType);
            }
            return model;
        }

        private static void WriteMatrix(System.IO.BinaryWriter writer, float[,] matrix)
        {
            int rows = matrix.GetLength(0);
            int cols = matrix.GetLength(1);
            writer.Write(rows);
            writer.Write(cols);
            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < cols; j++)
                {
                    writer.Write(matrix[i, j]);
                }
            }
        }

        private static void WriteVector(System.IO.BinaryWriter writer, float[] vector)
        {
            writer.Write(vector.Length);
            for (int i = 0; i < vector.Length; i++)
            {
                writer.Write(vector[i]);
            }
        }

        private static void WriteAttention(System.IO.BinaryWriter writer, MultiHeadAttention attn)
        {
            WriteMatrix(writer, attn.WQ);
            WriteMatrix(writer, attn.WK);
            WriteMatrix(writer, attn.WV);
            WriteMatrix(writer, attn.WO);
            WriteVector(writer, attn.BiasQ);
            WriteVector(writer, attn.BiasK);
            WriteVector(writer, attn.BiasV);
            WriteVector(writer, attn.BiasO);
        }

        private static void ReadMatrixInto(System.IO.BinaryReader reader, float[,] matrix)
        {
            int rows = reader.ReadInt32();
            int cols = reader.ReadInt32();
            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < cols; j++)
                {
                    matrix[i, j] = reader.ReadSingle();
                }
            }
        }

        private static void ReadVectorInto(System.IO.BinaryReader reader, float[] vector)
        {
            int len = reader.ReadInt32();
            for (int i = 0; i < len; i++)
            {
                vector[i] = reader.ReadSingle();
            }
        }

        private static void ReadAttentionInto(System.IO.BinaryReader reader, MultiHeadAttention attn)
        {
            ReadMatrixInto(reader, attn.WQ);
            ReadMatrixInto(reader, attn.WK);
            ReadMatrixInto(reader, attn.WV);
            ReadMatrixInto(reader, attn.WO);
            ReadVectorInto(reader, attn.BiasQ);
            ReadVectorInto(reader, attn.BiasK);
            ReadVectorInto(reader, attn.BiasV);
            ReadVectorInto(reader, attn.BiasO);
        }

        #endregion
    }
}
