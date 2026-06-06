using CallaghanDev.ML.AccelerationManagers;
using CallaghanDev.ML.Enums;
using CallaghanDev.ML.Extensions;
using CallaghanDev.ML.Transformers.Configuration;
using System;
using System.Collections.Generic;

namespace CallaghanDev.ML.Transformers.MultiTypeTransformer
{
    public class LanguageModel
    {
        private readonly TransformerConfig _config;
        private readonly Random _random;
        private readonly IAccelerationManager _accel;

        public TransformerConfig Config => _config;
        public IAccelerationManager AccelerationManager => _accel;

        public float[,] TokenEmbedding { get; set; }
        public float[,] InputProjection { get; set; }
        public float[] InputProjectionBias { get; set; }
        public TransformerBlock[] Blocks { get; set; }
        public float[,] OutputProjection { get; set; }
        public float[] OutputBias { get; set; }


        public LanguageModel(TransformerConfig config, Random random = null)
        {
            _config = config ?? throw new ArgumentNullException(nameof(config));
            _config.Validate();

            _random = random ?? new Random();
            _accel = AccelerationFactory.Create(_config.Runtime);

            InitInputLayer();
            InitBlocks();
            InitOutputLayer();
        }
        private void InitInputLayer()
        {
            if (_config.Data.UsesDiscreteTokens)
            {
                TokenEmbedding = new float[_config.VocabSize, _config.EmbeddingDim];
                float std = MathF.Sqrt(1.0f / _config.EmbeddingDim);
                for (int i = 0; i < _config.VocabSize; i++)
                    for (int j = 0; j < _config.EmbeddingDim; j++)
                        TokenEmbedding[i, j] = SampleGaussian() * std;
            }
            else
            {
                InputProjection = new float[_config.EmbeddingDim, _config.InputFeatureDim];
                InputProjectionBias = new float[_config.EmbeddingDim];
                float std = MathF.Sqrt(2.0f / (_config.InputFeatureDim + _config.EmbeddingDim));
                for (int i = 0; i < _config.EmbeddingDim; i++)
                    for (int j = 0; j < _config.InputFeatureDim; j++)
                        InputProjection[i, j] = SampleGaussian() * std;
            }
        }


        private void InitBlocks()
        {
            Blocks = new TransformerBlock[_config.NumLayers];
            for (int i = 0; i < _config.NumLayers; i++)
                Blocks[i] = CreateBlock();
        }

        private TransformerBlock CreateBlock()
        {
            return new TransformerBlock(

            embeddingDim: _config.EmbeddingDim,
            numHeads: _config.NumHeads,
            feedForwardDim: _config.FeedForwardDim,
            ffnActivationType: _config.Runtime.FFNActivationType,
            costFunction: _config.Data.CostFunction,
            activationDistribution: _config.Data.ActivationDistribution,
            l2RegulationLamda: _config.Regularization.L2RegulationLamda,
            gradientClippingThreshold: _config.Regularization.GradientClippingThreshold,
            accelerationType: _config.Runtime.AccelerationType,
            accelerationDeviceId: _config.Runtime.AccelerationDeviceId,
            accel: _accel,
            random: _random);
        }

        private void InitOutputLayer()
        {
            int outputDim = _config.EffectiveOutputDim;

            OutputProjection = new float[outputDim, _config.EmbeddingDim];
            OutputBias = new float[outputDim];

            float std = MathF.Sqrt(2.0f / (_config.EmbeddingDim + outputDim));

            for (int i = 0; i < outputDim; i++)
            {
                for (int j = 0; j < _config.EmbeddingDim; j++)
                {
                    OutputProjection[i, j] = SampleGaussian() * std;
                }
            }
        }

        private float SampleGaussian()
        {
            float u1 = 1.0f - _random.NextSingle();
            float u2 = 1.0f - _random.NextSingle();
            return MathF.Sqrt(-2.0f * MathF.Log(u1)) * MathF.Cos(2.0f * MathF.PI * u2);
        }


        public float[,] Forward(int[] tokenIds)
        {
            if (!_config.Data.UsesDiscreteTokens)
                throw new InvalidOperationException("Use Forward(float[,]) for continuous input data types.");

            ValidateTokenSequence(tokenIds);

            int seqLen = tokenIds.Length;
            var embedded = EmbedTokens(tokenIds, seqLen);
            return ForwardFromEmbedding(embedded);
        }

        public float[,] Forward(float[,] inputSequence)
        {
            if (_config.Data.UsesDiscreteTokens)
                throw new InvalidOperationException("Use Forward(int[]) for discrete token data types.");

            ValidateContinuousSequence(inputSequence);

            var embedded = EmbedContinuous(inputSequence);
            return ForwardFromEmbedding(embedded);
        }

        internal float[,] ForwardTokenSlice(int[] tokenIds, int tokenStart, int tokenCount)
        {
            if (!_config.Data.UsesDiscreteTokens)
                throw new InvalidOperationException("Use ForwardContinuousSlice(float[,], int, int) for continuous input data types.");

            ValidateTokenSlice(tokenIds, tokenStart, tokenCount);

            var embedded = _accel.EmbedTokenIds(tokenIds, tokenStart, tokenCount, TokenEmbedding, _config.EmbeddingDim);
            return ForwardFromEmbedding(embedded);
        }

        internal float[,] ForwardContinuousSlice(float[,] inputSequence, int rowStart, int rowCount)
        {
            if (_config.Data.UsesDiscreteTokens)
                throw new InvalidOperationException("Use ForwardTokenSlice(int[], int, int) for discrete token data types.");

            ValidateContinuousSlice(inputSequence, rowStart, rowCount);

            var embedded = _accel.BatchDotProductAddBias(InputProjection, inputSequence, rowStart, rowCount, InputProjectionBias);
            return ForwardFromEmbedding(embedded);
        }


        private float[,] EmbedTokens(int[] tokenIds, int seqLen)
        {
            // Exact int[] inputs can use the accelerator's row-copy implementation.
            // The IReadOnlyList overload below remains for List<int> generation state
            // so generation does not allocate tokens.ToArray() every step.
            if (seqLen == tokenIds.Length)
            {
                return _accel.EmbedTokenIds(tokenIds, TokenEmbedding, _config.EmbeddingDim);
            }

            return EmbedTokens((IReadOnlyList<int>)tokenIds, seqLen);
        }

        private float[,] EmbedTokens(IReadOnlyList<int> tokenIds, int seqLen)
        {
            var embedded = new float[seqLen, _config.EmbeddingDim];

            for (int i = 0; i < seqLen; i++)
            {
                int tokenId = tokenIds[i];
                for (int j = 0; j < _config.EmbeddingDim; j++)
                    embedded[i, j] = TokenEmbedding[tokenId, j];
            }

            return embedded;
        }




        private float[,] EmbedContinuous(float[,] inputSequence)
        {
            return _accel.BatchDotProductAddBias(InputProjection, inputSequence, InputProjectionBias);
        }
        private float[,] ForwardFromEmbedding(float[,] embedded)
        {
            return ProjectToOutput(ForwardHiddenFromEmbedding(embedded));
        }

        private float[,] ForwardHiddenFromTokens(IReadOnlyList<int> tokenIds)
        {
            int seqLen = tokenIds.Count;
            var embedded = EmbedTokens(tokenIds, seqLen);
            return ForwardHiddenFromEmbedding(embedded);
        }

        private float[,] ForwardHiddenFromContinuous(float[,] inputSequence)
        {
            var embedded = EmbedContinuous(inputSequence);
            return ForwardHiddenFromEmbedding(embedded);
        }

        private float[,] ForwardHiddenFromEmbedding(float[,] embedded)
        {
            bool causal = _config.UseDecoderOnly;
            var x = embedded;

            foreach (var block in Blocks)
            {
                // Avoid allocating an O(seqLen^2) causal bool[,] mask. The attention
                // backend already has a causal fast path.
                x = block.Forward(x, selfMask: null, causal: causal);
            }

            return x;
        }

        private float[,] ProjectToOutput(float[,] hidden)
        {
            return _accel.ProjectOutputBatch(
                hidden,
                OutputProjection,
                OutputBias,
                hidden.GetLength(0),
                _config.EffectiveOutputDim);
        }

        private float[] ProjectLastTokenOnly(float[,] hidden)
        {
            if (hidden == null)
            {
                throw new ArgumentNullException(nameof(hidden));
            }

            int seqLen = hidden.GetLength(0);
            int embeddingDim = hidden.GetLength(1);
            if (seqLen <= 0)
            {
                throw new ArgumentException("Hidden states must contain at least one row.", nameof(hidden));
            }
            if (embeddingDim != _config.EmbeddingDim)
            {
                throw new ArgumentException($"Hidden width must be {_config.EmbeddingDim}, got {embeddingDim}.", nameof(hidden));
            }

            int lastPos = seqLen - 1;
            int outputDim = _config.EffectiveOutputDim;
            var logits = new float[outputDim];

            for (int o = 0; o < outputDim; o++)
            {
                float sum = OutputBias[o];
                for (int e = 0; e < embeddingDim; e++)
                {
                    sum += OutputProjection[o, e] * hidden[lastPos, e];
                }
                logits[o] = sum;
            }

            return logits;
        }

        private sealed class DecoderGenerationLayerCache
        {
            public readonly float[,] Keys;
            public readonly float[,] Values;
            public int Length;

            public DecoderGenerationLayerCache(int maxSequenceLength, int embeddingDim)
            {
                Keys = new float[maxSequenceLength, embeddingDim];
                Values = new float[maxSequenceLength, embeddingDim];
                Length = 0;
            }
        }

        private DecoderGenerationLayerCache[] CreateDecoderGenerationCaches()
        {
            var caches = new DecoderGenerationLayerCache[_config.NumLayers];
            for (int i = 0; i < caches.Length; i++)
            {
                caches[i] = new DecoderGenerationLayerCache(_config.MaxSequenceLength, _config.EmbeddingDim);
            }
            return caches;
        }

        private float[,] ForwardDecoderTokenFromId(int tokenId, int absolutePosition, DecoderGenerationLayerCache[] caches)
        {
            var embedded = new float[1, _config.EmbeddingDim];
            for (int j = 0; j < _config.EmbeddingDim; j++)
            {
                embedded[0, j] = TokenEmbedding[tokenId, j];
            }

            return ForwardDecoderTokenFromEmbedding(embedded, absolutePosition, caches);
        }

        private float[,] ForwardDecoderTokenFromEmbedding(float[,] tokenEmbedding, int absolutePosition, DecoderGenerationLayerCache[] caches)
        {
            if (tokenEmbedding == null)
            {
                throw new ArgumentNullException(nameof(tokenEmbedding));
            }
            if (caches == null || caches.Length != _config.NumLayers)
            {
                throw new ArgumentException("Decoder cache does not match the model layer count.", nameof(caches));
            }
            if (tokenEmbedding.GetLength(0) != 1 || tokenEmbedding.GetLength(1) != _config.EmbeddingDim)
            {
                throw new ArgumentException($"Token embedding must have shape [1,{_config.EmbeddingDim}].", nameof(tokenEmbedding));
            }

            var x = tokenEmbedding;
            for (int layer = 0; layer < _config.NumLayers; layer++)
            {
                var block = Blocks[layer];
                var attentionOutput = DecoderSelfAttentionStep(block.Attention, x, absolutePosition, caches[layer]);
                var normedSelf = _accel.ResidualLayerNorm(x, attentionOutput, block.LN1Gamma, block.LN1Beta);

                var ffnOutput = _accel.FFNForwardBatch(normedSelf, 1, _config.EmbeddingDim, block.FeedForwardNetwork.ForwardPassOnly);
                x = _accel.ResidualLayerNorm(normedSelf, ffnOutput, block.LN2Gamma, block.LN2Beta);
            }

            return x;
        }

        private float[,] DecoderSelfAttentionStep(MultiHeadAttention attention, float[,] input, int absolutePosition, DecoderGenerationLayerCache cache)
        {
            if (attention == null)
            {
                throw new ArgumentNullException(nameof(attention));
            }
            if (cache == null)
            {
                throw new ArgumentNullException(nameof(cache));
            }
            if (cache.Length != absolutePosition)
            {
                throw new InvalidOperationException($"Decoder KV cache length {cache.Length} does not match requested absolute position {absolutePosition}.");
            }
            if (cache.Length >= _config.MaxSequenceLength)
            {
                throw new InvalidOperationException("Decoder KV cache is full.");
            }

            var (q, k, v) = _accel.ProjectQKV(input, attention.WQ, attention.BiasQ, attention.WK, attention.BiasK, attention.WV, attention.BiasV);
            ApplyRotaryPositionSingleRow(q, k, _config.NumHeads, absolutePosition);

            int writeRow = cache.Length;
            int embeddingDim = _config.EmbeddingDim;
            for (int col = 0; col < embeddingDim; col++)
            {
                cache.Keys[writeRow, col] = k[0, col];
                cache.Values[writeRow, col] = v[0, col];
            }
            cache.Length++;

            int keyLen = cache.Length;
            int numHeads = _config.NumHeads;
            int headDim = embeddingDim / numHeads;
            float scale = 1.0f / MathF.Sqrt(headDim);
            var concatenated = new float[1, embeddingDim];

            for (int head = 0; head < numHeads; head++)
            {
                int offset = head * headDim;
                float maxScore = float.NegativeInfinity;

                for (int key = 0; key < keyLen; key++)
                {
                    float dot = 0f;
                    for (int d = 0; d < headDim; d++)
                    {
                        int col = offset + d;
                        dot += q[0, col] * cache.Keys[key, col];
                    }

                    float score = dot * scale;
                    if (score > maxScore)
                    {
                        maxScore = score;
                    }
                }

                float sumExp = 0f;
                for (int key = 0; key < keyLen; key++)
                {
                    float dot = 0f;
                    for (int d = 0; d < headDim; d++)
                    {
                        int col = offset + d;
                        dot += q[0, col] * cache.Keys[key, col];
                    }

                    sumExp += MathF.Exp((dot * scale) - maxScore);
                }

                float invSumExp = sumExp > 0f ? 1.0f / sumExp : 0f;
                for (int key = 0; key < keyLen; key++)
                {
                    float dot = 0f;
                    for (int d = 0; d < headDim; d++)
                    {
                        int col = offset + d;
                        dot += q[0, col] * cache.Keys[key, col];
                    }

                    float weight = MathF.Exp((dot * scale) - maxScore) * invSumExp;
                    if (weight == 0f)
                    {
                        continue;
                    }

                    for (int d = 0; d < headDim; d++)
                    {
                        int col = offset + d;
                        concatenated[0, col] += weight * cache.Values[key, col];
                    }
                }
            }

            return _accel.ProjectOutputBatch(concatenated, attention.WO, attention.BiasO, 1, embeddingDim);
        }

        private static void ApplyRotaryPositionSingleRow(float[,] q, float[,] k, int numHeads, int absolutePosition)
        {
            if (q == null) throw new ArgumentNullException(nameof(q));
            if (k == null) throw new ArgumentNullException(nameof(k));
            if (q.GetLength(0) != 1 || k.GetLength(0) != 1)
            {
                throw new ArgumentException("Incremental RoPE expects single-row Q and K matrices.");
            }

            int embeddingDim = q.GetLength(1);
            if (k.GetLength(1) != embeddingDim)
            {
                throw new ArgumentException("Q and K must have the same embedding dimension.");
            }
            if (numHeads <= 0 || embeddingDim % numHeads != 0)
            {
                throw new ArgumentException("Embedding dimension must be divisible by number of heads.", nameof(numHeads));
            }

            int headDim = embeddingDim / numHeads;
            if ((headDim & 1) != 0)
            {
                throw new ArgumentException("RoPE requires an even per-head dimension.");
            }

            const float baseTheta = 10000f;
            for (int head = 0; head < numHeads; head++)
            {
                int offset = head * headDim;
                for (int pair = 0; pair < headDim / 2; pair++)
                {
                    int even = offset + pair * 2;
                    int odd = even + 1;
                    float theta = absolutePosition / MathF.Pow(baseTheta, (2.0f * pair) / headDim);
                    float cos = MathF.Cos(theta);
                    float sin = MathF.Sin(theta);

                    float q0 = q[0, even];
                    float q1 = q[0, odd];
                    q[0, even] = (q0 * cos) - (q1 * sin);
                    q[0, odd] = (q0 * sin) + (q1 * cos);

                    float k0 = k[0, even];
                    float k1 = k[0, odd];
                    k[0, even] = (k0 * cos) - (k1 * sin);
                    k[0, odd] = (k0 * sin) + (k1 * cos);
                }
            }
        }

        public int[] Generate(int[] promptTokens, int maxNewTokens, float temperature = 1.0f)
        {
            if (!_config.Data.UsesDiscreteTokens)
                throw new InvalidOperationException("Generate is only supported for discrete token data types.");

            ValidateTokenSequence(promptTokens);

            if (maxNewTokens < 0)
                throw new ArgumentOutOfRangeException(nameof(maxNewTokens), "maxNewTokens must be non-negative.");
            if (float.IsNaN(temperature) || float.IsInfinity(temperature) || temperature <= 0f)
                throw new ArgumentOutOfRangeException(nameof(temperature), "temperature must be finite and greater than zero.");

            var tokens = new List<int>(promptTokens);
            if (tokens.Count >= _config.MaxSequenceLength || maxNewTokens == 0)
                return tokens.ToArray();

            if (!_config.UseDecoderOnly)
            {
                for (int i = 0; i < maxNewTokens && tokens.Count < _config.MaxSequenceLength; i++)
                {
                    var hidden = ForwardHiddenFromTokens(tokens);
                    var lastLogits = ProjectLastTokenOnly(hidden);

                    for (int j = 0; j < lastLogits.Length; j++)
                        lastLogits[j] /= temperature;

                    tokens.Add(SampleFromLogits(lastLogits));
                }

                return tokens.ToArray();
            }

            var caches = CreateDecoderGenerationCaches();
            float[,] lastHidden = null;
            for (int pos = 0; pos < tokens.Count; pos++)
            {
                lastHidden = ForwardDecoderTokenFromId(tokens[pos], pos, caches);
            }

            for (int generated = 0; generated < maxNewTokens && tokens.Count < _config.MaxSequenceLength; generated++)
            {
                var lastLogits = ProjectLastTokenOnly(lastHidden);
                for (int j = 0; j < lastLogits.Length; j++)
                    lastLogits[j] /= temperature;

                int nextToken = SampleFromLogits(lastLogits);
                tokens.Add(nextToken);

                if (tokens.Count < _config.MaxSequenceLength && generated + 1 < maxNewTokens)
                {
                    lastHidden = ForwardDecoderTokenFromId(nextToken, tokens.Count - 1, caches);
                }
            }

            return tokens.ToArray();
        }



        public float[] PredictNext(float[,] inputSequence)
        {
            if (_config.Data.UsesDiscreteTokens)
                throw new InvalidOperationException("Must use Generate() for discrete token data types.");

            ValidateContinuousSequence(inputSequence);

            var hidden = ForwardHiddenFromContinuous(inputSequence);
            return ProjectLastTokenOnly(hidden);
        }
        private int SampleFromLogits(float[] logits)
        {
            if (logits == null) throw new ArgumentNullException(nameof(logits));
            if (logits.Length == 0) throw new ArgumentException("Logits cannot be empty.", nameof(logits));

            float max = float.NegativeInfinity;
            for (int i = 0; i < logits.Length; i++)
            {
                if (float.IsNaN(logits[i]) || float.IsInfinity(logits[i]))
                    throw new InvalidOperationException($"Logit at index {i} is not finite: {logits[i]}.");
                if (logits[i] > max) max = logits[i];
            }

            float sum = 0f;
            for (int i = 0; i < logits.Length; i++)
            {
                sum += MathF.Exp(logits[i] - max);
            }

            if (sum <= 0f || float.IsNaN(sum) || float.IsInfinity(sum))
                throw new InvalidOperationException("Cannot sample because softmax normalization is invalid.");

            float threshold = _random.NextSingle() * sum;
            float cumulative = 0f;

            for (int i = 0; i < logits.Length; i++)
            {
                cumulative += MathF.Exp(logits[i] - max);
                if (threshold < cumulative)
                    return i;
            }

            return logits.Length - 1;
        }
        private void ValidateTokenSequence(int[] tokenIds)
        {
            if (tokenIds == null)
                throw new ArgumentNullException(nameof(tokenIds));
            if (tokenIds.Length == 0)
                throw new ArgumentException("Token sequence must contain at least one token.", nameof(tokenIds));
            if (tokenIds.Length > _config.MaxSequenceLength)
                throw new ArgumentException($"Token sequence length {tokenIds.Length} exceeds MaxSequenceLength {_config.MaxSequenceLength}.", nameof(tokenIds));

            for (int i = 0; i < tokenIds.Length; i++)
            {
                int token = tokenIds[i];
                if ((uint)token >= (uint)_config.VocabSize)
                    throw new ArgumentOutOfRangeException(nameof(tokenIds), $"Token id {token} at position {i} is outside [0, {_config.VocabSize}).");
            }
        }

        private void ValidateTokenSlice(int[] tokenIds, int tokenStart, int tokenCount)
        {
            if (tokenIds == null)
                throw new ArgumentNullException(nameof(tokenIds));
            if (tokenCount <= 0)
                throw new ArgumentException("Token slice must contain at least one token.", nameof(tokenCount));
            if (tokenStart < 0 || tokenStart + tokenCount > tokenIds.Length)
                throw new ArgumentOutOfRangeException(nameof(tokenStart));
            if (tokenCount > _config.MaxSequenceLength)
                throw new ArgumentException($"Token sequence length {tokenCount} exceeds MaxSequenceLength {_config.MaxSequenceLength}.", nameof(tokenCount));

            for (int i = 0; i < tokenCount; i++)
            {
                int token = tokenIds[tokenStart + i];
                if ((uint)token >= (uint)_config.VocabSize)
                    throw new ArgumentOutOfRangeException(nameof(tokenIds), $"Token id {token} at position {tokenStart + i} is outside [0, {_config.VocabSize}).");
            }
        }

        private void ValidateContinuousSequence(float[,] inputSequence)
        {
            if (inputSequence == null)
                throw new ArgumentNullException(nameof(inputSequence));

            int seqLen = inputSequence.GetLength(0);
            int featureDim = inputSequence.GetLength(1);

            if (seqLen <= 0)
                throw new ArgumentException("Input sequence must contain at least one row.", nameof(inputSequence));
            if (seqLen > _config.MaxSequenceLength)
                throw new ArgumentException($"Input sequence length {seqLen} exceeds MaxSequenceLength {_config.MaxSequenceLength}.", nameof(inputSequence));
            if (featureDim != _config.InputFeatureDim)
                throw new ArgumentException($"Input feature dimension must be {_config.InputFeatureDim}, got {featureDim}.", nameof(inputSequence));

            for (int i = 0; i < seqLen; i++)
                for (int j = 0; j < featureDim; j++)
                    if (float.IsNaN(inputSequence[i, j]) || float.IsInfinity(inputSequence[i, j]))
                        throw new ArgumentException($"Input value at [{i},{j}] is not finite: {inputSequence[i, j]}.", nameof(inputSequence));
        }

        private void ValidateContinuousSlice(float[,] inputSequence, int rowStart, int rowCount)
        {
            if (inputSequence == null)
                throw new ArgumentNullException(nameof(inputSequence));

            int seqLen = inputSequence.GetLength(0);
            int featureDim = inputSequence.GetLength(1);

            if (rowCount <= 0)
                throw new ArgumentException("Input sequence slice must contain at least one row.", nameof(rowCount));
            if (rowStart < 0 || rowStart + rowCount > seqLen)
                throw new ArgumentOutOfRangeException(nameof(rowStart));
            if (rowCount > _config.MaxSequenceLength)
                throw new ArgumentException($"Input sequence length {rowCount} exceeds MaxSequenceLength {_config.MaxSequenceLength}.", nameof(rowCount));
            if (featureDim != _config.InputFeatureDim)
                throw new ArgumentException($"Input feature dimension must be {_config.InputFeatureDim}, got {featureDim}.", nameof(inputSequence));

            for (int i = 0; i < rowCount; i++)
            {
                int row = rowStart + i;
                for (int j = 0; j < featureDim; j++)
                {
                    if (float.IsNaN(inputSequence[row, j]) || float.IsInfinity(inputSequence[row, j]))
                        throw new ArgumentException($"Input value at [{row},{j}] is not finite: {inputSequence[row, j]}.", nameof(inputSequence));
                }
            }
        }
        #region Save / Load

        public void SaveFeedForwardNetworks(string directory)
        {
            Directory.CreateDirectory(directory);

            for (int i = 0; i < Blocks.Length; i++)
            {
                string path = Path.Combine(directory, $"ffn_layer_{i}.json");

                // overwrite always (avoid stale files)
                if (File.Exists(path))
                    File.Delete(path);

                Blocks[i].FeedForwardNetwork.Save(path);
            }
        }
        public void LoadFeedForwardNetworks(string directory)
        {
            for (int i = 0; i < Blocks.Length; i++)
            {
                string path = Path.Combine(directory, $"ffn_layer_{i}.json");

                if (!File.Exists(path))
                    throw new FileNotFoundException($"Missing FFN file: {path}");

                var loaded = NeuralNetwork.Load(
                    path,
                    _config.Runtime.AccelerationType,
                    _config.Runtime.AccelerationDeviceId);

                //  CRITICAL FIX: restore full internal state
                Blocks[i].FeedForwardNetwork.RestoreState(
                    loaded.GetInternalData()
                );
            }
        }
        public void Save(string directory)
        {
            Directory.CreateDirectory(directory);

            var configDict = new Dictionary<string, object>
            {
                ["VocabSize"] = _config.VocabSize,
                ["MaxSequenceLength"] = _config.MaxSequenceLength,
                ["EmbeddingDim"] = _config.EmbeddingDim,
                ["NumHeads"] = _config.NumHeads,
                ["NumLayers"] = _config.NumLayers,
                ["FeedForwardDim"] = _config.FeedForwardDim,
                ["UseDecoderOnly"] = _config.UseDecoderOnly,
                ["FFNActivationType"] = (int)_config.Runtime.FFNActivationType,
                ["AccelerationType"] = (int)_config.Runtime.AccelerationType,
                ["AccelerationDeviceId"] = _config.Runtime.AccelerationDeviceId,
                ["L2RegulationLamda"] = _config.Regularization.L2RegulationLamda,
                ["GradientClippingThreshold"] = _config.Regularization.GradientClippingThreshold,
                ["DataType"] = (int)_config.Data.DataType,
                ["InputFeatureDim"] = _config.InputFeatureDim,
                ["OutputDim"] = _config.OutputDim,
                ["CostFunction"] = (int)_config.Data.CostFunction,
                ["ActivationDistribution"] = (int)_config.Data.ActivationDistribution
            };

            File.WriteAllText(
                Path.Combine(directory, "config.json"),
                System.Text.Json.JsonSerializer.Serialize(
                    configDict,
                    new System.Text.Json.JsonSerializerOptions { WriteIndented = true }));

            using (var stream = new FileStream(Path.Combine(directory, "weights.bin"), FileMode.Create, FileAccess.Write))
            using (var writer = new BinaryWriter(stream))
            {
                writer.Write(_config.Data.UsesDiscreteTokens);

                if (_config.Data.UsesDiscreteTokens)
                {
                    WriteMatrix(writer, TokenEmbedding);
                }
                else
                {
                    WriteMatrix(writer, InputProjection);
                    WriteVector(writer, InputProjectionBias);
                }

                for (int layer = 0; layer < _config.NumLayers; layer++)
                {
                    var block = Blocks[layer];

                    WriteAttention(writer, block.Attention);
                    WriteVector(writer, block.LN1Gamma);
                    WriteVector(writer, block.LN1Beta);
                    WriteVector(writer, block.LN2Gamma);
                    WriteVector(writer, block.LN2Beta);
                }

                WriteMatrix(writer, OutputProjection);
                WriteVector(writer, OutputBias);
            }

            SaveFeedForwardNetworks(Path.Combine(directory, "ffn"));
        }
        public static LanguageModel Load(string directory)
        {
            var configJson = File.ReadAllText(Path.Combine(directory, "config.json"));
            var dict = System.Text.Json.JsonSerializer.Deserialize<Dictionary<string, System.Text.Json.JsonElement>>(configJson);

            var config = new TransformerConfig
            {
                VocabSize = dict["VocabSize"].GetInt32(),
                MaxSequenceLength = dict["MaxSequenceLength"].GetInt32(),
                EmbeddingDim = dict["EmbeddingDim"].GetInt32(),
                NumHeads = dict["NumHeads"].GetInt32(),
                NumLayers = dict["NumLayers"].GetInt32(),
                FeedForwardDim = dict["FeedForwardDim"].GetInt32(),
                UseDecoderOnly = dict["UseDecoderOnly"].GetBoolean(),
                InputFeatureDim = dict["InputFeatureDim"].GetInt32(),
                OutputDim = dict["OutputDim"].GetInt32(),
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
                },
                Data = new DataConfig
                {
                    DataType = (TransformerDataType)dict["DataType"].GetInt32(),
                    CostFunction = (CostFunctionType)dict["CostFunction"].GetInt32(),
                    ActivationDistribution = (ActivationDistribution)dict["ActivationDistribution"].GetInt32()
                }
            };

            config.Validate();

            var model = new LanguageModel(config);

            using (var stream = new FileStream(Path.Combine(directory, "weights.bin"), FileMode.Open, FileAccess.Read))
            using (var reader = new BinaryReader(stream))
            {
                bool savedAsDiscrete = reader.ReadBoolean();

                if (savedAsDiscrete)
                {
                    ReadMatrixInto(reader, model.TokenEmbedding);
                }
                else
                {
                    ReadMatrixInto(reader, model.InputProjection);
                    ReadVectorInto(reader, model.InputProjectionBias);
                }

                for (int layer = 0; layer < config.NumLayers; layer++)
                {
                    var block = model.Blocks[layer];
                    ReadAttentionInto(reader, block.Attention);
                    ReadVectorInto(reader, block.LN1Gamma);
                    ReadVectorInto(reader, block.LN1Beta);
                    ReadVectorInto(reader, block.LN2Gamma);
                    ReadVectorInto(reader, block.LN2Beta);
                }

                ReadMatrixInto(reader, model.OutputProjection);
                ReadVectorInto(reader, model.OutputBias);
            }

            model.LoadFeedForwardNetworks(Path.Combine(directory, "ffn"));
            return model;
        }
        private static void WriteMatrix(BinaryWriter w, float[,] m)
        {
            if (m == null)
                throw new ArgumentNullException(nameof(m));

            int rows = m.GetLength(0);
            int cols = m.GetLength(1);

            w.Write(rows);
            w.Write(cols);

            for (int i = 0; i < rows; i++)
                for (int j = 0; j < cols; j++)
                    w.Write(m[i, j]);
        }
        private static void WriteVector(BinaryWriter w, float[] v)
        {
            if (v == null)
                throw new ArgumentNullException(nameof(v));

            w.Write(v.Length);

            for (int i = 0; i < v.Length; i++)
                w.Write(v[i]);
        }
        private static void WriteAttention(System.IO.BinaryWriter w, MultiHeadAttention a)
        {
            WriteMatrix(w, a.WQ); WriteMatrix(w, a.WK); WriteMatrix(w, a.WV); WriteMatrix(w, a.WO);
            WriteVector(w, a.BiasQ); WriteVector(w, a.BiasK); WriteVector(w, a.BiasV); WriteVector(w, a.BiasO);
        }

        private static void ReadMatrixInto(BinaryReader r, float[,] m)
        {
            if (m == null)
                throw new ArgumentNullException(nameof(m));

            int rows = r.ReadInt32();
            int cols = r.ReadInt32();

            if (rows != m.GetLength(0) || cols != m.GetLength(1))
            {
                throw new InvalidOperationException(
                    $"Matrix shape mismatch. File has [{rows},{cols}], target is [{m.GetLength(0)},{m.GetLength(1)}].");
            }

            for (int i = 0; i < rows; i++)
                for (int j = 0; j < cols; j++)
                    m[i, j] = r.ReadSingle();
        }
        private static void ReadVectorInto(BinaryReader r, float[] v)
        {
            if (v == null)
                throw new ArgumentNullException(nameof(v));

            int len = r.ReadInt32();

            if (len != v.Length)
            {
                throw new InvalidOperationException(
                    $"Vector length mismatch. File has [{len}], target is [{v.Length}].");
            }

            for (int i = 0; i < len; i++)
                v[i] = r.ReadSingle();
        }
        private static void ReadAttentionInto(BinaryReader r, MultiHeadAttention a)
        {
            if (a == null)
                throw new ArgumentNullException(nameof(a));

            ReadMatrixInto(r, a.WQ);
            ReadMatrixInto(r, a.WK);
            ReadMatrixInto(r, a.WV);
            ReadMatrixInto(r, a.WO);

            ReadVectorInto(r, a.BiasQ);
            ReadVectorInto(r, a.BiasK);
            ReadVectorInto(r, a.BiasV);
            ReadVectorInto(r, a.BiasO);
        }
        #endregion
    }
}