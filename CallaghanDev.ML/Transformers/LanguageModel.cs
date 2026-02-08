using CallaghanDev.ML.AccelerationManagers;
using CallaghanDev.ML.Enums;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace CallaghanDev.ML.Transformers
{
    public class LanguageModel
    {
        private readonly TransformerConfig _config;
        private readonly Random _random;
        private readonly IAccelerationManager _accel;

        public TransformerConfig Config { get { return _config; } }


        public IAccelerationManager AccelerationManager
        {
            get { return _accel; }
        }
        public float[,] TokenEmbedding { get; set; }
        public float[,] PositionalEncoding { get; set; }
        public TransformerBlock[] Blocks { get; set; }
        public float[,] OutputProjection { get; set; }
        public float[] OutputBias { get; set; }

        public LanguageModel(TransformerConfig config, Random random = null)
        {
            _config = config;
            _random = random ?? new Random();
            if (_config.AccelerationType == AccelerationType.GPU || _config.AccelerationType == AccelerationType.CUDA)
            {
                _accel = new AccelerationGPU(_config.AccelerationType, _config.AccelerationDeviceId);
            }
            else if (_config.AccelerationType == AccelerationType.CPU)
            {
                _accel = new AccelerationCPU();
            }
            else if (_config.AccelerationType == AccelerationType.MultiThreadCPU)
            {
                _accel = new AccelerationMutliThreadCPU();
            }
            else
            {
                throw new Exception("Unsupported AccelerationType");
            }
            InitEmbeddings();
            InitBlocks();
            InitOutputLayer();
        }

        private void InitEmbeddings()
        {
            TokenEmbedding = new float[_config.VocabSize, _config.EmbeddingDim];
            float std = MathF.Sqrt(1.0f / _config.EmbeddingDim);

            for (int i = 0; i < _config.VocabSize; i++)
            {
                for (int j = 0; j < _config.EmbeddingDim; j++)
                {
                    float u1 = 1.0f - _random.NextSingle();
                    float u2 = 1.0f - _random.NextSingle();
                    float z = MathF.Sqrt(-2.0f * MathF.Log(u1)) * MathF.Cos(2.0f * MathF.PI * u2);
                    TokenEmbedding[i, j] = z * std;
                }
            }

            PositionalEncoding = new float[_config.MaxSequenceLength, _config.EmbeddingDim];
            for (int pos = 0; pos < _config.MaxSequenceLength; pos++)
            {
                for (int i = 0; i < _config.EmbeddingDim; i++)
                {
                    float angle = pos / MathF.Pow(10000, 2.0f * i / _config.EmbeddingDim);
                    PositionalEncoding[pos, i] = (i % 2 == 0) ? MathF.Sin(angle) : MathF.Cos(angle);
                }
            }
        }

        private void InitBlocks()
        {
            Blocks = new TransformerBlock[_config.NumLayers];
            for (int i = 0; i < _config.NumLayers; i++)
            {
                Blocks[i] = new TransformerBlock(_config, _accel, _random);
            }
        }

        private void InitOutputLayer()
        {
            OutputProjection = new float[_config.VocabSize, _config.EmbeddingDim];
            OutputBias = new float[_config.VocabSize];

            float std = MathF.Sqrt(2.0f / (_config.EmbeddingDim + _config.VocabSize));

            for (int i = 0; i < _config.VocabSize; i++)
            {
                for (int j = 0; j < _config.EmbeddingDim; j++)
                {
                    float u1 = 1.0f - _random.NextSingle();
                    float u2 = 1.0f - _random.NextSingle();
                    float z = MathF.Sqrt(-2.0f * MathF.Log(u1)) * MathF.Cos(2.0f * MathF.PI * u2);
                    OutputProjection[i, j] = z * std;
                }
            }
        }

        public float[,] Forward(int[] tokenIds)
        {
            int seqLen = tokenIds.Length;

            var embedded = new float[seqLen, _config.EmbeddingDim];
            for (int i = 0; i < seqLen; i++)
            {
                for (int j = 0; j < _config.EmbeddingDim; j++)
                {
                    embedded[i, j] = TokenEmbedding[tokenIds[i], j] + PositionalEncoding[i, j];
                }
            }

            bool[,] mask = null;
            if (_config.UseDecoderOnly)
            {
                mask = CreateCausalMask(seqLen);
            }

            var x = embedded;
            foreach (var block in Blocks)
            {
                x = block.Forward(x, mask);
            }
            return ProjectToVocab(x);
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

        private float[,] ProjectToVocab(float[,] hidden)
        {
            int seqLen = hidden.GetLength(0);
            var logits = new float[seqLen, _config.VocabSize];

            for (int i = 0; i < seqLen; i++)
            {
                var inputRow = new float[_config.EmbeddingDim];
                for (int k = 0; k < _config.EmbeddingDim; k++)
                {
                    inputRow[k] = hidden[i, k];
                }

                var outputRow = _accel.CalculateDotProduct(OutputProjection, inputRow);

                for (int j = 0; j < _config.VocabSize; j++)
                {
                    logits[i, j] = outputRow[j] + OutputBias[j];
                }
            }
            return logits;
        }
        public int[] Generate(int[] promptTokens, int maxNewTokens, float temperature = 1.0f)
        {
            var tokens = new List<int>(promptTokens);

            for (int i = 0; i < maxNewTokens; i++)
            {
                var logits = Forward(tokens.ToArray());

                var lastLogits = new float[_config.VocabSize];
                int lastPos = logits.GetLength(0) - 1;
                for (int j = 0; j < _config.VocabSize; j++)
                {
                    lastLogits[j] = logits[lastPos, j] / temperature;
                }

                int nextToken = SampleFromLogits(lastLogits);
                tokens.Add(nextToken);

                if (tokens.Count >= _config.MaxSequenceLength)
                {
                    break;
                }
            }

            return tokens.ToArray();
        }

        private int SampleFromLogits(float[] logits)
        {
            float max = logits.Max();
            var exp = logits.Select(x => MathF.Exp(x - max)).ToArray();
            float sum = exp.Sum();
            var probs = exp.Select(x => x / sum).ToArray();

            float r = _random.NextSingle();
            float cumulative = 0;
            for (int i = 0; i < probs.Length; i++)
            {
                cumulative += probs[i];
                if (r < cumulative)
                    return i;
            }
            return probs.Length - 1;
        }

        public NeuralNetwork GetFeedForwardNetwork(int blockIndex)
        {
            if (blockIndex < 0 || blockIndex >= Blocks.Length)
            {
                throw new ArgumentException($"Block index out of range");
            }

            return Blocks[blockIndex].FeedForwardNetwork;
        }

        public void SaveFeedForwardNetworks(string directory)
        {
            System.IO.Directory.CreateDirectory(directory);


            for (int i = 0; i < Blocks.Length; i++)
            {
                var path = System.IO.Path.Combine(directory, $"ffn_layer_{i}.json");
                Blocks[i].FeedForwardNetwork.Save(path);
            }
        }
        public void LoadFeedForwardNetworks(string directory)
        {
            for (int i = 0; i < Blocks.Length; i++)
            {
                var path = System.IO.Path.Combine(directory, $"ffn_layer_{i}.json");
                Blocks[i].FeedForwardNetwork = NeuralNetwork.Load(path, _config.AccelerationType, _config.AccelerationDeviceId);
            }
        }
    }
}
