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

        public float[,] InputProjection { get; set; }
        public float[] InputProjectionBias { get; set; }

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

            InitInputLayer();
            InitPositionalEncoding();
            InitBlocks();
            InitOutputLayer();
        }

        private void InitInputLayer()
        {
            if (_config.UsesDiscreteTokens)
            {
                TokenEmbedding = new float[_config.VocabSize, _config.EmbeddingDim];
                float std = MathF.Sqrt(1.0f / _config.EmbeddingDim);

                for (int i = 0; i < _config.VocabSize; i++)
                {
                    for (int j = 0; j < _config.EmbeddingDim; j++)
                    {
                        TokenEmbedding[i, j] = SampleGaussian() * std;
                    }
                }
            }
            else
            {
                InputProjection = new float[_config.EmbeddingDim, _config.InputFeatureDim];
                InputProjectionBias = new float[_config.EmbeddingDim];
                float std = MathF.Sqrt(2.0f / (_config.InputFeatureDim + _config.EmbeddingDim));

                for (int i = 0; i < _config.EmbeddingDim; i++)
                {
                    for (int j = 0; j < _config.InputFeatureDim; j++)
                    {
                        InputProjection[i, j] = SampleGaussian() * std;
                    }
                }
            }
        }

        private void InitPositionalEncoding()
        {
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
            if (!_config.UsesDiscreteTokens)
            {
                throw new InvalidOperationException("Use Forward(float[,]) for continuous input data types.");
            }

            int seqLen = tokenIds.Length;
            /*
            var embedded = new float[seqLen, _config.EmbeddingDim];
            for (int i = 0; i < seqLen; i++)
            {
                for (int j = 0; j < _config.EmbeddingDim; j++)
                {
                    embedded[i, j] = TokenEmbedding[tokenIds[i], j] + PositionalEncoding[i, j];
                }
            }*/
            
            var embedded = _accel.EmbedTokensWithPosition(TokenEmbedding, tokenIds, PositionalEncoding, seqLen, _config.EmbeddingDim);

            return ForwardFromEmbedding(embedded, seqLen);
        }

        public float[,] Forward(float[,] inputSequence)
        {
            if (_config.UsesDiscreteTokens)
            {
                throw new InvalidOperationException("Use Forward(int[]) for discrete token data types.");
            }

            int seqLen = inputSequence.GetLength(0);
            /*
            var projected = _accel.BatchDotProduct(InputProjection, inputSequence);
            var embedded = new float[seqLen, _config.EmbeddingDim];
            
            for (int i = 0; i < seqLen; i++)
            {
                for (int j = 0; j < _config.EmbeddingDim; j++)
                {
                    embedded[i, j] = projected[i, j] + InputProjectionBias[j] + PositionalEncoding[i, j];
                }
            }*/

            var projected = _accel.BatchDotProduct(InputProjection, inputSequence);
            var embedded = _accel.AddBiasAndPositionalEncoding(projected, InputProjectionBias, PositionalEncoding, seqLen, _config.EmbeddingDim);

            return ForwardFromEmbedding(embedded, seqLen);
        }

        private float[,] ForwardFromEmbedding(float[,] embedded, int seqLen)
        {
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

            return ProjectToOutput(x);
        }

        private bool[,] CreateCausalMask(int seqLen)
        {
            var mask = new bool[seqLen, seqLen];
            /*
            for (int i = 0; i < seqLen; i++)
            {
                for (int j = 0; j <= i; j++)
                {
                    mask[i, j] = true;
                }
            }*/


            return _accel.CreateCausalMask(seqLen);
        }

        private float[,] ProjectToOutput(float[,] hidden)
        {
            int seqLen = hidden.GetLength(0);
            int outputDim = _config.EffectiveOutputDim;
           /* var logits = new float[seqLen, outputDim];

            for (int i = 0; i < seqLen; i++)
            {
                var inputRow = new float[_config.EmbeddingDim];
                for (int k = 0; k < _config.EmbeddingDim; k++)
                {
                    inputRow[k] = hidden[i, k];
                }

                var outputRow = _accel.CalculateDotProduct(OutputProjection, inputRow);

                for (int j = 0; j < outputDim; j++)
                {
                    logits[i, j] = outputRow[j] + OutputBias[j];
                }
            }
            return logits;
            */

            var logits = _accel.BatchDotProduct(OutputProjection, hidden);
            return _accel.MatrixAddBias(logits, OutputBias);
        }

        public int[] Generate(int[] promptTokens, int maxNewTokens, float temperature = 1.0f)
        {
            if (!_config.UsesDiscreteTokens)
            {
                throw new InvalidOperationException("Generate is only supported for discrete token data types. So Text/SymbolicSequence only.");
            }

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

        public float[] PredictNext(float[,] inputSequence)
        {
            if (_config.UsesDiscreteTokens)
            {

                throw new InvalidOperationException("Must use Generate() for discrete token data types (Text/SymbolicSequence).");
            }

            var output = Forward(inputSequence);
            int lastPos = output.GetLength(0) - 1;
            int outputDim = _config.EffectiveOutputDim;

            var result = new float[outputDim];


            for (int j = 0; j < outputDim; j++)
            {
                result[j] = output[lastPos, j];
            }


            return result;
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
                {
                    return i;
                }
            }


            return probs.Length - 1;
        }

        #region Save Load

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


        public void Save(string directory)
        {
            if (!System.IO.Directory.Exists(directory))
            {
                System.IO.Directory.CreateDirectory(directory);
            }

            var configPath = System.IO.Path.Combine(directory, "config.json");
            var configDict = new Dictionary<string, object>
            {
                ["VocabSize"] = _config.VocabSize,
                ["MaxSequenceLength"] = _config.MaxSequenceLength,
                ["EmbeddingDim"] = _config.EmbeddingDim,
                ["NumHeads"] = _config.NumHeads,
                ["NumLayers"] = _config.NumLayers,
                ["FeedForwardDim"] = _config.FeedForwardDim,
                ["UseDecoderOnly"] = _config.UseDecoderOnly,
                ["FFNActivationType"] = (int)_config.FFNActivationType,
                ["AccelerationType"] = (int)_config.AccelerationType,
                ["AccelerationDeviceId"] = _config.AccelerationDeviceId,
                ["L2RegulationLamda"] = _config.L2RegulationLamda,
                ["DataType"] = (int)_config.DataType,
                ["InputFeatureDim"] = _config.InputFeatureDim,
                ["OutputDim"] = _config.OutputDim,
                ["CostFunction"] = (int)_config.CostFunction,
                ["ActivationDistribution"] = (int)_config.ActivationDistribution,
                ["GradientClippingThreshold"] = _config.GradientClippingThreshold
            };
            var configJson = System.Text.Json.JsonSerializer.Serialize(configDict, new System.Text.Json.JsonSerializerOptions { WriteIndented = true });

            System.IO.File.WriteAllText(configPath, configJson);

            var weightsPath = System.IO.Path.Combine(directory, "weights.bin");
            using (var stream = new System.IO.FileStream(weightsPath, System.IO.FileMode.Create))
            using (var writer = new System.IO.BinaryWriter(stream))
            {
                writer.Write(_config.UsesDiscreteTokens);

                if (_config.UsesDiscreteTokens)
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

            var ffnDir = System.IO.Path.Combine(directory, "ffn");
            System.IO.Directory.CreateDirectory(ffnDir);

            for (int layer = 0; layer < _config.NumLayers; layer++)
            {
                var ffnPath = System.IO.Path.Combine(ffnDir, $"ffn_layer_{layer}");
                Blocks[layer].FeedForwardNetwork.Save(ffnPath);
            }
        }

        public static LanguageModel Load(string directory)
        {
            var configPath = System.IO.Path.Combine(directory, "config.json");
            var configJson = System.IO.File.ReadAllText(configPath);

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
                FFNActivationType = (ActivationType)dict["FFNActivationType"].GetInt32(),
                AccelerationType = (AccelerationType)dict["AccelerationType"].GetInt32(),
                AccelerationDeviceId = dict["AccelerationDeviceId"].GetInt32(),
                L2RegulationLamda = dict["L2RegulationLamda"].GetSingle(),
                DataType = (TransformerDataType)dict["DataType"].GetInt32(),
                InputFeatureDim = dict["InputFeatureDim"].GetInt32(),
                OutputDim = dict["OutputDim"].GetInt32(),
                CostFunction = (CostFunctionType)dict["CostFunction"].GetInt32(),
                ActivationDistribution = (ActivationDistribution)dict["ActivationDistribution"].GetInt32(),
                GradientClippingThreshold = dict["GradientClippingThreshold"].GetSingle()
            };

            var model = new LanguageModel(config);

            var weightsPath = System.IO.Path.Combine(directory, "weights.bin");
            using (var stream = new System.IO.FileStream(weightsPath, System.IO.FileMode.Open))
            using (var reader = new System.IO.BinaryReader(stream))
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

            var ffnDir = System.IO.Path.Combine(directory, "ffn");

            for (int layer = 0; layer < config.NumLayers; layer++)
            {
                var ffnPath = System.IO.Path.Combine(ffnDir, $"ffn_layer_{layer}");

                model.Blocks[layer].FeedForwardNetwork = NeuralNetwork.Load(ffnPath, config.AccelerationType, config.AccelerationDeviceId);
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
