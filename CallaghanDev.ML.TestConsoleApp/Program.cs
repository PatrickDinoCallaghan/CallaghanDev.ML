
using CallaghanDev.ML.Enums;
using CallaghanDev.ML.Transformers;
using CallaghanDev.ML.Transformers.CrossAttentionMultimodal;

namespace CallaghanDev.ML.TestConsoleApp
{
    public class Program
    {
        public static void Main()
        {
            TACMATests tACMATests = new TACMATests();

            tACMATests.RunAllTests();

           
            AccelerationConsistencyTests accelerationConsistencyTests = new AccelerationConsistencyTests();
            accelerationConsistencyTests.RunAllTests();

            Tests tt = new Tests();
            tt.RunAllTests();

            CrossAttentionMultimodalExamples.RunAll();

            RunNeuralNetworkBooleanLogicTests();

            Console.WriteLine("Press any key to continue.");
            Console.ReadKey();
        }
        public static void RunNeuralNetworkBooleanLogicTests()
        {
            NeuralNetworkBooleanLogicTests neuralNetwork_Tests = new NeuralNetworkBooleanLogicTests();

            TryMethod(() => neuralNetwork_Tests.NeuralNetworkXorTestPolynomial());
            TryMethod(() => neuralNetwork_Tests.NeuralNetworkOrTest());
            TryMethod(() => neuralNetwork_Tests.NeuralNetworkAndCPUTest());
            TryMethod(() => neuralNetwork_Tests.NeuralNetworkBatchXorTestCUDA());


            TryMethod(() => neuralNetwork_Tests.NeuralNetworkAndGPUTest());
            TryMethod(()=>neuralNetwork_Tests.NeuralNetworkXorTestAutoTuneTest());

            Console.ReadLine();
        }

        private static void TryMethod(Action action)
        {
            try
            {
                action();   // invoke the method
            }
            catch (Exception ex)
            {
                Console.ForegroundColor = ConsoleColor.Red;
                Console.WriteLine(ex.Message);
                Console.ResetColor();
            }
        }


        public static class CrossAttentionMultimodalExamples
        {
            // ==========================================================================
            // Basic - Synthetic text and price data
            // ==========================================================================
            public static void BasicExample()
            {
                Console.WriteLine("=== Cross-Attention Multimodal: Basic Example ===\n");

                var random = new Random(42);

                // --- Step 1: Create a simple tokenizer ---
                string[] corpus = new[]
                {
                "stock price rose sharply today",
                "market crashed due to earnings miss",
                "bullish sentiment on tech sector",
                "bearish outlook for energy stocks",
                "neutral trading day with low volume",
                "strong earnings beat expectations",
                "weak guidance sent shares lower",
                "analysts upgrade rating to buy",
                "federal reserve raises interest rates",
                "inflation data came in hot"
            };

                var tokenizer = new BPETokenizer();
                tokenizer.Train(corpus, vocabSize: 200, minFrequency: 1);
                Console.WriteLine($"Text vocabulary size: {tokenizer.VocabSize}");

                // --- Step 2: Generate synthetic training data ---
                int numSamples = 20;
                int priceSeqLen = 10;
                int inputFeatures = 5; // OHLCV
                int outputDim = 5;     // predict next OHLCV

                var textSequences = new int[numSamples][];
                var priceInputs = new float[numSamples][,];
                var priceTargets = new float[numSamples][,];

                for (int s = 0; s < numSamples; s++)
                {
                    // Random text from corpus
                    string text = corpus[random.Next(corpus.Length)];
                    textSequences[s] = tokenizer.Encode(text, addSpecialTokens: true);

                    // Generate price data
                    priceInputs[s] = new float[priceSeqLen, inputFeatures];
                    priceTargets[s] = new float[priceSeqLen, outputDim];

                    bool bullish = text.Contains("rose") || text.Contains("bullish")
                                || text.Contains("beat") || text.Contains("upgrade");
                    float trend = bullish ? 0.02f : -0.01f;
                    float basePrice = 100f + (float)(random.NextDouble() * 50);

                    for (int t = 0; t < priceSeqLen; t++)
                    {
                        float noise = (float)(random.NextDouble() - 0.5) * 3f;
                        float close = basePrice + noise;
                        float open = close + (float)(random.NextDouble() - 0.5) * 2f;
                        float high = Math.Max(open, close) + (float)(random.NextDouble() * 2f);
                        float low = Math.Min(open, close) - (float)(random.NextDouble() * 2f);
                        float volume = 1000f + (float)(random.NextDouble() * 500f);

                        // Normalize to ~[0,1]
                        priceInputs[s][t, 0] = open / 200f;
                        priceInputs[s][t, 1] = high / 200f;
                        priceInputs[s][t, 2] = low / 200f;
                        priceInputs[s][t, 3] = close / 200f;
                        priceInputs[s][t, 4] = volume / 2000f;

                        // Target: next day's OHLCV (with trend influence from text)
                        float nextBase = basePrice + trend * basePrice;
                        float nextNoise = (float)(random.NextDouble() - 0.5) * 3f;
                        float nextClose = nextBase + nextNoise;
                        float nextOpen = nextClose + (float)(random.NextDouble() - 0.5) * 2f;
                        float nextHigh = Math.Max(nextOpen, nextClose) + (float)(random.NextDouble() * 2f);
                        float nextLow = Math.Min(nextOpen, nextClose) - (float)(random.NextDouble() * 2f);
                        float nextVol = 1000f + (float)(random.NextDouble() * 500f);

                        priceTargets[s][t, 0] = nextOpen / 200f;
                        priceTargets[s][t, 1] = nextHigh / 200f;
                        priceTargets[s][t, 2] = nextLow / 200f;
                        priceTargets[s][t, 3] = nextClose / 200f;
                        priceTargets[s][t, 4] = nextVol / 2000f;

                        basePrice = nextBase;
                    }
                }

                Console.WriteLine($"Training samples: {numSamples}");
                Console.WriteLine($"Price sequence length: {priceSeqLen}");
                Console.WriteLine($"Input features: {inputFeatures}");

                // --- Step 3: Configure the model ---
                var config = new Config
                {
                    // Text encoder
                    TextVocabSize = tokenizer.VocabSize +2,
                    TextMaxSequenceLength = 32,
                    TextEmbeddingDim = 32,
                    TextNumHeads = 2,
                    TextNumLayers = 2,
                    TextFeedForwardDim = 64,
                    TextUseDecoderOnly = false, // bidirectional for encoding

                    // Price decoder with cross-attention
                    PriceInputFeatureDim = inputFeatures,
                    PriceMaxSequenceLength = priceSeqLen + 2,
                    PriceEmbeddingDim = 32,  // must match TextEmbeddingDim
                    PriceNumHeads = 2,
                    PriceNumLayers = 2,
                    PriceFeedForwardDim = 64,
                    PriceUseDecoderOnly = true, // causal for price prediction

                    // Output
                    OutputDim = outputDim,
                    UseConfidenceHead = true,

                    // Hardware
                    FFNActivationType = ActivationType.Relu,
                    AccelerationType = AccelerationType.CPU,
                    L2RegulationLamda = 0f,
                    GradientClippingThreshold = 1.0f
                };

                var model = new Model(config, new Random(42));

                // --- Step 4: Train ---
                var trainConfig = new MultimodalTrainingConfig
                {
                    LearningRate = 0.001f,
                    BatchSize = 4,
                    Epochs = 30,
                    UseGradientClipping = true,
                    GradientClipThreshold = 1.0f,
                    ConfidenceLossWeight = 0.1f,
                    Verbose = true
                };

                var trainer = new Trainer(model, trainConfig);
                trainer.Train(textSequences, priceInputs, priceTargets);

                // --- Step 5: Validate ---
                float valLoss = trainer.Validate(textSequences, priceInputs, priceTargets);
                Console.WriteLine($"\nValidation MSE Loss: {valLoss:F6}");

                // --- Step 6: Predict ---
                string testText = "strong earnings beat expectations";
                int[] testTokens = tokenizer.Encode(testText, addSpecialTokens: true);
                var testPrice = priceInputs[0];

                var (prediction, confidence) = model.PredictNext(testTokens, testPrice);
                Console.WriteLine($"\nText context: \"{testText}\"");
                Console.WriteLine($"Predicted OHLCV (normalized):");
                Console.WriteLine($"  Open:   {prediction[0]:F4}  (${prediction[0] * 200:F2})");
                Console.WriteLine($"  High:   {prediction[1]:F4}  (${prediction[1] * 200:F2})");
                Console.WriteLine($"  Low:    {prediction[2]:F4}  (${prediction[2] * 200:F2})");
                Console.WriteLine($"  Close:  {prediction[3]:F4}  (${prediction[3] * 200:F2})");
                Console.WriteLine($"  Volume: {prediction[4]:F4}  ({prediction[4] * 2000:F0})");
                Console.WriteLine($"  Confidence: {confidence:P2}");
            }

            // ==========================================================================
            // Frozen text encoder and fine-tune price decoder only
            // ==========================================================================
            public static void FrozenTextEncoderExample()
            {
                Console.WriteLine("=== Cross-Attention Multimodal: Frozen Text Encoder ===\n");

                var random = new Random(42);

                string[] corpus = new[]
                {
                "price up trend", "price down trend",
                "bullish signal", "bearish signal",
                "neutral market", "volatile trading"
            };

                var tokenizer = new BPETokenizer();
                tokenizer.Train(corpus, vocabSize: 100, minFrequency: 1);

                int numSamples = 15;
                int seqLen = 8;
                var textSeqs = new int[numSamples][];
                var priceInputs = new float[numSamples][,];
                var priceTargets = new float[numSamples][,];

                for (int s = 0; s < numSamples; s++)
                {
                    textSeqs[s] = tokenizer.Encode(corpus[random.Next(corpus.Length)], addSpecialTokens: true);
                    priceInputs[s] = new float[seqLen, 3]; // simplified: open, close, volume
                    priceTargets[s] = new float[seqLen, 3];

                    float price = 50f + (float)(random.NextDouble() * 50);
                    for (int t = 0; t < seqLen; t++)
                    {
                        float n = (float)(random.NextDouble() - 0.5) * 5f;
                        priceInputs[s][t, 0] = (price + n) / 100f;
                        priceInputs[s][t, 1] = price / 100f;
                        priceInputs[s][t, 2] = (500f + (float)random.NextDouble() * 500f) / 1000f;

                        price += (float)(random.NextDouble() - 0.48) * 3f;
                        priceTargets[s][t, 0] = (price + n) / 100f;
                        priceTargets[s][t, 1] = price / 100f;
                        priceTargets[s][t, 2] = (500f + (float)random.NextDouble() * 500f) / 1000f;
                    }
                }

                var config = new Config
                {
                    TextVocabSize = tokenizer.VocabSize,
                    TextMaxSequenceLength = 16,
                    TextEmbeddingDim = 16,
                    TextNumHeads = 2,
                    TextNumLayers = 1,
                    TextFeedForwardDim = 32,
                    TextUseDecoderOnly = false,

                    PriceInputFeatureDim = 3,
                    PriceMaxSequenceLength = seqLen + 2,
                    PriceEmbeddingDim = 16,
                    PriceNumHeads = 2,
                    PriceNumLayers = 1,
                    PriceFeedForwardDim = 32,

                    OutputDim = 3,
                    UseConfidenceHead = false,
                    FreezeTextEncoder = true, // <-- KEY: text encoder is frozen

                    FFNActivationType = ActivationType.Relu,
                    AccelerationType = AccelerationType.CPU
                };

                var model = new Model(config, new Random(42));

                var trainConfig = new MultimodalTrainingConfig
                {
                    LearningRate = 0.001f,
                    BatchSize = 5,
                    Epochs = 20,
                    Verbose = true
                };

                var trainer = new Trainer(model, trainConfig);

                // Verify text encoder doesn't change
                float textEmbBefore = model.TextTokenEmbedding[1, 0];

                trainer.Train(textSeqs, priceInputs, priceTargets);

                float textEmbAfter = model.TextTokenEmbedding[1, 0];
                Console.WriteLine($"\nText embedding[1,0] before: {textEmbBefore:F6}, after: {textEmbAfter:F6}");
                Console.WriteLine($"Text encoder frozen: {textEmbBefore == textEmbAfter}");

                float valLoss = trainer.Validate(textSeqs, priceInputs, priceTargets);
                Console.WriteLine($"Validation Loss: {valLoss:F6}");
            }

            public static void RunAll()
            {
                BasicExample();
                Console.WriteLine("\n" + new string('=', 60) + "\n");
                FrozenTextEncoderExample();
            }
        }
    }
}

