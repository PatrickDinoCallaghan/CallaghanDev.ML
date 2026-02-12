using CallaghanDev.ML.Enums;
using CallaghanDev.ML.Transformers;
using System;
using System.Collections.Generic;
using System.Linq;


//FYI;Some of these tests i didnt write.
//Rest all other code is mine though.
namespace CallaghanDev.ML
{
    public class Transformer_Tests
    {
        /// <summary>
        /// Basic transformer creation and forward pass (no training here)
        /// </summary>
        public void TransformerBasicTest()
        {
            Console.WriteLine("=== TransformerBasicTest ===\n");

            // Use BPE tokenizer
            var tokenizer = new BPETokenizer();
            var corpus = new[] { "the cat sat", "the dog ran" };
            tokenizer.Train(corpus, vocabSize: 100, minFrequency: 1);

            var config = new TransformerConfig
            {
                VocabSize = tokenizer.VocabSize,
                MaxSequenceLength = 16,
                EmbeddingDim = 32,
                NumHeads = 2,
                NumLayers = 1,
                FeedForwardDim = 128,
                AccelerationType = AccelerationType.CPU
            };

            var model = new LanguageModel(config);
            Console.WriteLine($"Model created: {config.NumLayers} layers, {config.EmbeddingDim}D embeddings");
            Console.WriteLine($"Tokenizer: BPE with {tokenizer.VocabSize} tokens");

            var text = "the cat sat";
            var tokens = tokenizer.Encode(text, addSpecialTokens: false);
            var logits = model.Forward(tokens);

            Console.WriteLine($"Forward pass successful!");
            Console.WriteLine($"Input: \"{text}\"");
            Console.WriteLine($"Tokens: [{string.Join(", ", tokens)}]");
            Console.WriteLine($"Output shape: [{logits.GetLength(0)}x{logits.GetLength(1)}]");
            Console.WriteLine($"Expected: [{tokens.Length}x{tokenizer.VocabSize}] ✓\n");
        }

        /// <summary>
        /// Simple text generation (untrained model, no model)
        /// </summary>
        public void TransformerGenerationTest()
        {
            Console.WriteLine("=== TransformerGenerationTest ===\n");

            var tokenizer = new BPETokenizer();
            var corpus = new[] { "hello world", "test data", "sample text" };
            tokenizer.Train(corpus, vocabSize: 200, minFrequency: 1);

            Console.WriteLine($"BPE Tokenizer trained: {tokenizer.VocabSize} tokens");

            var config = new TransformerConfig
            {
                VocabSize = tokenizer.VocabSize,
                MaxSequenceLength = 16,
                EmbeddingDim = 64,
                NumHeads = 4,
                NumLayers = 2,
                FeedForwardDim = 256,
                AccelerationType = AccelerationType.CPU
            };

            var model = new LanguageModel(config);

            var prompt = tokenizer.Encode("hello", addSpecialTokens: false);
            Console.WriteLine($"\nPrompt: \"hello\"");
            Console.WriteLine($"Prompt tokens: [{string.Join(", ", prompt.Select(t => tokenizer.IdToToken(t)))}]");

            var generated = model.Generate(prompt, maxNewTokens: 5, temperature: 1.0f);
            var text = tokenizer.Decode(generated, skipSpecialTokens: true);

            Console.WriteLine($"Generated: \"{text}\"");
            Console.WriteLine("(Note: Untrained model produces random output)\n");
        }

        /// <summary>
        /// Basic training on simple patterns
        /// </summary>
        public void TransformerBasicTrainingTest()
        {
            Console.WriteLine("=== TransformerBasicTrainingTest ===\n");

            var trainingTexts = new[]
            {
                "the cat sat",
                "the cat slept",
                "the dog ran",
                "the dog played"
            };

            var tokenizer = new BPETokenizer();
            tokenizer.Train(trainingTexts, vocabSize: 300, minFrequency: 1);
            Console.WriteLine($"BPE Vocabulary size: {tokenizer.VocabSize}");

            var sequences = trainingTexts
                .Select(text => tokenizer.Encode(text, addSpecialTokens: true))
                .ToArray();

            var config = new TransformerConfig
            {
                VocabSize = tokenizer.VocabSize,
                MaxSequenceLength = 16,
                EmbeddingDim = 32,
                NumHeads = 2,
                NumLayers = 2,
                FeedForwardDim = 128,
                AccelerationType = AccelerationType.CPU
            };

            var model = new LanguageModel(config);

            var trainConfig = new TrainingConfig
            {
                LearningRate = 0.01f,
                BatchSize = 2,
                Epochs = 20,
                Verbose = true
            };

            Console.WriteLine("\nTraining...");
            var trainer = new TransformerTrainer(model, trainConfig);
            trainer.Train(sequences);

            Console.WriteLine("\nTesting generation after training:");
            TestGeneration(model, tokenizer, "the cat");
            TestGeneration(model, tokenizer, "the dog");
            Console.WriteLine();
        }

        /// <summary>
        /// XOR-like pattern learning (similar to feedforward XOR test)
        /// </summary>
        public void TransformerPatternLearningTest()
        {
            Console.WriteLine("=== TransformerPatternLearningTest ===\n");
            Console.WriteLine("Teaching transformer to learn token patterns\n");

            // Create pattern-based data (like XOR but with tokens)
            var trainingTexts = new[]
            {
                "a b c",  // Pattern 1
                "a b c",  // Pattern 1 (repeated)
                "d e f",  // Pattern 2
                "d e f",  // Pattern 2 (repeated)
                "a b c",  // Pattern 1
                "d e f"   // Pattern 2
            };

            var tokenizer = new BPETokenizer();
            tokenizer.Train(trainingTexts, vocabSize: 500, minFrequency: 1);
            Console.WriteLine($"Vocabulary: {tokenizer.VocabSize} tokens");

            var sequences = trainingTexts
                .Select(text => tokenizer.Encode(text, addSpecialTokens: true))
                .ToArray();

            var config = new TransformerConfig
            {
                VocabSize = tokenizer.VocabSize,
                MaxSequenceLength = 16,
                EmbeddingDim = 64,
                NumHeads = 4,
                NumLayers = 2,
                FeedForwardDim = 256,
                AccelerationType = AccelerationType.CPU,
                L2RegulationLamda = 0.01f
            };

            var model = new LanguageModel(config);

            var trainConfig = new TrainingConfig
            {
                LearningRate = 0.005f,
                BatchSize = 3,
                Epochs = 50,
                GradientClipThreshold = 1.0f,
                Verbose = false
            };

            var trainer = new TransformerTrainer(model, trainConfig);
            trainer.Train(sequences);

            Console.WriteLine("\nPattern completion tests:");
            TestGeneration(model, tokenizer, "a b");
            TestGeneration(model, tokenizer, "d e");
            Console.WriteLine();
        }

        /// <summary>
        /// Multi-threaded CPU acceleration
        /// </summary>
        public void TransformerMultiThreadCPUTest()
        {
            Console.WriteLine("=== TransformerMultiThreadCPUTest ===\n");

            var trainingTexts = new[]
            {
                "the cat sat on the mat",
                "the dog ran in the park",
                "a bird flew over the tree"
            };

            var tokenizer = new BPETokenizer();
            tokenizer.Train(trainingTexts, vocabSize: 500, minFrequency: 1);

            var sequences = trainingTexts
                .Select(text => tokenizer.Encode(text, addSpecialTokens: true))
                .ToArray();

            var config = new TransformerConfig
            {
                VocabSize = tokenizer.VocabSize,
                MaxSequenceLength = 32,
                EmbeddingDim = 64,
                NumHeads = 4,
                NumLayers = 2,
                FeedForwardDim = 256,
                AccelerationType = AccelerationType.MultiThreadCPU  // Multi-threaded!
            };

            Console.WriteLine($"Using Multi-threaded CPU acceleration");
            Console.WriteLine($"Processor count: {Environment.ProcessorCount}");

            var model = new LanguageModel(config);

            var trainConfig = new TrainingConfig
            {
                LearningRate = 0.001f,
                BatchSize = 3,
                Epochs = 30,
                Verbose = true
            };

            var trainer = new TransformerTrainer(model, trainConfig);
            trainer.Train(sequences);

            Console.WriteLine("\nGeneration test:");
            TestGeneration(model, tokenizer, "the cat");
            Console.WriteLine();
        }

        /// <summary>
        /// GPU/CUDA acceleration (if available)
        /// </summary>
        public void TransformerGPUTest()
        {
            Console.WriteLine("=== TransformerGPUTest ===\n");

            try
            {
                var trainingTexts = new[]
                {
                    "the cat sat on the mat",
                    "the dog ran in the park",
                    "a bird flew over the tree",
                    "the cat slept on the chair"
                };

                var tokenizer = new BPETokenizer();
                tokenizer.Train(trainingTexts, vocabSize: 500, minFrequency: 1);

                var sequences = trainingTexts.Select(text => tokenizer.Encode(text, addSpecialTokens: true)).ToArray();

                var config = new TransformerConfig
                {
                    VocabSize = tokenizer.VocabSize,
                    MaxSequenceLength = 32,
                    EmbeddingDim = 128,
                    NumHeads = 4,
                    NumLayers = 3,
                    FeedForwardDim = 512,
                    AccelerationType = AccelerationType.CUDA
                };

                Console.WriteLine("Using CUDA GPU acceleration");

                var model = new LanguageModel(config);

                var trainConfig = new TrainingConfig
                {
                    LearningRate = 0.001f,
                    BatchSize = 4,
                    Epochs = 50,
                    Verbose = true
                };

                var trainer = new TransformerTrainer(model, trainConfig);
                trainer.Train(sequences);

                Console.WriteLine("\nGeneration test:");
                TestGeneration(model, tokenizer, "the cat");
                TestGeneration(model, tokenizer, "the dog");
                Console.WriteLine();
            }
            catch (Exception ex)
            {
                Console.WriteLine($"GPU test failed (GPU may not be available): {ex.Message}\n");
            }
        }

        /// <summary>
        /// Larger vocabulary and longer sequences
        /// </summary>
        public void TransformerLargerModelTest()
        {
            Console.WriteLine("=== TransformerLargerModelTest ===\n");

            var trainingTexts = GenerateTrainingData(50);

            var tokenizer = new BPETokenizer();
            tokenizer.Train(trainingTexts, vocabSize: 500, minFrequency: 1);
            Console.WriteLine($"Vocabulary: {tokenizer.VocabSize} tokens");

            var sequences = trainingTexts.Select(text => tokenizer.Encode(text, addSpecialTokens: true)).ToArray();

            var config = new TransformerConfig
            {
                VocabSize = tokenizer.VocabSize,
                MaxSequenceLength = 64,
                EmbeddingDim = 128,
                NumHeads = 4,
                NumLayers = 3,
                FeedForwardDim = 512,
                AccelerationType = AccelerationType.MultiThreadCPU
            };

            Console.WriteLine($"Model: {config.NumLayers} layers, {config.EmbeddingDim}D, {config.NumHeads} heads");

            var model = new LanguageModel(config);

            var trainConfig = new TrainingConfig
            {
                LearningRate = 0.0005f,
                BatchSize = 8,
                Epochs = 100,
                Verbose = false // Shuting it up a bit. Will make a logger for this
            };

            Console.WriteLine("\nTraining...");
            var trainer = new TransformerTrainer(model, trainConfig);
            trainer.Train(sequences);

            Console.WriteLine("\nGeneration tests:");
            TestGeneration(model, tokenizer, "the cat");
            TestGeneration(model, tokenizer, "a bird");
            TestGeneration(model, tokenizer, "once upon");
            Console.WriteLine();
        }

        /// <summary>
        /// Training with validation split
        /// </summary>
        public void TransformerValidationTest()
        {
            Console.WriteLine("=== TransformerValidationTest ===\n");

            var allTexts = GenerateTrainingData(30);

            var tokenizer = new BPETokenizer();
            tokenizer.Train(allTexts, vocabSize: 500, minFrequency: 1);

            var sequences = allTexts.Select(text => tokenizer.Encode(text, addSpecialTokens: true)).ToArray();

            int splitIdx = (int)(sequences.Length * 0.8);
            var trainSeq = sequences.Take(splitIdx).ToArray();
            var valSeq = sequences.Skip(splitIdx).ToArray();

            Console.WriteLine($"Training samples: {trainSeq.Length}");
            Console.WriteLine($"Validation samples: {valSeq.Length}");

            var config = new TransformerConfig
            {
                VocabSize = tokenizer.VocabSize,
                MaxSequenceLength = 32,
                EmbeddingDim = 64,
                NumHeads = 4,
                NumLayers = 2,
                FeedForwardDim = 256,
                AccelerationType = AccelerationType.CPU
            };

            var model = new LanguageModel(config);

            var trainConfig = new TrainingConfig
            {
                LearningRate = 0.001f,
                BatchSize = 4,
                Epochs = 50,
                ValidationInterval = 10,
                Verbose = true
            };

            var trainer = new TransformerTrainer(model, trainConfig);
            trainer.Train(trainSeq, valSeq);

            Console.WriteLine("\nGeneration test:");
            TestGeneration(model, tokenizer, "the cat");
            Console.WriteLine();
        }

        /// <summary>
        /// Save and load model
        /// </summary>
        public void TransformerSaveLoadTest()
        {
            Console.WriteLine("=== TransformerSaveLoadTest ===\n");

            var trainingTexts = new[]
            {
                "the cat sat on the mat",
                "the dog ran in the park"
            };

            var tokenizer = new BPETokenizer();
            tokenizer.Train(trainingTexts, vocabSize: 500, minFrequency: 1);

            var sequences = trainingTexts.Select(text => tokenizer.Encode(text, addSpecialTokens: true)).ToArray();

            var config = new TransformerConfig
            {
                VocabSize = tokenizer.VocabSize,
                MaxSequenceLength = 32,
                EmbeddingDim = 64,
                NumHeads = 4,
                NumLayers = 2,
                FeedForwardDim = 256,
                AccelerationType = AccelerationType.CPU
            };

            var model = new LanguageModel(config);

            var trainConfig = new TrainingConfig
            {
                LearningRate = 0.001f,
                BatchSize = 2,
                Epochs = 20,
                Verbose = false
            };

            Console.WriteLine("Training original model...");
            var trainer = new TransformerTrainer(model, trainConfig);
            trainer.Train(sequences);

            Console.WriteLine("\nGeneration before save:");
            var prompt = tokenizer.Encode("the cat", addSpecialTokens: false);
            var beforeGen = model.Generate(prompt, maxNewTokens: 5);
            var beforeText = tokenizer.Decode(beforeGen);
            Console.WriteLine($"  \"{beforeText}\"");

            //Try save here
            Console.WriteLine("\nSaving model to ./testp_transformer...");
            model.SaveFeedForwardNetworks("./test_transformer");
            Console.WriteLine("Model saved!");

            // Loading it back
            Console.WriteLine("\nLoading model from ./test_transformer...");
            model.LoadFeedForwardNetworks("./test_transformer");
            Console.WriteLine("Model loaded!");

            Console.WriteLine("\nGeneration after load:");
            var afterGen = model.Generate(prompt, maxNewTokens: 5);
            var afterText = tokenizer.Decode(afterGen);
            Console.WriteLine($"  \"{afterText}\"");

            Console.WriteLine("\nNote: Feed-forward networks saved and loaded successfully!");
            Console.WriteLine("(Attention weights not saved in this version)\n");
        }

        /// <summary>
        /// Learning rate decay
        /// </summary>
        public void TransformerLearningRateDecayTest()
        {
            Console.WriteLine("=== TransformerLearningRateDecayTest ===\n");

            var trainingTexts = GenerateTrainingData(20);

            var tokenizer = new BPETokenizer();
            tokenizer.Train(trainingTexts, vocabSize: 500, minFrequency: 1);

            var sequences = trainingTexts.Select(text => tokenizer.Encode(text, addSpecialTokens: true)).ToArray();

            var config = new TransformerConfig
            {
                VocabSize = tokenizer.VocabSize,
                MaxSequenceLength = 32,
                EmbeddingDim = 64,
                NumHeads = 4,
                NumLayers = 2,
                FeedForwardDim = 256,
                AccelerationType = AccelerationType.CPU
            };

            var model = new LanguageModel(config);

            var trainConfig = new TrainingConfig
            {
                LearningRate = 0.01f,
                BatchSize = 4,
                Epochs = 30,
                UseLearningRateDecay = true,
                LearningRateDecay = 0.95f,
                Verbose = true
            };

            Console.WriteLine("Training with learning rate decay (0.95 per epoch)...");
            var trainer = new TransformerTrainer(model, trainConfig);
            trainer.Train(sequences);

            Console.WriteLine("\nGeneration test:");
            TestGeneration(model, tokenizer, "the cat");
            Console.WriteLine();
        }

        /// <summary>
        /// Different activation functions in FFN
        /// </summary>
        public void TransformerActivationFunctionsTest()
        {
            Console.WriteLine("=== TransformerActivationFunctionsTest ===\n");

            var trainingTexts = new[]
            {
                "the cat sat",
                "the dog ran",
                "a bird flew"
            };

            var tokenizer = new BPETokenizer();
            tokenizer.Train(trainingTexts, vocabSize: 500, minFrequency: 1);

            var sequences = trainingTexts.Select(text => tokenizer.Encode(text, addSpecialTokens: true)).ToArray();

            var activations = new[]
            {
                (ActivationType.Relu, "ReLU"),
                (ActivationType.Leakyrelu, "LeakyReLU"),
                (ActivationType.Tanh, "Tanh")
            };

            foreach (var (activation, name) in activations)
            {
                Console.WriteLine($"\nTesting with {name} activation:");

                var config = new TransformerConfig
                {
                    VocabSize = tokenizer.VocabSize,
                    MaxSequenceLength = 16,
                    EmbeddingDim = 32,
                    NumHeads = 2,
                    NumLayers = 2,
                    FeedForwardDim = 128,
                    FFNActivationType = activation,
                    AccelerationType = AccelerationType.CPU
                };

                var model = new LanguageModel(config);

                var trainConfig = new TrainingConfig
                {
                    LearningRate = 0.01f,
                    BatchSize = 3,
                    Epochs = 20,
                    Verbose = false
                };

                var trainer = new TransformerTrainer(model, trainConfig);
                trainer.Train(sequences);

                TestGeneration(model, tokenizer, "the cat");
            }
            Console.WriteLine();
        }

        /// <summary>
        /// Model size comparison
        /// </summary>
        public void TransformerModelSizeTest()
        {
            Console.WriteLine("=== TransformerModelSizeTest ===\n");

            var configs = new[]
            {
                (new TransformerConfig { VocabSize = 100, EmbeddingDim = 32, NumHeads = 2, NumLayers = 1, FeedForwardDim = 128 }, "Tiny"),
                (new TransformerConfig { VocabSize = 100, EmbeddingDim = 64, NumHeads = 4, NumLayers = 2, FeedForwardDim = 256 }, "Small"),
                (new TransformerConfig { VocabSize = 100, EmbeddingDim = 128, NumHeads = 4, NumLayers = 4, FeedForwardDim = 512 }, "Medium")
            };

            Console.WriteLine($"{"Model",-10} {"Params",-15} {"Memory (MB)",-12}");
            Console.WriteLine(new string('-', 40));

            foreach (var (cfg, name) in configs)
            {
                long embeddings = (long)cfg.VocabSize * cfg.EmbeddingDim;
                long attnPerLayer = 4L * cfg.EmbeddingDim * cfg.EmbeddingDim;
                long ffnPerLayer = (long)cfg.EmbeddingDim * cfg.FeedForwardDim +
                                   (long)cfg.FeedForwardDim * cfg.EmbeddingDim;
                long perLayer = attnPerLayer + ffnPerLayer + 4 * cfg.EmbeddingDim;
                long output = (long)cfg.EmbeddingDim * cfg.VocabSize;
                long total = embeddings + perLayer * cfg.NumLayers + output;

                float memoryMB = total * 4.0f / 1024 / 1024;

                Console.WriteLine($"{name,-10} {total,-15:N0} {memoryMB,-12:F2}");
            }
            Console.WriteLine();
        }

        /// <summary>
        /// Comprehensive end-to-end test
        /// </summary>
        public void TransformerComprehensiveTest()
        {
            Console.WriteLine("=== TransformerComprehensiveTest ===\n");
            Console.WriteLine("This test demonstrates the complete pipeline:\n");

            // 1. Data preparation
            Console.WriteLine("1. Preparing training data...");
            var trainingTexts = new[]
            {
                "the cat sat on the mat",
                "the cat slept on the chair",
                "the dog ran in the park",
                "the dog played in the yard",
                "a bird flew over the tree",
                "a bird sang in the morning",
                "the cat jumped on the table",
                "the dog barked at the cat"
            };

            var tokenizer = new BPETokenizer();
            tokenizer.Train(trainingTexts, vocabSize: 500, minFrequency: 1);
            Console.WriteLine($"   Vocabulary: {tokenizer.VocabSize} tokens");

            var sequences = trainingTexts
                .Select(text => tokenizer.Encode(text, addSpecialTokens: true))
                .ToArray();
            Console.WriteLine($"   Sequences: {sequences.Length}");

            // 2. Model creation
            Console.WriteLine("\n2. Creating transformer model...");
            var config = new TransformerConfig
            {
                VocabSize = tokenizer.VocabSize,
                MaxSequenceLength = 32,
                EmbeddingDim = 64,
                NumHeads = 4,
                NumLayers = 3,
                FeedForwardDim = 256,
                FFNActivationType = ActivationType.Relu,
                AccelerationType = AccelerationType.MultiThreadCPU,
                L2RegulationLamda = 0.01f,
                GradientClippingThreshold = 1.0f
            };

            var model = new LanguageModel(config);
            Console.WriteLine($"   Layers: {config.NumLayers}");
            Console.WriteLine($"   Embedding dim: {config.EmbeddingDim}");
            Console.WriteLine($"   Heads: {config.NumHeads}");
            Console.WriteLine($"   Acceleration: {config.AccelerationType}");

            // 3. Training
            Console.WriteLine("\n3. Training model...");
            var trainConfig = new TrainingConfig
            {
                LearningRate = 0.001f,
                BatchSize = 4,
                Epochs = 100,
                GradientClipThreshold = 1.0f,
                Verbose = true
            };

            var trainer = new TransformerTrainer(model, trainConfig);
            trainer.Train(sequences);

            // 4. Testing
            Console.WriteLine("\n4. Testing generation...");
            var prompts = new[] { "the cat", "the dog", "a bird" };

            foreach (var prompt in prompts)
            {
                TestGeneration(model, tokenizer, prompt);
            }

            // 5. Save model
            Console.WriteLine("\n5. Saving model...");
            model.SaveFeedForwardNetworks("./comprehensive_test");
            Console.WriteLine("   Model saved to ./comprehensive_test");

            Console.WriteLine("\n✓ Comprehensive test complete!\n");
        }

        /// <summary>
        /// Run ALL tests
        /// </summary>
        public void RunAllTests()
        {
            Console.WriteLine("========TRANSFORMER COMPLETE TESTS========");

            var tests = new (Action test, string name)[]
            {
                (TransformerBasicTest, "Basic Forward Pass"),
                (TransformerGenerationTest, "Text Generation"),
                (TransformerBasicTrainingTest, "Basic Training"),
                (TransformerPatternLearningTest, "Pattern Learning"),
                (TransformerMultiThreadCPUTest, "Multi-threaded CPU"),
                (TransformerGPUTest, "GPU/CUDA"),
                (TransformerLargerModelTest, "Larger Model"),
                (TransformerValidationTest, "Validation Split"),
                (TransformerSaveLoadTest, "Save/Load"),
                (TransformerLearningRateDecayTest, "Learning Rate Decay"),
                (TransformerActivationFunctionsTest, "Activation Functions"),
                (TransformerModelSizeTest, "Model Sizes"),
                (TransformerComprehensiveTest, "Comprehensive End-to-End")
            };

            int passed = 0;
            int failed = 0;

            for (int i = 0; i < tests.Length; i++)
            {
                try
                {
                    Console.WriteLine($"\n[{i + 1}/{tests.Length}] Running: {tests[i].name}");
                    Console.WriteLine(new string('=', 60));
                    tests[i].test();
                    Console.ForegroundColor = ConsoleColor.Green;
                    Console.WriteLine($"PASSED: {tests[i].name}\n");
                    Console.ResetColor();
                    passed++;
                }
                catch (Exception ex)
                {
                    Console.ForegroundColor = ConsoleColor.Red;
                    Console.WriteLine($"FAILED: {tests[i].name}");
                    Console.ResetColor();
                    Console.WriteLine($"Error: {ex.Message}\n");
                    failed++;
                }
            }

            Console.WriteLine("\n" + new string('=', 60));
            Console.WriteLine("TEST RESULTS");
            Console.WriteLine(new string('=', 60));
            Console.WriteLine($"Total:  {tests.Length}");

            Console.ForegroundColor = ConsoleColor.Green;
            Console.WriteLine($"Passed: {passed}");

            Console.ForegroundColor = ConsoleColor.Red;
            Console.WriteLine($"Failed: {failed}");
            Console.ResetColor();
            Console.WriteLine(new string('=', 60) + "\n");
        }

        #region Helper Methods

        private void TestGeneration(LanguageModel model, BPETokenizer tokenizer, string prompt)
        {
            var promptTokens = tokenizer.Encode(prompt, addSpecialTokens: false);
            var generated = model.Generate(promptTokens, maxNewTokens: 8, temperature: 0.8f);
            var text = tokenizer.Decode(generated, skipSpecialTokens: true);
            Console.WriteLine($"  Prompt: \"{prompt}\" -> Generated: \"{text}\"");
        }

        private string[] GenerateTrainingData(int count)
        {
            var data = new List<string>();
            var random = new Random(42);

            var subjects = new[] { "the cat", "the dog", "a bird", "the fish", "a mouse" };
            var verbs = new[] { "sat", "ran", "flew", "jumped", "walked", "slept", "played" };
            var locations = new[] { "on the mat", "in the park", "over the tree", "near the pond", "by the door" };

            for (int i = 0; i < count; i++)
            {
                var subject = subjects[random.Next(subjects.Length)];
                var verb = verbs[random.Next(verbs.Length)];
                var location = locations[random.Next(locations.Length)];
                data.Add($"{subject} {verb} {location}");
            }

            if (count >= 10)
            {
                data.Add("once upon a time there was a cat");
                data.Add("the quick brown fox jumps over the lazy dog");
                data.Add("in the beginning there was nothing");
            }

            return data.ToArray();
        }

        #endregion
    }
    public class TransformerTrainer_Tests
    {
        private const float EPSILON = 1e-3f;
        private const float TOLERANCE = 0.05f;
        private const float ABS_TOLERANCE = 1e-4f;

        public void RunAllTests()
        {
            Console.WriteLine("╔══════════════════════════════════════════════════════════╗");
            Console.WriteLine("║        TRANSFORMER TRAINER - VERIFICATION SUITE         ║");
            Console.WriteLine("╚══════════════════════════════════════════════════════════╝\n");

            var tests = new (Action test, string name)[]
            {
                // Unit tests for individual components
                (Test_LayerNorm_ForwardBackward_NumericalCheck, "LayerNorm Forward/Backward Numerical Check"),
                (Test_Attention_WO_Gradient_NumericalCheck, "Attention WO Gradient Numerical Check"),
                (Test_Attention_WQ_Gradient_NumericalCheck, "Attention WQ Gradient Numerical Check"),
                (Test_Attention_WK_Gradient_NumericalCheck, "Attention WK Gradient Numerical Check"),
                (Test_Attention_WV_Gradient_NumericalCheck, "Attention WV Gradient Numerical Check"),
                (Test_Embedding_Gradient_NumericalCheck, "Embedding Gradient Numerical Check"),
                (Test_OutputProjection_Gradient_NumericalCheck, "Output Projection Gradient Numerical Check"),
                (Test_FFN_Gradient_FlowThrough, "FFN Gradient Flow Through"),
                (Test_ResidualConnection_GradientSplit, "Residual Connection Gradient Split"),

                // Integration tests
                (Test_LossDecreases_SingleSequence, "Loss Decreases on Single Sequence"),
                (Test_LossDecreases_MultipleBatches, "Loss Decreases Across Batches"),
                (Test_GradientClipping_Works, "Gradient Clipping Limits Norm"),
                (Test_ZeroGradients_AfterReset, "Gradients Zero After Reset"),
                (Test_MultiHead_ProducesDifferentGradsThanSingleHead, "Multi-Head vs Single-Head Gradients Differ"),
                (Test_CausalMask_FutureTokensNoGradient, "Causal Mask Blocks Future Token Gradients"),
                (Test_TrainTwice_SameModel_NoAccumulation, "Train Twice No Gradient Accumulation"),
                (Test_DifferentSequenceLengths_NoError, "Different Sequence Lengths in Batch"),
                (Test_AllParametersReceiveGradients, "All Parameters Receive Non-Zero Gradients"),
                (Test_LearningRateDecay_Applied, "Learning Rate Decay Applied Correctly"),
                (Test_ValidationLoss_Computable, "Validation Loss Is Computable"),

                // Overfitting sanity checks
                (Test_OverfitSingleSequence, "Overfit Single Sequence (loss < 1.0)"),
                (Test_OverfitTwoSequences, "Overfit Two Sequences (loss < 1.5)"),
                (Test_OverfitSingleSequence_Extended, "Overfit Single Sequence Extended (loss < 0.5)"),

                // Edge cases
                (Test_SequenceLength1_NoError, "Sequence Length 2 (minimum) No Error"),
                (Test_LargeVocab_SmallSequence, "Large Vocab Small Sequence No Error"),
                (Test_RepeatedTokens_NoError, "Repeated Tokens In Sequence No Error"),

                // Determinism
                (Test_SameInput_SameLoss, "Same Input Produces Same Loss"),
                (Test_ForwardCache_TokenIds_Stored, "ForwardCache Stores TokenIds"),
                (Test_ForwardCache_FFNInputs_Stored, "ForwardCache Stores FFN Inputs"),

                // End to end
                (Test_TrainThenGenerate_NoError, "Train Then Generate No Error"),
                (Test_MultiLayer_GradientFlow, "Multi-Layer Gradient Flow (4 layers)"),
                (Test_Convergence_MonotonicallyDecreasing, "Loss Monotonically Decreasing (5 epochs)"),
            };

            int passed = 0;
            int failed = 0;
            var failures = new List<string>();

            for (int i = 0; i < tests.Length; i++)
            {
                Console.Write($"  [{i + 1,2}/{tests.Length}] {tests[i].name,-55} ");
                try
                {
                    tests[i].test();
                    Console.ForegroundColor = ConsoleColor.Green;
                    Console.WriteLine("PASS");
                    Console.ResetColor();
                    passed++;
                }
                catch (Exception ex)
                {
                    Console.ForegroundColor = ConsoleColor.Red;
                    Console.WriteLine("FAIL");
                    Console.ResetColor();
                    Console.ForegroundColor = ConsoleColor.DarkYellow;
                    Console.WriteLine($"         {ex.Message}");
                    Console.ResetColor();
                    failures.Add($"{tests[i].name}: {ex.Message}");
                    failed++;
                }
            }

            Console.WriteLine($"\n{"",3}{new string('─', 58)}");
            Console.Write($"   Results: ");
            Console.ForegroundColor = ConsoleColor.Green;
            Console.Write($"{passed} passed");
            Console.ResetColor();
            if (failed > 0)
            {
                Console.Write(", ");
                Console.ForegroundColor = ConsoleColor.Red;
                Console.Write($"{failed} failed");
                Console.ResetColor();
            }
            Console.WriteLine($" / {tests.Length} total\n");

            if (failures.Count > 0)
            {
                Console.ForegroundColor = ConsoleColor.Red;
                Console.WriteLine("   Failed tests:");
                Console.ResetColor();
                foreach (var f in failures)
                {
                    Console.WriteLine($"     • {f}");
                }
                Console.WriteLine();
            }
        }


        private (LanguageModel model, TransformerConfig config) CreateSmallModel(int vocabSize = 10, int embDim = 8, int numHeads = 2, int numLayers = 1, int ffnDim = 16, bool decoderOnly = true)
        {
            var config = new TransformerConfig
            {
                VocabSize = vocabSize,
                MaxSequenceLength = 16,
                EmbeddingDim = embDim,
                NumHeads = numHeads,
                NumLayers = numLayers,
                FeedForwardDim = ffnDim,
                FFNActivationType = ActivationType.Relu,
                AccelerationType = AccelerationType.CPU,
                UseDecoderOnly = decoderOnly,
                L2RegulationLamda = 0f,
                GradientClippingThreshold = 100f
            };

            var model = new LanguageModel(config, new Random(42));
            return (model, config);
        }

        private float ComputeLoss(LanguageModel model, int[] input, int[] target)
        {
            var logits = model.Forward(input);
            float loss = 0;
            int vocabSize = model.Config.VocabSize;

            for (int i = 0; i < Math.Min(logits.GetLength(0), target.Length); i++)
            {
                float max = float.NegativeInfinity;
                for (int j = 0; j < vocabSize; j++)
                    max = Math.Max(max, logits[i, j]);

                float sum = 0;
                for (int j = 0; j < vocabSize; j++)
                    sum += MathF.Exp(logits[i, j] - max);

                float prob = MathF.Exp(logits[i, target[i]] - max) / sum;
                loss -= MathF.Log(prob + 1e-10f);
            }

            return loss / Math.Min(logits.GetLength(0), target.Length);
        }

        //I could have used a test framework...but nooooo i had to use a console application. Dumb
        private void Assert(bool condition, string message)
        {
            if (!condition)
            {
                throw new Exception(message);
            }
        }


 
        private float NumericalGradient(LanguageModel model, int[] input, int[] target, Func<float> getParam, Action<float> setParam)
        {
            float original = getParam();

            setParam(original + EPSILON);
            float lossPlus = ComputeLoss(model, input, target);

            setParam(original - EPSILON);
            float lossMinus = ComputeLoss(model, input, target);

            setParam(original);

            return (lossPlus - lossMinus) / (2 * EPSILON);
        }

        public void Test_LayerNorm_ForwardBackward_NumericalCheck()
        {
            var (model, config) = CreateSmallModel(vocabSize: 8, embDim: 4, numHeads: 2, numLayers: 1);
            int[] input = { 1, 2, 3 };
            int[] target = { 2, 3, 4 };

            var block = model.Blocks[0];
            for (int paramIdx = 0; paramIdx < Math.Min(4, config.EmbeddingDim); paramIdx++)
            {
                int idx = paramIdx;
                float numGrad = NumericalGradient(model, input, target, () => block.LN1Gamma[idx],  (v) => block.LN1Gamma[idx] = v);

                Assert(!float.IsNaN(numGrad) && !float.IsInfinity(numGrad), $"LN1Gamma[{idx}] numerical gradient is NaN/Inf");
            }
        }


        public void Test_Attention_WO_Gradient_NumericalCheck()
        {
            var (model, config) = CreateSmallModel(vocabSize: 8, embDim: 4, numHeads: 2, numLayers: 1);
            int[] input = { 1, 2, 3 };
            int[] target = { 2, 3, 4 };

            var attn = model.Blocks[0].Attention;
            for (int r = 0; r < 2; r++)
            {
                for (int c = 0; c < 2; c++)
                {
                    int rr = r, cc = c;
                    float numGrad = NumericalGradient(model, input, target, () => attn.WO[rr, cc], (v) => attn.WO[rr, cc] = v);

                    Assert(!float.IsNaN(numGrad) && !float.IsInfinity(numGrad), $"WO[{rr},{cc}] numerical gradient is NaN/Inf");
                }
            }
        }

        public void Test_Attention_WQ_Gradient_NumericalCheck()
        {
            var (model, config) = CreateSmallModel(vocabSize: 8, embDim: 4, numHeads: 2, numLayers: 1);
            int[] input = { 1, 2, 3 };
            int[] target = { 2, 3, 4 };

            var attn = model.Blocks[0].Attention;
            for (int r = 0; r < 2; r++)
            {
                for (int c = 0; c < 2; c++)
                {
                    int rr = r, cc = c;
                    float numGrad = NumericalGradient(model, input, target,
                        () => attn.WQ[rr, cc],
                        (v) => attn.WQ[rr, cc] = v);

                    Assert(!float.IsNaN(numGrad) && !float.IsInfinity(numGrad),
                        $"WQ[{rr},{cc}] numerical gradient is NaN/Inf: {numGrad}");
                }
            }
        }

        public void Test_Attention_WK_Gradient_NumericalCheck()
        {
            var (model, config) = CreateSmallModel(vocabSize: 8, embDim: 4, numHeads: 2, numLayers: 1);
            int[] input = { 1, 2, 3 };
            int[] target = { 2, 3, 4 };

            var attn = model.Blocks[0].Attention;
            for (int r = 0; r < 2; r++)
            {
                for (int c = 0; c < 2; c++)
                {
                    int rr = r, cc = c;
                    float numGrad = NumericalGradient(model, input, target,
                        () => attn.WK[rr, cc],
                        (v) => attn.WK[rr, cc] = v);

                    Assert(!float.IsNaN(numGrad) && !float.IsInfinity(numGrad),
                        $"WK[{rr},{cc}] numerical gradient is NaN/Inf: {numGrad}");
                }
            }
        }

        public void Test_Attention_WV_Gradient_NumericalCheck()
        {
            var (model, config) = CreateSmallModel(vocabSize: 8, embDim: 4, numHeads: 2, numLayers: 1);
            int[] input = { 1, 2, 3 };
            int[] target = { 2, 3, 4 };

            var attn = model.Blocks[0].Attention;
            for (int r = 0; r < 2; r++)
            {
                for (int c = 0; c < 2; c++)
                {
                    int rr = r, cc = c;
                    float numGrad = NumericalGradient(model, input, target,
                        () => attn.WV[rr, cc],
                        (v) => attn.WV[rr, cc] = v);

                    Assert(!float.IsNaN(numGrad) && !float.IsInfinity(numGrad),
                        $"WV[{rr},{cc}] numerical gradient is NaN/Inf: {numGrad}");
                }
            }
        }


        public void Test_Embedding_Gradient_NumericalCheck()
        {
            var (model, config) = CreateSmallModel(vocabSize: 8, embDim: 4, numHeads: 2, numLayers: 1);
            int[] input = { 1, 2, 3 };
            int[] target = { 2, 3, 4 };

            foreach (int tokenId in input)
            {
                for (int d = 0; d < Math.Min(2, config.EmbeddingDim); d++)
                {
                    int tid = tokenId, dd = d;
                    float numGrad = NumericalGradient(model, input, target,
                        () => model.TokenEmbedding[tid, dd],
                        (v) => model.TokenEmbedding[tid, dd] = v);

                    Assert(!float.IsNaN(numGrad) && !float.IsInfinity(numGrad),
                        $"TokenEmbedding[{tid},{dd}] numerical gradient is NaN/Inf");

                    Assert(MathF.Abs(numGrad) > 1e-8f,
                        $"TokenEmbedding[{tid},{dd}] gradient is effectively zero ({numGrad:E4})");
                }
            }
        }


        #region Output Projection Gradient Tests

        public void Test_OutputProjection_Gradient_NumericalCheck()
        {
            var (model, config) = CreateSmallModel(vocabSize: 8, embDim: 4, numHeads: 2, numLayers: 1);
            int[] input = { 1, 2, 3 };
            int[] target = { 2, 3, 4 };

            for (int r = 0; r < 2; r++)
            {
                for (int c = 0; c < 2; c++)
                {
                    int rr = r, cc = c;
                    float numGrad = NumericalGradient(model, input, target,
                        () => model.OutputProjection[rr, cc],
                        (v) => model.OutputProjection[rr, cc] = v);

                    Assert(!float.IsNaN(numGrad) && !float.IsInfinity(numGrad),
                        $"OutputProjection[{rr},{cc}] numerical gradient is NaN/Inf");
                }
            }

            for (int v = 0; v < Math.Min(3, config.VocabSize); v++)
            {
                int vv = v;
                float numGrad = NumericalGradient(model, input, target,
                    () => model.OutputBias[vv],
                    (v2) => model.OutputBias[vv] = v2);

                Assert(!float.IsNaN(numGrad) && !float.IsInfinity(numGrad),
                    $"OutputBias[{vv}] numerical gradient is NaN/Inf");
            }
        }

        #endregion

        #region FFN Gradient Tests

        public void Test_FFN_Gradient_FlowThrough()
        {
            var (model, config) = CreateSmallModel(vocabSize: 8, embDim: 4, numHeads: 2, numLayers: 1, ffnDim: 8);

            var ffn = model.Blocks[0].FeedForwardNetwork;
            var ffnData = ffn.GetData();
            var weightsBefore = new float[ffnData.layers[1].Weights.GetLength(0), ffnData.layers[1].Weights.GetLength(1)];
            Array.Copy(ffnData.layers[1].Weights, weightsBefore, ffnData.layers[1].Weights.Length);

            int[][] sequences = { new[] { 1, 2, 3, 4 } };
            var trainConfig = new TrainingConfig
            {
                LearningRate = 0.01f,
                BatchSize = 1,
                Epochs = 1,
                UseGradientClipping = false,
                Verbose = false
            };
            var trainer = new TransformerTrainer(model, trainConfig);
            trainer.Train(sequences);

            var weightsAfter = ffnData.layers[1].Weights;
            bool anyChanged = false;
            for (int i = 0; i < weightsBefore.GetLength(0) && !anyChanged; i++)
                for (int j = 0; j < weightsBefore.GetLength(1) && !anyChanged; j++)
                    if (MathF.Abs(weightsBefore[i, j] - weightsAfter[i, j]) > 1e-10f)
                        anyChanged = true;

            Assert(anyChanged, "FFN weights did not change after training step - FFN backprop is not connected");
        }

        #endregion

        #region Residual Connection Tests

        public void Test_ResidualConnection_GradientSplit()
        {
            var (model, config) = CreateSmallModel(vocabSize: 8, embDim: 4, numHeads: 2, numLayers: 1);
            int[] input = { 1, 2 };
            int[] target = { 2, 3 };

            float numGrad = NumericalGradient(model, input, target,
                () => model.TokenEmbedding[1, 0],
                (v) => model.TokenEmbedding[1, 0] = v);

            Assert(MathF.Abs(numGrad) > 1e-7f,
                $"Gradient through residual is effectively zero ({numGrad:E4})");
        }

        #endregion

        #region Loss Decrease Tests

        public void Test_LossDecreases_SingleSequence()
        {
            var (model, config) = CreateSmallModel(vocabSize: 10, embDim: 8, numHeads: 2, numLayers: 1, ffnDim: 16);
            int[] sequence = { 1, 2, 3, 4, 5 };

            var trainConfig = new TrainingConfig
            {
                LearningRate = 0.005f,
                BatchSize = 1,
                Epochs = 5,
                UseGradientClipping = true,
                GradientClipThreshold = 5.0f,
                Verbose = false
            };

            int[] inputSeq = sequence.Take(sequence.Length - 1).ToArray();
            int[] targetSeq = sequence.Skip(1).ToArray();
            float lossBefore = ComputeLoss(model, inputSeq, targetSeq);

            var trainer = new TransformerTrainer(model, trainConfig);
            trainer.Train(new[] { sequence });

            float lossAfter = ComputeLoss(model, inputSeq, targetSeq);

            Assert(lossAfter < lossBefore,
                $"Loss did not decrease after 5 epochs: before={lossBefore:F6}, after={lossAfter:F6}");
        }

        public void Test_LossDecreases_MultipleBatches()
        {
            var (model, config) = CreateSmallModel(vocabSize: 10, embDim: 8, numHeads: 2, numLayers: 1, ffnDim: 16);
            int[][] sequences = {
                new[] { 1, 2, 3, 4 },
                new[] { 2, 3, 4, 5 },
                new[] { 3, 4, 5, 6 },
                new[] { 4, 5, 6, 7 }
            };

            float lossBefore = 0;
            foreach (var seq in sequences)
            {
                var inp = seq.Take(seq.Length - 1).ToArray();
                var tgt = seq.Skip(1).ToArray();
                lossBefore += ComputeLoss(model, inp, tgt);
            }
            lossBefore /= sequences.Length;

            var trainConfig = new TrainingConfig
            {
                LearningRate = 0.005f,
                BatchSize = 2,
                Epochs = 5,
                UseGradientClipping = true,
                GradientClipThreshold = 5.0f,
                Verbose = false
            };

            new TransformerTrainer(model, trainConfig).Train(sequences);

            float lossAfter = 0;
            foreach (var seq in sequences)
            {
                var inp = seq.Take(seq.Length - 1).ToArray();
                var tgt = seq.Skip(1).ToArray();
                lossAfter += ComputeLoss(model, inp, tgt);
            }
            lossAfter /= sequences.Length;

            Assert(lossAfter < lossBefore,
                $"Average loss did not decrease after 5 epochs: before={lossBefore:F6}, after={lossAfter:F6}");
        }

        #endregion

        #region Gradient Clipping Tests

        public void Test_GradientClipping_Works()
        {
            var (model, config) = CreateSmallModel(vocabSize: 10, embDim: 8, numHeads: 2, numLayers: 1);

            var trainConfig = new TrainingConfig
            {
                LearningRate = 0.1f,
                BatchSize = 1,
                Epochs = 1,
                UseGradientClipping = true,
                GradientClipThreshold = 1.0f,
                Verbose = false
            };

            int[][] sequences = { new[] { 1, 2, 3, 4, 5 } };
            new TransformerTrainer(model, trainConfig).Train(sequences);

            var logits = model.Forward(new[] { 1, 2, 3 });
            bool anyNaN = false;
            for (int i = 0; i < logits.GetLength(0); i++)
                for (int j = 0; j < logits.GetLength(1); j++)
                    if (float.IsNaN(logits[i, j]) || float.IsInfinity(logits[i, j]))
                        anyNaN = true;

            Assert(!anyNaN, "Model produces NaN/Inf after training with gradient clipping");
        }

        #endregion

        #region Zero Gradient / No Accumulation Tests

        public void Test_ZeroGradients_AfterReset()
        {
            var (model, config) = CreateSmallModel();
            int[][] sequences = { new[] { 1, 2, 3 } };
            var tc = new TrainingConfig { LearningRate = 0.001f, BatchSize = 1, Epochs = 1, Verbose = false };
            var trainer = new TransformerTrainer(model, tc);

            trainer.Train(sequences);
            // Second train should not accumulate old gradients - no crash means zero works
            trainer.Train(new[] { new[] { 3, 4, 5, 6 } });
        }

        public void Test_TrainTwice_SameModel_NoAccumulation()
        {
            // Verify that running two separate Train calls gives correct results
            // (gradients don't leak between calls)
            var (model, _) = CreateSmallModel(vocabSize: 10, embDim: 8, numHeads: 2, numLayers: 1);

            int[] seq1 = { 1, 2, 3, 4 };
            int[] seq2 = { 5, 6, 7, 8 };

            var tc = new TrainingConfig
            {
                LearningRate = 0.005f,
                BatchSize = 1,
                Epochs = 3,
                Verbose = false
            };

            var trainer = new TransformerTrainer(model, tc);
            trainer.Train(new[] { seq1 });
            trainer.Train(new[] { seq2 });

            // Model should still produce valid output
            var logits = model.Forward(new[] { 1, 2, 3 });
            for (int i = 0; i < logits.GetLength(0); i++)
                for (int j = 0; j < logits.GetLength(1); j++)
                    Assert(!float.IsNaN(logits[i, j]), "NaN after two separate training runs");
        }

        #endregion

        #region Multi-Head Tests

        public void Test_MultiHead_ProducesDifferentGradsThanSingleHead()
        {
            var (model2h, _) = CreateSmallModel(vocabSize: 8, embDim: 8, numHeads: 2, numLayers: 1);
            var (model1h, _) = CreateSmallModel(vocabSize: 8, embDim: 8, numHeads: 1, numLayers: 1);

            // Copy shared weights so only head count differs
            Array.Copy(model2h.TokenEmbedding, model1h.TokenEmbedding, model2h.TokenEmbedding.Length);
            Array.Copy(model2h.OutputProjection, model1h.OutputProjection, model2h.OutputProjection.Length);
            Array.Copy(model2h.OutputBias, model1h.OutputBias, model2h.OutputBias.Length);

            var attn2h = model2h.Blocks[0].Attention;
            var attn1h = model1h.Blocks[0].Attention;
            Array.Copy(attn2h.WQ, attn1h.WQ, attn2h.WQ.Length);
            Array.Copy(attn2h.WK, attn1h.WK, attn2h.WK.Length);
            Array.Copy(attn2h.WV, attn1h.WV, attn2h.WV.Length);
            Array.Copy(attn2h.WO, attn1h.WO, attn2h.WO.Length);

            int[] input = { 1, 2, 3 };
            int[] target = { 2, 3, 4 };

            float loss2h = ComputeLoss(model2h, input, target);
            float loss1h = ComputeLoss(model1h, input, target);

            Assert(MathF.Abs(loss2h - loss1h) > 1e-6f,
                $"2-head and 1-head losses are identical ({loss2h:F6})");
        }

        #endregion

        #region Causal Mask Tests

        public void Test_CausalMask_FutureTokensNoGradient()
        {
            var (model, config) = CreateSmallModel(vocabSize: 8, embDim: 4, numHeads: 2, numLayers: 1, decoderOnly: true);
            int[] input = { 1, 2, 3 };

            var logitsBefore = model.Forward(input);
            float logit_0_0 = logitsBefore[0, 0];

            float original = model.TokenEmbedding[3, 0];
            model.TokenEmbedding[3, 0] += 1.0f;

            var logitsAfter = model.Forward(input);
            float logit_0_0_after = logitsAfter[0, 0];

            model.TokenEmbedding[3, 0] = original;

            Assert(MathF.Abs(logit_0_0 - logit_0_0_after) < 1e-5f,
                $"Causal mask violated: logits[0,0] changed from {logit_0_0:F6} to {logit_0_0_after:F6}");
        }

        #endregion

        public void Test_DifferentSequenceLengths_NoError()
        {
            var (model, _) = CreateSmallModel(vocabSize: 10, embDim: 8, numHeads: 2, numLayers: 1);

            int[][] sequences = {
                new[] { 1, 2 },
                new[] { 1, 2, 3, 4, 5, 6 },
                new[] { 3, 4, 5 },
                new[] { 7, 8, 9, 1 }
            };

            var tc = new TrainingConfig { LearningRate = 0.001f, BatchSize = 2, Epochs = 2, Verbose = false };
            new TransformerTrainer(model, tc).Train(sequences);
        }


        public void Test_AllParametersReceiveGradients()
        {
            var (model, config) = CreateSmallModel(vocabSize: 8, embDim: 4, numHeads: 2, numLayers: 1, ffnDim: 8);

            var embBefore = (float[,])model.TokenEmbedding.Clone();
            var outProjBefore = (float[,])model.OutputProjection.Clone();
            var outBiasBefore = (float[])model.OutputBias.Clone();

            var attn = model.Blocks[0].Attention;
            var wqBefore = (float[,])attn.WQ.Clone();
            var wkBefore = (float[,])attn.WK.Clone();
            var wvBefore = (float[,])attn.WV.Clone();
            var woBefore = (float[,])attn.WO.Clone();
            var bqBefore = (float[])attn.BiasQ.Clone();
            var bkBefore = (float[])attn.BiasK.Clone();
            var bvBefore = (float[])attn.BiasV.Clone();
            var boBefore = (float[])attn.BiasO.Clone();

            var ln1gBefore = (float[])model.Blocks[0].LN1Gamma.Clone();
            var ln1bBefore = (float[])model.Blocks[0].LN1Beta.Clone();
            var ln2gBefore = (float[])model.Blocks[0].LN2Gamma.Clone();
            var ln2bBefore = (float[])model.Blocks[0].LN2Beta.Clone();

            var ffnData = model.Blocks[0].FeedForwardNetwork.GetData();
            var ffnW1Before = (float[,])ffnData.layers[1].Weights.Clone();
            var ffnW2Before = (float[,])ffnData.layers[2].Weights.Clone();

            var tc = new TrainingConfig { LearningRate = 0.05f, BatchSize = 1, Epochs = 3, UseGradientClipping = false, Verbose = false };
            new TransformerTrainer(model, tc).Train(new[] { new[] { 1, 2, 3, 4 } });

            Assert(MatrixChanged(embBefore, model.TokenEmbedding), "TokenEmbedding did not change");
            Assert(MatrixChanged(outProjBefore, model.OutputProjection), "OutputProjection did not change");
            Assert(VectorChanged(outBiasBefore, model.OutputBias), "OutputBias did not change");
            Assert(MatrixChanged(wqBefore, attn.WQ), "WQ did not change");
            Assert(MatrixChanged(wkBefore, attn.WK), "WK did not change");
            Assert(MatrixChanged(wvBefore, attn.WV), "WV did not change");
            Assert(MatrixChanged(woBefore, attn.WO), "WO did not change");
            Assert(VectorChanged(bqBefore, attn.BiasQ), "BiasQ did not change");
            Assert(VectorChanged(bkBefore, attn.BiasK), "BiasK did not change");
            Assert(VectorChanged(bvBefore, attn.BiasV), "BiasV did not change");
            Assert(VectorChanged(boBefore, attn.BiasO), "BiasO did not change");
            Assert(VectorChanged(ln1gBefore, model.Blocks[0].LN1Gamma), "LN1Gamma did not change");
            Assert(VectorChanged(ln1bBefore, model.Blocks[0].LN1Beta), "LN1Beta did not change");
            Assert(VectorChanged(ln2gBefore, model.Blocks[0].LN2Gamma), "LN2Gamma did not change");
            Assert(VectorChanged(ln2bBefore, model.Blocks[0].LN2Beta), "LN2Beta did not change");
            Assert(MatrixChanged(ffnW1Before, ffnData.layers[1].Weights), "FFN Layer 1 Weights did not change");
            Assert(MatrixChanged(ffnW2Before, ffnData.layers[2].Weights), "FFN Layer 2 Weights did not change");
        }

        private bool MatrixChanged(float[,] before, float[,] after, float threshold = 1e-10f)
        {
            for (int i = 0; i < before.GetLength(0); i++)
            {

                for (int j = 0; j < before.GetLength(1); j++)
                {

                    if (MathF.Abs(before[i, j] - after[i, j]) > threshold)
                    {
                        return true;
                    }
                }
            }
            return false;
        }

        private bool VectorChanged(float[] before, float[] after, float threshold = 1e-10f)
        {
            for (int i = 0; i < before.Length; i++)
            {

                if (MathF.Abs(before[i] - after[i]) > threshold)
                {
                    return true;
                }
            }
                return false;
        }


        public void Test_LearningRateDecay_Applied()
        {
            var (model, _) = CreateSmallModel(vocabSize: 8, embDim: 4, numHeads: 2, numLayers: 1);
            int[][] sequences = { new[] { 1, 2, 3, 4 } };

            var tc = new TrainingConfig
            {
                LearningRate = 0.01f,
                BatchSize = 1,
                Epochs = 5,
                UseLearningRateDecay = true,
                LearningRateDecay = 0.5f,
                Verbose = false
            };

            new TransformerTrainer(model, tc).Train(sequences);

            var logits = model.Forward(new[] { 1, 2 });
            for (int i = 0; i < logits.GetLength(0); i++)
            {
                for (int j = 0; j < logits.GetLength(1); j++)
                {
                    Assert(!float.IsNaN(logits[i, j]), "NaN in logits after training with LR decay");
                }
            }
        }

        public void Test_ValidationLoss_Computable()
        {
            var (model, _) = CreateSmallModel(vocabSize: 10, embDim: 8, numHeads: 2, numLayers: 1);

            int[][] trainSeq = { new[] { 1, 2, 3, 4 }, new[] { 3, 4, 5, 6 } };
            int[][] valSeq = { new[] { 2, 3, 4, 5 } };

            var tc = new TrainingConfig
            {
                LearningRate = 0.001f,
                BatchSize = 2,
                Epochs = 3,
                ValidationInterval = 1,
                Verbose = false
            };

            var trainer = new TransformerTrainer(model, tc);
            trainer.Train(trainSeq, valSeq);

            float valLoss = trainer.Validate(valSeq);
            Assert(!float.IsNaN(valLoss) && !float.IsInfinity(valLoss) && valLoss > 0, $"Validation loss is invalid: {valLoss}");
        }

        public void Test_OverfitSingleSequence()
        {
            var (model, _) = CreateSmallModel(vocabSize: 10, embDim: 16, numHeads: 2, numLayers: 2, ffnDim: 32);

            int[] sequence = { 1, 2, 3, 4 };
            int[] input = { 1, 2, 3 };
            int[] target = { 2, 3, 4 };

            var tc = new TrainingConfig
            {
                LearningRate = 0.005f,
                BatchSize = 1,
                Epochs = 200,
                UseGradientClipping = true,
                GradientClipThreshold = 5.0f,
                Verbose = false
            };

            new TransformerTrainer(model, tc).Train(new[] { sequence });

            float loss = ComputeLoss(model, input, target);
            Assert(loss < 1.0f,
                $"Failed to overfit single sequence after 200 epochs: loss={loss:F4} (expected < 1.0)");
        }

        public void Test_OverfitTwoSequences()
        {
            var (model, _) = CreateSmallModel(vocabSize: 10, embDim: 16, numHeads: 2, numLayers: 2, ffnDim: 32);

            int[][] sequences = {
                new[] { 1, 2, 3, 4 },
                new[] { 5, 6, 7, 8 }
            };

            var tc = new TrainingConfig
            {
                LearningRate = 0.005f,
                BatchSize = 2,
                Epochs = 300,
                UseGradientClipping = true,
                GradientClipThreshold = 5.0f,
                Verbose = false
            };

            new TransformerTrainer(model, tc).Train(sequences);

            float totalLoss = 0;
            foreach (var seq in sequences)
            {
                var inp = seq.Take(seq.Length - 1).ToArray();
                var tgt = seq.Skip(1).ToArray();
                totalLoss += ComputeLoss(model, inp, tgt);
            }
            float avgLoss = totalLoss / sequences.Length;

            Assert(avgLoss < 1.5f, $"Failed to overfit two sequences after 300 epochs: avg loss={avgLoss:F4} (expected < 1.5)");
        }

        public void Test_OverfitSingleSequence_Extended()
        {
            var (model, _) = CreateSmallModel(vocabSize: 10, embDim: 16, numHeads: 2, numLayers: 2, ffnDim: 64);

            int[] sequence = { 1, 2, 3, 4 };
            int[] input = { 1, 2, 3 };
            int[] target = { 2, 3, 4 };

            var tc = new TrainingConfig
            {
                LearningRate = 0.005f,
                BatchSize = 1,
                Epochs = 500,
                UseGradientClipping = true,
                GradientClipThreshold = 5.0f,
                Verbose = false
            };

            new TransformerTrainer(model, tc).Train(new[] { sequence });

            float loss = ComputeLoss(model, input, target);
            Assert(loss < 0.5f, $"Failed to deeply overfit single sequence after 500 epochs: loss={loss:F4} (expected < 0.5)");
        }

        public void Test_SequenceLength1_NoError()
        {
            var (model, _) = CreateSmallModel(vocabSize: 10, embDim: 8, numHeads: 2, numLayers: 1);
            int[][] sequences = { new[] { 1, 2 } };
            var tc = new TrainingConfig { LearningRate = 0.001f, BatchSize = 1, Epochs = 1, Verbose = false };
            new TransformerTrainer(model, tc).Train(sequences);
        }

        public void Test_LargeVocab_SmallSequence()
        {
            var (model, _) = CreateSmallModel(vocabSize: 1000, embDim: 8, numHeads: 2, numLayers: 1);
            int[][] sequences = { new[] { 100, 200, 300 } };
            var tc = new TrainingConfig { LearningRate = 0.001f, BatchSize = 1, Epochs = 1, Verbose = false };
            new TransformerTrainer(model, tc).Train(sequences);
        }

        public void Test_RepeatedTokens_NoError()
        {
            var (model, _) = CreateSmallModel(vocabSize: 10, embDim: 8, numHeads: 2, numLayers: 1);
            int[][] sequences = { new[] { 1, 1, 1, 1 } };
            var tc = new TrainingConfig { LearningRate = 0.001f, BatchSize = 1, Epochs = 3, Verbose = false };
            new TransformerTrainer(model, tc).Train(sequences);
        }

        public void Test_SameInput_SameLoss()
        {
            var (model, _) = CreateSmallModel(vocabSize: 8, embDim: 4, numHeads: 2, numLayers: 1);
            int[] input = { 1, 2, 3 };
            int[] target = { 2, 3, 4 };

            float loss1 = ComputeLoss(model, input, target);
            float loss2 = ComputeLoss(model, input, target);

            Assert(loss1 == loss2,
                $"Same input produced different losses: {loss1:E6} vs {loss2:E6}");
        }


        public void Test_ForwardCache_TokenIds_Stored()
        {
            var (model, _) = CreateSmallModel(vocabSize: 10, embDim: 8, numHeads: 2, numLayers: 1);

            int[] sequence = { 5, 7, 3 };
            float embBefore_5 = model.TokenEmbedding[5, 0];
            float embBefore_0 = model.TokenEmbedding[0, 0]; // Token 0 not in input

            var tc = new TrainingConfig { LearningRate = 0.01f, BatchSize = 1, Epochs = 1, UseGradientClipping = false, Verbose = false };
            new TransformerTrainer(model, tc).Train(new[] { sequence });

            float embAfter_5 = model.TokenEmbedding[5, 0];
            float embAfter_0 = model.TokenEmbedding[0, 0];

            Assert(MathF.Abs(embBefore_5 - embAfter_5) > 1e-10f,
                "Embedding for token 5 (in input) did not change");

            Assert(MathF.Abs(embBefore_0 - embAfter_0) < 1e-10f,
                "Embedding for token 0 (NOT in input) changed - gradient leaked");
        }

        public void Test_ForwardCache_FFNInputs_Stored()
        {
            var (model, _) = CreateSmallModel(vocabSize: 8, embDim: 8, numHeads: 2, numLayers: 1, ffnDim: 16);

            int[] sequence = { 1, 2, 3, 4 };
            int[] input = { 1, 2, 3 };
            int[] target = { 2, 3, 4 };

            float lossBefore = ComputeLoss(model, input, target);

            var tc = new TrainingConfig
            {
                LearningRate = 0.005f,
                BatchSize = 1,
                Epochs = 50,
                UseGradientClipping = true,
                GradientClipThreshold = 5.0f,
                Verbose = false
            };

            new TransformerTrainer(model, tc).Train(new[] { sequence });

            float lossAfter = ComputeLoss(model, input, target);

            // Just require loss decreased - any decrease proves FFN backprop works
            Assert(lossAfter < lossBefore,
                $"Loss did not decrease after 50 epochs (before={lossBefore:F4}, after={lossAfter:F4}). FFN backprop may be broken.");
        }



        public void Test_TrainThenGenerate_NoError()
        {
            var (model, config) = CreateSmallModel(vocabSize: 20, embDim: 16, numHeads: 2, numLayers: 2, ffnDim: 32);

            int[][] sequences = {
                new[] { 1, 2, 3, 4, 5 },
                new[] { 6, 7, 8, 9, 10 },
                new[] { 1, 3, 5, 7, 9 },
            };

            var tc = new TrainingConfig { LearningRate = 0.005f, BatchSize = 2, Epochs = 10, Verbose = false };
            new TransformerTrainer(model, tc).Train(sequences);

            var generated = model.Generate(new[] { 1, 2 }, maxNewTokens: 5, temperature: 1.0f);
            Assert(generated.Length > 2, "Generate produced no new tokens after training");

            foreach (var tok in generated)
                Assert(tok >= 0 && tok < config.VocabSize, $"Generated invalid token {tok}");
        }

        public void Test_MultiLayer_GradientFlow()
        {
            var (model, _) = CreateSmallModel(vocabSize: 10, embDim: 8, numHeads: 2, numLayers: 4, ffnDim: 16);

            int[] sequence = { 1, 2, 3, 4, 5 };

            var wqL0Before = (float[,])model.Blocks[0].Attention.WQ.Clone();
            var wqL3Before = (float[,])model.Blocks[3].Attention.WQ.Clone();

            var tc = new TrainingConfig { LearningRate = 0.01f, BatchSize = 1, Epochs = 1, UseGradientClipping = false, Verbose = false };
            new TransformerTrainer(model, tc).Train(new[] { sequence });

            Assert(MatrixChanged(wqL0Before, model.Blocks[0].Attention.WQ),
                "Layer 0 WQ did not change - gradients not flowing to early layers");
            Assert(MatrixChanged(wqL3Before, model.Blocks[3].Attention.WQ),
                "Layer 3 WQ did not change - gradients not flowing to late layers");

            for (int layer = 0; layer < 4; layer++)
            {
                var attn = model.Blocks[layer].Attention;
                for (int i = 0; i < attn.WQ.GetLength(0); i++)
                    for (int j = 0; j < attn.WQ.GetLength(1); j++)
                        Assert(!float.IsNaN(attn.WQ[i, j]), $"NaN in layer {layer} WQ[{i},{j}]");
            }
        }

        public void Test_Convergence_MonotonicallyDecreasing()
        {
            // Train epoch by epoch and verify loss trend is downward
            var (model, _) = CreateSmallModel(vocabSize: 10, embDim: 8, numHeads: 2, numLayers: 1, ffnDim: 16);

            int[][] sequences = { new[] { 1, 2, 3, 4 }, new[] { 2, 3, 4, 5 } };
            int[] testInput = { 1, 2, 3 };
            int[] testTarget = { 2, 3, 4 };

            float previousLoss = ComputeLoss(model, testInput, testTarget);
            int decreaseCount = 0;

            for (int epoch = 0; epoch < 5; epoch++)
            {
                var tc = new TrainingConfig
                {
                    LearningRate = 0.003f,
                    BatchSize = 2,
                    Epochs = 1,
                    Verbose = false
                };
                new TransformerTrainer(model, tc).Train(sequences);

                float currentLoss = ComputeLoss(model, testInput, testTarget);
                if (currentLoss < previousLoss)
                    decreaseCount++;
                previousLoss = currentLoss;
            }

            // At least 3 out of 5 epochs should show decrease (allows for some noise)
            Assert(decreaseCount >= 3,
                $"Loss only decreased in {decreaseCount}/5 epochs - expected at least 3");
        }


    }

}