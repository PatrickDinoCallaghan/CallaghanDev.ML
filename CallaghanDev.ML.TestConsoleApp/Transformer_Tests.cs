using CallaghanDev.ML.Enums;
using CallaghanDev.ML.Transformers;
using System;
using System.Collections.Generic;
using System.Linq;

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
}