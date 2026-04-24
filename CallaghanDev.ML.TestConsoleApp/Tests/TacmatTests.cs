using CallaghanDev.ML.Enums;
using CallaghanDev.ML.Transformers;
using CallaghanDev.ML.Transformers.Configuration;
using CallaghanDev.ML.Transformers.CrossAttentionMultimodal;
using CallaghanDev.ML.Transformers.TACAMT;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Reflection;
using Gradients = CallaghanDev.ML.Transformers.TACAMT.Gradients;
using MultimodalForwardCache = CallaghanDev.ML.Transformers.TACAMT.MultimodalForwardCache;
using TACAMT_Model = CallaghanDev.ML.Transformers.TACAMT.Model;
using TACAMT_Trainer = CallaghanDev.ML.Transformers.TACAMT.Trainer;

namespace CallaghanDev.ML.TestConsoleApp.Tests
{
    /// <summary>
    /// TACAMT regression/contract tests.
    ///
    /// This suite intentionally avoids brittle checks such as "every parameter changed"
    /// or exact loss thresholds on random data.  It tests contracts that should remain
    /// true across refactors: temporal coordinate conventions, cache hygiene, gradient
    /// routing, memory semantics, save/load fidelity, and small deterministic learning
    /// problems that prove the trainer can actually optimize the model.
    /// </summary>
    internal sealed class TacmatTests : TestBase
    { 
        private int _passed;
        private int _failed;
        private readonly List<string> _failures = new List<string>();

        public void RunAllTests()
        {
            CountNumber++;
            TransformerTestSuite.PrintBanner($"{CountNumber} * TACAMT trainer/model contract tests");
            Console.ForegroundColor = ConsoleColor.DarkCyan;
            Console.WriteLine("  Running targeted TACAMT regression tests …");
            Console.ResetColor();
            Console.WriteLine();

            _passed = 0;
            _failed = 0;
            _failures.Clear();

            var tests = new (Action Test, string Name)[]
            {
                // Configuration / construction
                (Test_Config_GoodConfig_Validates, "Config: good config validates"),
                (Test_Config_BadHeadDims_Throws, "Config: embedding dims must divide by heads"),
                (Test_Config_PriceContextDisabled_IsPreserved, "Config: PriceContext.Enabled=false is preserved"),
                (Test_Model_Dimensions_MatchConfig, "Model: public tensors match config dimensions"),
                (Test_Model_UsesCpuAcceleration_WhenConfigured, "Model: CPU acceleration manager selected"),

                // Tokenizer / text contracts
                (Test_Tokenizer_WithoutTokenizer_Throws, "Tokenizer: raw text requires attached tokenizer"),
                (Test_Tokenizer_TooLargeVocab_Throws, "Tokenizer: vocab larger than embedding table throws"),
                (Test_Tokenizer_SaveLoad_RoundTrip, "Tokenizer: saved and loaded with model"),
                (Test_Text_InvalidTokenId_Throws, "Text: invalid token ID throws"),
                (Test_Text_EmptyStoryTokens_Throws, "Text: empty story token list throws"),
                (Test_Text_LongStoryCacheEmbeddings_AreRawTokenEmbeddings, "Text: long story cache stores raw embeddings, not block outputs"),
                (Test_Text_MeanPooling_FixedShape, "Text: story mean-pooling produces fixed context shape"),

                // Forward-path contracts
                (Test_Forward_NullStories_FiniteShape, "Forward: null stories finite shape"),
                (Test_Forward_EmptyStories_EqualsNullStories, "Forward: empty stories equals null stories"),
                (Test_Forward_SingleStory_ActivatesCrossAttention, "Forward: one story activates cross-attention"),
                (Test_Forward_MultipleStories_AllContextRowsPresent, "Forward: multiple stories create all context rows"),
                (Test_ForwardWithCache_EqualsForward_Inference, "ForwardWithCache: inference output equals Forward"),
                (Test_ForwardWithCache_ReusedCache_ClearsStoryState, "ForwardWithCache: reused cache clears stale story state"),
                (Test_ForwardWithCache_RowSlice_TimeAlignmentMatchesExplicitSlice, "ForwardWithCache: rowStart shifts story times correctly"),
                (Test_Forward_PriceContext_OutputShapeAndCacheMetadata, "ForwardPriceContext: shape and cache metadata"),
                (Test_Forward_PriceContext_ChangesPrediction, "ForwardPriceContext: price context affects prediction"),
                (Test_ConfidenceHead_Range_AndDefaultConfidence, "Confidence: sigmoid range and default confidence"),
                (Test_CausalMask_FutureRows_DoNotAffectPastRows, "Causal mask: future price rows do not affect past outputs"),
                (Test_Deterministic_Inference_IsRepeatable, "Determinism: inference is repeatable"),
                (Test_Stability_ExtremeInputs_NoNaN, "Stability: extreme finite inputs do not produce NaN"),

                // Temporal decay / memory semantics
                (Test_DecayNetwork_ZeroTimeDiff_ProducesNearZeroBias, "Decay: zero time diff produces near-zero bias"),
                (Test_DecayNetwork_Gates_AreWithinConfiguredRange, "Decay: gates stay inside configured range"),
                (Test_Decay_RecentIdenticalStory_GetsHigherAttentionThanOld, "Decay: recent identical story gets higher attention"),
                (Test_Decay_TimeDiffCache_UsesPastRelativeConvention, "Decay: cached time diffs use past-relative key times"),
                (Test_PredictWithMemory_ExistingNewsAge_MatchesDirectForward, "Memory: existing news age matches direct forward convention"),
                (Test_PredictWithMemory_StoresAbsoluteTimes_AndLastPriceTimestamp, "Memory: stores absolute timestamps correctly"),
                (Test_PredictWithMemory_AccumulatesAndPrunes, "Memory: accumulates then prunes news and price memories"),
                (Test_Pruning_MaxZero_ClearsNewsAndPrice, "Pruning: max size zero clears memory"),
                (Test_Pruning_Disabled_KeepsNewestEntries, "Pruning: disabled attention pruning keeps newest entries"),
                (Test_Pruning_AttentionBased_KeepsHighScoreAndReservedNewest, "Pruning: attention pruning keeps high score and reserved newest"),
                (Test_Pruning_ColdStartEntries_AreProtected, "Pruning: cold-start entries are protected"),

                // Save/load fidelity
                (Test_SaveLoad_ForwardMatches_AndWeightsPreserved, "SaveLoad: forward output and weights round-trip"),
                (Test_SaveLoad_MemoryScoresTokenizerAndPruningConfig_Preserved, "SaveLoad: memory, tokenizer, pruning config preserved"),

                // Trainer validation and optimization contracts
                (Test_Validate_PriceContextDisabled_MatchesManualNoContextMse, "Validate: PriceContext.Enabled=false avoids split/context path"),
                (Test_Validate_PriceContextEnabled_UsesNegativeHistoryKeyTimes, "Validate: price-context history times are negative relative key times"),
                (Test_Train_PriceOnly_EasyConstantTarget_LossDrops, "Train: price-only easy constant target loss drops"),
                (Test_Train_TextFrozen_DoesNotMutateTextEmbeddings, "Train: frozen text encoder stays frozen"),
                (Test_Train_ConfidenceHead_UpdatesWithTargets, "Train: confidence head updates with confidence targets"),
                (Test_TrainSequential_AccumulatesMemoryAndRespectsLimits, "TrainSequential: accumulates memory and respects limits"),
                (Test_TrainSequential_TimeUnitsPerPositionZero_Throws, "TrainSequential: zero timeUnitsPerPosition throws"),
                (Test_Train_MixedNullAndShortSamples_NoNaN, "Train: mixed null/short samples do not corrupt model"),

                // Backward/gradient routing contracts
                (Test_Backward_PriceOnly_ProducesCoreGradients, "Backward: price-only produces core gradients"),
                (Test_Backward_NewsContext_ProducesTextCrossAndDecayGradients, "Backward: news context produces text/cross/decay gradients"),
                (Test_Backward_FrozenText_HasNoTextEmbeddingGradient, "Backward: frozen text has no text embedding gradient"),
                (Test_Backward_PriceContext_ProducesBothTypeEmbeddingGradients, "Backward: price context produces news and price type gradients"),
                (Test_BackpropMultiStory_UsesFreshStoryOffset_WhenNewsMemoryExists, "Backward: fresh story gradients use offset after news memory"),
                (Test_BackpropMultiStory_DoesNotRouteExistingMemoryGradientToFreshText, "Backward: existing memory gradient does not hit fresh text"),
                (Test_GradientClipping_ReducesNorm_AndIncludesContextTypeEmbedding, "GradientClip: reduces norm and includes context type embedding"),

                // End-to-end learning on deterministic synthetic problems
                (Test_E2E_SingleSample_OverfitsEasyConstantTarget, "E2E: single sample overfits easy target"),
                (Test_E2E_LearnsBullVsBearNewsSignal, "E2E: learns bull vs bear text signal"),
                (Test_E2E_LearnsRecencyUnderCompetition, "E2E: learns recency under competing stories"),

                // Additional high-confidence hardening tests
                (Test_Forward_NullPriceSequence_Throws, "Hardening: null price sequence throws"),
                (Test_Forward_ZeroLengthPriceSequence_Throws, "Hardening: zero-length price sequence throws"),
                (Test_Forward_WrongFeatureDim_Throws, "Hardening: wrong price feature dimension throws"),
                (Test_ForwardWithCache_InvalidSliceBounds_Throw, "Hardening: invalid cached slice bounds throw"),
                (Test_ForwardWithPriceContext_InvalidContextShape_Throws, "Hardening: bad price-context hidden shape throws"),
                (Test_ForwardWithPriceContext_InvalidTimeLength_Throws, "Hardening: bad price-context time length throws"),
                (Test_ForwardWithPriceContext_ReusedCache_ClearsStaleContext, "Hardening: price-context reused cache clears stale context"),
                (Test_Tokenizer_LengthMismatch_Throws, "Hardening: TokenizeStories length mismatch throws"),
                (Test_TrainTokenizer_EmptyCorpusThrows_NonEmptyAttachesTokenizer, "Hardening: TrainTokenizer guards empty corpus and attaches tokenizer"),
                (Test_Memory_ContextTypeEmbeddingDoesNotMutateStoredMemory, "Hardening: context type embeddings do not mutate stored memory"),
                (Test_UpdateMemoryAttentionScores_MapsNewsAndPriceRowsCorrectly, "Hardening: memory score update maps news and price rows correctly"),
                (Test_UpdateNewsMemory_MaxZero_ClearsFreshEntry, "Hardening: UpdateNewsMemory max zero clears fresh entry"),
                (Test_PredictWithMemory_TimeUnitsPerPositionScalesAbsoluteStoryTimes, "Hardening: PredictWithMemory scales absolute story/price timestamps"),
                (Test_ClearAllMemory_ResetsBothMemoriesAndTimestamp, "Hardening: ClearAllMemory resets memories and timestamp"),
                (Test_Train_DuplicateSamplesBatchAveraging_EqualsSingleSampleUpdate, "Hardening: duplicate-sample batch averaging equals single sample update"),
                (Test_Train_InvalidSample_DoesNotContaminateValidSampleUpdate, "Hardening: invalid sample does not contaminate valid update"),
                (Test_Validate_UsesNextStepTargets_NotCurrentRow, "Hardening: Validate uses next-step targets, not current row"),
                (Test_Train_IntArrayOverload_EqualsNewsStoryOverload, "Hardening: Train int[][] overload equals NewsStory overload"),
                (Test_Validate_IntArrayOverload_EqualsNewsStoryOverload, "Hardening: Validate int[][] overload equals NewsStory overload"),
                (Test_Gradient_OutputProjection_MatchesFiniteDifference, "Hardening: OutputProjection gradient matches finite difference"),
                (Test_Gradient_PriceInputProjection_MatchesFiniteDifference, "Hardening: PriceInputProjection gradient matches finite difference"),
                (Test_Gradient_ConfidenceProjection_MatchesFiniteDifference, "Hardening: ConfidenceProjection gradient matches finite difference"),
                (Test_Dropout_TrainingSameSeedRepeatableDifferentSeedDiffers, "Hardening: decay dropout is seeded/reproducible and stochastic"),
                (Test_Backward_LongStory_AllChunksReceiveTokenGradient, "Hardening: long story gradients reach every token chunk"),
                (Test_Backward_NewsOnlyContext_OnlyNewsTypeGradient, "Hardening: news-only context updates only news type embedding"),
                (Test_Backward_PriceOnlyContext_OnlyPriceTypeGradient, "Hardening: price-only context updates only price type embedding"),
            };

            for (int i = 0; i < tests.Length; i++)
            {
                Console.Write($"  [{i + 1,2}/{tests.Length}] {tests[i].Name,-76} ");
                try
                {
                    tests[i].Test();
                    Console.ForegroundColor = ConsoleColor.Green;
                    Console.WriteLine("PASS");
                    Console.ResetColor();
                    _passed++;
                }
                catch (Exception ex)
                {
                    Console.ForegroundColor = ConsoleColor.Red;
                    Console.WriteLine("FAIL");
                    Console.ResetColor();
                    Console.ForegroundColor = ConsoleColor.DarkYellow;
                    Console.WriteLine($"         {ex.Message}");
                    Console.ResetColor();
                    _failures.Add($"{tests[i].Name}: {ex.Message}");
                    _failed++;
                }
            }

            Console.WriteLine($"\n   {new string('-', 80)}");
            Console.Write("   Results: ");
            Console.ForegroundColor = ConsoleColor.Green;
            Console.Write($"{_passed} passed");
            Console.ResetColor();
            if (_failed > 0)
            {
                Console.Write(", ");
                Console.ForegroundColor = ConsoleColor.Red;
                Console.Write($"{_failed} failed");
                Console.ResetColor();
            }
            Console.WriteLine($" / {tests.Length} total\n");

            if (_failures.Count > 0)
            {
                Console.ForegroundColor = ConsoleColor.Red;
                Console.WriteLine("   Failed tests:");
                Console.ResetColor();
                foreach (string failure in _failures)
                {
                    Console.WriteLine($"     * {failure}");
                }
                Console.WriteLine();
            }
        }

        #region Configuration / construction

        private void Test_Config_GoodConfig_Validates()
        {
            var config = CreateConfig();
            config.Validate();
        }

        private void Test_Config_BadHeadDims_Throws()
        {
            var config = CreateConfig(embDim: 15, numHeads: 4, validate: false);
            AssertThrows<ArgumentException>(() => config.Validate(), "Expected bad embedding/head dimensions to throw.");
        }

        private void Test_Config_PriceContextDisabled_IsPreserved()
        {
            var config = CreateConfig(priceContextEnabled: false);
            Assert(!config.PriceContext.Enabled, "PriceContext.Enabled should be false.");
            config.Validate();
        }

        private void Test_Model_Dimensions_MatchConfig()
        {
            var config = CreateConfig(textVocabSize: 73, embDim: 24, numHeads: 4, inputFeatures: 7, outputDim: 3, useConfidence: true);
            var model = new TACAMT_Model(config, new Random(42));

            Assert(model.TextTokenEmbedding.GetLength(0) == 73, "Text embedding rows mismatch.");
            Assert(model.TextTokenEmbedding.GetLength(1) == 24, "Text embedding cols mismatch.");
            Assert(model.PriceInputProjection.GetLength(0) == 24, "Price projection rows mismatch.");
            Assert(model.PriceInputProjection.GetLength(1) == 7, "Price projection cols mismatch.");
            Assert(model.OutputProjection.GetLength(0) == 3, "Output projection rows mismatch.");
            Assert(model.OutputProjection.GetLength(1) == 24, "Output projection cols mismatch.");
            Assert(model.ContextTypeEmbedding.GetLength(0) == 2, "Context type rows should be [news, price].");
            Assert(model.ContextTypeEmbedding.GetLength(1) == 24, "Context type embedding dim mismatch.");
            Assert(model.ConfidenceProjection != null, "Confidence projection should exist when enabled.");
            Assert(model.ConfidenceProjection.GetLength(0) == 1, "Confidence projection should have one row.");
            Assert(model.ConfidenceProjection.GetLength(1) == 24, "Confidence projection dim mismatch.");
        }

        private void Test_Model_UsesCpuAcceleration_WhenConfigured()
        {
            var model = new TACAMT_Model(CreateConfig(), new Random(42));
            Assert(model.AccelerationManager.GetType().Name.Contains("CPU"), $"Expected CPU acceleration manager, got {model.AccelerationManager.GetType().Name}.");
        }

        #endregion

        #region Tokenizer / text contracts

        private void Test_Tokenizer_WithoutTokenizer_Throws()
        {
            var model = new TACAMT_Model(CreateConfig(), new Random(42));
            AssertThrows<InvalidOperationException>(() => model.TokenizeStories(new[] { "hello" }, new[] { 0f }), "Expected TokenizeStories to require a tokenizer.");
        }

        private void Test_Tokenizer_TooLargeVocab_Throws()
        {
            var tokenizer = new BPETokenizer();
            tokenizer.Train(new[] { "alpha beta gamma delta epsilon zeta eta theta" }, vocabSize: 80, minFrequency: 1);

            var model = new TACAMT_Model(CreateConfig(textVocabSize: Math.Max(2, tokenizer.VocabSize - 1)), new Random(42));
            AssertThrows<ArgumentException>(() => model.SetTokenizer(tokenizer), "Tokenizer with too-large vocab must be rejected.");
        }

        private void Test_Tokenizer_SaveLoad_RoundTrip()
        {
            var tokenizer = new BPETokenizer();
            tokenizer.Train(new[] { "stock price rose", "market fell sharply" }, vocabSize: 60, minFrequency: 1);

            var model = new TACAMT_Model(CreateConfig(textVocabSize: tokenizer.VocabSize + 2), new Random(42));
            model.SetTokenizer(tokenizer);

            string dir = GetTempDir();
            try
            {
                model.Save(dir);
                var loaded = TACAMT_Model.Load(dir);
                Assert(loaded.Tokenizer != null, "Tokenizer should be loaded.");
                Assert(loaded.Tokenizer.Encode("stock price").Length > 0, "Loaded tokenizer should encode text.");
            }
            finally
            {
                CleanupDir(dir);
            }
        }

        private void Test_Text_InvalidTokenId_Throws()
        {
            var model = new TACAMT_Model(CreateConfig(textVocabSize: 10), new Random(42));
            var story = new[] { new NewsStory(new[] { 0, 9, 10 }, -1f) };
            var price = RandomMatrix(4, 5, new Random(1), 0.2f);

            AssertThrows<InvalidOperationException>(() => model.Forward(story, price), "Token ID equal to vocab size should throw.");
        }

        private void Test_Text_EmptyStoryTokens_Throws()
        {
            var model = new TACAMT_Model(CreateConfig(), new Random(42));
            var story = new[] { new NewsStory(Array.Empty<int>(), -1f) };
            var price = RandomMatrix(4, 5, new Random(1), 0.2f);

            AssertThrows<Exception>(() => model.Forward(story, price), "Empty story tokens should throw instead of silently producing invalid hidden states.");
        }

        private void Test_Text_LongStoryCacheEmbeddings_AreRawTokenEmbeddings()
        {
            var config = CreateConfig(textVocabSize: 80, embDim: 16, numHeads: 2, textMaxSeqLen: 5);
            var model = new TACAMT_Model(config, new Random(42));
            var tokens = new[] { 3, 4, 5, 6, 7, 8, 9, 10 };
            var stories = new[] { new NewsStory(tokens, -1f) };
            var price = RandomMatrix(4, 5, new Random(2), 0.1f);
            var cache = new MultimodalForwardCache(config.Text.NumLayers, config.Price.NumLayers);

            model.ForwardWithCache(stories, price, cache, isTraining: false);

            Assert(cache.StoryCaches != null && cache.StoryCaches.Count == 1, "Expected one story cache.");
            var storyCache = cache.StoryCaches[0];
            Assert(storyCache.TextEmbedded != null, "Story cache should store TextEmbedded.");
            Assert(storyCache.TextEmbedded.GetLength(0) == tokens.Length, "TextEmbedded row count should equal token count.");

            for (int i = 0; i < tokens.Length; i++)
            {
                for (int d = 0; d < config.Text.EmbeddingDim; d++)
                {
                    AssertClose(model.TextTokenEmbedding[tokens[i], d], storyCache.TextEmbedded[i, d], 1e-6f,
                        $"TextEmbedded[{i},{d}] should be raw token embedding, not a transformer output.");
                }
            }
        }

        private void Test_Text_MeanPooling_FixedShape()
        {
            var tokenizer = CreateTinyTokenizer(out var corpus);
            var config = CreateConfig(textVocabSize: tokenizer.VocabSize + 2, embDim: 16, numHeads: 2);
            var model = new TACAMT_Model(config, new Random(42));

            var stories = new[]
            {
                new NewsStory(tokenizer.Encode(corpus[0], addSpecialTokens: true), -2f),
                new NewsStory(tokenizer.Encode(corpus[1], addSpecialTokens: true), -1f),
                new NewsStory(new[] { 3 }, 0f)
            };

            var cache = new MultimodalForwardCache(config.Text.NumLayers, config.Price.NumLayers);
            model.ForwardWithCache(stories, RandomMatrix(5, 5, new Random(2), 0.2f), cache);

            Assert(cache.TextFinalHidden != null, "TextFinalHidden should exist.");
            Assert(cache.TextFinalHidden.GetLength(0) == stories.Length, "One pooled row per story expected.");
            Assert(cache.TextFinalHidden.GetLength(1) == config.Text.EmbeddingDim, "Pooled hidden dimension mismatch.");
            AssertFinite(cache.TextFinalHidden, "Pooled story hidden states");
        }

        #endregion

        #region Forward path contracts

        private void Test_Forward_NullStories_FiniteShape()
        {
            var config = CreateConfig(outputDim: 3);
            var model = new TACAMT_Model(config, new Random(42));
            var price = RandomMatrix(7, 5, new Random(1), 0.2f);

            var (pred, conf) = model.Forward((NewsStory[])null, price);

            Assert(pred.GetLength(0) == 7, "Prediction row count should match price rows.");
            Assert(pred.GetLength(1) == 3, "Prediction output dim mismatch.");
            Assert(conf == null, "Confidence matrix should be null when disabled.");
            AssertFinite(pred, "price-only predictions");
        }

        private void Test_Forward_EmptyStories_EqualsNullStories()
        {
            var model = new TACAMT_Model(CreateConfig(), new Random(42));
            var price = RandomMatrix(6, 5, new Random(1), 0.2f);

            var (nullPred, _) = model.Forward((NewsStory[])null, price);
            var (emptyPred, _) = model.Forward(Array.Empty<NewsStory>(), price);

            AssertMatrixClose(nullPred, emptyPred, 1e-6f, "null stories and empty story array should be equivalent.");
        }

        private void Test_Forward_SingleStory_ActivatesCrossAttention()
        {
            var tokenizer = CreateTinyTokenizer(out var corpus);
            var config = CreateConfig(textVocabSize: tokenizer.VocabSize + 2);
            var model = new TACAMT_Model(config, new Random(42));
            var price = RandomMatrix(7, 5, new Random(2), 0.2f);
            var stories = new[] { new NewsStory(tokenizer.Encode(corpus[0], addSpecialTokens: true), -1f) };

            var cache = new MultimodalForwardCache(config.Text.NumLayers, config.Price.NumLayers);
            model.ForwardWithCache(stories, price, cache, isTraining: false);

            Assert(cache.PriceBlockCaches[0].CrossQ != null, "Cross attention Q should be cached with story context.");
            Assert(cache.PriceBlockCaches[0].CrossAttentionWeights != null, "Cross attention weights should be cached with story context.");
        }

        private void Test_Forward_MultipleStories_AllContextRowsPresent()
        {
            var tokenizer = CreateTinyTokenizer(out var corpus);
            var config = CreateConfig(textVocabSize: tokenizer.VocabSize + 2);
            var model = new TACAMT_Model(config, new Random(42));
            var stories = new[]
            {
                new NewsStory(tokenizer.Encode(corpus[0], addSpecialTokens: true), -3f),
                new NewsStory(tokenizer.Encode(corpus[1], addSpecialTokens: true), -2f),
                new NewsStory(tokenizer.Encode(corpus[2], addSpecialTokens: true), -1f)
            };

            var cache = new MultimodalForwardCache(config.Text.NumLayers, config.Price.NumLayers);
            model.ForwardWithCache(stories, RandomMatrix(5, 5, new Random(3), 0.2f), cache, isTraining: false);

            Assert(cache.TextFinalHidden.GetLength(0) == 3, "Expected one context row per story.");
            Assert(cache.StoryArrivalTimes.Length == 3, "Expected one arrival time per story.");
            Assert(cache.PriceBlockCaches[0].CrossK.GetLength(0) == 3, "Cross-attention key rows should match story count.");
        }

        private void Test_ForwardWithCache_EqualsForward_Inference()
        {
            var tokenizer = CreateTinyTokenizer(out var corpus);
            var config = CreateConfig(textVocabSize: tokenizer.VocabSize + 2);
            var model = new TACAMT_Model(config, new Random(42));
            var price = RandomMatrix(8, 5, new Random(4), 0.2f);
            var stories = new[]
            {
                new NewsStory(tokenizer.Encode(corpus[0], addSpecialTokens: true), -3f),
                new NewsStory(tokenizer.Encode(corpus[1], addSpecialTokens: true), -1f)
            };

            var (direct, _) = model.Forward(stories, price);
            var cache = new MultimodalForwardCache(config.Text.NumLayers, config.Price.NumLayers);
            var (cached, _) = model.ForwardWithCache(stories, price, cache, isTraining: false);

            AssertMatrixClose(direct, cached, 1e-5f, "Forward and ForwardWithCache should match in inference mode.");
        }

        private void Test_ForwardWithCache_ReusedCache_ClearsStoryState()
        {
            var tokenizer = CreateTinyTokenizer(out var corpus);
            var config = CreateConfig(textVocabSize: tokenizer.VocabSize + 2);
            var model = new TACAMT_Model(config, new Random(42));
            var price = RandomMatrix(6, 5, new Random(5), 0.2f);
            var stories = new[] { new NewsStory(tokenizer.Encode(corpus[0], addSpecialTokens: true), -1f) };
            var cache = new MultimodalForwardCache(config.Text.NumLayers, config.Price.NumLayers);

            model.ForwardWithCache(stories, price, cache, isTraining: false);
            Assert(cache.StoryCaches != null && cache.StoryCaches.Count == 1, "Precondition: story cache should be populated.");

            var (reusedNoStoryPred, _) = model.ForwardWithCache(null, price, cache, isTraining: false);
            var freshCache = new MultimodalForwardCache(config.Text.NumLayers, config.Price.NumLayers);
            var (freshNoStoryPred, _) = model.ForwardWithCache(null, price, freshCache, isTraining: false);

            AssertMatrixClose(freshNoStoryPred, reusedNoStoryPred, 1e-6f, "Reusing a cache after story context should not leak old context into no-story forward.");
            Assert(cache.TextFinalHidden == null, "Reused no-story cache should clear TextFinalHidden.");
            Assert(cache.StoryArrivalTimes == null || cache.StoryArrivalTimes.Length == 0, "Reused no-story cache should clear StoryArrivalTimes.");
            Assert(cache.StoryCaches == null || cache.StoryCaches.Count == 0, "Reused no-story cache should clear StoryCaches.");
            Assert(cache.StoryTokenCounts == null || cache.StoryTokenCounts.Length == 0, "Reused no-story cache should clear StoryTokenCounts.");
            Assert(cache.PriceBlockCaches[0].CrossQ == null, "No-story forward should clear cross-attention cache.");
            Assert(cache.PriceBlockCaches[0].CrossAttentionWeights == null, "No-story forward should clear cached cross-attention weights.");
        }

        private void Test_ForwardWithCache_RowSlice_TimeAlignmentMatchesExplicitSlice()
        {
            var tokenizer = CreateTinyTokenizer(out var corpus);
            var config = CreateConfig(textVocabSize: tokenizer.VocabSize + 2, priceSeqLen: 20);
            var model = new TACAMT_Model(config, new Random(42));
            var fullPrice = RandomMatrix(14, 5, new Random(6), 0.2f);
            int rowStart = 5;
            int rowCount = 6;

            var fullFrameStories = new[]
            {
                new NewsStory(tokenizer.Encode(corpus[0], addSpecialTokens: true), 2f),
                new NewsStory(tokenizer.Encode(corpus[1], addSpecialTokens: true), 4f)
            };

            var adjustedSliceStories = fullFrameStories
                .Select(s => new NewsStory((int[])s.TokenIds.Clone(), s.ArrivalTime - rowStart))
                .ToArray();

            var cacheA = new MultimodalForwardCache(config.Text.NumLayers, config.Price.NumLayers);
            var (fromFullSlice, _) = model.ForwardWithCache(fullFrameStories, fullPrice, rowStart, rowCount, cacheA, isTraining: false);

            var explicitPriceSlice = SliceRows(fullPrice, rowStart, rowStart + rowCount);
            var cacheB = new MultimodalForwardCache(config.Text.NumLayers, config.Price.NumLayers);
            var (fromExplicitSlice, _) = model.ForwardWithCache(adjustedSliceStories, explicitPriceSlice, cacheB, isTraining: false);

            AssertMatrixClose(fromExplicitSlice, fromFullSlice, 1e-5f,
                "ForwardWithCache(rowStart,rowCount) should be equivalent to forwarding an explicit price slice with story times shifted by -rowStart.");
        }

        private void Test_Forward_PriceContext_OutputShapeAndCacheMetadata()
        {
            var tokenizer = CreateTinyTokenizer(out var corpus);
            var config = CreateConfig(textVocabSize: tokenizer.VocabSize + 2, priceSeqLen: 20);
            var model = new TACAMT_Model(config, new Random(42));
            var price = RandomMatrix(12, 5, new Random(7), 0.2f);
            var history = SliceRows(price, 0, 5);
            var current = SliceRows(price, 5, 11);
            var priceCtx = model.EncodePriceHistory(history);
            var priceTimes = Enumerable.Range(0, history.GetLength(0)).Select(t => (float)(t - history.GetLength(0))).ToArray();
            var stories = new[] { new NewsStory(tokenizer.Encode(corpus[0], addSpecialTokens: true), -1f) };
            var cache = new MultimodalForwardCache(config.Text.NumLayers, config.Price.NumLayers);

            var (pred, _) = model.ForwardWithPriceContextAndCache(stories, current, priceCtx, priceTimes, cache, isTraining: false);

            Assert(pred.GetLength(0) == current.GetLength(0), "Prediction rows should match current chunk rows.");
            Assert(pred.GetLength(1) == config.Output.OutputDim, "Prediction cols mismatch.");
            Assert(cache.NumNewsContext == 1, "Cache NumNewsContext mismatch.");
            Assert(cache.NumPriceContext == priceCtx.GetLength(0), "Cache NumPriceContext mismatch.");
            Assert(cache.TextFinalHidden.GetLength(0) == 1 + priceCtx.GetLength(0), "Combined context row count mismatch.");
            AssertFinite(pred, "price-context predictions");
        }

        private void Test_Forward_PriceContext_ChangesPrediction()
        {
            var config = CreateConfig(priceSeqLen: 20);
            var model = new TACAMT_Model(config, new Random(42));
            var price = RandomMatrix(12, 5, new Random(8), 0.2f);
            var current = SliceRows(price, 5, 11);

            var (withoutContext, _) = model.Forward((NewsStory[])null, current);

            var history = SliceRows(price, 0, 5);
            var priceCtx = model.EncodePriceHistory(history);
            var priceTimes = Enumerable.Range(0, history.GetLength(0)).Select(t => (float)(t - history.GetLength(0))).ToArray();
            var cache = new MultimodalForwardCache(config.Text.NumLayers, config.Price.NumLayers);
            var (withContext, _) = model.ForwardWithPriceContextAndCache(null, current, priceCtx, priceTimes, cache, isTraining: false);

            Assert(MatrixMaxAbsDiff(withoutContext, withContext) > 1e-6f, "Price context should affect predictions.");
        }

        private void Test_ConfidenceHead_Range_AndDefaultConfidence()
        {
            var price = RandomMatrix(5, 5, new Random(9), 0.2f);

            var withConf = new TACAMT_Model(CreateConfig(useConfidence: true), new Random(42));
            var (_, confMatrix) = withConf.Forward((NewsStory[])null, price);
            Assert(confMatrix != null, "Confidence matrix should exist when enabled.");
            for (int i = 0; i < confMatrix.GetLength(0); i++)
                Assert(confMatrix[i, 0] >= 0f && confMatrix[i, 0] <= 1f, $"Confidence[{i}] outside [0,1]: {confMatrix[i, 0]}");

            var noConf = new TACAMT_Model(CreateConfig(useConfidence: false), new Random(42));
            var (_, defaultConfidence) = noConf.PredictNext((NewsStory[])null, price);
            AssertClose(1.0f, defaultConfidence, 0f, "PredictNext confidence should be 1.0 when confidence head is disabled.");
        }

        private void Test_CausalMask_FutureRows_DoNotAffectPastRows()
        {
            var config = CreateConfig(priceSeqLen: 14);
            var model = new TACAMT_Model(config, new Random(42));
            var p1 = RandomMatrix(10, 5, new Random(10), 0.2f);
            var p2 = CloneMatrix(p1);

            for (int t = 7; t < 10; t++)
                for (int f = 0; f < 5; f++)
                    p2[t, f] += 3.0f;

            var (r1, _) = model.Forward((NewsStory[])null, p1);
            var (r2, _) = model.Forward((NewsStory[])null, p2);

            for (int t = 0; t < 7; t++)
                for (int j = 0; j < config.Output.OutputDim; j++)
                    AssertClose(r1[t, j], r2[t, j], 1e-5f, $"Future rows affected output at [{t},{j}].");
        }

        private void Test_Deterministic_Inference_IsRepeatable()
        {
            var tokenizer = CreateTinyTokenizer(out var corpus);
            var config = CreateConfig(textVocabSize: tokenizer.VocabSize + 2);
            var model = new TACAMT_Model(config, new Random(42));
            var stories = new[] { new NewsStory(tokenizer.Encode(corpus[0], addSpecialTokens: true), -1f) };
            var price = RandomMatrix(6, 5, new Random(11), 0.2f);

            var (a, _) = model.Forward(stories, price);
            var (b, _) = model.Forward(stories, price);
            AssertMatrixClose(a, b, 0f, "Inference should be deterministic when dropout is not enabled.");
        }

        private void Test_Stability_ExtremeInputs_NoNaN()
        {
            var model = new TACAMT_Model(CreateConfig(priceSeqLen: 10), new Random(42));
            var cases = new[]
            {
                FilledMatrix(6, 5, 0f),
                FilledMatrix(6, 5, 1e-7f),
                FilledMatrix(6, 5, -100f),
                FilledMatrix(6, 5, 100f),
                MixedMagnitudeMatrix(6, 5, new Random(12))
            };

            foreach (var price in cases)
            {
                var (pred, _) = model.Forward((NewsStory[])null, price);
                AssertFinite(pred, "extreme-input predictions");
            }
        }

        #endregion

        #region Temporal decay / memory semantics

        private void Test_DecayNetwork_ZeroTimeDiff_ProducesNearZeroBias()
        {
            var rng = new Random(42);
            var accel = new CallaghanDev.ML.AccelerationManagers.AccelerationCPU();
            var net = new ContentAwareDecayNetwork(2, 16, 8, 16, rng, memAttnDropout: 0f, mlpDropout: 0f);
            var query = RandomMatrix(3, 16, rng, 0.2f);
            var key = RandomMatrix(2, 16, rng, 0.2f);
            var timeDiffs = new float[3, 2];
            var keyTimes = new[] { 0f, 0f };

            var (bias, _) = accel.ContentAwareDecayForward(query, key, timeDiffs, keyTimes, net, isTraining: false);

            for (int q = 0; q < bias.GetLength(0); q++)
                for (int k = 0; k < bias.GetLength(1); k++)
                    for (int h = 0; h < bias.GetLength(2); h++)
                        Assert(MathF.Abs(bias[q, k, h]) < 1e-5f, $"Bias should be near zero at zero time diff, got {bias[q, k, h]}.");
        }

        private void Test_DecayNetwork_Gates_AreWithinConfiguredRange()
        {
            var rng = new Random(42);
            var accel = new CallaghanDev.ML.AccelerationManagers.AccelerationCPU();
            var net = new ContentAwareDecayNetwork(2, 16, 8, 16, rng, memAttnDropout: 0f, mlpDropout: 0f, minGate: 0.05f, maxGate: 0.95f);
            var query = RandomMatrix(4, 16, rng, 0.4f);
            var key = RandomMatrix(3, 16, rng, 0.4f);
            var timeDiffs = new float[4, 3];
            var keyTimes = new[] { -3f, -2f, -1f };
            for (int q = 0; q < 4; q++)
                for (int k = 0; k < 3; k++)
                    timeDiffs[q, k] = q - keyTimes[k];

            var (_, cache) = accel.ContentAwareDecayForward(query, key, timeDiffs, keyTimes, net, isTraining: false);

            for (int q = 0; q < 4; q++)
                for (int k = 0; k < 3; k++)
                    for (int h = 0; h < 2; h++)
                        Assert(cache.Gates[q, k, h] >= net.MinGate && cache.Gates[q, k, h] <= net.MaxGate,
                            $"Gate[{q},{k},{h}]={cache.Gates[q, k, h]} outside configured range.");
        }

        private void Test_Decay_RecentIdenticalStory_GetsHigherAttentionThanOld()
        {
            var config = CreateConfig(priceSeqLen: 8);
            var model = new TACAMT_Model(config, new Random(42));
            var price = FilledMatrix(6, 5, 0.5f);
            var tokens = new[] { 4, 5, 6, 7 };
            var stories = new[]
            {
                new NewsStory(tokens, -1f),
                new NewsStory(tokens, -1000f)
            };

            var cache = new MultimodalForwardCache(config.Text.NumLayers, config.Price.NumLayers);
            model.ForwardWithCache(stories, price, cache, isTraining: false);

            var weights = cache.PriceBlockCaches[0].CrossAttentionWeights;
            Assert(weights != null, "Cross-attention weights should be cached.");

            float recent = 0f;
            float old = 0f;
            for (int h = 0; h < weights.Length; h++)
            {
                var w = weights[h];
                for (int q = 0; q < w.GetLength(0); q++)
                {
                    recent += w[q, 0];
                    old += w[q, 1];
                }
            }

            Assert(recent > old, $"Recent identical story should get more attention. recent={recent:F6}, old={old:F6}");
        }

        private void Test_Decay_TimeDiffCache_UsesPastRelativeConvention()
        {
            var config = CreateConfig(priceSeqLen: 8);
            var model = new TACAMT_Model(config, new Random(42));
            var price = FilledMatrix(5, 5, 0.5f);
            var stories = new[]
            {
                new NewsStory(new[] { 4, 5, 6 }, -2f),
                new NewsStory(new[] { 7, 8, 9 }, -5f)
            };

            var cache = new MultimodalForwardCache(config.Text.NumLayers, config.Price.NumLayers);
            model.ForwardWithCache(stories, price, cache, isTraining: false);

            var td = cache.PriceBlockCaches[0].TimeDiffs;
            Assert(td != null, "TimeDiffs should be cached when stories have times.");
            AssertClose(2f, td[0, 0], 1e-5f, "At query row 0, key time -2 should have age 2.");
            AssertClose(5f, td[0, 1], 1e-5f, "At query row 0, key time -5 should have age 5.");
            AssertClose(4f, td[2, 0], 1e-5f, "At query row 2, key time -2 should have age 4.");
            AssertClose(7f, td[2, 1], 1e-5f, "At query row 2, key time -5 should have age 7.");
        }

        private void Test_PredictWithMemory_ExistingNewsAge_MatchesDirectForward()
        {
            var config = CreateConfig(priceSeqLen: 8);
            var price = RandomMatrix(6, 5, new Random(13), 0.2f);
            var oldTokens = new[] { 4, 5, 6, 7 };
            var recentTokens = new[] { 8, 9, 10, 11 };

            var directStories = new[]
            {
                new NewsStory(oldTokens, -10f),
                new NewsStory(recentTokens, -1f)
            };

            var directModel = new TACAMT_Model(config, new Random(42));
            var (directPredMatrix, _) = directModel.Forward(directStories, price);
            var directLast = GetLastRow(directPredMatrix);

            var memoryModel = new TACAMT_Model(config, new Random(42));
            memoryModel.UpdateNewsMemory(new[] { new NewsStory((int[])oldTokens.Clone(), 0f) }, ts: 90.0);
            memoryModel.UpdateNewsMemory(new[] { new NewsStory((int[])recentTokens.Clone(), 0f) }, ts: 99.0);
            var (memoryPred, _) = memoryModel.PredictWithMemory(null, price, currentAbsoluteTimestamp: 100.0, maxNewsMemorySize: 100, maxPriceMemorySize: 0);

            AssertVectorClose(directLast, memoryPred, 1e-5f,
                "Memory stories at t=90 and t=99 queried at t=100 should equal direct forward with ArrivalTime -10 and -1.");
        }

        private void Test_PredictWithMemory_StoresAbsoluteTimes_AndLastPriceTimestamp()
        {
            var config = CreateConfig(priceSeqLen: 8);
            var model = new TACAMT_Model(config, new Random(42));
            var stories = new[]
            {
                new NewsStory(new[] { 4, 5, 6 }, -3f),
                new NewsStory(new[] { 7, 8, 9 }, 0f)
            };
            var price = RandomMatrix(5, 5, new Random(14), 0.2f);

            model.PredictWithMemory(stories, price, currentAbsoluteTimestamp: 100.0, timeUnitsPerPosition: 2.0, maxNewsMemorySize: 10, maxPriceMemorySize: 10);

            Assert(model.NewsMemory.Count == 2, "Expected two news memory entries.");
            AssertClose(94.0, model.NewsMemory[0].AbsoluteTimestamp, 1e-8, "First story absolute timestamp should be current + arrival * timeUnits.");
            AssertClose(100.0, model.NewsMemory[1].AbsoluteTimestamp, 1e-8, "Second story absolute timestamp mismatch.");
            Assert(model.PriceMemory.Count == price.GetLength(0), "One price memory entry per price row expected.");
            AssertClose(100.0, model.PriceMemory[0].AbsoluteTimestamp, 1e-8, "Price row 0 absolute timestamp mismatch.");
            AssertClose(108.0, model.PriceMemory[4].AbsoluteTimestamp, 1e-8, "Last price row absolute timestamp mismatch.");
            AssertClose(108.0, model.LastPriceTimestamp, 1e-8, "LastPriceTimestamp should equal timestamp of last price row.");
        }

        private void Test_PredictWithMemory_AccumulatesAndPrunes()
        {
            var config = CreateConfig(priceSeqLen: 8);
            var model = new TACAMT_Model(config, new Random(42));
            var price = RandomMatrix(5, 5, new Random(15), 0.2f);

            for (int i = 0; i < 6; i++)
            {
                model.PredictWithMemory(
                    new[] { new NewsStory(new[] { 4, 5, 6, 7 }, -1f) },
                    price,
                    currentAbsoluteTimestamp: 100 + i * 10,
                    maxNewsMemorySize: 3,
                    maxPriceMemorySize: 7);
            }

            Assert(model.NewsMemory.Count <= 3, $"News memory should be pruned to <=3, got {model.NewsMemory.Count}.");
            Assert(model.PriceMemory.Count <= 7, $"Price memory should be pruned to <=7, got {model.PriceMemory.Count}.");
            Assert(model.NewsMemory.Count > 0, "News memory should retain some entries.");
            Assert(model.PriceMemory.Count > 0, "Price memory should retain some entries.");
        }

        private void Test_Pruning_MaxZero_ClearsNewsAndPrice()
        {
            var config = CreateConfig();
            var model = new TACAMT_Model(config, new Random(42));
            model.PruningConfig.UseAttentionBasedPruning = true;

            for (int i = 0; i < 5; i++)
            {
                model.NewsMemory.Add(new NewsMemoryEntry { HiddenState = new float[config.Price.EmbeddingDim], AbsoluteTimestamp = i, AttentionScore = 1f, QueryCount = 10 });
                model.PriceMemory.Add(new PriceMemoryEntry { HiddenState = new float[config.Price.EmbeddingDim], AbsoluteTimestamp = i, AttentionScore = 1f, QueryCount = 10 });
            }

            InvokePruneNews(model, 0);
            InvokePrunePrice(model, 0);

            Assert(model.NewsMemory.Count == 0, $"News memory should be cleared, got {model.NewsMemory.Count}.");
            Assert(model.PriceMemory.Count == 0, $"Price memory should be cleared, got {model.PriceMemory.Count}.");
        }

        private void Test_Pruning_Disabled_KeepsNewestEntries()
        {
            var config = CreateConfig();
            var model = new TACAMT_Model(config, new Random(42));
            model.PruningConfig.UseAttentionBasedPruning = false;

            for (int i = 0; i < 10; i++)
                model.NewsMemory.Add(new NewsMemoryEntry { HiddenState = new float[config.Price.EmbeddingDim], AbsoluteTimestamp = i * 10.0, AttentionScore = 0.01f, QueryCount = 10 });

            InvokePruneNews(model, 4);

            Assert(model.NewsMemory.Count == 4, "Expected exactly four entries.");
            Assert(model.NewsMemory.Min(e => e.AbsoluteTimestamp) >= 60.0, $"Disabled pruning should keep newest entries. Oldest kept={model.NewsMemory.Min(e => e.AbsoluteTimestamp)}.");
            Assert(IsSortedAscending(model.NewsMemory.Select(e => e.AbsoluteTimestamp)), "Memory should be returned in chronological order after pruning.");
        }

        private void Test_Pruning_AttentionBased_KeepsHighScoreAndReservedNewest()
        {
            var config = CreateConfig();
            var model = new TACAMT_Model(config, new Random(42));
            model.PruningConfig.UseAttentionBasedPruning = true;
            model.PruningConfig.MinQueryCountForPruning = 1;
            model.PruningConfig.NewEntryReserveFraction = 0.25f;

            for (int i = 0; i < 12; i++)
            {
                model.NewsMemory.Add(new NewsMemoryEntry
                {
                    HiddenState = new float[config.Price.EmbeddingDim],
                    AbsoluteTimestamp = i * 10.0,
                    AttentionScore = i == 1 || i == 2 ? 10.0f : 0.01f,
                    QueryCount = 10
                });
            }

            InvokePruneNews(model, 4);

            Assert(model.NewsMemory.Count == 4, "Expected exactly four entries.");
            Assert(model.NewsMemory.Any(e => Math.Abs(e.AbsoluteTimestamp - 10.0) < 1e-8), "High-score old entry at t=10 should survive.");
            Assert(model.NewsMemory.Any(e => Math.Abs(e.AbsoluteTimestamp - 20.0) < 1e-8), "High-score old entry at t=20 should survive.");
            Assert(model.NewsMemory.Any(e => Math.Abs(e.AbsoluteTimestamp - 110.0) < 1e-8), "Newest reserved entry should survive.");
        }

        private void Test_Pruning_ColdStartEntries_AreProtected()
        {
            var config = CreateConfig();
            var model = new TACAMT_Model(config, new Random(42));
            model.PruningConfig.UseAttentionBasedPruning = true;
            model.PruningConfig.MinQueryCountForPruning = 5;
            model.PruningConfig.NewEntryReserveFraction = 0f;

            for (int i = 0; i < 8; i++)
            {
                model.NewsMemory.Add(new NewsMemoryEntry
                {
                    HiddenState = new float[config.Price.EmbeddingDim],
                    AbsoluteTimestamp = i,
                    AttentionScore = i < 3 ? 0.0f : 1.0f,
                    QueryCount = i < 3 ? 0 : 10
                });
            }

            InvokePruneNews(model, 4);

            int coldStartSurvivors = model.NewsMemory.Count(e => e.QueryCount < 5);
            Assert(coldStartSurvivors > 0, "At least one cold-start entry should be protected from immediate pruning.");
        }

        #endregion

        #region Save / load

        private void Test_SaveLoad_ForwardMatches_AndWeightsPreserved()
        {
            var tokenizer = CreateTinyTokenizer(out var corpus);
            var config = CreateConfig(textVocabSize: tokenizer.VocabSize + 2, embDim: 16, numHeads: 2, numLayers: 2);
            var model = new TACAMT_Model(config, new Random(42));
            var stories = new[] { new NewsStory(tokenizer.Encode(corpus[0], addSpecialTokens: true), -1f) };
            var price = RandomMatrix(6, 5, new Random(16), 0.2f);
            var (before, _) = model.Forward(stories, price);
            var outputProjectionBefore = CloneMatrix(model.OutputProjection);
            var contextTypeBefore = CloneMatrix(model.ContextTypeEmbedding);

            string dir = GetTempDir();
            try
            {
                model.Save(dir);
                var loaded = TACAMT_Model.Load(dir);
                var (after, _) = loaded.Forward(stories, price);

                AssertMatrixClose(before, after, 1e-5f, "Forward output mismatch after save/load.");
                AssertMatrixClose(outputProjectionBefore, loaded.OutputProjection, 0f, "OutputProjection should be exactly preserved.");
                AssertMatrixClose(contextTypeBefore, loaded.ContextTypeEmbedding, 0f, "ContextTypeEmbedding should be exactly preserved.");
            }
            finally
            {
                CleanupDir(dir);
            }
        }

        private void Test_SaveLoad_MemoryScoresTokenizerAndPruningConfig_Preserved()
        {
            var tokenizer = CreateTinyTokenizer(out _);
            var config = CreateConfig(textVocabSize: tokenizer.VocabSize + 2);
            var model = new TACAMT_Model(config, new Random(42));
            model.SetTokenizer(tokenizer);
            model.PruningConfig.AttentionScoreAlpha = 0.33f;
            model.PruningConfig.MinQueryCountForPruning = 9;
            model.PruningConfig.NewEntryReserveFraction = 0.2f;
            model.PruningConfig.UseAttentionBasedPruning = false;
            model.NewsMemory.Add(new NewsMemoryEntry { HiddenState = RampVector(config.Price.EmbeddingDim, 0.1f), AbsoluteTimestamp = 12.0, AttentionScore = 0.42f, QueryCount = 7 });
            model.PriceMemory.Add(new PriceMemoryEntry { HiddenState = RampVector(config.Price.EmbeddingDim, 0.2f), AbsoluteTimestamp = 13.0, AttentionScore = 0.77f, QueryCount = 3 });
            model.LastPriceTimestamp = 99.0;

            string dir = GetTempDir();
            try
            {
                model.Save(dir);
                var loaded = TACAMT_Model.Load(dir);

                Assert(loaded.Tokenizer != null, "Tokenizer should be preserved.");
                AssertClose(0.33f, loaded.PruningConfig.AttentionScoreAlpha, 1e-6f, "Pruning alpha mismatch.");
                Assert(loaded.PruningConfig.MinQueryCountForPruning == 9, "Pruning min query count mismatch.");
                AssertClose(0.2f, loaded.PruningConfig.NewEntryReserveFraction, 1e-6f, "Pruning reserve fraction mismatch.");
                Assert(!loaded.PruningConfig.UseAttentionBasedPruning, "Pruning enabled flag mismatch.");
                AssertClose(99.0, loaded.LastPriceTimestamp, 1e-8, "LastPriceTimestamp mismatch.");
                Assert(loaded.NewsMemory.Count == 1, "News memory count mismatch.");
                Assert(loaded.PriceMemory.Count == 1, "Price memory count mismatch.");
                AssertClose(0.42f, loaded.NewsMemory[0].AttentionScore, 1e-6f, "News attention score mismatch.");
                Assert(loaded.NewsMemory[0].QueryCount == 7, "News query count mismatch.");
                AssertClose(0.77f, loaded.PriceMemory[0].AttentionScore, 1e-6f, "Price attention score mismatch.");
                Assert(loaded.PriceMemory[0].QueryCount == 3, "Price query count mismatch.");
                AssertVectorClose(model.NewsMemory[0].HiddenState, loaded.NewsMemory[0].HiddenState, 0f, "News hidden state mismatch.");
                AssertVectorClose(model.PriceMemory[0].HiddenState, loaded.PriceMemory[0].HiddenState, 0f, "Price hidden state mismatch.");
            }
            finally
            {
                CleanupDir(dir);
            }
        }

        #endregion

        #region Trainer validation / optimization

        private void Test_Validate_PriceContextDisabled_MatchesManualNoContextMse()
        {
            var config = CreateConfig(inputFeatures: 3, outputDim: 2, priceSeqLen: 20, priceContextEnabled: false, priceCtxMinHist: 4, priceCtxMinCurrent: 4);
            var model = new TACAMT_Model(config, new Random(42));
            var trainer = new TACAMT_Trainer(model, new TrainingConfig { LearningRate = 0.001f, BatchSize = 1, Epochs = 1, Verbose = false });
            var rng = new Random(17);
            var stories = new NewsStory[1][] { null };
            var priceInputs = new[] { RandomMatrix(14, 3, rng, 0.2f) };
            var priceTargets = new[] { RandomMatrix(14, 2, rng, 0.2f) };

            float validateLoss = trainer.Validate(stories, priceInputs, priceTargets);
            float manualLoss = ManualNoContextNextStepLoss(model, stories[0], priceInputs[0], priceTargets[0]);

            AssertClose(manualLoss, validateLoss, 1e-5f,
                "Validate should not use the price-context split path when PriceContext.Enabled=false.");
        }

        private void Test_Validate_PriceContextEnabled_UsesNegativeHistoryKeyTimes()
        {
            var tokenizer = CreateTinyTokenizer(out var corpus);
            var config = CreateConfig(textVocabSize: tokenizer.VocabSize + 2, inputFeatures: 3, outputDim: 1, priceSeqLen: 20, priceContextEnabled: true, priceCtxMinHist: 4, priceCtxMinCurrent: 4);
            var model = new TACAMT_Model(config, new Random(42));
            var trainer = new TACAMT_Trainer(model, new TrainingConfig { LearningRate = 0.001f, BatchSize = 1, Epochs = 1, Verbose = false });
            var rng = new Random(18);
            var stories = new[] { new[] { new NewsStory(tokenizer.Encode(corpus[0], addSpecialTokens: true), -1f) } };
            var priceInputs = new[] { RandomMatrix(14, 3, rng, 0.2f) };
            var priceTargets = new[] { RandomMatrix(14, 1, rng, 0.2f) };

            float validateLoss = trainer.Validate(stories, priceInputs, priceTargets);
            float manualLoss = ManualPriceContextValidationLoss(model, stories[0], priceInputs[0], priceTargets[0], config.PriceContext.MinHistoryLength, config.PriceContext.MinCurrentLength);

            AssertClose(manualLoss, validateLoss, 1e-5f,
                "Validate price-context path should use history key times t - splitPoint, not positive ages.");
        }

        private void Test_Train_PriceOnly_EasyConstantTarget_LossDrops()
        {
            var config = CreateConfig(inputFeatures: 2, outputDim: 1, embDim: 12, numHeads: 3, priceSeqLen: 10, priceContextEnabled: false);
            var model = new TACAMT_Model(config, new Random(42));
            var (stories, inputs, targets) = CreateConstantTargetPriceData(samples: 6, seqLen: 8, inputDim: 2, outputDim: 1, targetValue: 0.35f, seed: 19);

            var trainer = new TACAMT_Trainer(model, new TrainingConfig { LearningRate = 0.01f, BatchSize = 3, Epochs = 1, UseGradientClipping = true, GradientClipThreshold = 1f, Verbose = false });
            float before = trainer.Validate(stories, inputs, targets);

            trainer = new TACAMT_Trainer(model, new TrainingConfig { LearningRate = 0.01f, BatchSize = 3, Epochs = 60, UseGradientClipping = true, GradientClipThreshold = 1f, Verbose = false });
            trainer.Train(stories, inputs, targets);
            float after = trainer.Validate(stories, inputs, targets);

            AssertFinite(after, "loss after training");
            Assert(after < before * 0.8f, $"Expected easy constant-target loss to drop meaningfully. before={before:F6}, after={after:F6}");
        }

        private void Test_Train_TextFrozen_DoesNotMutateTextEmbeddings()
        {
            var tokenizer = CreateTinyTokenizer(out var corpus);
            var config = CreateConfig(textVocabSize: tokenizer.VocabSize + 2, freezeText: true, priceContextEnabled: false, outputDim: 1, inputFeatures: 2, priceSeqLen: 10);
            var model = new TACAMT_Model(config, new Random(42));
            var before = CloneMatrix(model.TextTokenEmbedding);
            var (stories, inputs, targets) = CreateBullBearData(tokenizer, corpus[0], corpus[1], samples: 12, seqLen: 8, inputDim: 2);

            var trainer = new TACAMT_Trainer(model, new TrainingConfig { LearningRate = 0.01f, BatchSize = 4, Epochs = 20, UseGradientClipping = true, GradientClipThreshold = 1f, Verbose = false });
            trainer.Train(stories, inputs, targets);

            AssertMatrixClose(before, model.TextTokenEmbedding, 0f, "Frozen text embeddings should not change during training.");
        }

        private void Test_Train_ConfidenceHead_UpdatesWithTargets()
        {
            var config = CreateConfig(inputFeatures: 2, outputDim: 1, embDim: 12, numHeads: 3, priceSeqLen: 10, useConfidence: true, priceContextEnabled: false);
            var model = new TACAMT_Model(config, new Random(42));
            var before = CloneMatrix(model.ConfidenceProjection);
            var (stories, inputs, targets) = CreateConstantTargetPriceData(samples: 8, seqLen: 8, inputDim: 2, outputDim: 1, targetValue: 0.25f, seed: 20);
            var confidenceTargets = Enumerable.Range(0, 8).Select(s => Enumerable.Range(0, 8).Select(t => s < 4 ? 1f : 0f).ToArray()).ToArray();

            var trainer = new TACAMT_Trainer(model, new TrainingConfig { LearningRate = 0.01f, BatchSize = 4, Epochs = 40, ConfidenceLossWeight = 1f, UseGradientClipping = true, GradientClipThreshold = 1f, Verbose = false });
            trainer.Train(stories, inputs, targets, confidenceTargets);

            Assert(MatrixChanged(before, model.ConfidenceProjection, 1e-8f), "ConfidenceProjection should update when confidence targets are supplied.");
            var (_, conf) = model.Forward((NewsStory[])null, inputs[0]);
            AssertFinite(conf, "confidence output after training");
        }

        private void Test_TrainSequential_AccumulatesMemoryAndRespectsLimits()
        {
            var tokenizer = CreateTinyTokenizer(out var corpus);
            var config = CreateConfig(textVocabSize: tokenizer.VocabSize + 2, inputFeatures: 2, outputDim: 1, embDim: 12, numHeads: 3, priceSeqLen: 10);
            var model = new TACAMT_Model(config, new Random(42));
            var (stories, inputs, targets) = CreateBullBearData(tokenizer, corpus[0], corpus[1], samples: 5, seqLen: 8, inputDim: 2);
            var timestamps = Enumerable.Range(0, 5).Select(i => 100.0 + i * 20.0).ToArray();

            var trainer = new TACAMT_Trainer(model, new TrainingConfig { LearningRate = 0.005f, BatchSize = 1, Epochs = 1, UseGradientClipping = true, GradientClipThreshold = 1f, Verbose = false });
            trainer.TrainSequential(stories, inputs, targets, timestamps, timeUnitsPerPosition: 1.0, maxNewsMemory: 3, maxPriceMemory: 6);

            Assert(model.NewsMemory.Count > 0 && model.NewsMemory.Count <= 3, $"News memory limit not respected: {model.NewsMemory.Count}.");
            Assert(model.PriceMemory.Count > 0 && model.PriceMemory.Count <= 6, $"Price memory limit not respected: {model.PriceMemory.Count}.");
            AssertFinite(model.PriceMemory[0].HiddenState, "stored price memory hidden state");
        }

        private void Test_TrainSequential_TimeUnitsPerPositionZero_Throws()
        {
            var config = CreateConfig(priceContextEnabled: false);
            var model = new TACAMT_Model(config, new Random(42));
            var trainer = new TACAMT_Trainer(model, new TrainingConfig { LearningRate = 0.001f, BatchSize = 1, Epochs = 1, Verbose = false });
            var (stories, inputs, targets) = CreateConstantTargetPriceData(samples: 1, seqLen: 4, inputDim: 5, outputDim: 5, targetValue: 0f, seed: 21);

            AssertThrows<ArgumentOutOfRangeException>(() => trainer.TrainSequential(stories, inputs, targets, new[] { 100.0 }, timeUnitsPerPosition: 0.0),
                "TrainSequential should reject zero timeUnitsPerPosition.");
        }

        private void Test_Train_MixedNullAndShortSamples_NoNaN()
        {
            var config = CreateConfig(inputFeatures: 2, outputDim: 1, embDim: 12, numHeads: 3, priceSeqLen: 8, priceContextEnabled: false);
            var model = new TACAMT_Model(config, new Random(42));
            var stories = new NewsStory[4][];
            var inputs = new float[4][,];
            var targets = new float[4][,];
            inputs[0] = RandomMatrix(1, 2, new Random(22), 0.2f); targets[0] = FilledMatrix(1, 1, 0.1f);
            inputs[1] = RandomMatrix(5, 2, new Random(23), 0.2f); targets[1] = FilledMatrix(5, 1, 0.2f);
            inputs[2] = RandomMatrix(6, 2, new Random(24), 0.2f); targets[2] = FilledMatrix(6, 1, 0.3f);
            inputs[3] = RandomMatrix(2, 2, new Random(25), 0.2f); targets[3] = FilledMatrix(2, 1, 0.4f);

            var trainer = new TACAMT_Trainer(model, new TrainingConfig { LearningRate = 0.005f, BatchSize = 2, Epochs = 5, UseGradientClipping = true, GradientClipThreshold = 1f, Verbose = false });
            trainer.Train(stories, inputs, targets);
            float loss = trainer.Validate(stories, inputs, targets);

            AssertFinite(loss, "mixed-sample validation loss");
        }

        #endregion

        #region Backward / gradient routing

        private void Test_Backward_PriceOnly_ProducesCoreGradients()
        {
            var config = CreateConfig(inputFeatures: 3, outputDim: 2, embDim: 12, numHeads: 3, priceSeqLen: 8, priceContextEnabled: false);
            var model = new TACAMT_Model(config, new Random(42));
            var trainer = new TACAMT_Trainer(model, new TrainingConfig { LearningRate = 0.001f, BatchSize = 1, Epochs = 1, Verbose = false });
            var price = RandomMatrix(5, 3, new Random(26), 0.2f);
            var target = FilledMatrix(5, 2, 0.4f);
            var cache = new MultimodalForwardCache(config.Text.NumLayers, config.Price.NumLayers);
            var (pred, conf) = model.ForwardWithCache(null, price, cache, isTraining: true, dropoutRng: new Random(1));

            RunBackward(trainer, pred, conf, target, null, cache);
            var grads = GetGradients(trainer);

            Assert(SumAbs(grads.OutputProjectionGrad) > 1e-9f, "OutputProjectionGrad should be non-zero.");
            Assert(SumAbs(grads.OutputBiasGrad) > 1e-9f, "OutputBiasGrad should be non-zero.");
            Assert(SumAbs(grads.PriceInputProjectionGrad) > 1e-9f, "PriceInputProjectionGrad should be non-zero.");
            Assert(AttentionGradSum(grads.PriceBlockGrads[0].SelfAttnGrads) > 1e-9f, "Self-attention gradients should be non-zero.");
            Assert(AttentionGradSum(grads.PriceBlockGrads[0].CrossAttnGrads) == 0f, "Cross-attention grads should be zero without context.");
        }

        private void Test_Backward_NewsContext_ProducesTextCrossAndDecayGradients()
        {
            var tokenizer = CreateTinyTokenizer(out var corpus);
            var config = CreateConfig(textVocabSize: tokenizer.VocabSize + 2, inputFeatures: 3, outputDim: 2, embDim: 12, numHeads: 3, priceSeqLen: 8, freezeText: false, priceContextEnabled: false);
            var model = new TACAMT_Model(config, new Random(42));
            var trainer = new TACAMT_Trainer(model, new TrainingConfig { LearningRate = 0.001f, BatchSize = 1, Epochs = 1, Verbose = false });
            var price = RandomMatrix(5, 3, new Random(27), 0.2f);
            var target = FilledMatrix(5, 2, 0.4f);
            var stories = new[]
            {
                new NewsStory(tokenizer.Encode(corpus[0], addSpecialTokens: true), -3f),
                new NewsStory(tokenizer.Encode(corpus[1], addSpecialTokens: true), -1f)
            };
            var cache = new MultimodalForwardCache(config.Text.NumLayers, config.Price.NumLayers);
            var (pred, conf) = model.ForwardWithCache(stories, price, cache, isTraining: true, dropoutRng: new Random(1));

            RunBackward(trainer, pred, conf, target, null, cache);
            var grads = GetGradients(trainer);

            Assert(AttentionGradSum(grads.PriceBlockGrads[0].CrossAttnGrads) > 1e-9f, "Cross-attention gradients should be non-zero with news context.");
            Assert(DecayGradSum(grads.PriceBlockGrads[0].DecayGrads) > 1e-9f, "Decay network gradients should be non-zero with timed context.");
            Assert(SumAbs(grads.TextEmbeddingGrad) > 1e-9f, "Text embedding gradients should be non-zero when text is not frozen.");
            Assert(SumAbsRow(grads.ContextTypeEmbeddingGrad, 0) > 1e-9f, "News context type embedding gradient should be non-zero.");
        }

        private void Test_Backward_FrozenText_HasNoTextEmbeddingGradient()
        {
            var tokenizer = CreateTinyTokenizer(out var corpus);
            var config = CreateConfig(textVocabSize: tokenizer.VocabSize + 2, inputFeatures: 3, outputDim: 2, embDim: 12, numHeads: 3, priceSeqLen: 8, freezeText: true, priceContextEnabled: false);
            var model = new TACAMT_Model(config, new Random(42));
            var trainer = new TACAMT_Trainer(model, new TrainingConfig { LearningRate = 0.001f, BatchSize = 1, Epochs = 1, Verbose = false });
            var price = RandomMatrix(5, 3, new Random(28), 0.2f);
            var target = FilledMatrix(5, 2, 0.4f);
            var stories = new[] { new NewsStory(tokenizer.Encode(corpus[0], addSpecialTokens: true), -1f) };
            var cache = new MultimodalForwardCache(config.Text.NumLayers, config.Price.NumLayers);
            var (pred, conf) = model.ForwardWithCache(stories, price, cache, isTraining: true, dropoutRng: new Random(1));

            RunBackward(trainer, pred, conf, target, null, cache);
            var grads = GetGradients(trainer);

            Assert(SumAbs(grads.TextEmbeddingGrad) == 0f, "Text embedding gradients should remain zero when text is frozen.");
            Assert(AttentionGradSum(grads.PriceBlockGrads[0].CrossAttnGrads) > 1e-9f, "Cross-attention gradients should still flow when text is frozen.");
        }

        private void Test_Backward_PriceContext_ProducesBothTypeEmbeddingGradients()
        {
            var tokenizer = CreateTinyTokenizer(out var corpus);
            var config = CreateConfig(textVocabSize: tokenizer.VocabSize + 2, inputFeatures: 3, outputDim: 2, embDim: 12, numHeads: 3, priceSeqLen: 12, priceContextEnabled: true);
            var model = new TACAMT_Model(config, new Random(42));
            var trainer = new TACAMT_Trainer(model, new TrainingConfig { LearningRate = 0.001f, BatchSize = 1, Epochs = 1, Verbose = false });
            var price = RandomMatrix(9, 3, new Random(29), 0.2f);
            var history = SliceRows(price, 0, 4);
            var current = SliceRows(price, 4, 8);
            var priceCtx = model.EncodePriceHistory(history);
            var priceTimes = Enumerable.Range(0, history.GetLength(0)).Select(t => (float)(t - history.GetLength(0))).ToArray();
            var stories = new[] { new NewsStory(tokenizer.Encode(corpus[0], addSpecialTokens: true), -1f) };
            var target = FilledMatrix(current.GetLength(0), 2, 0.3f);
            var cache = new MultimodalForwardCache(config.Text.NumLayers, config.Price.NumLayers);
            var (pred, conf) = model.ForwardWithPriceContextAndCache(stories, current, priceCtx, priceTimes, cache, isTraining: true, dropoutRng: new Random(1));

            RunBackward(trainer, pred, conf, target, null, cache);
            var grads = GetGradients(trainer);

            Assert(SumAbsRow(grads.ContextTypeEmbeddingGrad, 0) > 1e-9f, "News context type embedding gradient should be non-zero.");
            Assert(SumAbsRow(grads.ContextTypeEmbeddingGrad, 1) > 1e-9f, "Price context type embedding gradient should be non-zero.");
        }

        private void Test_BackpropMultiStory_UsesFreshStoryOffset_WhenNewsMemoryExists()
        {
            var config = CreateConfig(textVocabSize: 50, embDim: 12, numHeads: 3, priceContextEnabled: false);
            var model = new TACAMT_Model(config, new Random(42));
            var trainer = new TACAMT_Trainer(model, new TrainingConfig { LearningRate = 0.001f, BatchSize = 1, Epochs = 1, Verbose = false });
            var story = new[] { new NewsStory(new[] { 4, 5, 6 }, -1f) };
            var cache = new MultimodalForwardCache(config.Text.NumLayers, config.Price.NumLayers);
            model.ForwardWithCache(story, RandomMatrix(4, 5, new Random(30), 0.2f), cache, isTraining: true, dropoutRng: new Random(1));

            cache.NumNewsContext = 2; // one existing news-memory row + one fresh story row
            var dShared = new float[2, config.Text.EmbeddingDim];
            for (int d = 0; d < config.Text.EmbeddingDim; d++)
                dShared[1, d] = 1f; // gradient belongs to the fresh story row, not memory row 0

            InvokePrivate(trainer, "ZeroAllGradients");
            InvokePrivate(trainer, "BackpropMultiStoryTextEncoder", dShared, cache);
            var grads = GetGradients(trainer);
            float tokenGrad = story[0].TokenIds.Sum(id => SumAbsRow(grads.TextEmbeddingGrad, id));

            Assert(tokenGrad > 1e-9f, "Fresh story token gradients should be read from dShared row offset after existing news memory rows.");
        }

        private void Test_BackpropMultiStory_DoesNotRouteExistingMemoryGradientToFreshText()
        {
            var config = CreateConfig(textVocabSize: 50, embDim: 12, numHeads: 3, priceContextEnabled: false);
            var model = new TACAMT_Model(config, new Random(42));
            var trainer = new TACAMT_Trainer(model, new TrainingConfig { LearningRate = 0.001f, BatchSize = 1, Epochs = 1, Verbose = false });
            var story = new[] { new NewsStory(new[] { 4, 5, 6 }, -1f) };
            var cache = new MultimodalForwardCache(config.Text.NumLayers, config.Price.NumLayers);
            model.ForwardWithCache(story, RandomMatrix(4, 5, new Random(31), 0.2f), cache, isTraining: true, dropoutRng: new Random(1));

            cache.NumNewsContext = 2; // row 0 existing memory, row 1 fresh story
            var dShared = new float[2, config.Text.EmbeddingDim];
            for (int d = 0; d < config.Text.EmbeddingDim; d++)
                dShared[0, d] = 1f; // gradient belongs only to existing memory, which should be detached

            InvokePrivate(trainer, "ZeroAllGradients");
            InvokePrivate(trainer, "BackpropMultiStoryTextEncoder", dShared, cache);
            var grads = GetGradients(trainer);
            float tokenGrad = story[0].TokenIds.Sum(id => SumAbsRow(grads.TextEmbeddingGrad, id));

            Assert(tokenGrad == 0f, "Existing memory gradient must not be routed into fresh story text encoder tokens.");
        }

        private void Test_GradientClipping_ReducesNorm_AndIncludesContextTypeEmbedding()
        {
            var config = CreateConfig(embDim: 12, numHeads: 3);
            var model = new TACAMT_Model(config, new Random(42));
            var trainer = new TACAMT_Trainer(model, new TrainingConfig { LearningRate = 0.001f, BatchSize = 1, Epochs = 1, Verbose = false });
            var grads = GetGradients(trainer);

            InvokePrivate(trainer, "ZeroAllGradients");
            for (int d = 0; d < config.Price.EmbeddingDim; d++)
            {
                grads.ContextTypeEmbeddingGrad[0, d] = 100f;
                grads.ContextTypeEmbeddingGrad[1, d] = -50f;
            }

            float before = (float)InvokePrivate(trainer, "ComputeGradientNorm");
            InvokePrivate(trainer, "ClipGradients", 1.0f);
            float after = (float)InvokePrivate(trainer, "ComputeGradientNorm");

            Assert(before > 1.0f, "Precondition: gradient norm should exceed clip threshold.");
            Assert(after <= 1.0001f, $"Gradient norm should be clipped to <=1, got {after}.");
            Assert(MathF.Abs(grads.ContextTypeEmbeddingGrad[0, 0]) < 100f, "Context type embedding gradients should have been scaled by clipping.");
            Assert(SumAbs(grads.ContextTypeEmbeddingGrad) > 0f, "Clipping should scale, not zero, context type gradients.");
        }

        #endregion


        #region Additional high-confidence hardening tests

        private void Test_Forward_NullPriceSequence_Throws()
        {
            var model = new TACAMT_Model(CreateConfig(), new Random(42));
            AssertThrows<ArgumentNullException>(() => model.Forward((NewsStory[])null, null),
                "Forward should reject a null price sequence with an argument exception.");
        }

        private void Test_Forward_ZeroLengthPriceSequence_Throws()
        {
            var model = new TACAMT_Model(CreateConfig(), new Random(42));
            AssertThrows<ArgumentException>(() => model.Forward((NewsStory[])null, new float[0, 5]),
                "Forward should reject an empty price sequence instead of producing invalid last-row indexing.");
        }

        private void Test_Forward_WrongFeatureDim_Throws()
        {
            var model = new TACAMT_Model(CreateConfig(inputFeatures: 5), new Random(42));
            AssertThrows<Exception>(() => model.Forward((NewsStory[])null, RandomMatrix(4, 4, new Random(101), 0.2f)),
                "Forward should not silently accept a price matrix with the wrong feature dimension.");
        }

        private void Test_ForwardWithCache_InvalidSliceBounds_Throw()
        {
            var config = CreateConfig(priceSeqLen: 8);
            var model = new TACAMT_Model(config, new Random(42));
            var price = RandomMatrix(6, 5, new Random(102), 0.2f);
            var cache = new MultimodalForwardCache(config.Text.NumLayers, config.Price.NumLayers);

            AssertThrows<ArgumentOutOfRangeException>(() => model.ForwardWithCache((NewsStory[])null, price, -1, 2, cache),
                "Negative rowStart should throw.");
            AssertThrows<ArgumentOutOfRangeException>(() => model.ForwardWithCache((NewsStory[])null, price, 1, 0, cache),
                "rowCount=0 should throw because the decoder cannot produce a last timestep.");
            AssertThrows<ArgumentException>(() => model.ForwardWithCache((NewsStory[])null, price, 4, 3, cache),
                "rowStart + rowCount beyond the price matrix should throw.");
        }

        private void Test_ForwardWithPriceContext_InvalidContextShape_Throws()
        {
            var config = CreateConfig(inputFeatures: 3, outputDim: 1, embDim: 12, numHeads: 3, priceSeqLen: 10);
            var model = new TACAMT_Model(config, new Random(42));
            var current = RandomMatrix(4, 3, new Random(103), 0.2f);
            var badContext = new float[2, config.Price.EmbeddingDim + 1];
            var times = new[] { -2f, -1f };
            var cache = new MultimodalForwardCache(config.Text.NumLayers, config.Price.NumLayers);

            AssertThrows<ArgumentException>(() => model.ForwardWithPriceContextAndCache(null, current, badContext, times, cache),
                "Price context hidden states with the wrong embedding dim should throw.");
        }

        private void Test_ForwardWithPriceContext_InvalidTimeLength_Throws()
        {
            var config = CreateConfig(inputFeatures: 3, outputDim: 1, embDim: 12, numHeads: 3, priceSeqLen: 10);
            var model = new TACAMT_Model(config, new Random(42));
            var current = RandomMatrix(4, 3, new Random(104), 0.2f);
            var context = RandomMatrix(3, config.Price.EmbeddingDim, new Random(105), 0.2f);
            var badTimes = new[] { -1f, -2f };
            var cache = new MultimodalForwardCache(config.Text.NumLayers, config.Price.NumLayers);

            AssertThrows<ArgumentException>(() => model.ForwardWithPriceContextAndCache(null, current, context, badTimes, cache),
                "priceCtxTimes length must match priceCtxHidden rows.");
        }

        private void Test_ForwardWithPriceContext_ReusedCache_ClearsStaleContext()
        {
            var tokenizer = CreateTinyTokenizer(out var corpus);
            var config = CreateConfig(textVocabSize: tokenizer.VocabSize + 2, inputFeatures: 3, outputDim: 1, embDim: 12, numHeads: 3, priceSeqLen: 12);
            var model = new TACAMT_Model(config, new Random(42));
            var cache = new MultimodalForwardCache(config.Text.NumLayers, config.Price.NumLayers);
            var price = RandomMatrix(10, 3, new Random(106), 0.2f);
            var history = SliceRows(price, 0, 4);
            var current = SliceRows(price, 4, 9);
            var priceCtx = model.EncodePriceHistory(history);
            var priceTimes = Enumerable.Range(0, 4).Select(t => (float)(t - 4)).ToArray();
            var stories = new[] { new NewsStory(tokenizer.Encode(corpus[0], addSpecialTokens: true), -1f) };

            model.ForwardWithPriceContextAndCache(stories, current, priceCtx, priceTimes, cache, isTraining: false);
            Assert(cache.TextFinalHidden != null && cache.NumNewsContext == 1 && cache.NumPriceContext == 4,
                "Precondition: first call should populate cache context.");

            model.ForwardWithPriceContextAndCache(null, current, null, null, cache, isTraining: false);

            Assert(cache.TextFinalHidden == null, "Reused price-context cache should clear stale combined context hidden states.");
            Assert(cache.StoryArrivalTimes == null, "Reused price-context cache should clear stale story/context times.");
            Assert(cache.NumNewsContext == 0, "Reused price-context cache should reset NumNewsContext.");
            Assert(cache.NumPriceContext == 0, "Reused price-context cache should reset NumPriceContext.");
            Assert(cache.PriceContextHidden == null, "Reused price-context cache should clear stale price context reference.");
            Assert(cache.PriceBlockCaches.All(bc => bc.CrossAttentionWeights == null && bc.CrossQ == null && bc.DecayCache == null),
                "Price block caches should clear stale cross-attention state when no context is provided.");
        }

        private void Test_Tokenizer_LengthMismatch_Throws()
        {
            var tokenizer = CreateTinyTokenizer(out var _);
            var model = new TACAMT_Model(CreateConfig(textVocabSize: tokenizer.VocabSize + 2), new Random(42));
            model.SetTokenizer(tokenizer);

            AssertThrows<ArgumentException>(() => model.TokenizeStories(new[] { "one", "two" }, new[] { 0f }),
                "TokenizeStories should reject text/time length mismatches.");
        }

        private void Test_TrainTokenizer_EmptyCorpusThrows_NonEmptyAttachesTokenizer()
        {
            var model = new TACAMT_Model(CreateConfig(textVocabSize: 80), new Random(42));
            AssertThrows<ArgumentException>(() => model.TrainTokenizer(Array.Empty<string>(), minFrequency: 1),
                "TrainTokenizer should reject an empty corpus.");

            model.TrainTokenizer(new[] { "alpha beta gamma", "beta gamma delta" }, minFrequency: 1);
            Assert(model.Tokenizer != null, "TrainTokenizer should attach a tokenizer to the model.");
            var stories = model.TokenizeStories(new[] { "alpha gamma" }, new[] { -1f });
            Assert(stories.Length == 1 && stories[0].TokenIds.Length > 0, "Tokenizer trained by the model should be usable immediately.");
        }

        private void Test_Memory_ContextTypeEmbeddingDoesNotMutateStoredMemory()
        {
            var tokenizer = CreateTinyTokenizer(out var corpus);
            var config = CreateConfig(textVocabSize: tokenizer.VocabSize + 2, embDim: 12, numHeads: 3, inputFeatures: 2, outputDim: 1, priceSeqLen: 8);
            var model = new TACAMT_Model(config, new Random(42));
            var story = new[] { new NewsStory(tokenizer.Encode(corpus[0], addSpecialTokens: true), -1f) };
            model.UpdateNewsMemory(story, 100.0, max: 10);

            var priceHidden = new float[config.Price.EmbeddingDim];
            for (int d = 0; d < priceHidden.Length; d++)
                priceHidden[d] = d * 0.01f;
            model.PriceMemory.Add(new PriceMemoryEntry { HiddenState = CloneVector(priceHidden), AbsoluteTimestamp = 90.0 });

            var newsBefore = CloneVector(model.NewsMemory[0].HiddenState);
            var priceBefore = CloneVector(model.PriceMemory[0].HiddenState);

            model.PredictWithMemory(null, FilledMatrix(5, 2, 0.2f), 110.0, maxNewsMemorySize: 10, maxPriceMemorySize: 10);

            AssertVectorClose(newsBefore, model.NewsMemory[0].HiddenState, 0f,
                "Applying context type embeddings during inference must not mutate stored news memory hidden states.");
            AssertVectorClose(priceBefore, model.PriceMemory[0].HiddenState, 0f,
                "Applying context type embeddings during inference must not mutate stored price memory hidden states.");
        }

        private void Test_UpdateMemoryAttentionScores_MapsNewsAndPriceRowsCorrectly()
        {
            var config = CreateConfig(embDim: 12, numHeads: 3);
            var model = new TACAMT_Model(config, new Random(42));
            model.PruningConfig.AttentionScoreAlpha = 1f;

            model.NewsMemory.Add(new NewsMemoryEntry { HiddenState = new float[config.Price.EmbeddingDim], AbsoluteTimestamp = 1.0, AttentionScore = 0f, QueryCount = 0 });
            model.PriceMemory.Add(new PriceMemoryEntry { HiddenState = new float[config.Price.EmbeddingDim], AbsoluteTimestamp = 2.0, AttentionScore = 0f, QueryCount = 0 });

            var priceHidden = new float[2, config.Price.EmbeddingDim];
            var contextHidden = new float[2, config.Price.EmbeddingDim];
            for (int d = 0; d < config.Price.EmbeddingDim; d++)
            {
                priceHidden[1, d] = 1f;
                contextHidden[0, d] = 1f;
                contextHidden[1, d] = -1f;
            }

            InvokePrivate(model, "UpdateMemoryAttentionScores", priceHidden, contextHidden, 2);

            Assert(model.NewsMemory[0].QueryCount == 1, "News memory query count should increment.");
            Assert(model.PriceMemory[0].QueryCount == 1, "Price memory query count should increment.");
            Assert(model.NewsMemory[0].AttentionScore > model.PriceMemory[0].AttentionScore,
                $"News row should map to context row 0 and price row to context row 1. news={model.NewsMemory[0].AttentionScore:F6}, price={model.PriceMemory[0].AttentionScore:F6}");
        }

        private void Test_UpdateNewsMemory_MaxZero_ClearsFreshEntry()
        {
            var config = CreateConfig(textVocabSize: 50, embDim: 12, numHeads: 3);
            var model = new TACAMT_Model(config, new Random(42));
            model.UpdateNewsMemory(new[] { new NewsStory(new[] { 4, 5, 6 }, -1f) }, 100.0, max: 0);
            Assert(model.NewsMemory.Count == 0, $"UpdateNewsMemory(max:0) should not retain entries, got {model.NewsMemory.Count}.");
        }

        private void Test_PredictWithMemory_TimeUnitsPerPositionScalesAbsoluteStoryTimes()
        {
            var config = CreateConfig(textVocabSize: 50, embDim: 12, numHeads: 3, inputFeatures: 2, outputDim: 1, priceSeqLen: 8);
            var model = new TACAMT_Model(config, new Random(42));
            var story = new[] { new NewsStory(new[] { 4, 5, 6 }, -2f) };
            var price = FilledMatrix(4, 2, 0.1f);

            model.PredictWithMemory(story, price, currentAbsoluteTimestamp: 100.0, timeUnitsPerPosition: 5.0, maxNewsMemorySize: 10, maxPriceMemorySize: 10);

            Assert(model.NewsMemory.Count == 1, "Expected one stored news memory entry.");
            AssertClose(90.0, model.NewsMemory[0].AbsoluteTimestamp, 1e-6,
                "Story absolute timestamp should be current timestamp plus ArrivalTime * timeUnitsPerPosition.");
            Assert(model.PriceMemory.Count == 4, "Expected one price memory entry per price row.");
            AssertClose(100.0, model.PriceMemory[0].AbsoluteTimestamp, 1e-6, "First price memory timestamp mismatch.");
            AssertClose(115.0, model.PriceMemory[3].AbsoluteTimestamp, 1e-6, "Last price memory timestamp mismatch.");
            AssertClose(115.0, model.LastPriceTimestamp, 1e-6,
                "LastPriceTimestamp should point at the final stored price row, not the first row.");
        }

        private void Test_ClearAllMemory_ResetsBothMemoriesAndTimestamp()
        {
            var config = CreateConfig();
            var model = new TACAMT_Model(config, new Random(42));
            model.NewsMemory.Add(new NewsMemoryEntry { HiddenState = new float[config.Price.EmbeddingDim], AbsoluteTimestamp = 1.0 });
            model.PriceMemory.Add(new PriceMemoryEntry { HiddenState = new float[config.Price.EmbeddingDim], AbsoluteTimestamp = 2.0 });
            model.LastPriceTimestamp = 123.0;

            model.ClearAllMemory();

            Assert(model.NewsMemory.Count == 0, "News memory should be cleared.");
            Assert(model.PriceMemory.Count == 0, "Price memory should be cleared.");
            Assert(model.LastPriceTimestamp == 0.0, "LastPriceTimestamp should reset to zero.");
        }


        private void Test_Train_DuplicateSamplesBatchAveraging_EqualsSingleSampleUpdate()
        {
            var config = CreateConfig(inputFeatures: 2, outputDim: 1, embDim: 12, numHeads: 3, priceSeqLen: 8, priceContextEnabled: false);
            var input = RandomMatrix(6, 2, new Random(107), 0.2f);
            var target = FilledMatrix(6, 1, 0.35f);

            var modelSingle = new TACAMT_Model(config, new Random(42));
            var trainerSingle = new TACAMT_Trainer(modelSingle, new TrainingConfig { LearningRate = 0.002f, BatchSize = 1, Epochs = 1, UseGradientClipping = false, Verbose = false });
            trainerSingle.Train(new NewsStory[1][], new[] { input }, new[] { target });

            var modelDuplicateBatch = new TACAMT_Model(config, new Random(42));
            var trainerDuplicate = new TACAMT_Trainer(modelDuplicateBatch, new TrainingConfig { LearningRate = 0.002f, BatchSize = 2, Epochs = 1, UseGradientClipping = false, Verbose = false });
            trainerDuplicate.Train(new NewsStory[2][], new[] { input, CloneMatrix(input) }, new[] { target, CloneMatrix(target) });

            AssertMatrixClose(modelSingle.OutputProjection, modelDuplicateBatch.OutputProjection, 2e-5f,
                "A batch of two identical samples should average gradients and match one copy of that sample.");
            AssertVectorClose(modelSingle.OutputBias, modelDuplicateBatch.OutputBias, 2e-5f,
                "OutputBias update should also be batch-averaged.");
            AssertMatrixClose(modelSingle.PriceInputProjection, modelDuplicateBatch.PriceInputProjection, 2e-5f,
                "PriceInputProjection update should be batch-averaged.");
        }

        private void Test_Train_InvalidSample_DoesNotContaminateValidSampleUpdate()
        {
            var config = CreateConfig(textVocabSize: 20, inputFeatures: 2, outputDim: 1, embDim: 12, numHeads: 3, priceSeqLen: 8, priceContextEnabled: false);
            var validInput = RandomMatrix(6, 2, new Random(108), 0.2f);
            var validTarget = FilledMatrix(6, 1, 0.4f);
            var invalidInput = RandomMatrix(6, 2, new Random(109), 0.2f);
            var invalidTarget = FilledMatrix(6, 1, 0.1f);
            var invalidStory = new[] { new NewsStory(new[] { 999 }, -1f) };

            var cleanModel = new TACAMT_Model(config, new Random(42));
            var cleanTrainer = new TACAMT_Trainer(cleanModel, new TrainingConfig { LearningRate = 0.002f, BatchSize = 1, Epochs = 1, UseGradientClipping = false, Verbose = false });
            cleanTrainer.Train(new NewsStory[1][], new[] { validInput }, new[] { validTarget });

            var mixedModel = new TACAMT_Model(config, new Random(42));
            var mixedTrainer = new TACAMT_Trainer(mixedModel, new TrainingConfig { LearningRate = 0.002f, BatchSize = 1, Epochs = 1, UseGradientClipping = false, Verbose = false });
            mixedTrainer.Train(
                new[] { invalidStory, null },
                new[] { invalidInput, CloneMatrix(validInput) },
                new[] { invalidTarget, CloneMatrix(validTarget) });

            AssertMatrixClose(cleanModel.OutputProjection, mixedModel.OutputProjection, 2e-5f,
                "A skipped invalid sample should not leave partial gradients that alter the valid sample update.");
            AssertMatrixClose(cleanModel.PriceInputProjection, mixedModel.PriceInputProjection, 2e-5f,
                "Invalid-sample cleanup should protect lower-layer parameter updates too.");
        }

        private void Test_Validate_UsesNextStepTargets_NotCurrentRow()
        {
            var config = CreateConfig(inputFeatures: 2, outputDim: 1, embDim: 12, numHeads: 3, priceSeqLen: 8, priceContextEnabled: false);
            var model = new TACAMT_Model(config, new Random(42));
            var trainer = new TACAMT_Trainer(model, new TrainingConfig { LearningRate = 0.001f, BatchSize = 1, Epochs = 1, Verbose = false });
            var input = RandomMatrix(6, 2, new Random(110), 0.2f);
            var target = FilledMatrix(6, 1, 0f);
            target[0, 0] = 1000f; // Sentinel: validation must ignore row 0 and compare prediction row 0 to target row 1.

            float validate = trainer.Validate(new NewsStory[1][], new[] { input }, new[] { target });
            float manualNextStep = ManualNoContextNextStepLoss(model, null, input, target);

            AssertClose(manualNextStep, validate, 1e-5f, "Validate should use target rows [1..] for next-step training.");
            Assert(validate < 10000f, $"If target row 0 leaked into validation, loss would be huge; got {validate:F6}.");
        }

        private void Test_Train_IntArrayOverload_EqualsNewsStoryOverload()
        {
            var config = CreateConfig(textVocabSize: 50, inputFeatures: 2, outputDim: 1, embDim: 12, numHeads: 3, priceSeqLen: 8, priceContextEnabled: false);
            var tokens = new[] { 4, 5, 6 };
            var inputs = new[]
            {
                RandomMatrix(6, 2, new Random(111), 0.2f),
                RandomMatrix(6, 2, new Random(112), 0.2f)
            };
            var targets = new[] { FilledMatrix(6, 1, 0.3f), FilledMatrix(6, 1, 0.6f) };

            var modelInt = new TACAMT_Model(config, new Random(42));
            var trainerInt = new TACAMT_Trainer(modelInt, new TrainingConfig { LearningRate = 0.002f, BatchSize = 2, Epochs = 1, UseGradientClipping = false, Verbose = false });
            trainerInt.Train(new[] { tokens, tokens }, inputs, targets);

            var modelStory = new TACAMT_Model(config, new Random(42));
            var trainerStory = new TACAMT_Trainer(modelStory, new TrainingConfig { LearningRate = 0.002f, BatchSize = 2, Epochs = 1, UseGradientClipping = false, Verbose = false });
            trainerStory.Train(new[] { new[] { new NewsStory(tokens, 0f) }, new[] { new NewsStory(tokens, 0f) } }, inputs, targets);

            AssertMatrixClose(modelStory.OutputProjection, modelInt.OutputProjection, 2e-5f,
                "Train(int[][]) should be equivalent to wrapping each token array in a single NewsStory at arrival time 0.");
            AssertMatrixClose(modelStory.TextTokenEmbedding, modelInt.TextTokenEmbedding, 2e-5f,
                "Text embedding updates should match between Train overloads.");
        }

        private void Test_Validate_IntArrayOverload_EqualsNewsStoryOverload()
        {
            var config = CreateConfig(textVocabSize: 50, inputFeatures: 2, outputDim: 1, embDim: 12, numHeads: 3, priceSeqLen: 8, priceContextEnabled: false);
            var model = new TACAMT_Model(config, new Random(42));
            var trainer = new TACAMT_Trainer(model, new TrainingConfig { LearningRate = 0.001f, BatchSize = 2, Epochs = 1, Verbose = false });
            var tokens = new[] { 4, 5, 6 };
            var inputs = new[] { RandomMatrix(6, 2, new Random(113), 0.2f), RandomMatrix(6, 2, new Random(114), 0.2f) };
            var targets = new[] { FilledMatrix(6, 1, 0.2f), FilledMatrix(6, 1, 0.7f) };

            float intLoss = trainer.Validate(new[] { tokens, tokens }, inputs, targets);
            float storyLoss = trainer.Validate(new[] { new[] { new NewsStory(tokens, 0f) }, new[] { new NewsStory(tokens, 0f) } }, inputs, targets);

            AssertClose(storyLoss, intLoss, 1e-6f, "Validate(int[][]) should match the NewsStory overload.");
        }

        private void Test_Gradient_OutputProjection_MatchesFiniteDifference()
        {
            var config = CreateConfig(inputFeatures: 2, outputDim: 2, embDim: 8, numHeads: 2, priceSeqLen: 8, priceContextEnabled: false);
            var model = new TACAMT_Model(config, new Random(42));
            var trainer = new TACAMT_Trainer(model, new TrainingConfig { LearningRate = 0.001f, BatchSize = 1, Epochs = 1, Verbose = false });
            var price = RandomMatrix(5, 2, new Random(115), 0.2f);
            var target = RandomMatrix(5, 2, new Random(116), 0.2f);
            var cache = new MultimodalForwardCache(config.Text.NumLayers, config.Price.NumLayers);
            var (pred, conf) = model.ForwardWithCache((NewsStory[])null, price, cache, isTraining: false);

            RunBackward(trainer, pred, conf, target, null, cache);
            var grads = GetGradients(trainer);
            float analytic = grads.OutputProjectionGrad[0, 0];
            float fd = FiniteDifferenceLoss(model, null, price, target, null, null, m => m.OutputProjection[0, 0], (m, v) => m.OutputProjection[0, 0] = v);

            AssertRelativeClose(analytic, fd, 0.05f, "OutputProjection[0,0] gradient should match finite difference.");
        }

        private void Test_Gradient_PriceInputProjection_MatchesFiniteDifference()
        {
            var config = CreateConfig(inputFeatures: 2, outputDim: 2, embDim: 8, numHeads: 2, priceSeqLen: 8, priceContextEnabled: false);
            var model = new TACAMT_Model(config, new Random(42));
            var trainer = new TACAMT_Trainer(model, new TrainingConfig { LearningRate = 0.001f, BatchSize = 1, Epochs = 1, Verbose = false });
            var price = RandomMatrix(5, 2, new Random(117), 0.2f);
            var target = RandomMatrix(5, 2, new Random(118), 0.2f);
            var cache = new MultimodalForwardCache(config.Text.NumLayers, config.Price.NumLayers);
            var (pred, conf) = model.ForwardWithCache((NewsStory[])null, price, cache, isTraining: false);

            RunBackward(trainer, pred, conf, target, null, cache);
            var grads = GetGradients(trainer);
            var (row, col, analytic) = MaxAbsIndex(grads.PriceInputProjectionGrad);
            Assert(MathF.Abs(analytic) > 1e-8f, "Expected a non-zero PriceInputProjection gradient to check.");

            float fd = FiniteDifferenceLoss(model, null, price, target, null, null, m => m.PriceInputProjection[row, col], (m, v) => m.PriceInputProjection[row, col] = v);
            AssertRelativeClose(analytic, fd, 0.20f, $"PriceInputProjection[{row},{col}] gradient should roughly match finite difference.");
        }

        private void Test_Gradient_ConfidenceProjection_MatchesFiniteDifference()
        {
            var config = CreateConfig(inputFeatures: 2, outputDim: 1, embDim: 8, numHeads: 2, priceSeqLen: 8, useConfidence: true, priceContextEnabled: false);
            var model = new TACAMT_Model(config, new Random(42));
            var trainer = new TACAMT_Trainer(model, new TrainingConfig { LearningRate = 0.001f, BatchSize = 1, Epochs = 1, ConfidenceLossWeight = 1f, Verbose = false });
            var price = RandomMatrix(5, 2, new Random(119), 0.2f);
            var target = FilledMatrix(5, 1, 0.4f);
            var confidenceTarget = new[] { 1f, 0f, 1f, 0f, 1f };
            var cache = new MultimodalForwardCache(config.Text.NumLayers, config.Price.NumLayers);
            var (pred, conf) = model.ForwardWithCache((NewsStory[])null, price, cache, isTraining: false);

            RunBackward(trainer, pred, conf, target, confidenceTarget, cache);
            var grads = GetGradients(trainer);
            var (row, col, analytic) = MaxAbsIndex(grads.ConfidenceProjectionGrad);
            Assert(MathF.Abs(analytic) > 1e-8f, "Expected a non-zero ConfidenceProjection gradient to check.");

            float fd = FiniteDifferenceLoss(model, null, price, target, confidenceTarget, 1f, m => m.ConfidenceProjection[row, col], (m, v) => m.ConfidenceProjection[row, col] = v);
            AssertRelativeClose(analytic, fd, 0.08f, $"ConfidenceProjection[{row},{col}] gradient should match finite difference.");
        }


        private void Test_Dropout_TrainingSameSeedRepeatableDifferentSeedDiffers()
        {
            var tokenizer = CreateTinyTokenizer(out var corpus);
            var config = CreateConfig(textVocabSize: tokenizer.VocabSize + 2, inputFeatures: 2, outputDim: 1, embDim: 16, numHeads: 4, priceSeqLen: 8, validate: false);
            config.DecayNetwork.MemAttentionDropout = 0.5f;
            config.DecayNetwork.MlpDropout = 0.5f;
            config.Validate();

            var model = new TACAMT_Model(config, new Random(42));
            var stories = new[]
            {
                new NewsStory(tokenizer.Encode(corpus[0], addSpecialTokens: true), -1f),
                new NewsStory(tokenizer.Encode(corpus[1], addSpecialTokens: true), -2f)
            };
            var price = RandomMatrix(6, 2, new Random(120), 0.2f);

            var cacheA = new MultimodalForwardCache(config.Text.NumLayers, config.Price.NumLayers);
            var (trainA, _) = model.ForwardWithCache(stories, price, cacheA, isTraining: true, dropoutRng: new Random(999));
            var cacheB = new MultimodalForwardCache(config.Text.NumLayers, config.Price.NumLayers);
            var (trainB, _) = model.ForwardWithCache(stories, price, cacheB, isTraining: true, dropoutRng: new Random(999));
            var cacheC = new MultimodalForwardCache(config.Text.NumLayers, config.Price.NumLayers);
            var (trainC, _) = model.ForwardWithCache(stories, price, cacheC, isTraining: true, dropoutRng: new Random(1000));
            var cacheInf1 = new MultimodalForwardCache(config.Text.NumLayers, config.Price.NumLayers);
            var (inf1, _) = model.ForwardWithCache(stories, price, cacheInf1, isTraining: false);
            var cacheInf2 = new MultimodalForwardCache(config.Text.NumLayers, config.Price.NumLayers);
            var (inf2, _) = model.ForwardWithCache(stories, price, cacheInf2, isTraining: false);

            AssertMatrixClose(trainA, trainB, 1e-6f, "Training dropout with the same RNG seed should be reproducible.");
            Assert(MatrixMaxAbsDiff(trainA, trainC) > 1e-7f,
                "Training dropout with different RNG seeds should alter the output when dropout is enabled.");
            AssertMatrixClose(inf1, inf2, 1e-6f, "Inference should be deterministic even when dropout rates are configured.");
        }

        private void Test_Backward_LongStory_AllChunksReceiveTokenGradient()
        {
            var config = CreateConfig(textVocabSize: 50, textMaxSeqLen: 3, embDim: 12, numHeads: 3, inputFeatures: 2, outputDim: 1, priceSeqLen: 8, priceContextEnabled: false);
            var model = new TACAMT_Model(config, new Random(42));
            var trainer = new TACAMT_Trainer(model, new TrainingConfig { LearningRate = 0.001f, BatchSize = 1, Epochs = 1, Verbose = false });
            var tokens = new[] { 4, 5, 6, 7, 8, 9, 10, 11 };
            var stories = new[] { new NewsStory(tokens, -1f) };
            var price = RandomMatrix(5, 2, new Random(121), 0.2f);
            var target = FilledMatrix(5, 1, 0.25f);
            var cache = new MultimodalForwardCache(config.Text.NumLayers, config.Price.NumLayers);
            var (pred, conf) = model.ForwardWithCache(stories, price, cache, isTraining: true, dropoutRng: new Random(1));

            RunBackward(trainer, pred, conf, target, null, cache);
            var grads = GetGradients(trainer);

            foreach (int tokenId in tokens)
            {
                float grad = SumAbsRow(grads.TextEmbeddingGrad, tokenId);
                Assert(grad > 1e-10f, $"Token {tokenId} should receive gradient even when the story spans multiple text chunks.");
            }
        }

        private void Test_Backward_NewsOnlyContext_OnlyNewsTypeGradient()
        {
            var config = CreateConfig(textVocabSize: 50, embDim: 12, numHeads: 3, inputFeatures: 2, outputDim: 1, priceSeqLen: 8, priceContextEnabled: false);
            var model = new TACAMT_Model(config, new Random(42));
            var trainer = new TACAMT_Trainer(model, new TrainingConfig { LearningRate = 0.001f, BatchSize = 1, Epochs = 1, Verbose = false });
            var stories = new[] { new NewsStory(new[] { 4, 5, 6 }, -1f) };
            var price = RandomMatrix(5, 2, new Random(122), 0.2f);
            var target = FilledMatrix(5, 1, 0.2f);
            var cache = new MultimodalForwardCache(config.Text.NumLayers, config.Price.NumLayers);
            var (pred, conf) = model.ForwardWithCache(stories, price, cache, isTraining: true, dropoutRng: new Random(1));

            RunBackward(trainer, pred, conf, target, null, cache);
            var grads = GetGradients(trainer);

            Assert(SumAbsRow(grads.ContextTypeEmbeddingGrad, 0) > 1e-10f, "News-only context should update news type embedding row.");
            Assert(SumAbsRow(grads.ContextTypeEmbeddingGrad, 1) == 0f, "News-only context should not update price type embedding row.");
        }

        private void Test_Backward_PriceOnlyContext_OnlyPriceTypeGradient()
        {
            var config = CreateConfig(embDim: 12, numHeads: 3, inputFeatures: 2, outputDim: 1, priceSeqLen: 10, priceContextEnabled: true);
            var model = new TACAMT_Model(config, new Random(42));
            var trainer = new TACAMT_Trainer(model, new TrainingConfig { LearningRate = 0.001f, BatchSize = 1, Epochs = 1, Verbose = false });
            var price = RandomMatrix(8, 2, new Random(123), 0.2f);
            var history = SliceRows(price, 0, 3);
            var current = SliceRows(price, 3, 7);
            var priceCtx = model.EncodePriceHistory(history);
            var priceTimes = Enumerable.Range(0, 3).Select(t => (float)(t - 3)).ToArray();
            var target = FilledMatrix(current.GetLength(0), 1, 0.2f);
            var cache = new MultimodalForwardCache(config.Text.NumLayers, config.Price.NumLayers);
            var (pred, conf) = model.ForwardWithPriceContextAndCache(null, current, priceCtx, priceTimes, cache, isTraining: true, dropoutRng: new Random(1));

            RunBackward(trainer, pred, conf, target, null, cache);
            var grads = GetGradients(trainer);

            Assert(SumAbsRow(grads.ContextTypeEmbeddingGrad, 0) == 0f, "Price-only context should not update news type embedding row.");
            Assert(SumAbsRow(grads.ContextTypeEmbeddingGrad, 1) > 1e-10f, "Price-only context should update price type embedding row.");
        }

        #endregion

        #region End-to-end synthetic learning tests

        private void Test_E2E_SingleSample_OverfitsEasyConstantTarget()
        {
            var config = CreateConfig(inputFeatures: 2, outputDim: 1, embDim: 16, numHeads: 4, numLayers: 1, ffnDim: 32, priceSeqLen: 8, priceContextEnabled: false);
            var model = new TACAMT_Model(config, new Random(42));
            var (stories, inputs, targets) = CreateConstantTargetPriceData(samples: 1, seqLen: 6, inputDim: 2, outputDim: 1, targetValue: 0.8f, seed: 32);
            var trainer = new TACAMT_Trainer(model, new TrainingConfig { LearningRate = 0.01f, BatchSize = 1, Epochs = 1, UseGradientClipping = true, GradientClipThreshold = 1f, Verbose = false });
            float before = trainer.Validate(stories, inputs, targets);

            trainer = new TACAMT_Trainer(model, new TrainingConfig { LearningRate = 0.01f, BatchSize = 1, Epochs = 120, UseGradientClipping = true, GradientClipThreshold = 1f, Verbose = false });
            trainer.Train(stories, inputs, targets);
            float after = trainer.Validate(stories, inputs, targets);

            Assert(after < before * 0.25f, $"Single-sample easy target should overfit. before={before:F6}, after={after:F6}");
        }

        private void Test_E2E_LearnsBullVsBearNewsSignal()
        {
            var tokenizer = new BPETokenizer();
            tokenizer.Train(new[] { "bull bull bull rally upside", "bear bear bear crash downside" }, vocabSize: 80, minFrequency: 1);
            var config = CreateConfig(textVocabSize: tokenizer.VocabSize + 2, inputFeatures: 2, outputDim: 1, embDim: 24, numHeads: 4, numLayers: 1, ffnDim: 48, priceSeqLen: 8, priceContextEnabled: false);
            var model = new TACAMT_Model(config, new Random(42));
            var (stories, inputs, targets) = CreateBullBearData(tokenizer, "bull bull bull rally upside", "bear bear bear crash downside", samples: 48, seqLen: 6, inputDim: 2);

            var trainer = new TACAMT_Trainer(model, new TrainingConfig { LearningRate = 0.01f, BatchSize = 8, Epochs = 120, UseGradientClipping = true, GradientClipThreshold = 1f, Verbose = false });
            trainer.Train(stories, inputs, targets);

            var testPrice = FilledMatrix(6, 2, 0.1f);
            float bull = AveragePrediction(model.Forward(new[] { new NewsStory(tokenizer.Encode("bull bull bull rally upside", addSpecialTokens: true), -1f) }, testPrice).predictions);
            float bear = AveragePrediction(model.Forward(new[] { new NewsStory(tokenizer.Encode("bear bear bear crash downside", addSpecialTokens: true), -1f) }, testPrice).predictions);

            Assert(bull > bear + 0.1f, $"Bull prediction should exceed bear prediction after training. bull={bull:F4}, bear={bear:F4}");
        }

        private void Test_E2E_LearnsRecencyUnderCompetition()
        {
            var tokenizer = new BPETokenizer();
            tokenizer.Train(new[] { "positive catalyst positive catalyst", "neutral background neutral background" }, vocabSize: 80, minFrequency: 1);
            var positive = tokenizer.Encode("positive catalyst positive catalyst", addSpecialTokens: true);
            var neutral = tokenizer.Encode("neutral background neutral background", addSpecialTokens: true);

            var config = CreateConfig(textVocabSize: tokenizer.VocabSize + 2, inputFeatures: 2, outputDim: 1, embDim: 24, numHeads: 4, numLayers: 1, ffnDim: 48, priceSeqLen: 8, priceContextEnabled: false);
            var model = new TACAMT_Model(config, new Random(42));

            int samples = 64;
            var stories = new NewsStory[samples][];
            var inputs = new float[samples][,];
            var targets = new float[samples][,];

            for (int s = 0; s < samples; s++)
            {
                bool positiveRecent = s % 2 == 0;
                stories[s] = positiveRecent
                    ? new[] { new NewsStory(positive, -1f), new NewsStory(neutral, -20f) }
                    : new[] { new NewsStory(neutral, -1f), new NewsStory(positive, -20f) };

                inputs[s] = FilledMatrix(6, 2, 0.1f);
                targets[s] = FilledMatrix(6, 1, positiveRecent ? 0.9f : 0.1f);
            }

            var trainer = new TACAMT_Trainer(model, new TrainingConfig { LearningRate = 0.01f, BatchSize = 8, Epochs = 150, UseGradientClipping = true, GradientClipThreshold = 1f, Verbose = false });
            trainer.Train(stories, inputs, targets);

            var testPrice = FilledMatrix(6, 2, 0.1f);
            float recentPositive = AveragePrediction(model.Forward(new[] { new NewsStory(positive, -1f), new NewsStory(neutral, -20f) }, testPrice).predictions);
            float oldPositive = AveragePrediction(model.Forward(new[] { new NewsStory(neutral, -1f), new NewsStory(positive, -20f) }, testPrice).predictions);

            Assert(recentPositive > oldPositive + 0.05f,
                $"Recent positive story should produce higher prediction than old positive story under competition. recent={recentPositive:F4}, old={oldPositive:F4}");
        }

        #endregion

        #region Helpers

        private new void Assert(bool condition, string message)
        {
            if (!condition)
                throw new Exception(message);
        }

        private void AssertThrows<TException>(Action action, string message) where TException : Exception
        {
            try
            {
                action();
            }
            catch (TargetInvocationException tie) when (tie.InnerException is TException)
            {
                return;
            }
            catch (TException)
            {
                return;
            }
            catch (Exception ex)
            {
                throw new Exception($"{message} Expected {typeof(TException).Name}, got {ex.GetType().Name}: {ex.Message}");
            }

            throw new Exception(message);
        }

        private MultimodalTransformerConfig CreateConfig(
            int textVocabSize = 50,
            int textMaxSeqLen = 32,
            int embDim = 16,
            int numHeads = 2,
            int numLayers = 1,
            int ffnDim = 32,
            int inputFeatures = 5,
            int outputDim = 5,
            int priceSeqLen = 20,
            bool useConfidence = false,
            bool freezeText = false,
            bool priceContextEnabled = true,
            int priceCtxMinHist = 5,
            int priceCtxMinCurrent = 5,
            bool validate = true)
        {
            var config = new MultimodalTransformerConfig
            {
                Text = new TextEncoderConfig
                {
                    VocabSize = textVocabSize,
                    MaxSequenceLength = textMaxSeqLen,
                    EmbeddingDim = embDim,
                    NumHeads = numHeads,
                    NumLayers = numLayers,
                    FeedForwardDim = ffnDim,
                    UseDecoderOnly = false,
                    Freeze = freezeText
                },
                Price = new PriceDecoderConfig
                {
                    InputFeatureDim = inputFeatures,
                    MaxSequenceLength = priceSeqLen + 2,
                    EmbeddingDim = embDim,
                    NumHeads = numHeads,
                    NumLayers = numLayers,
                    FeedForwardDim = ffnDim,
                    UseDecoderOnly = true
                },
                Output = new OutputHeadConfig
                {
                    OutputDim = outputDim,
                    UseConfidenceHead = useConfidence
                },
                Runtime = new RuntimeConfig
                {
                    FFNActivationType = ActivationType.Relu,
                    AccelerationType = AccelerationType.CPU,
                    AccelerationDeviceId = 0
                },
                Regularization = new RegularizationConfig
                {
                    L2RegulationLamda = 0f,
                    GradientClippingThreshold = 1f
                },
                PriceContext = new PriceContextConfig
                {
                    Enabled = priceContextEnabled,
                    MinHistoryLength = priceCtxMinHist,
                    MinCurrentLength = priceCtxMinCurrent
                },
                DecayNetwork = new DecayNetworkConfig
                {
                    Enabled = true,
                    ProjectionDim = Math.Max(4, Math.Min(8, embDim)),
                    HiddenDim = Math.Max(8, Math.Min(16, embDim * 2)),
                    TimeEncodingBases = 8,
                    MemAttentionDropout = 0f,
                    MlpDropout = 0f,
                    WeightDecay = 0f
                },
                RequireSharedCrossAttentionEmbeddingDim = true
            };

            if (validate)
                config.Validate();

            return config;
        }

        private BPETokenizer CreateTinyTokenizer(out string[] corpus)
        {
            corpus = new[]
            {
                "stock price rose sharply today",
                "market crashed due to earnings miss",
                "bullish sentiment on tech sector",
                "neutral background market commentary"
            };
            var tokenizer = new BPETokenizer();
            tokenizer.Train(corpus, vocabSize: 100, minFrequency: 1);
            return tokenizer;
        }

        private (NewsStory[][] stories, float[][,] inputs, float[][,] targets) CreateConstantTargetPriceData(int samples, int seqLen, int inputDim, int outputDim, float targetValue, int seed)
        {
            var rng = new Random(seed);
            var stories = new NewsStory[samples][];
            var inputs = new float[samples][,];
            var targets = new float[samples][,];

            for (int s = 0; s < samples; s++)
            {
                stories[s] = null;
                inputs[s] = RandomMatrix(seqLen, inputDim, rng, 0.25f);
                targets[s] = FilledMatrix(seqLen, outputDim, targetValue);
            }

            return (stories, inputs, targets);
        }

        private (NewsStory[][] stories, float[][,] inputs, float[][,] targets) CreateBullBearData(BPETokenizer tokenizer, string bullText, string bearText, int samples, int seqLen, int inputDim)
        {
            var stories = new NewsStory[samples][];
            var inputs = new float[samples][,];
            var targets = new float[samples][,];
            int[] bull = tokenizer.Encode(bullText, addSpecialTokens: true);
            int[] bear = tokenizer.Encode(bearText, addSpecialTokens: true);

            for (int s = 0; s < samples; s++)
            {
                bool isBull = s % 2 == 0;
                stories[s] = new[] { new NewsStory(isBull ? bull : bear, -1f) };
                inputs[s] = FilledMatrix(seqLen, inputDim, 0.1f);
                targets[s] = FilledMatrix(seqLen, 1, isBull ? 0.9f : 0.1f);
            }

            return (stories, inputs, targets);
        }

        private float ManualNoContextNextStepLoss(TACAMT_Model model, NewsStory[] stories, float[,] priceInput, float[,] priceTarget)
        {
            int seqLen = priceInput.GetLength(0);
            int effective = seqLen - 1;
            var inputWindow = SliceRows(priceInput, 0, effective);
            var cache = new MultimodalForwardCache(model.Config.Text.NumLayers, model.Config.Price.NumLayers);
            var (pred, _) = model.ForwardWithCache(stories, inputWindow, cache, isTraining: false);
            return MseAgainstTargetSlice(pred, priceTarget, 1, effective, model.Config.Output.OutputDim);
        }

        private float ManualPriceContextValidationLoss(TACAMT_Model model, NewsStory[] stories, float[,] priceInput, float[,] priceTarget, int minHistory, int minCurrent)
        {
            int seqLen = priceInput.GetLength(0);
            int maxHistory = seqLen - minCurrent - 1;
            int splitPoint = (minHistory + maxHistory) / 2;

            var history = SliceRows(priceInput, 0, splitPoint);
            var currentInput = SliceRows(priceInput, splitPoint, seqLen - 1);
            var priceCtxHidden = model.EncodePriceHistory(history);
            var priceCtxTimes = new float[splitPoint];
            for (int t = 0; t < splitPoint; t++)
                priceCtxTimes[t] = t - splitPoint;

            NewsStory[] adjustedStories = null;
            if (stories != null && stories.Length > 0)
            {
                adjustedStories = stories
                    .Select(s => new NewsStory((int[])s.TokenIds.Clone(), s.ArrivalTime - splitPoint))
                    .ToArray();
            }

            var cache = new MultimodalForwardCache(model.Config.Text.NumLayers, model.Config.Price.NumLayers);
            var (pred, _) = model.ForwardWithPriceContextAndCache(adjustedStories, currentInput, priceCtxHidden, priceCtxTimes, cache, isTraining: false);
            return MseAgainstTargetSlice(pred, priceTarget, splitPoint + 1, currentInput.GetLength(0), model.Config.Output.OutputDim);
        }

        private float MseAgainstTargetSlice(float[,] pred, float[,] target, int targetStart, int targetCount, int outputDim)
        {
            float loss = 0f;
            for (int t = 0; t < targetCount; t++)
            {
                int targetRow = targetStart + t;
                for (int j = 0; j < outputDim; j++)
                {
                    float diff = pred[t, j] - target[targetRow, j];
                    loss += diff * diff;
                }
            }

            return loss / (targetCount * outputDim);
        }

        private float RunBackward(TACAMT_Trainer trainer, float[,] pred, float[,] conf, float[,] target, float[] confidenceTarget, MultimodalForwardCache cache)
        {
            InvokePrivate(trainer, "ZeroAllGradients");

            var method = typeof(TACAMT_Trainer).GetMethod(
                "BackwardPass",
                BindingFlags.Instance | BindingFlags.NonPublic,
                binder: null,
                types: new[] { typeof(float[,]), typeof(float[,]), typeof(float[,]), typeof(float[]), typeof(MultimodalForwardCache) },
                modifiers: null);

            if (method == null)
                throw new MissingMethodException("Could not find BackwardPass(pred, conf, target, confTarget, cache).");

            return (float)method.Invoke(trainer, new object[] { pred, conf, target, confidenceTarget, cache });
        }

        private Gradients GetGradients(TACAMT_Trainer trainer)
        {
            var field = typeof(TACAMT_Trainer).GetField("_gradients", BindingFlags.Instance | BindingFlags.NonPublic);
            if (field == null)
                throw new MissingFieldException("Trainer._gradients not found.");
            return (Gradients)field.GetValue(trainer);
        }

        private object InvokePrivate(object instance, string methodName, params object[] args)
        {
            var methods = instance.GetType().GetMethods(BindingFlags.Instance | BindingFlags.NonPublic | BindingFlags.Public)
                .Where(m => m.Name == methodName && m.GetParameters().Length == args.Length)
                .ToArray();

            foreach (var method in methods)
            {
                var parameters = method.GetParameters();
                bool compatible = true;
                for (int i = 0; i < parameters.Length; i++)
                {
                    if (args[i] == null)
                        continue;
                    if (!parameters[i].ParameterType.IsInstanceOfType(args[i]))
                    {
                        compatible = false;
                        break;
                    }
                }

                if (compatible)
                    return method.Invoke(instance, args);
            }

            throw new MissingMethodException($"Method {methodName} with {args.Length} arguments not found on {instance.GetType().Name}.");
        }

        private void InvokePruneNews(TACAMT_Model model, int maxSize)
        {
            InvokePrivate(model, "PruneNewsMemory", maxSize);
        }

        private void InvokePrunePrice(TACAMT_Model model, int maxSize)
        {
            InvokePrivate(model, "PricePruneMemory", maxSize);
        }

        private float[,] RandomMatrix(int rows, int cols, Random rng, float scale)
        {
            var matrix = new float[rows, cols];
            for (int i = 0; i < rows; i++)
                for (int j = 0; j < cols; j++)
                    matrix[i, j] = ((float)rng.NextDouble() - 0.5f) * 2f * scale;
            return matrix;
        }

        private float[,] FilledMatrix(int rows, int cols, float value)
        {
            var matrix = new float[rows, cols];
            for (int i = 0; i < rows; i++)
                for (int j = 0; j < cols; j++)
                    matrix[i, j] = value;
            return matrix;
        }

        private float[,] MixedMagnitudeMatrix(int rows, int cols, Random rng)
        {
            var matrix = new float[rows, cols];
            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < cols; j++)
                {
                    float sign = rng.NextDouble() > 0.5 ? 1f : -1f;
                    float magnitude = (float)Math.Pow(10, rng.Next(-4, 4));
                    matrix[i, j] = sign * magnitude;
                }
            }
            return matrix;
        }

        private float[,] SliceRows(float[,] matrix, int startInclusive, int endExclusive)
        {
            int rows = endExclusive - startInclusive;
            int cols = matrix.GetLength(1);
            var result = new float[rows, cols];
            for (int i = 0; i < rows; i++)
                for (int j = 0; j < cols; j++)
                    result[i, j] = matrix[startInclusive + i, j];
            return result;
        }

        private float[,] CloneMatrix(float[,] matrix)
        {
            return (float[,])matrix.Clone();
        }

        private float[] RampVector(int length, float scale)
        {
            var vector = new float[length];
            for (int i = 0; i < length; i++)
                vector[i] = i * scale;
            return vector;
        }

        private float[] GetLastRow(float[,] matrix)
        {
            int last = matrix.GetLength(0) - 1;
            int cols = matrix.GetLength(1);
            var result = new float[cols];
            for (int j = 0; j < cols; j++)
                result[j] = matrix[last, j];
            return result;
        }

        private float AveragePrediction(float[,] pred)
        {
            float sum = 0f;
            int count = pred.GetLength(0) * pred.GetLength(1);
            for (int i = 0; i < pred.GetLength(0); i++)
                for (int j = 0; j < pred.GetLength(1); j++)
                    sum += pred[i, j];
            return sum / count;
        }

        private bool MatrixChanged(float[,] before, float[,] after, float threshold)
        {
            return MatrixMaxAbsDiff(before, after) > threshold;
        }

        private float MatrixMaxAbsDiff(float[,] a, float[,] b)
        {
            Assert(a.GetLength(0) == b.GetLength(0) && a.GetLength(1) == b.GetLength(1), "Matrix shape mismatch.");
            float max = 0f;
            for (int i = 0; i < a.GetLength(0); i++)
                for (int j = 0; j < a.GetLength(1); j++)
                    max = MathF.Max(max, MathF.Abs(a[i, j] - b[i, j]));
            return max;
        }

        private bool IsSortedAscending(IEnumerable<double> values)
        {
            double previous = double.NegativeInfinity;
            foreach (double value in values)
            {
                if (value < previous)
                    return false;
                previous = value;
            }
            return true;
        }

        private float SumAbs(float[] vector)
        {
            float sum = 0f;
            for (int i = 0; i < vector.Length; i++)
                sum += MathF.Abs(vector[i]);
            return sum;
        }

        private float SumAbs(float[,] matrix)
        {
            float sum = 0f;
            for (int i = 0; i < matrix.GetLength(0); i++)
                for (int j = 0; j < matrix.GetLength(1); j++)
                    sum += MathF.Abs(matrix[i, j]);
            return sum;
        }

        private float SumAbs(float[,,] tensor)
        {
            float sum = 0f;
            for (int i = 0; i < tensor.GetLength(0); i++)
                for (int j = 0; j < tensor.GetLength(1); j++)
                    for (int k = 0; k < tensor.GetLength(2); k++)
                        sum += MathF.Abs(tensor[i, j, k]);
            return sum;
        }

        private float SumAbsRow(float[,] matrix, int row)
        {
            float sum = 0f;
            for (int j = 0; j < matrix.GetLength(1); j++)
                sum += MathF.Abs(matrix[row, j]);
            return sum;
        }

        private float AttentionGradSum(dynamic gradients)
        {
            return SumAbs((float[,])gradients.WQ_Grad)
                 + SumAbs((float[,])gradients.WK_Grad)
                 + SumAbs((float[,])gradients.WV_Grad)
                 + SumAbs((float[,])gradients.WO_Grad)
                 + SumAbs((float[])gradients.BiasQ_Grad)
                 + SumAbs((float[])gradients.BiasK_Grad)
                 + SumAbs((float[])gradients.BiasV_Grad)
                 + SumAbs((float[])gradients.BiasO_Grad);
        }

        private float DecayGradSum(ContentAwareDecayGradients gradients)
        {
            if (gradients == null)
                return 0f;

            return SumAbs(gradients.LogBaseDecayRateGrad)
                 + SumAbs(gradients.B2Grad)
                 + SumAbs(gradients.QueryProjectionGrad)
                 + SumAbs(gradients.QueryProjectionBiasGrad)
                 + SumAbs(gradients.KeyProjectionGrad)
                 + SumAbs(gradients.KeyProjectionBiasGrad)
                 + SumAbs(gradients.TimeLogFreqGrad)
                 + SumAbs(gradients.TimeProjGrad)
                 + SumAbs(gradients.TimeProjBiasGrad)
                 + SumAbs(gradients.MemAttnOutputWGrad)
                 + SumAbs(gradients.MemAttnOutputBGrad)
                 + SumAbs(gradients.W1Grad)
                 + SumAbs(gradients.B1Grad)
                 + SumAbs(gradients.W2Grad);
        }


        private float[] CloneVector(float[] vector)
        {
            return (float[])vector.Clone();
        }

        private (int row, int col, float value) MaxAbsIndex(float[,] matrix)
        {
            int bestRow = 0;
            int bestCol = 0;
            float bestAbs = -1f;
            float bestValue = 0f;

            for (int i = 0; i < matrix.GetLength(0); i++)
            {
                for (int j = 0; j < matrix.GetLength(1); j++)
                {
                    float abs = MathF.Abs(matrix[i, j]);
                    if (abs > bestAbs)
                    {
                        bestAbs = abs;
                        bestValue = matrix[i, j];
                        bestRow = i;
                        bestCol = j;
                    }
                }
            }

            return (bestRow, bestCol, bestValue);
        }

        private float LossForPrediction(float[,] pred, float[,] conf, float[,] target, float[] confidenceTarget, float confidenceLossWeight)
        {
            int seqLen = pred.GetLength(0);
            int outputDim = pred.GetLength(1);
            float mse = 0f;

            for (int t = 0; t < seqLen; t++)
            {
                for (int j = 0; j < outputDim; j++)
                {
                    float diff = pred[t, j] - target[t, j];
                    mse += diff * diff;
                }
            }

            mse /= seqLen * outputDim;

            if (conf == null || confidenceLossWeight == 0f)
                return mse;

            float confidenceLoss = 0f;
            for (int t = 0; t < seqLen; t++)
            {
                float targetConfidence;
                if (confidenceTarget != null)
                {
                    targetConfidence = confidenceTarget[t];
                }
                else
                {
                    float sumSq = 0f;
                    for (int j = 0; j < outputDim; j++)
                    {
                        float diff = pred[t, j] - target[t, j];
                        sumSq += diff * diff;
                    }

                    float rmse = MathF.Sqrt(sumSq / outputDim);
                    targetConfidence = MathF.Exp(-5f * rmse);
                }

                float prediction = Math.Clamp(conf[t, 0], 1e-7f, 1f - 1e-7f);
                confidenceLoss -= targetConfidence * MathF.Log(prediction) + (1f - targetConfidence) * MathF.Log(1f - prediction);
            }

            confidenceLoss /= seqLen;
            return mse + confidenceLossWeight * confidenceLoss;
        }

        private float FiniteDifferenceLoss(
            TACAMT_Model model,
            NewsStory[] stories,
            float[,] price,
            float[,] target,
            float[] confidenceTarget,
            float? confidenceLossWeight,
            Func<TACAMT_Model, float> getter,
            Action<TACAMT_Model, float> setter,
            float epsilon = 1e-3f)
        {
            float original = getter(model);

            setter(model, original + epsilon);
            var (predPlus, confPlus) = model.Forward(stories, price);
            float lossPlus = LossForPrediction(predPlus, confPlus, target, confidenceTarget, confidenceLossWeight ?? 0f);

            setter(model, original - epsilon);
            var (predMinus, confMinus) = model.Forward(stories, price);
            float lossMinus = LossForPrediction(predMinus, confMinus, target, confidenceTarget, confidenceLossWeight ?? 0f);

            setter(model, original);
            return (lossPlus - lossMinus) / (2f * epsilon);
        }

        private void AssertRelativeClose(float expected, float actual, float maxRelativeError, string message)
        {
            float denom = MathF.Abs(expected) + MathF.Abs(actual) + 1e-8f;
            float rel = MathF.Abs(expected - actual) / denom;
            if (rel > maxRelativeError)
                throw new Exception($"{message} expected={expected:E6}, actual={actual:E6}, relativeError={rel:F6}, max={maxRelativeError:F6}.");
        }

        private void AssertClose(float expected, float actual, float tolerance, string message)
        {
            if (MathF.Abs(expected - actual) > tolerance)
                throw new Exception($"{message} Expected {expected}, got {actual}, tol {tolerance}.");
        }

        private void AssertClose(double expected, double actual, double tolerance, string message)
        {
            if (Math.Abs(expected - actual) > tolerance)
                throw new Exception($"{message} Expected {expected}, got {actual}, tol {tolerance}.");
        }

        private void AssertMatrixClose(float[,] expected, float[,] actual, float tolerance, string message)
        {
            Assert(expected.GetLength(0) == actual.GetLength(0), $"{message} Row mismatch: {expected.GetLength(0)} vs {actual.GetLength(0)}.");
            Assert(expected.GetLength(1) == actual.GetLength(1), $"{message} Col mismatch: {expected.GetLength(1)} vs {actual.GetLength(1)}.");

            for (int i = 0; i < expected.GetLength(0); i++)
            {
                for (int j = 0; j < expected.GetLength(1); j++)
                {
                    if (MathF.Abs(expected[i, j] - actual[i, j]) > tolerance)
                        throw new Exception($"{message} Mismatch at [{i},{j}]: expected {expected[i, j]}, got {actual[i, j]}, tol {tolerance}.");
                }
            }
        }

        private void AssertVectorClose(float[] expected, float[] actual, float tolerance, string message)
        {
            Assert(expected.Length == actual.Length, $"{message} Length mismatch: {expected.Length} vs {actual.Length}.");
            for (int i = 0; i < expected.Length; i++)
            {
                if (MathF.Abs(expected[i] - actual[i]) > tolerance)
                    throw new Exception($"{message} Mismatch at [{i}]: expected {expected[i]}, got {actual[i]}, tol {tolerance}.");
            }
        }

        private void AssertFinite(float value, string name)
        {
            Assert(!float.IsNaN(value) && !float.IsInfinity(value), $"{name} should be finite, got {value}.");
        }

        private void AssertFinite(float[] vector, string name)
        {
            for (int i = 0; i < vector.Length; i++)
                AssertFinite(vector[i], $"{name}[{i}]");
        }

        private void AssertFinite(float[,] matrix, string name)
        {
            for (int i = 0; i < matrix.GetLength(0); i++)
                for (int j = 0; j < matrix.GetLength(1); j++)
                    AssertFinite(matrix[i, j], $"{name}[{i},{j}]");
        }

        private string GetTempDir()
        {
            string dir = Path.Combine(Path.GetTempPath(), $"tacamt_tests_{Guid.NewGuid():N}");
            Directory.CreateDirectory(dir);
            return dir;
        }

        private void CleanupDir(string dir)
        {
            try
            {
                if (Directory.Exists(dir))
                    Directory.Delete(dir, recursive: true);
            }
            catch
            {
                // Test cleanup best-effort only.
            }
        }

        #endregion
    }
}
