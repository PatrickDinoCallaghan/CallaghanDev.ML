using CallaghanDev.ML.Transformers.CrossAttentionMultimodal;
using CallaghanDev.ML.Transformers.TACAMT;
public class TimeDecayNewsExample
{
    public static void Run()
    {
        // ================================================================
        // SHARED CONFIG (same for both approaches)
        // ================================================================
        var config = new CrossAttentionMultimodalConfig
        {
            TextVocabSize = 500,
            TextEmbeddingDim = 64,
            TextNumHeads = 4,
            TextNumLayers = 2,
            TextFeedForwardDim = 128,
            TextMaxSequenceLength = 64,

            PriceInputFeatureDim = 5,   // OHLCV
            PriceEmbeddingDim = 64,
            PriceNumHeads = 4,
            PriceNumLayers = 2,
            PriceFeedForwardDim = 128,
            PriceMaxSequenceLength = 32,

            OutputDim = 5,
            UseConfidenceHead = true
        };

        // Fake tokenized headlines (in real use, from BPETokenizer.Encode)
        int[] headline1_tokens = { 10, 42, 88, 3, 55 };   // "Fed announces rate hold"
        int[] headline2_tokens = { 7, 19, 200, 33 };       // "Tech stocks rally"  
        int[] headline3_tokens = { 15, 91, 67, 44, 12 };   // "Oil prices surge today"

        // 15 minutes of OHLCV price data
        int minutes = 15;
        var priceData = new float[minutes, 5];
        var priceTargets = new float[minutes, 5];
        var rng = new Random(42);
        for (int t = 0; t < minutes; t++)
        {
            for (int f = 0; f < 5; f++)
            {
                priceData[t, f] = 100f + rng.NextSingle() * 10f;
                priceTargets[t, f] = priceData[t, f] + rng.NextSingle() * 2f - 1f;
            }
        }

        // ================================================================
        // OLD WAY: Single concatenated text, no time awareness
        // ================================================================
        Console.WriteLine("=== OLD WAY: Single text, no time decay ===");
        {
            var model = new CallaghanDev.ML.Transformers.CrossAttentionMultimodal.CrossAttentionMultimodalModel(config);

            // Concatenate all headlines into one token sequence
            // Problem: model has NO IDEA when each headline arrived
            int[] allTokens = headline1_tokens
                .Concat(headline2_tokens)
                .Concat(headline3_tokens)
                .ToArray();

            // Old API: single int[] for text
            var (pred, conf) = model.PredictNext(allTokens, priceData);

            Console.WriteLine($"Prediction: [{string.Join(", ", pred.Select(p => p.ToString("F3")))}]");
            Console.WriteLine($"Confidence: {conf:F3}");
            Console.WriteLine();
            Console.WriteLine("Problems with old approach:");
            Console.WriteLine("  - Headline 1 arrived at minute 2, but model doesn't know that");
            Console.WriteLine("  - Headline 3 arrived at minute 14, but gets same weight as headline 1");
            Console.WriteLine("  - Every price position sees all headlines with equal access");

            // Old training API still works (backward compatible)
            var trainer = new CallaghanDev.ML.Transformers.CrossAttentionMultimodal.CrossAttentionMultimodalTrainer(model, new MultimodalTrainingConfig
            {
                Epochs = 2, BatchSize = 1, Verbose = false
            });

            trainer.Train(
                textSequences: new[] { allTokens },
                priceInputs: new[] { priceData },
                priceTargets: new[] { priceTargets }
            );
        }

        // ================================================================
        // NEW WAY: Multiple stories with arrival times + learned decay
        // ================================================================
        Console.WriteLine("\n=== NEW WAY: Multi-story with learned time decay ===");
        {
            var model = new CallaghanDev.ML.Transformers.TACAMT.CrossAttentionMultimodalModel(config);

            // Each headline is a separate NewsStory with its arrival time
            // ArrivalTime is in the same units as price positions (here: minutes)
            var stories = new NewsStory[]
            {
                new NewsStory(headline1_tokens, arrivalTime: 2.0f),   // Arrived at minute 2
                new NewsStory(headline2_tokens, arrivalTime: 9.0f),   // Arrived at minute 9
                new NewsStory(headline3_tokens, arrivalTime: 14.0f),  // Arrived at minute 14
            };

            // New API: NewsStory[] for text
            var (pred, conf) = model.PredictNext(stories, priceData);

            Console.WriteLine($"Prediction: [{string.Join(", ", pred.Select(p => p.ToString("F3")))}]");
            Console.WriteLine($"Confidence: {conf:F3}");
            Console.WriteLine();
            Console.WriteLine("What the model now understands:");
            Console.WriteLine("  - At minute 3:  headline 1 is 1 min old (high weight), 2 & 3 don't exist yet");
            Console.WriteLine("  - At minute 10: headline 2 is 1 min old (high weight), headline 1 is 8 min old (decayed)");
            Console.WriteLine("  - At minute 14: headline 3 is fresh, headline 1 is very old (heavily decayed)");
            Console.WriteLine();
            Console.WriteLine("Each attention head learns its OWN decay rate:");
            for (int layer = 0; layer < config.PriceNumLayers; layer++)
            {
                var block = model.PriceBlocks[layer];
                Console.Write($"  Layer {layer} rates: ");
                for (int h = 0; h < config.PriceNumHeads; h++)
                {
                    float rate = MathF.Exp(block.LogDecayRate[h]);
                    Console.Write($"head{h}={rate:F3}  ");
                }
                Console.WriteLine();
            }

            // New training API
            var trainer = new CallaghanDev.ML.Transformers.TACAMT.CrossAttentionMultimodalTrainer(model, new MultimodalTrainingConfig
            {
                Epochs = 5, BatchSize = 1, Verbose = true
            });

            // Training data: one NewsStory[] per sample
            var trainStories = new NewsStory[][] { stories };

            trainer.Train(
                storiesPerSample: trainStories,
                priceInputs: new[] { priceData },
                priceTargets: new[] { priceTargets }
            );

            Console.WriteLine("\nAfter training, heads learn different decay speeds:");
            for (int layer = 0; layer < config.PriceNumLayers; layer++)
            {
                var block = model.PriceBlocks[layer];
                Console.Write($"  Layer {layer}: ");
                for (int h = 0; h < config.PriceNumHeads; h++)
                {
                    float rate = MathF.Exp(block.LogDecayRate[h]);
                    Console.Write($"head{h}={rate:F3}  ");
                }
                Console.WriteLine();
            }
        }

        // ================================================================
        // NEWS MEMORY: Save, wait a week, reload with correct decay
        // ================================================================
        Console.WriteLine("\n=== NEWS MEMORY: Persistence across save/load ===");
        {
            var model = new CallaghanDev.ML.Transformers.TACAMT.CrossAttentionMultimodalModel(config);
            string savePath = "/tmp/multimodal_model";

            // --- Day 1: Train and save ---
            double day1Timestamp = 1000.0; // minutes since epoch (simplified)
            var day1Stories = new NewsStory[]
            {
                new NewsStory(headline1_tokens, arrivalTime: 0f),
                new NewsStory(headline2_tokens, arrivalTime: 5f),
            };

            // Store stories in memory with absolute timestamps
            model.UpdateNewsMemory(day1Stories, currentAbsoluteTimestamp: day1Timestamp);
            model.Save(savePath);
            Console.WriteLine($"Saved model with {model.NewsMemory.Count} stories in memory");

            // --- Day 8 (one week later): Load and predict ---
            var loadedModel = CallaghanDev.ML.Transformers.TACAMT.CrossAttentionMultimodalModel.Load(savePath);
            Console.WriteLine($"Loaded model with {loadedModel.NewsMemory.Count} stories in memory");
            Console.WriteLine($"Last price timestamp: {loadedModel.LastPriceTimestamp}");

            double day8Timestamp = day1Timestamp + 7 * 24 * 60; // One week later (in minutes)

            var day8Stories = new NewsStory[]
            {
                new NewsStory(headline3_tokens, arrivalTime: 0f), // Fresh story today
            };

            // PredictWithMemory automatically:
            //   1. Loads old story hidden states from memory
            //   2. Computes their age: 7 * 24 * 60 = 10080 minutes old
            //   3. Assigns them very negative arrival times (heavily decayed)
            //   4. Combines with today's fresh story
            //   5. The learned decay naturally suppresses week-old news
            var (pred, conf) = loadedModel.PredictWithMemory(
                newStories: day8Stories,
                priceSequence: priceData,
                currentAbsoluteTimestamp: day8Timestamp,
                timeUnitsPerPosition: 1.0  // 1 minute per price position
            );

            Console.WriteLine($"Prediction with memory: [{string.Join(", ", pred.Select(p => p.ToString("F3")))}]");
            Console.WriteLine($"Confidence: {conf:F3}");
            Console.WriteLine("Old stories from a week ago are heavily decayed but still influence predictions");

            // Clean up
            if (System.IO.Directory.Exists(savePath))
                System.IO.Directory.Delete(savePath, true);
        }

        // ================================================================
        // MIXED: Some samples have text, some don't (still works)
        // ================================================================
        Console.WriteLine("\n=== MIXED: Optional text per sample ===");
        {
            var model = new CallaghanDev.ML.Transformers.TACAMT.CrossAttentionMultimodalModel(config);
            var trainer = new CallaghanDev.ML.Transformers.TACAMT.CrossAttentionMultimodalTrainer(model, new MultimodalTrainingConfig
            {
                Epochs = 3, BatchSize = 2, Verbose = true
            });

            // Sample 0: has 3 news stories with timestamps
            // Sample 1: no news available (null)
            // Sample 2: just 1 story
            var stories = new NewsStory[][]
            {
                new[]
                {
                    new NewsStory(headline1_tokens, 0f),
                    new NewsStory(headline2_tokens, 5f),
                    new NewsStory(headline3_tokens, 12f),
                },
                null,  // No text for this sample - model uses price-only mode
                new[]
                {
                    new NewsStory(headline1_tokens, 7f),
                },
            };

            var prices = new float[3][,];
            var targets = new float[3][,];

            for (int s = 0; s < 3; s++)
            {
                prices[s] = new float[minutes, 5];
                targets[s] = new float[minutes, 5];

                for (int t = 0; t < minutes; t++)
                {
                    for (int f = 0; f < 5; f++)
                    {
                        prices[s][t, f] = 100f + rng.NextSingle() * 10f;
                        targets[s][t, f] = prices[s][t, f] + rng.NextSingle() * 2f - 1f;
                    }
                }
            }

            trainer.Train(stories, prices, targets);
            Console.WriteLine("Training with mixed text/no-text samples works seamlessly");
        }
    }
}
