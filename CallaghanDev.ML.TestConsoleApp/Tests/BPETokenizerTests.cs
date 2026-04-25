using CallaghanDev.ML.AccelerationManagers;
using CallaghanDev.ML.Transformers;
using System.Reflection;

namespace CallaghanDev.ML.TestConsoleApp.Tests
{
    internal sealed class BPETokenizerTests : TestBase
    {
        public void RunAllTests()
        {
            CountNumber++;
            Run(Tests(), $"{CountNumber} * BPE Tokenizer");
        }

        private (Action, string)[] Tests() => new (Action, string)[]
        {
            (Test_BPE_LearnsFrequentWholeWordMerge, "BPE: frequent word becomes compact tokenization"),
            (Test_BPE_MinFrequencyPreventsRareMerge, "BPE: minFrequency prevents rare merges"),
            (Test_BPE_MergePriorityAffectsEncoding, "BPE: learned merge priority is respected"),
            (Test_Punctuation_IsPreservedAsSeparateTokens, "PreTokenize: punctuation is preserved"),
            (Test_SaveLoad_MergePriorityPreserved, "SaveLoad: merge priority preserved"),
            (Test_Train_RetrainClearsOldVocabularyAndCache, "Train: retrain clears old vocab/cache"),
            (Test_Construction_DefaultTokensCorrect, "Construction: special token ids are correct"),
            (Test_Construction_NullAccelerationFallsBackOrThrowsClearly, "Construction: acceleration manager is usable"),
            (Test_Train_InvalidArgumentsThrow, "Train: invalid arguments throw"),
            (Test_Train_BuildsVocabularyAndMerges, "Train: builds vocab and learns merges"),
            (Test_Encode_EmptyStringBehaviour, "Encode: empty string special-token behaviour"),
            (Test_Encode_WithAndWithoutSpecialTokens, "Encode: special token wrapping"),
            (Test_Encode_DeterministicAndCacheSafe, "Encode: deterministic and cache stable"),
            (Test_Encode_UnknownCharactersFallbackToUnk, "Encode: unknown characters fallback"),
            (Test_Decode_SkipsSpecialTokens, "Decode: skips special tokens"),
            (Test_Decode_IncludesSpecialTokensWhenRequested, "Decode: includes special tokens when requested"),
            (Test_EncodeBatch_NoPaddingMatchesEncode, "EncodeBatch: no maxLength matches Encode"),
            (Test_EncodeBatch_PadsAndTruncates, "EncodeBatch: pads and truncates correctly"),
            (Test_LowerCase_ChangesVocabularyAndEncoding, "LowerCase: normalizes text"),
            (Test_IdToTokenAndTokenToId, "Lookup: IdToToken and TokenToId work"),
            (Test_GetVocabulary_IsIdOrdered, "Vocabulary: returned in id order"),
            (Test_SaveLoad_RoundTripPreservesEncoding, "SaveLoad: preserves encoding"),
            (Test_SaveLoad_PreservesConfig, "SaveLoad: preserves tokenizer config"),
            (Test_Acceleration_PrimitivesMatchTokenizerUse, "Acceleration: tokenizer primitive functions behave correctly"),
            (Test_ParallelEncodeBatch_ReadOnlyDeterministic, "Future parallelism: parallel Encode calls are deterministic"),
        };
        private void Test_BPE_LearnsFrequentWholeWordMerge()
        {
            var tok = NewTokenizer();

            tok.Train(
                Enumerable.Repeat("hello", 20).ToArray(),
                vocabSize: 50,
                minFrequency: 1);

            int[] encoded = tok.Encode("hello", addSpecialTokens: false);

            Assert(encoded.Length == 1,
                $"Expected frequent word 'hello' to merge into one token, got {encoded.Length}");

            Assert(tok.IdToToken(encoded[0]) == "hello",
                $"Expected token to decode to 'hello', got {tok.IdToToken(encoded[0])}");
        }

        private void Test_BPE_MinFrequencyPreventsRareMerge()
        {
            var tok = NewTokenizer();

            tok.Train(
                new[] { "hello" },
                vocabSize: 50,
                minFrequency: 10);

            int[] encoded = tok.Encode("hello", addSpecialTokens: false);

            Assert(encoded.Length > 1,
                $"Expected minFrequency to prevent full merge, got one token: {tok.IdToToken(encoded[0])}");
        }

        private void Test_BPE_MergePriorityAffectsEncoding()
        {
            var tok = NewTokenizer();

            tok.Train(
                Enumerable.Repeat("abab", 20).ToArray(),
                vocabSize: 50,
                minFrequency: 1);

            int[] encoded = tok.Encode("abab", addSpecialTokens: false);
            string decoded = tok.Decode(encoded, skipSpecialTokens: true);

            Assert(decoded == "abab", $"Decoded text mismatch. decoded={decoded}");
            Assert(encoded.Length <= 2,
                $"Expected learned BPE merges to compact 'abab', got {encoded.Length} tokens");
        }

        private void Test_Punctuation_IsPreservedAsSeparateTokens()
        {
            var tok = NewTokenizer();

            tok.Train(
                new[] { "hello, world!" },
                vocabSize: 50,
                minFrequency: 1);

            var vocab = tok.GetVocabulary();

            Assert(vocab.Contains(","), "Vocabulary should contain comma token");
            Assert(vocab.Contains("!"), "Vocabulary should contain exclamation token");

            string decoded = tok.Decode(tok.Encode("hello, world!", true), true);

            Assert(decoded == "hello,world!",
                $"Expected punctuation-preserving decode without whitespace reconstruction. decoded={decoded}");
        }

        private void Test_SaveLoad_MergePriorityPreserved()
        {
            var tok = NewTokenizer();

            tok.Train(
                new[]
                {
            "low lower lowest",
            "low lower lowest",
            "low lower lowest"
                },
                vocabSize: 80,
                minFrequency: 1);

            var before = tok.Encode("lowest lower low", addSpecialTokens: true);

            string dir = TmpDir();

            try
            {
                tok.Save(dir);
                var loaded = BPETokenizer.Load(dir, new AccelerationCPU());

                var after = loaded.Encode("lowest lower low", addSpecialTokens: true);

                Assert(before.SequenceEqual(after),
                    "Encoding changed after save/load, so merge priority may not be preserved.");
            }
            finally
            {
                DeleteDir(dir);
            }
        }

        private void Test_Train_RetrainClearsOldVocabularyAndCache()
        {
            var tok = NewTokenizer();

            tok.Train(new[] { "abc abc abc" }, vocabSize: 30, minFrequency: 1);
            _ = tok.Encode("abc", addSpecialTokens: false);

            Assert(tok.TokenToId("a") != tok.UnkTokenId, "Precondition failed: old vocab missing 'a'");

            tok.Train(new[] { "xyz xyz xyz" }, vocabSize: 30, minFrequency: 1);

            Assert(tok.TokenToId("x") != tok.UnkTokenId, "New vocab missing 'x'");
            Assert(tok.TokenToId("a") == tok.UnkTokenId, "Old vocab token 'a' should be gone after retrain");

            var encoded = tok.Encode("abc", addSpecialTokens: false);

            Assert(encoded.All(id => id == tok.UnkTokenId),
                "Old cached encoding appears to have survived retrain.");
        }
        private void Test_Construction_DefaultTokensCorrect()
        {
            var tok = NewTokenizer();

            Assert(tok.PadTokenId == 0, "PadTokenId should be 0");
            Assert(tok.StartTokenId == 1, "StartTokenId should be 1");
            Assert(tok.EndTokenId == 2, "EndTokenId should be 2");
            Assert(tok.UnkTokenId == 3, "UnkTokenId should be 3");

            Assert(tok.TokenToId(BPETokenizer.PAD_TOKEN) == 0, "PAD token lookup failed");
            Assert(tok.TokenToId(BPETokenizer.START_TOKEN) == 1, "START token lookup failed");
            Assert(tok.TokenToId(BPETokenizer.END_TOKEN) == 2, "END token lookup failed");
            Assert(tok.TokenToId(BPETokenizer.UNK_TOKEN) == 3, "UNK token lookup failed");

            Assert(tok.IdToToken(0) == BPETokenizer.PAD_TOKEN, "id 0 lookup failed");
            Assert(tok.IdToToken(1) == BPETokenizer.START_TOKEN, "id 1 lookup failed");
            Assert(tok.IdToToken(2) == BPETokenizer.END_TOKEN, "id 2 lookup failed");
            Assert(tok.IdToToken(3) == BPETokenizer.UNK_TOKEN, "id 3 lookup failed");

            Assert(tok.VocabSize == 4, $"Initial vocab size should be 4, got {tok.VocabSize}");
        }

        private void Test_Construction_NullAccelerationFallsBackOrThrowsClearly()
        {
            var ctor = typeof(BPETokenizer).GetConstructor(new[] { typeof(IAccelerationManager) });
            Assert(ctor != null, "BPETokenizer(IAccelerationManager) constructor missing");

            var tok = new BPETokenizer(new AccelerationCPU());
            tok.Train(new[] { "hello world" }, vocabSize: 32, minFrequency: 1);

            var ids = tok.Encode("hello");
            Assert(ids.Length > 0, "Tokenizer with AccelerationCPU failed to encode");

            var defaultCtor = typeof(BPETokenizer).GetConstructor(Type.EmptyTypes);
            Assert(defaultCtor != null,
                "BPETokenizer needs a parameterless constructor because Load() calls new BPETokenizer(). Add public BPETokenizer() : this(new AccelerationCPU()) { }");
        }

        private void Test_Train_InvalidArgumentsThrow()
        {
            var tok = NewTokenizer();

            AssertThrows<ArgumentException>(
                () => tok.Train(null, vocabSize: 32, minFrequency: 1),
                "Train should reject null corpus");

            AssertThrows<ArgumentException>(
                () => tok.Train(Array.Empty<string>(), vocabSize: 32, minFrequency: 1),
                "Train should reject empty corpus");

            AssertThrows<ArgumentOutOfRangeException>(
                () => tok.Train(new[] { "abc" }, vocabSize: 4, minFrequency: 1),
                "Train should reject vocabSize <= special-token count");

            AssertThrows<ArgumentOutOfRangeException>(
                () => tok.Train(new[] { "abc" }, vocabSize: 32, minFrequency: 0),
                "Train should reject minFrequency <= 0");
        }

        private void Test_Train_BuildsVocabularyAndMerges()
        {
            var tok = NewTokenizer();

            tok.Train(
                new[]
                {
                    "low lower lowest",
                    "low lower lowest",
                    "newer wider lower",
                    "market lower low"
                },
                vocabSize: 80,
                minFrequency: 1);

            Assert(tok.VocabSize > 4, "Vocabulary did not grow beyond special tokens");
            Assert(tok.TokenToId("l") != tok.UnkTokenId, "Expected character token 'l'");
            Assert(tok.TokenToId("o") != tok.UnkTokenId, "Expected character token 'o'");

            var encoded = tok.Encode("lower", addSpecialTokens: false);
            Assert(encoded.Length > 0, "Encoding trained word returned no tokens");
            Assert(encoded.All(id => id >= 0 && id < tok.VocabSize), "Encoded ids out of vocab range");
        }

        private void Test_Encode_EmptyStringBehaviour()
        {
            var tok = NewTokenizer();

            var withSpecial = tok.Encode("", addSpecialTokens: true);
            Assert(withSpecial.SequenceEqual(new[] { tok.StartTokenId, tok.EndTokenId }),
                "Empty string with special tokens should be [START, END]");

            var withoutSpecial = tok.Encode("", addSpecialTokens: false);
            Assert(withoutSpecial.Length == 0, "Empty string without special tokens should be empty");
        }

        private void Test_Encode_WithAndWithoutSpecialTokens()
        {
            var tok = TrainedTokenizer();

            var withSpecial = tok.Encode("stock rose", addSpecialTokens: true);
            var withoutSpecial = tok.Encode("stock rose", addSpecialTokens: false);

            Assert(withSpecial.Length == withoutSpecial.Length + 2, "Special-token length mismatch");
            Assert(withSpecial[0] == tok.StartTokenId, "First token should be START");
            Assert(withSpecial[^1] == tok.EndTokenId, "Last token should be END");

            Assert(withSpecial.Skip(1).Take(withSpecial.Length - 2).SequenceEqual(withoutSpecial),
                "Inner tokens should match no-special encoding");
        }

        private void Test_Encode_DeterministicAndCacheSafe()
        {
            var tok = TrainedTokenizer();

            var a = tok.Encode("central bank support growth", addSpecialTokens: true);
            var b = tok.Encode("central bank support growth", addSpecialTokens: true);
            var c = tok.Encode("central bank support growth", addSpecialTokens: true);

            Assert(a.SequenceEqual(b), "Repeated encode mismatch 1");
            Assert(a.SequenceEqual(c), "Repeated encode mismatch 2");

            var cache = GetPrivateField<Dictionary<string, List<int>>>(tok, "_encodeCache");
            Assert(cache.Count > 0, "Encode cache did not populate");
            Assert(cache.ContainsKey("central"), "Expected word not found in encode cache");
        }

        private void Test_Encode_UnknownCharactersFallbackToUnk()
        {
            var tok = NewTokenizer();
            tok.Train(new[] { "abc abc abc" }, vocabSize: 16, minFrequency: 1);

            var encoded = tok.Encode("€", addSpecialTokens: false);

            Assert(encoded.Length == 1, $"Expected one unknown token, got {encoded.Length}");
            Assert(encoded[0] == tok.UnkTokenId, $"Expected UnkTokenId, got {encoded[0]}");
        }

        private void Test_Decode_SkipsSpecialTokens()
        {
            var tok = TrainedTokenizer();

            var ids = tok.Encode("stock rose sharply", addSpecialTokens: true);
            string decoded = tok.Decode(ids, skipSpecialTokens: true);

            Assert(!decoded.Contains("<|"), $"Decoded text should skip special tokens. decoded={decoded}");
            Assert(decoded.Contains("stock"), $"Decoded text should contain original content. decoded={decoded}");
            Assert(decoded.Contains("rose"), $"Decoded text should contain original content. decoded={decoded}");
        }

        private void Test_Decode_IncludesSpecialTokensWhenRequested()
        {
            var tok = TrainedTokenizer();

            var ids = tok.Encode("stock", addSpecialTokens: true);
            string decoded = tok.Decode(ids, skipSpecialTokens: false);

            Assert(decoded.Contains(BPETokenizer.START_TOKEN), "Decoded text should include START token");
            Assert(decoded.Contains(BPETokenizer.END_TOKEN), "Decoded text should include END token");
        }

        private void Test_EncodeBatch_NoPaddingMatchesEncode()
        {
            var tok = TrainedTokenizer();
            string[] texts =
            {
                "stock rose",
                "market crashed",
                "central bank support"
            };

            var batch = tok.EncodeBatch(texts, addSpecialTokens: true, maxLength: null);

            Assert(batch.Length == texts.Length, "Batch length mismatch");

            for (int i = 0; i < texts.Length; i++)
            {
                var expected = tok.Encode(texts[i], addSpecialTokens: true);
                Assert(batch[i].SequenceEqual(expected), $"Batch item {i} does not match Encode()");
            }
        }

        private void Test_EncodeBatch_PadsAndTruncates()
        {
            var tok = TrainedTokenizer();

            var batch = tok.EncodeBatch(
                new[]
                {
                    "stock",
                    "central bank support growth demand sharply today"
                },
                addSpecialTokens: true,
                maxLength: 6);

            Assert(batch.Length == 2, "Batch length mismatch");
            Assert(batch[0].Length == 6, "Padded row length mismatch");
            Assert(batch[1].Length == 6, "Truncated row length mismatch");

            Assert(batch[0][0] == tok.StartTokenId, "Padded row should start with START");
            Assert(batch[0].Contains(tok.EndTokenId), "Padded row should contain END");
            Assert(batch[0].Last() == tok.PadTokenId, "Short row should end with PAD");

            Assert(batch[1][0] == tok.StartTokenId, "Truncated row should start with START");
            Assert(batch[1][^1] == tok.EndTokenId, "Truncated row should force final END token");
        }

        private void Test_LowerCase_ChangesVocabularyAndEncoding()
        {
            var tok = NewTokenizer();
            tok.LowerCase = true;

            tok.Train(new[] { "Stock STOCK stock" }, vocabSize: 40, minFrequency: 1);

            var a = tok.Encode("STOCK", addSpecialTokens: false);
            var b = tok.Encode("stock", addSpecialTokens: false);

            Assert(a.SequenceEqual(b), "LowerCase=true should make STOCK and stock encode identically");
            Assert(tok.Decode(tok.Encode("STOCK", addSpecialTokens: false)).Contains("stock"),
                "Decoded lowercase text should contain normalized lowercase form");
        }

        private void Test_IdToTokenAndTokenToId()
        {
            var tok = TrainedTokenizer();

            foreach (var token in tok.GetVocabulary())
            {
                int id = tok.TokenToId(token);
                Assert(id >= 0, $"Invalid id for token {token}");
                Assert(tok.IdToToken(id) == token, $"Round-trip token/id lookup failed for {token}");
            }

            Assert(tok.TokenToId("__missing_token__") == tok.UnkTokenId,
                "Missing token should map to UnkTokenId");

            Assert(tok.IdToToken(int.MaxValue) == BPETokenizer.UNK_TOKEN,
                "Missing id should map to UNK token");
        }

        private void Test_GetVocabulary_IsIdOrdered()
        {
            var tok = TrainedTokenizer();
            var vocab = tok.GetVocabulary();

            Assert(vocab.Count == tok.VocabSize, "Vocabulary count mismatch");

            for (int i = 0; i < vocab.Count; i++)
            {
                Assert(tok.IdToToken(i) == vocab[i], $"Vocabulary not ordered by id at {i}");
            }
        }

        private void Test_SaveLoad_RoundTripPreservesEncoding()
        {
            var tok = TrainedTokenizer();
            tok.MaxTokenLength = 77;
            tok.LowerCase = true;

            string[] texts =
            {
                "stock rose sharply",
                "central bank support growth",
                "credit stress recession risk",
                "unknown € token"
            };

            var before = texts
                .Select(t => tok.Encode(t, addSpecialTokens: true))
                .ToArray();

            string dir = TmpDir();

            try
            {
                tok.Save(dir);

                var loaded = BPETokenizer.Load(dir, new AccelerationCPU());

                Assert(loaded.VocabSize == tok.VocabSize, "Loaded vocab size mismatch");

                for (int i = 0; i < texts.Length; i++)
                {
                    var after = loaded.Encode(texts[i], addSpecialTokens: true);
                    Assert(before[i].SequenceEqual(after), $"Encoding mismatch after load for text index {i}");
                }

                Assert(tok.GetVocabulary().SequenceEqual(loaded.GetVocabulary()),
                    "Vocabulary differs after load");
            }
            finally
            {
                DeleteDir(dir);
            }
        }

        private void Test_SaveLoad_PreservesConfig()
        {
            var tok = TrainedTokenizer();
            tok.MaxTokenLength = 123;
            tok.LowerCase = true;

            string dir = TmpDir();

            try
            {
                tok.Save(dir);
                var loaded = BPETokenizer.Load(dir, new AccelerationCPU());

                Assert(loaded.MaxTokenLength == 123, "MaxTokenLength not preserved");
                Assert(loaded.LowerCase == true, "LowerCase not preserved");
            }
            finally
            {
                DeleteDir(dir);
            }
        }

        private void Test_Acceleration_PrimitivesMatchTokenizerUse()
        {
            var accel = new AccelerationCPU();

            var tokens = accel.PreTokenize("Hello, world! 123");
            Assert(tokens.SequenceEqual(new[] { "Hello", ",", "world", "!", "123" }),
                "PreTokenize produced unexpected tokens");

            var freqs = accel.GetWordFrequencies(
                new[] { "aa aa ab", "aa!" },
                lowerCase: false);

            Assert(freqs["a a"] == 3, "Word frequency for 'aa' should be 3");
            Assert(freqs["a b"] == 1, "Word frequency for 'ab' should be 1");
            Assert(freqs["!"] == 1, "Word frequency for punctuation should be 1");

            var chars = accel.BuildCharacterVocabulary(freqs);
            Assert(chars.Contains("a"), "Character vocab should contain 'a'");
            Assert(chars.Contains("b"), "Character vocab should contain 'b'");
            Assert(chars.Contains("!"), "Character vocab should contain '!'");

            var merged = accel.ApplyMerge(new List<string> { "a", "a", "b" }, "a", "a");
            Assert(merged.SequenceEqual(new[] { "aa", "b" }), "ApplyMerge failed");

            var words = new Dictionary<List<string>, int>(new ListEqualityComparer<string>())
            {
                [new List<string> { "a", "a" }] = 3,
                [new List<string> { "a", "b" }] = 2
            };

            var pairCounts = accel.CountPairFrequencies(words);
            Assert(pairCounts[("a", "a")] == 3, "Pair count aa mismatch");
            Assert(pairCounts[("a", "b")] == 2, "Pair count ab mismatch");

            var (pair, frequency) = accel.SelectBestPair(pairCounts, minFrequency: 1);
            Assert(pair == ("a", "a") && frequency == 3, "SelectBestPair failed");

            var padded = accel.PadOrTruncate(new[] { 1, 2 }, 4, true, 0, 2);
            Assert(padded.SequenceEqual(new[] { 1, 2, 0, 0 }), "PadOrTruncate padding failed");

            var truncated = accel.PadOrTruncate(new[] { 1, 5, 6, 7, 2 }, 3, true, 0, 2);
            Assert(truncated.SequenceEqual(new[] { 1, 5, 2 }), "PadOrTruncate truncation failed");
        }

        private void Test_ParallelEncodeBatch_ReadOnlyDeterministic()
        {
            var tok = TrainedTokenizer();

            string[] texts = Enumerable.Range(0, 500)
          .Select(i => (i % 4) switch
          {
              0 => "stock rose sharply",
              1 => "market crashed today",
              2 => "central bank support growth",
              _ => "credit stress recession risk"
          })
          .ToArray();

            var expected = texts
                .Select(t => tok.Encode(t, addSpecialTokens: true))
                .ToArray();

            var actual = new int[texts.Length][];

            Parallel.For(0, texts.Length, i =>
            {
                actual[i] = tok.Encode(texts[i], addSpecialTokens: true);
            });

            for (int i = 0; i < texts.Length; i++)
            {
                Assert(actual[i].SequenceEqual(expected[i]), $"Parallel encode mismatch at index {i}");
            }
        }

        private static BPETokenizer NewTokenizer()
        {
            return new BPETokenizer(new AccelerationCPU());
        }

        private static BPETokenizer TrainedTokenizer()
        {
            var tok = NewTokenizer();

            tok.Train(
                new[]
                {
                    "stock rose sharply",
                    "stock rose sharply",
                    "market crashed today",
                    "central bank support growth",
                    "credit stress recession risk",
                    "routine market update commentary",
                    "demand growth support",
                    "bearish data weak",
                    "bullish outlook strong"
                },
                vocabSize: 120,
                minFrequency: 1);

            return tok;
        }

        private static T GetPrivateField<T>(object instance, string fieldName)
        {
            var field = instance.GetType().GetField(
                fieldName,
                BindingFlags.Instance | BindingFlags.NonPublic);

            if (field == null)
                throw new InvalidOperationException($"Could not find private field {fieldName}");

            return (T)field.GetValue(instance);
        }

        private static string TmpDir()
        {
            string dir = Path.Combine(
                Path.GetTempPath(),
                "bpe_tokenizer_test_" + Guid.NewGuid().ToString("N"));

            Directory.CreateDirectory(dir);
            return dir;
        }

        private static void DeleteDir(string dir)
        {
            if (Directory.Exists(dir))
                Directory.Delete(dir, recursive: true);
        }

        private void AssertThrows<T>(Action action, string message) where T : Exception
        {
            try
            {
                action();
            }
            catch (TargetInvocationException ex) when (ex.InnerException is T)
            {
                return;
            }
            catch (T)
            {
                return;
            }
            catch (Exception ex)
            {
                Assert(false, $"{message}; expected {typeof(T).Name}, got {ex.GetType().Name}: {ex.Message}");
                return;
            }

            Assert(false, $"{message}; expected {typeof(T).Name}, got no exception");
        }
    }
}