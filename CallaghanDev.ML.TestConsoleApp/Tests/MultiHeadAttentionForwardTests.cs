using CallaghanDev.ML.AccelerationManagers;
using System.Reflection;

namespace CallaghanDev.ML.TestConsoleApp.Tests
{
    internal sealed class MultiHeadAttentionForwardTests : TestBase
    {
        private const float Tolerance = 1e-4f;

        public void RunAllTests()
        {
            CountNumber++;
            Run(Tests(), $"{CountNumber} * Multi-Head Attention Forward Equivalence");
        }

        private (Action, string)[] Tests() => new (Action, string)[]
        {
            (Test_Construction_DefaultStateCorrect, "Construction: AccelerationCPU constructs"),
            (Test_DeterministicKnownValues_NoMask, "Known values: single-head no-mask output correct"),
            (Test_DeterministicKnownValues_WithMask, "Known values: single-head masked output correct"),
            (Test_OldAndNewMatch_ExhaustiveSmallShapes_NoMask, "Equivalence: exhaustive small valid shapes without mask"),
            (Test_OldAndNewMatch_ExhaustiveSmallShapes_AllValidMasks, "Equivalence: exhaustive small valid masks"),
            (Test_OldAndNewMatch_CausalMasks, "Equivalence: causal masks"),
            (Test_OldAndNewMatch_RandomizedMediumCases, "Equivalence: randomized medium cases"),
            (Test_OldAndNewMatch_DifferentScales, "Equivalence: different scale values"),
            (Test_OldAndNewMatch_CrossAttentionShapes, "Equivalence: seqLenQ != seqLenK"),
            (Test_NewMethod_FullyMaskedRowReturnsZero, "New method: fully masked row returns zero"),
            (Test_NewMethod_NullArgumentsThrow, "New method: null arguments throw"),
            (Test_NewMethod_InvalidShapesThrow, "New method: invalid shapes throw"),
            (Test_Methods_DoNotMutateInputs, "Both methods: inputs are not mutated"),
            (Test_ParallelCalls_NewMethodDeterministic, "Concurrency: new method deterministic"),
        };

        private void Test_Construction_DefaultStateCorrect()
        {
            var cpu = NewCpu();

            Assert(cpu != null, "AccelerationCPU should construct");
        }

        private void Test_DeterministicKnownValues_NoMask()
        {
            var cpu = NewCpu();

            var q = new float[,]
            {
                { 1f, 0f }
            };

            var k = new float[,]
            {
                { 1f, 0f },
                { 0f, 1f }
            };

            var v = new float[,]
            {
                { 10f, 20f },
                { 30f, 40f }
            };

            var actual = cpu.MultiHeadAttentionForward(q, k, v, numHeads: 1, scale: 1f);

            float e1 = MathF.Exp(1f);
            float e0 = MathF.Exp(0f);
            float w0 = e1 / (e1 + e0);
            float w1 = e0 / (e1 + e0);

            float expected0 = (w0 * 10f) + (w1 * 30f);
            float expected1 = (w0 * 20f) + (w1 * 40f);

            AssertNearlyEqual(actual[0, 0], expected0, "Output[0,0] mismatch");
            AssertNearlyEqual(actual[0, 1], expected1, "Output[0,1] mismatch");
        }

        private void Test_DeterministicKnownValues_WithMask()
        {
            var cpu = NewCpu();

            var q = new float[,]
            {
                { 1f, 0f }
            };

            var k = new float[,]
            {
                { 1f, 0f },
                { 0f, 1f }
            };

            var v = new float[,]
            {
                { 10f, 20f },
                { 30f, 40f }
            };

            var mask = new bool[,]
            {
                { true, false }
            };

            var actual = cpu.MultiHeadAttentionForward(q, k, v, numHeads: 1, scale: 1f, mask: mask);

            AssertNearlyEqual(actual[0, 0], 10f, "Masked output[0,0] mismatch");
            AssertNearlyEqual(actual[0, 1], 20f, "Masked output[0,1] mismatch");
        }

        private void Test_OldAndNewMatch_ExhaustiveSmallShapes_NoMask()
        {
            var cpu = NewCpu();

            for (int seqLenQ = 1; seqLenQ <= 5; seqLenQ++)
            {
                for (int seqLenK = 1; seqLenK <= 5; seqLenK++)
                {
                    for (int embeddingDim = 1; embeddingDim <= 12; embeddingDim++)
                    {
                        for (int numHeads = 1; numHeads <= embeddingDim; numHeads++)
                        {
                            if (embeddingDim % numHeads != 0)
                            {
                                continue;
                            }

                            var q = DeterministicMatrix(seqLenQ, embeddingDim, seed: 100 + seqLenQ + embeddingDim);
                            var k = DeterministicMatrix(seqLenK, embeddingDim, seed: 200 + seqLenK + embeddingDim);
                            var v = DeterministicMatrix(seqLenK, embeddingDim, seed: 300 + seqLenK + embeddingDim);

                            float scale = 1f / MathF.Sqrt(embeddingDim / numHeads);

                            AssertOldAndNewMatch(cpu, q, k, v, numHeads, scale, null,
                                $"No-mask mismatch; Q={seqLenQ}, K={seqLenK}, D={embeddingDim}, H={numHeads}");
                        }
                    }
                }
            }
        }

        private void Test_OldAndNewMatch_ExhaustiveSmallShapes_AllValidMasks()
        {
            var cpu = NewCpu();

            for (int seqLenQ = 1; seqLenQ <= 3; seqLenQ++)
            {
                for (int seqLenK = 1; seqLenK <= 3; seqLenK++)
                {
                    for (int embeddingDim = 1; embeddingDim <= 8; embeddingDim++)
                    {
                        for (int numHeads = 1; numHeads <= embeddingDim; numHeads++)
                        {
                            if (embeddingDim % numHeads != 0)
                            {
                                continue;
                            }

                            var q = DeterministicMatrix(seqLenQ, embeddingDim, seed: 400 + seqLenQ + embeddingDim);
                            var k = DeterministicMatrix(seqLenK, embeddingDim, seed: 500 + seqLenK + embeddingDim);
                            var v = DeterministicMatrix(seqLenK, embeddingDim, seed: 600 + seqLenK + embeddingDim);

                            float scale = 1f / MathF.Sqrt(embeddingDim / numHeads);

                            foreach (var mask in EnumerateMasksWithoutFullyMaskedRows(seqLenQ, seqLenK))
                            {
                                AssertOldAndNewMatch(cpu, q, k, v, numHeads, scale, mask,
                                    $"Mask mismatch; Q={seqLenQ}, K={seqLenK}, D={embeddingDim}, H={numHeads}");
                            }
                        }
                    }
                }
            }
        }

        private void Test_OldAndNewMatch_CausalMasks()
        {
            var cpu = NewCpu();

            for (int seqLenQ = 1; seqLenQ <= 8; seqLenQ++)
            {
                for (int seqLenK = 1; seqLenK <= 8; seqLenK++)
                {
                    for (int embeddingDim = 2; embeddingDim <= 16; embeddingDim += 2)
                    {
                        for (int numHeads = 1; numHeads <= embeddingDim; numHeads++)
                        {
                            if (embeddingDim % numHeads != 0)
                            {
                                continue;
                            }

                            var q = DeterministicMatrix(seqLenQ, embeddingDim, seed: 700 + seqLenQ);
                            var k = DeterministicMatrix(seqLenK, embeddingDim, seed: 800 + seqLenK);
                            var v = DeterministicMatrix(seqLenK, embeddingDim, seed: 900 + embeddingDim);

                            float scale = 1f / MathF.Sqrt(embeddingDim / numHeads);
                            var mask = CausalMask(seqLenQ, seqLenK);

                            AssertOldAndNewMatch(cpu, q, k, v, numHeads, scale, mask,
                                $"Causal mask mismatch; Q={seqLenQ}, K={seqLenK}, D={embeddingDim}, H={numHeads}");
                        }
                    }
                }
            }
        }

        private void Test_OldAndNewMatch_RandomizedMediumCases()
        {
            var cpu = NewCpu();
            var random = new Random(12345);

            for (int test = 0; test < 250; test++)
            {
                int seqLenQ = random.Next(1, 16);
                int seqLenK = random.Next(1, 16);

                int[] embeddingDims = new[] { 2, 4, 6, 8, 12, 16, 24, 32 };
                int embeddingDim = embeddingDims[random.Next(embeddingDims.Length)];

                var validHeads = ValidHeadCounts(embeddingDim);
                int numHeads = validHeads[random.Next(validHeads.Length)];

                var q = RandomMatrix(seqLenQ, embeddingDim, random);
                var k = RandomMatrix(seqLenK, embeddingDim, random);
                var v = RandomMatrix(seqLenK, embeddingDim, random);

                float scale = 1f / MathF.Sqrt(embeddingDim / numHeads);
                bool[,] mask = random.Next(0, 2) == 0
                    ? null
                    : RandomMaskWithoutFullyMaskedRows(seqLenQ, seqLenK, random);

                AssertOldAndNewMatch(cpu, q, k, v, numHeads, scale, mask,
                    $"Random mismatch at test {test}; Q={seqLenQ}, K={seqLenK}, D={embeddingDim}, H={numHeads}");
            }
        }

        private void Test_OldAndNewMatch_DifferentScales()
        {
            var cpu = NewCpu();

            float[] scales = new[]
            {
                0f,
                0.01f,
                0.1f,
                0.5f,
                1f,
                2f,
                -0.5f
            };

            foreach (float scale in scales)
            {
                var q = DeterministicMatrix(4, 8, seed: 1000);
                var k = DeterministicMatrix(5, 8, seed: 1100);
                var v = DeterministicMatrix(5, 8, seed: 1200);

                AssertOldAndNewMatch(cpu, q, k, v, numHeads: 2, scale: scale, mask: null,
                    $"Scale mismatch for scale {scale}");
            }
        }

        private void Test_OldAndNewMatch_CrossAttentionShapes()
        {
            var cpu = NewCpu();

            var cases = new[]
            {
                (SeqQ: 1, SeqK: 7, EmbeddingDim: 8, Heads: 2),
                (SeqQ: 7, SeqK: 1, EmbeddingDim: 8, Heads: 2),
                (SeqQ: 3, SeqK: 9, EmbeddingDim: 12, Heads: 3),
                (SeqQ: 9, SeqK: 3, EmbeddingDim: 12, Heads: 4),
            };

            foreach (var testCase in cases)
            {
                var q = DeterministicMatrix(testCase.SeqQ, testCase.EmbeddingDim, seed: 1300);
                var k = DeterministicMatrix(testCase.SeqK, testCase.EmbeddingDim, seed: 1400);
                var v = DeterministicMatrix(testCase.SeqK, testCase.EmbeddingDim, seed: 1500);

                float scale = 1f / MathF.Sqrt(testCase.EmbeddingDim / testCase.Heads);

                AssertOldAndNewMatch(cpu, q, k, v, testCase.Heads, scale, null,
                    $"Cross-attention mismatch; Q={testCase.SeqQ}, K={testCase.SeqK}, D={testCase.EmbeddingDim}, H={testCase.Heads}");
            }
        }

        private void Test_NewMethod_FullyMaskedRowReturnsZero()
        {
            var cpu = NewCpu();

            var q = DeterministicMatrix(2, 4, seed: 1600);
            var k = DeterministicMatrix(3, 4, seed: 1700);
            var v = DeterministicMatrix(3, 4, seed: 1800);

            var mask = new bool[,]
            {
                { true, false, true },
                { false, false, false }
            };

            var actual = cpu.MultiHeadAttentionForward(q, k, v, numHeads: 2, scale: 0.5f, mask: mask);

            AssertNearlyEqual(actual[1, 0], 0f, "Fully masked row col 0 should be zero");
            AssertNearlyEqual(actual[1, 1], 0f, "Fully masked row col 1 should be zero");
            AssertNearlyEqual(actual[1, 2], 0f, "Fully masked row col 2 should be zero");
            AssertNearlyEqual(actual[1, 3], 0f, "Fully masked row col 3 should be zero");
        }

        private void Test_NewMethod_NullArgumentsThrow()
        {
            var cpu = NewCpu();
            var matrix = DeterministicMatrix(2, 4, seed: 1900);

            AssertThrows<ArgumentNullException>(
                () => cpu.MultiHeadAttentionForward(null, matrix, matrix, 2, 0.5f),
                "New method should reject null Q");

            AssertThrows<ArgumentNullException>(
                () => cpu.MultiHeadAttentionForward(matrix, null, matrix, 2, 0.5f),
                "New method should reject null K");

            AssertThrows<ArgumentNullException>(
                () => cpu.MultiHeadAttentionForward(matrix, matrix, null, 2, 0.5f),
                "New method should reject null V");
        }

        private void Test_NewMethod_InvalidShapesThrow()
        {
            var cpu = NewCpu();

            AssertThrows<ArgumentException>(
                () => cpu.MultiHeadAttentionForward(
                    DeterministicMatrix(2, 5, 1),
                    DeterministicMatrix(2, 5, 2),
                    DeterministicMatrix(2, 5, 3),
                    numHeads: 2,
                    scale: 1f),
                "New method should reject embedding dimension not divisible by heads");

            AssertThrows<ArgumentException>(
                () => cpu.MultiHeadAttentionForward(
                    DeterministicMatrix(2, 4, 1),
                    DeterministicMatrix(2, 5, 2),
                    DeterministicMatrix(2, 4, 3),
                    numHeads: 2,
                    scale: 1f),
                "New method should reject K embedding dimension mismatch");

            AssertThrows<ArgumentException>(
                () => cpu.MultiHeadAttentionForward(
                    DeterministicMatrix(2, 4, 1),
                    DeterministicMatrix(2, 4, 2),
                    DeterministicMatrix(2, 5, 3),
                    numHeads: 2,
                    scale: 1f),
                "New method should reject V embedding dimension mismatch");

            AssertThrows<ArgumentException>(
                () => cpu.MultiHeadAttentionForward(
                    DeterministicMatrix(2, 4, 1),
                    DeterministicMatrix(3, 4, 2),
                    DeterministicMatrix(2, 4, 3),
                    numHeads: 2,
                    scale: 1f),
                "New method should reject K/V sequence length mismatch");

            AssertThrows<ArgumentException>(
                () => cpu.MultiHeadAttentionForward(
                    DeterministicMatrix(2, 4, 1),
                    DeterministicMatrix(3, 4, 2),
                    DeterministicMatrix(3, 4, 3),
                    numHeads: 2,
                    scale: 1f,
                    mask: new bool[3, 3]),
                "New method should reject mask shape mismatch");
        }

        private void Test_Methods_DoNotMutateInputs()
        {
            var cpu = NewCpu();

            var q = DeterministicMatrix(4, 8, seed: 2000);
            var k = DeterministicMatrix(5, 8, seed: 2100);
            var v = DeterministicMatrix(5, 8, seed: 2200);

            var qOriginal = Clone(q);
            var kOriginal = Clone(k);
            var vOriginal = Clone(v);
            cpu.MultiHeadAttentionForward_Obsolete(q, k, v, numHeads: 2, scale: 0.5f);

            AssertMatrixNearlyEqual(q, qOriginal, "Old method mutated Q");
            AssertMatrixNearlyEqual(k, kOriginal, "Old method mutated K");
            AssertMatrixNearlyEqual(v, vOriginal, "Old method mutated V");

            cpu.MultiHeadAttentionForward(q, k, v, numHeads: 2, scale: 0.5f);

            AssertMatrixNearlyEqual(q, qOriginal, "New method mutated Q");
            AssertMatrixNearlyEqual(k, kOriginal, "New method mutated K");
            AssertMatrixNearlyEqual(v, vOriginal, "New method mutated V");
        }

        private void Test_ParallelCalls_NewMethodDeterministic()
        {
            var cpu = NewCpu();

            var q = DeterministicMatrix(16, 32, seed: 2300);
            var k = DeterministicMatrix(16, 32, seed: 2400);
            var v = DeterministicMatrix(16, 32, seed: 2500);
            var mask = CausalMask(16, 16);

            var expected = cpu.MultiHeadAttentionForward(q, k, v, numHeads: 4, scale: 0.5f, mask: mask);

            var results = new float[32][,];

            Parallel.For(0, results.Length, i =>
            {
                var localCpu = NewCpu();

                results[i] = localCpu.MultiHeadAttentionForward(
                    Clone(q),
                    Clone(k),
                    Clone(v),
                    numHeads: 4,
                    scale: 0.5f,
                    mask: Clone(mask));
            });

            for (int i = 0; i < results.Length; i++)
            {
                AssertMatrixNearlyEqual(results[i], expected, $"Parallel result mismatch at index {i}");
            }
        }

        private static void AssertOldAndNewMatch(
            AccelerationCPU cpu,
            float[,] q,
            float[,] k,
            float[,] v,
            int numHeads,
            float scale,
            bool[,] mask,
            string message)
        {
            var qForOld = Clone(q);
            var kForOld = Clone(k);
            var vForOld = Clone(v);
            var maskForOld = mask == null ? null : Clone(mask);

            var qForNew = Clone(q);
            var kForNew = Clone(k);
            var vForNew = Clone(v);
            var maskForNew = mask == null ? null : Clone(mask);

            var expected = cpu.MultiHeadAttentionForward_Obsolete(qForOld, kForOld, vForOld, numHeads, scale, maskForOld);
            var actual = cpu.MultiHeadAttentionForward(qForNew, kForNew, vForNew, numHeads, scale, maskForNew);

            AssertMatrixNearlyEqualStatic(actual, expected, message);
        }

        private static AccelerationCPU NewCpu()
        {
            return new AccelerationCPU();
        }

        private static float[,] DeterministicMatrix(int rows, int cols, int seed)
        {
            var matrix = new float[rows, cols];

            int state = seed;

            for (int r = 0; r < rows; r++)
            {
                for (int c = 0; c < cols; c++)
                {
                    state = unchecked((state * 1103515245) + 12345);

                    int value = (state >> 8) & 0xFFFF;

                    matrix[r, c] = ((value / 65535f) - 0.5f) * 2f;
                }
            }

            return matrix;
        }

        private static float[,] RandomMatrix(int rows, int cols, Random random)
        {
            var matrix = new float[rows, cols];

            for (int r = 0; r < rows; r++)
            {
                for (int c = 0; c < cols; c++)
                {
                    matrix[r, c] = ((float)random.NextDouble() - 0.5f) * 2f;
                }
            }

            return matrix;
        }

        private static bool[,] CausalMask(int seqLenQ, int seqLenK)
        {
            var mask = new bool[seqLenQ, seqLenK];

            for (int i = 0; i < seqLenQ; i++)
            {
                for (int k = 0; k < seqLenK; k++)
                {
                    mask[i, k] = k <= i;
                }
            }

            return mask;
        }

        private static bool[,] RandomMaskWithoutFullyMaskedRows(int seqLenQ, int seqLenK, Random random)
        {
            var mask = new bool[seqLenQ, seqLenK];

            for (int i = 0; i < seqLenQ; i++)
            {
                bool anyTrue = false;

                for (int k = 0; k < seqLenK; k++)
                {
                    bool value = random.Next(0, 2) == 1;
                    mask[i, k] = value;
                    anyTrue |= value;
                }

                if (!anyTrue)
                {
                    mask[i, random.Next(seqLenK)] = true;
                }
            }

            return mask;
        }

        private static IEnumerable<bool[,]> EnumerateMasksWithoutFullyMaskedRows(int seqLenQ, int seqLenK)
        {
            int bits = seqLenQ * seqLenK;
            int combinations = 1 << bits;

            for (int combination = 0; combination < combinations; combination++)
            {
                var mask = new bool[seqLenQ, seqLenK];

                for (int bit = 0; bit < bits; bit++)
                {
                    int r = bit / seqLenK;
                    int c = bit % seqLenK;

                    mask[r, c] = ((combination >> bit) & 1) == 1;
                }

                if (HasNoFullyMaskedRows(mask))
                {
                    yield return mask;
                }
            }
        }

        private static bool HasNoFullyMaskedRows(bool[,] mask)
        {
            int rows = mask.GetLength(0);
            int cols = mask.GetLength(1);

            for (int r = 0; r < rows; r++)
            {
                bool anyTrue = false;

                for (int c = 0; c < cols; c++)
                {
                    anyTrue |= mask[r, c];
                }

                if (!anyTrue)
                {
                    return false;
                }
            }

            return true;
        }

        private static int[] ValidHeadCounts(int embeddingDim)
        {
            var values = new List<int>();

            for (int h = 1; h <= embeddingDim; h++)
            {
                if (embeddingDim % h == 0)
                {
                    values.Add(h);
                }
            }

            return values.ToArray();
        }

        private static float[,] Clone(float[,] source)
        {
            var clone = new float[source.GetLength(0), source.GetLength(1)];

            for (int r = 0; r < source.GetLength(0); r++)
            {
                for (int c = 0; c < source.GetLength(1); c++)
                {
                    clone[r, c] = source[r, c];
                }
            }

            return clone;
        }

        private static bool[,] Clone(bool[,] source)
        {
            var clone = new bool[source.GetLength(0), source.GetLength(1)];

            for (int r = 0; r < source.GetLength(0); r++)
            {
                for (int c = 0; c < source.GetLength(1); c++)
                {
                    clone[r, c] = source[r, c];
                }
            }

            return clone;
        }

        private void AssertMatrixNearlyEqual(float[,] actual, float[,] expected, string message)
        {
            AssertMatrixNearlyEqualStatic(actual, expected, message);
        }

        private static void AssertMatrixNearlyEqualStatic(float[,] actual, float[,] expected, string message)
        {
            if (actual.GetLength(0) != expected.GetLength(0))
            {
                throw new Exception($"{message}; row count mismatch. Expected {expected.GetLength(0)}, got {actual.GetLength(0)}");
            }

            if (actual.GetLength(1) != expected.GetLength(1))
            {
                throw new Exception($"{message}; column count mismatch. Expected {expected.GetLength(1)}, got {actual.GetLength(1)}");
            }

            for (int r = 0; r < actual.GetLength(0); r++)
            {
                for (int c = 0; c < actual.GetLength(1); c++)
                {
                    float a = actual[r, c];
                    float e = expected[r, c];

                    if (float.IsNaN(a) || float.IsNaN(e))
                    {
                        if (!(float.IsNaN(a) && float.IsNaN(e)))
                        {
                            throw new Exception($"{message}; NaN mismatch at [{r},{c}]. Expected {e}, got {a}");
                        }

                        continue;
                    }

                    float diff = MathF.Abs(a - e);

                    if (diff > Tolerance)
                    {
                        throw new Exception($"{message}; mismatch at [{r},{c}]. Expected {e}, got {a}, diff {diff}");
                    }
                }
            }
        }

        private void AssertNearlyEqual(float actual, float expected, string message)
        {
            float diff = MathF.Abs(actual - expected);

            Assert(diff <= Tolerance,
                $"{message}; expected {expected}, got {actual}, diff {diff}");
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