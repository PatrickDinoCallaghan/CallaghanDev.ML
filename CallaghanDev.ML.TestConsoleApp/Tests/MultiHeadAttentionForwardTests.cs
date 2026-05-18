using CallaghanDev.ML.AccelerationManagers;
using CallaghanDev.ML.Enums;
using System.Reflection;

namespace CallaghanDev.ML.TestConsoleApp.Tests
{
    internal sealed class MultiHeadAttentionForwardTests : TestBase
    {
        private const float Tolerance = 1e-4f;
        AccelerationType _accelerationType;
        public MultiHeadAttentionForwardTests(AccelerationType accelerationType)
        {
            _accelerationType = accelerationType;
        }
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
            (Test_NewMethod_FullyMaskedRowReturnsZero, "New method: fully masked row returns zero"),
            (Test_NewMethod_NullArgumentsThrow, "New method: null arguments throw"),
            (Test_NewMethod_InvalidShapesThrow, "New method: invalid shapes throw"),
            (Test_ParallelCalls_NewMethodDeterministic, "Concurrency: new method deterministic"),
        };

        private void Test_Construction_DefaultStateCorrect()
        {
            var cpu = NewAccellerator();

            Assert(cpu != null, "AccelerationCPU should construct");
        }

        private void Test_DeterministicKnownValues_NoMask()
        {
            var cpu = NewAccellerator();

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
            var cpu = NewAccellerator();

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


        private void Test_NewMethod_FullyMaskedRowReturnsZero()
        {
            var cpu = NewAccellerator();

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
            var cpu = NewAccellerator();
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
            var cpu = NewAccellerator();

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


        private void Test_ParallelCalls_NewMethodDeterministic()
        {
            var cpu = NewAccellerator();

            var q = DeterministicMatrix(16, 32, seed: 2300);
            var k = DeterministicMatrix(16, 32, seed: 2400);
            var v = DeterministicMatrix(16, 32, seed: 2500);
            var mask = CausalMask(16, 16);

            var expected = cpu.MultiHeadAttentionForward(q, k, v, numHeads: 4, scale: 0.5f, mask: mask);

            var results = new float[32][,];

            Parallel.For(0, results.Length, i =>
            {
                var localCpu = NewAccellerator();

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



        private IAccelerationManager NewAccellerator()
        {
            return AccelerationFactory.Create(_accelerationType);
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