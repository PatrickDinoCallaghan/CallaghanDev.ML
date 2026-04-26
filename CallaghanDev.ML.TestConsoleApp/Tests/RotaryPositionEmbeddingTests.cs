using CallaghanDev.ML.AccelerationManagers;
using CallaghanDev.ML.Transformers;
using System.Reflection;

namespace CallaghanDev.ML.TestConsoleApp.Tests
{
    internal sealed class RotaryPositionEmbeddingTests : TestBase
    {
        private const float Tolerance = 1e-5f;

        public void RunAllTests()
        {
            CountNumber++;
            Run(Tests(), $"{CountNumber} * Rotary Position Embedding");
        }

        private (Action, string)[] Tests() => new (Action, string)[]
        {
            (Test_Construction_DefaultStateCorrect, "Construction: default state is usable"),
            (Test_ApplyInPlace_NullArgumentsThrow, "ApplyInPlace: null arguments throw"),
            (Test_ApplyInPlace_InvalidHeadArgumentsThrow, "ApplyInPlace: invalid head arguments throw"),
            (Test_ApplyInPlace_QKDimensionMismatchThrows, "ApplyInPlace: Q/K dimension mismatch throws"),
            (Test_ApplyBackwardInPlace_NullArgumentsThrow, "ApplyBackwardInPlace: null arguments throw"),
            (Test_ApplyBackwardInPlace_InvalidHeadArgumentsThrow, "ApplyBackwardInPlace: invalid head arguments throw"),
            (Test_ApplyBackwardInPlace_DQDKDimensionMismatchThrows, "ApplyBackwardInPlace: dQ/dK dimension mismatch throws"),
            (Test_ApplyInPlace_PositionZeroUnchanged, "ApplyInPlace: position zero is unchanged"),
            (Test_ApplyInPlace_DeterministicKnownValues, "ApplyInPlace: deterministic known values"),
            (Test_ApplyInPlace_MultipleHeadsRotatedIndependently, "ApplyInPlace: multiple heads rotated independently"),
            (Test_ApplyInPlace_QAndKBothMutated, "ApplyInPlace: Q and K both mutated"),
            (Test_ApplyBackwardInPlace_ReversesApplyInPlace, "ApplyBackwardInPlace: reverses ApplyInPlace"),
            (Test_ApplyBackwardInPlace_QKReversesApplyInPlaceQK, "ApplyBackwardInPlace: reverses Q/K ApplyInPlace"),
            (Test_ApplyInPlace_ZeroRowsDoesNotThrow, "ApplyInPlace: zero rows accepted"),
            (Test_ApplyInPlace_ZeroColumnsDoesNotThrowWhenValid, "ApplyInPlace: zero columns accepted when mathematically valid"),
            (Test_ApplyInPlace_DoesNotChangeShape, "ApplyInPlace: shape preserved"),
            (Test_ParallelIndependentInstances_Deterministic, "Concurrency: independent instances deterministic"),
        };

        private void Test_Construction_DefaultStateCorrect()
        {
            var rope = NewRotaryPositionEmbedding();

            Assert(rope != null, "RotaryPositionEmbedding should construct");
        }

        private void Test_ApplyInPlace_NullArgumentsThrow()
        {
            var rope = NewRotaryPositionEmbedding();
            var matrix = Matrix(1, 2);

            AssertThrows<ArgumentNullException>(
                () => rope.ApplyInPlace((float[,])null, 1),
                "ApplyInPlace(x) should reject null x");

            AssertThrows<ArgumentNullException>(
                () => rope.ApplyInPlace(null, matrix, 1),
                "ApplyInPlace(q, k) should reject null q");

            AssertThrows<ArgumentNullException>(
                () => rope.ApplyInPlace(matrix, null, 1),
                "ApplyInPlace(q, k) should reject null k");
        }

        private void Test_ApplyInPlace_InvalidHeadArgumentsThrow()
        {
            var rope = NewRotaryPositionEmbedding();

            AssertThrows<DivideByZeroException>(
                () => rope.ApplyInPlace(Matrix(2, 4), 0),
                "ApplyInPlace should reject zero heads");

            AssertThrows<ArgumentException>(
                () => rope.ApplyInPlace(Matrix(2, 5), 2),
                "ApplyInPlace should reject embedding dimensions not divisible by heads");

            AssertThrows<ArgumentException>(
                () => rope.ApplyInPlace(Matrix(2, 6), 2),
                "ApplyInPlace should reject odd per-head dimensions");
        }

        private void Test_ApplyInPlace_QKDimensionMismatchThrows()
        {
            var rope = NewRotaryPositionEmbedding();

            AssertThrows<ArgumentException>(
                () => rope.ApplyInPlace(Matrix(2, 4), Matrix(2, 6), 2),
                "ApplyInPlace(q, k) should reject mismatched embedding dimensions");
        }

        private void Test_ApplyBackwardInPlace_NullArgumentsThrow()
        {
            var rope = NewRotaryPositionEmbedding();
            var matrix = Matrix(1, 2);

            AssertThrows<ArgumentNullException>(
                () => rope.ApplyBackwardInPlace((float[,])null, 1),
                "ApplyBackwardInPlace(dX) should reject null dX");

            AssertThrows<ArgumentNullException>(
                () => rope.ApplyBackwardInPlace(null, matrix, 1),
                "ApplyBackwardInPlace(dQ, dK) should reject null dQ");

            AssertThrows<ArgumentNullException>(
                () => rope.ApplyBackwardInPlace(matrix, null, 1),
                "ApplyBackwardInPlace(dQ, dK) should reject null dK");
        }

        private void Test_ApplyBackwardInPlace_InvalidHeadArgumentsThrow()
        {
            var rope = NewRotaryPositionEmbedding();

            AssertThrows<DivideByZeroException>(
                () => rope.ApplyBackwardInPlace(Matrix(2, 4), 0),
                "ApplyBackwardInPlace should reject zero heads");

            AssertThrows<ArgumentException>(
                () => rope.ApplyBackwardInPlace(Matrix(2, 5), 2),
                "ApplyBackwardInPlace should reject embedding dimensions not divisible by heads");

            AssertThrows<ArgumentException>(
                () => rope.ApplyBackwardInPlace(Matrix(2, 6), 2),
                "ApplyBackwardInPlace should reject odd per-head dimensions");
        }

        private void Test_ApplyBackwardInPlace_DQDKDimensionMismatchThrows()
        {
            var rope = NewRotaryPositionEmbedding();

            AssertThrows<ArgumentException>(
                () => rope.ApplyBackwardInPlace(Matrix(2, 4), Matrix(2, 6), 2),
                "ApplyBackwardInPlace(dQ, dK) should reject mismatched embedding dimensions");
        }

        private void Test_ApplyInPlace_PositionZeroUnchanged()
        {
            var rope = NewRotaryPositionEmbedding();

            var x = new float[,]
            {
                { 1f, 2f, 3f, 4f },
                { 5f, 6f, 7f, 8f }
            };

            rope.ApplyInPlace(x, numHeads: 1);

            AssertNearlyEqual(x[0, 0], 1f, "Position 0 col 0 changed");
            AssertNearlyEqual(x[0, 1], 2f, "Position 0 col 1 changed");
            AssertNearlyEqual(x[0, 2], 3f, "Position 0 col 2 changed");
            AssertNearlyEqual(x[0, 3], 4f, "Position 0 col 3 changed");
        }

        private void Test_ApplyInPlace_DeterministicKnownValues()
        {
            var rope = NewRotaryPositionEmbedding();

            var x = new float[,]
            {
                { 1f, 0f },
                { 1f, 0f },
                { 0f, 1f }
            };

            rope.ApplyInPlace(x, numHeads: 1);

            AssertNearlyEqual(x[0, 0], 1f, "Position 0 even mismatch");
            AssertNearlyEqual(x[0, 1], 0f, "Position 0 odd mismatch");

            AssertNearlyEqual(x[1, 0], MathF.Cos(1f), "Position 1 even mismatch");
            AssertNearlyEqual(x[1, 1], MathF.Sin(1f), "Position 1 odd mismatch");

            AssertNearlyEqual(x[2, 0], -MathF.Sin(2f), "Position 2 even mismatch");
            AssertNearlyEqual(x[2, 1], MathF.Cos(2f), "Position 2 odd mismatch");
        }

        private void Test_ApplyInPlace_MultipleHeadsRotatedIndependently()
        {
            var rope = NewRotaryPositionEmbedding();

            var x = new float[,]
            {
                { 1f, 2f, 3f, 4f },
                { 1f, 0f, 0f, 1f }
            };

            rope.ApplyInPlace(x, numHeads: 2);

            AssertNearlyEqual(x[1, 0], MathF.Cos(1f), "Head 0 even mismatch");
            AssertNearlyEqual(x[1, 1], MathF.Sin(1f), "Head 0 odd mismatch");

            AssertNearlyEqual(x[1, 2], -MathF.Sin(1f), "Head 1 even mismatch");
            AssertNearlyEqual(x[1, 3], MathF.Cos(1f), "Head 1 odd mismatch");
        }

        private void Test_ApplyInPlace_QAndKBothMutated()
        {
            var rope = NewRotaryPositionEmbedding();

            var q = new float[,]
            {
                { 1f, 0f },
                { 1f, 0f }
            };

            var k = new float[,]
            {
                { 0f, 1f },
                { 0f, 1f }
            };

            rope.ApplyInPlace(q, k, numHeads: 1);

            AssertNearlyEqual(q[1, 0], MathF.Cos(1f), "Q even mismatch");
            AssertNearlyEqual(q[1, 1], MathF.Sin(1f), "Q odd mismatch");

            AssertNearlyEqual(k[1, 0], -MathF.Sin(1f), "K even mismatch");
            AssertNearlyEqual(k[1, 1], MathF.Cos(1f), "K odd mismatch");
        }

        private void Test_ApplyBackwardInPlace_ReversesApplyInPlace()
        {
            var rope = NewRotaryPositionEmbedding();

            var x = new float[,]
            {
                { 1f, 2f, 3f, 4f },
                { 5f, 6f, 7f, 8f },
                { -1f, 0.5f, 2.5f, -3f }
            };

            var original = Clone(x);

            rope.ApplyInPlace(x, numHeads: 1);
            rope.ApplyBackwardInPlace(x, numHeads: 1);

            AssertMatrixNearlyEqual(x, original, "Backward pass should reverse forward RoPE");
        }

        private void Test_ApplyBackwardInPlace_QKReversesApplyInPlaceQK()
        {
            var rope = NewRotaryPositionEmbedding();

            var q = new float[,]
            {
                { 1f, 2f },
                { 3f, 4f }
            };

            var k = new float[,]
            {
                { 5f, 6f },
                { 7f, 8f }
            };

            var qOriginal = Clone(q);
            var kOriginal = Clone(k);

            rope.ApplyInPlace(q, k, numHeads: 1);
            rope.ApplyBackwardInPlace(q, k, numHeads: 1);

            AssertMatrixNearlyEqual(q, qOriginal, "Backward Q should reverse forward Q");
            AssertMatrixNearlyEqual(k, kOriginal, "Backward K should reverse forward K");
        }

        private void Test_ApplyInPlace_ZeroRowsDoesNotThrow()
        {
            var rope = NewRotaryPositionEmbedding();

            var x = new float[0, 4];

            rope.ApplyInPlace(x, numHeads: 2);

            Assert(x.GetLength(0) == 0, "Row count should remain zero");
            Assert(x.GetLength(1) == 4, "Column count should remain unchanged");
        }

        private void Test_ApplyInPlace_ZeroColumnsDoesNotThrowWhenValid()
        {
            var rope = NewRotaryPositionEmbedding();

            var x = new float[3, 0];

            rope.ApplyInPlace(x, numHeads: 1);

            Assert(x.GetLength(0) == 3, "Row count should remain unchanged");
            Assert(x.GetLength(1) == 0, "Column count should remain zero");
        }

        private void Test_ApplyInPlace_DoesNotChangeShape()
        {
            var rope = NewRotaryPositionEmbedding();

            var x = Matrix(7, 8);

            rope.ApplyInPlace(x, numHeads: 4);

            Assert(x.GetLength(0) == 7, "Row count changed");
            Assert(x.GetLength(1) == 8, "Column count changed");
        }

        private void Test_ParallelIndependentInstances_Deterministic()
        {
            var source = Matrix(64, 8);
            var expected = Clone(source);

            NewRotaryPositionEmbedding().ApplyInPlace(expected, numHeads: 2);

            var results = new float[32][,];

            Parallel.For(0, results.Length, i =>
            {
                var local = Clone(source);
                NewRotaryPositionEmbedding().ApplyInPlace(local, numHeads: 2);
                results[i] = local;
            });

            for (int i = 0; i < results.Length; i++)
            {
                AssertMatrixNearlyEqual(results[i], expected, $"Parallel result mismatch at index {i}");
            }
        }

        private static RotaryPositionEmbedding NewRotaryPositionEmbedding()
        {
            return new RotaryPositionEmbedding(new AccelerationCPU());
        }

        private static float[,] Matrix(int rows, int cols)
        {
            var matrix = new float[rows, cols];

            float value = 1f;
            for (int r = 0; r < rows; r++)
            {
                for (int c = 0; c < cols; c++)
                {
                    matrix[r, c] = value;
                    value += 1f;
                }
            }

            return matrix;
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

        private void AssertMatrixNearlyEqual(float[,] actual, float[,] expected, string message)
        {
            Assert(actual.GetLength(0) == expected.GetLength(0), $"{message}; row count mismatch");
            Assert(actual.GetLength(1) == expected.GetLength(1), $"{message}; column count mismatch");

            for (int r = 0; r < actual.GetLength(0); r++)
            {
                for (int c = 0; c < actual.GetLength(1); c++)
                {
                    AssertNearlyEqual(actual[r, c], expected[r, c], $"{message}; mismatch at [{r},{c}]");
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