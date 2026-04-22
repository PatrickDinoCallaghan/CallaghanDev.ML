namespace CallaghanDev.ML.TestConsoleApp.Tests
{
    internal abstract class TestBase
    {
        protected int _passed, _failed;
        protected readonly List<string> _failures = new();
        protected static int CountNumber = 0;
        protected void Assert(bool cond, string msg)
        {
            if (!cond)
            {
                throw new Exception(msg);
            }
        }

        protected void AssertLossImproved(float before, float after, float minImprovementRatio = 0.90f)
        {
            Assert(after < before * minImprovementRatio, $"Weak improvement: {before:F6} → {after:F6} (expected < {before * minImprovementRatio:F6})");
        }

        protected void AssertLossImprovedByAbsolute(float before, float after, float minAbsoluteDrop)
        {
            Assert(after <= before - minAbsoluteDrop, $"Weak absolute improvement: {before:F6} → {after:F6} (expected drop ≥ {minAbsoluteDrop:F6})");
        }

        protected void AssertOverfitStrong(float loss, float threshold = 0.10f)
        {
            Assert(loss < threshold, $"Failed to strongly overfit: loss={loss:F6} (expected < {threshold:F6})");
        }

        protected void AssertFinite(float value, string name)
        {
            Assert(!float.IsNaN(value) && !float.IsInfinity(value), $"{name} is not finite: {value}");
        }

        protected void AssertBetweenInclusive(float value, float lo, float hi, string name)
        {
            Assert(value >= lo && value <= hi, $"{name}={value} outside [{lo}, {hi}]");
        }

        protected static bool MatrixChanged(float[,] a, float[,] b, float tol = 1e-10f)
        {
            for (int i = 0; i < a.GetLength(0); i++)
            {
                for (int j = 0; j < a.GetLength(1); j++)
                {
                    if (MathF.Abs(a[i, j] - b[i, j]) > tol)
                    {
                        return true;
                    }
                }
            }
            return false;
        }

        protected static bool VectorChanged(float[] a, float[] b, float tol = 1e-10f)
        {
            for (int i = 0; i < a.Length; i++)
            {
                if (MathF.Abs(a[i] - b[i]) > tol)
                {
                    return true;
                }
            }
            return false;
        }

        protected static bool MatrixEquals(float[,] a, float[,] b, float tol = 1e-10f)
        {
            if (a.GetLength(0) != b.GetLength(0) || a.GetLength(1) != b.GetLength(1))
                return false;

            for (int i = 0; i < a.GetLength(0); i++)
                for (int j = 0; j < a.GetLength(1); j++)
                    if (MathF.Abs(a[i, j] - b[i, j]) > tol) return false;

            return true;
        }

        protected static bool VectorEquals(float[] a, float[] b, float tol = 1e-10f)
        {
            if (a.Length != b.Length) return false;

            for (int i = 0; i < a.Length; i++)
                if (MathF.Abs(a[i] - b[i]) > tol) return false;

            return true;
        }

        protected static bool HasNaN(float[,] m)
        {
            for (int i = 0; i < m.GetLength(0); i++)
                for (int j = 0; j < m.GetLength(1); j++)
                    if (float.IsNaN(m[i, j]) || float.IsInfinity(m[i, j])) return true;
            return false;
        }

        protected static bool HasNaN(float[] v)
        {
            foreach (var f in v)
                if (float.IsNaN(f) || float.IsInfinity(f)) return true;
            return false;
        }

        protected static float MaxAbs(float[,] m)
        {
            float max = 0f;
            for (int i = 0; i < m.GetLength(0); i++)
                for (int j = 0; j < m.GetLength(1); j++)
                    max = Math.Max(max, MathF.Abs(m[i, j]));
            return max;
        }

        protected static float MaxAbs(float[] v)
        {
            float max = 0f;
            for (int i = 0; i < v.Length; i++)
                max = Math.Max(max, MathF.Abs(v[i]));
            return max;
        }

        protected static float MeanAbs(float[,] m)
        {
            float sum = 0f;
            int n = 0;
            for (int i = 0; i < m.GetLength(0); i++)
                for (int j = 0; j < m.GetLength(1); j++)
                {
                    sum += MathF.Abs(m[i, j]);
                    n++;
                }

            return n == 0 ? 0f : sum / n;
        }

        protected static float[,] RandMatrix(int rows, int cols, Random rng, float scale = 0.5f)
        {
            var m = new float[rows, cols];
            for (int i = 0; i < rows; i++)
                for (int j = 0; j < cols; j++)
                    m[i, j] = ((float)rng.NextDouble() - 0.5f) * 2f * scale;
            return m;
        }

        protected static float[,] SliceRows(float[,] m, int start, int end)
        {
            int cols = m.GetLength(1), rows = end - start;
            var r = new float[rows, cols];
            for (int i = 0; i < rows; i++)
                for (int j = 0; j < cols; j++)
                    r[i, j] = m[start + i, j];
            return r;
        }

        protected string TempDir()
        {
            var d = Path.Combine(Path.GetTempPath(), $"tf_test_{Guid.NewGuid():N}");
            Directory.CreateDirectory(d);
            return d;
        }

        protected void Cleanup(string dir)
        {
            try { if (Directory.Exists(dir)) Directory.Delete(dir, true); } catch { }
        }

        protected void Run((Action test, string name)[] tests, string suiteName)
        {
            TransformerTestSuite.PrintBanner(suiteName);
            _passed = _failed = 0;
            _failures.Clear();

            for (int i = 0; i < tests.Length; i++)
            {
                Console.Write($"  [{i + 1,3}/{tests.Length}] {tests[i].name,-62} ");
                try
                {
                    tests[i].test();
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
                    Console.WriteLine($"         ↳ {ex.Message}");
                    Console.ResetColor();
                    _failures.Add($"{tests[i].name}: {ex.Message}");
                    _failed++;
                }
            }

            Console.WriteLine($"\n  {"",3}{new string('-', 68)}");
            Console.Write("  Results: ");
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

            Console.WriteLine($" / {tests.Length} total");

            if (_failures.Count > 0)
            {
                Console.ForegroundColor = ConsoleColor.Red;
                Console.WriteLine("\n  Failed tests:");
                Console.ResetColor();
                foreach (var f in _failures)
                    Console.WriteLine($"    • {f}");
            }

            Console.WriteLine();
        }
    }

}
