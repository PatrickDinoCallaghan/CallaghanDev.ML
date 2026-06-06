using System;
using System.Collections.Generic;

namespace CallaghanDev.ML.Transformers
{
    /// <summary>
    /// Tracks rows of a very large gradient matrix that have been touched by the current
    /// batch. Token embedding gradients are sparse by construction; untouched rows are
    /// exactly zero, so full-matrix zero/norm/scale/update work can be skipped without
    /// changing the mathematical update.
    /// </summary>
    internal sealed class SparseRowGradientTracker
    {
        private readonly HashSet<int> _rows = new HashSet<int>();

        public int Count => _rows.Count;
        public IEnumerable<int> Rows => _rows;

        public void Mark(int row, int rowLimit)
        {
            if ((uint)row >= (uint)rowLimit)
            {
                throw new ArgumentOutOfRangeException(nameof(row), $"Sparse gradient row {row} is outside [0, {rowLimit}).");
            }

            _rows.Add(row);
        }

        public void MarkRows(int[] ids, int start, int count, int rowLimit)
        {
            if (ids == null)
            {
                throw new ArgumentNullException(nameof(ids));
            }
            if (start < 0 || count < 0 || start + count > ids.Length)
            {
                throw new ArgumentOutOfRangeException(nameof(start));
            }

            for (int i = 0; i < count; i++)
            {
                Mark(ids[start + i], rowLimit);
            }
        }

        public void MergeFrom(SparseRowGradientTracker other)
        {
            if (other == null)
            {
                return;
            }

            foreach (int row in other._rows)
            {
                _rows.Add(row);
            }
        }

        public void ZeroTrackedRowsAndClear(float[,] matrix)
        {
            if (matrix == null)
            {
                _rows.Clear();
                return;
            }

            int rows = matrix.GetLength(0);
            int cols = matrix.GetLength(1);
            foreach (int row in _rows)
            {
                if ((uint)row >= (uint)rows)
                {
                    continue;
                }

                for (int col = 0; col < cols; col++)
                {
                    matrix[row, col] = 0f;
                }
            }

            _rows.Clear();
        }

        public float SquaredNorm(float[,] matrix)
        {
            if (matrix == null)
            {
                return 0f;
            }

            int rows = matrix.GetLength(0);
            int cols = matrix.GetLength(1);
            float sum = 0f;

            foreach (int row in _rows)
            {
                if ((uint)row >= (uint)rows)
                {
                    continue;
                }

                for (int col = 0; col < cols; col++)
                {
                    float value = matrix[row, col];
                    sum += value * value;
                }
            }

            return sum;
        }

        public void Scale(float[,] matrix, float scale)
        {
            if (matrix == null)
            {
                return;
            }

            int rows = matrix.GetLength(0);
            int cols = matrix.GetLength(1);
            foreach (int row in _rows)
            {
                if ((uint)row >= (uint)rows)
                {
                    continue;
                }

                for (int col = 0; col < cols; col++)
                {
                    matrix[row, col] *= scale;
                }
            }
        }

        public void UpdateRows(float[,] weights, float[,] gradients, float learningRate)
        {
            if (weights == null)
            {
                throw new ArgumentNullException(nameof(weights));
            }
            if (gradients == null)
            {
                throw new ArgumentNullException(nameof(gradients));
            }
            if (weights.GetLength(0) != gradients.GetLength(0) || weights.GetLength(1) != gradients.GetLength(1))
            {
                throw new ArgumentException("Weights and gradients must have the same shape.", nameof(gradients));
            }

            int rows = weights.GetLength(0);
            int cols = weights.GetLength(1);
            foreach (int row in _rows)
            {
                if ((uint)row >= (uint)rows)
                {
                    continue;
                }

                for (int col = 0; col < cols; col++)
                {
                    weights[row, col] -= learningRate * gradients[row, col];
                }
            }
        }

        public void AddRowsTo(float[,] target, float[,] source)
        {
            if (target == null)
            {
                throw new ArgumentNullException(nameof(target));
            }
            if (source == null)
            {
                throw new ArgumentNullException(nameof(source));
            }
            if (target.GetLength(0) != source.GetLength(0) || target.GetLength(1) != source.GetLength(1))
            {
                throw new ArgumentException("Target and source must have the same shape.", nameof(source));
            }

            int rows = target.GetLength(0);
            int cols = target.GetLength(1);
            foreach (int row in _rows)
            {
                if ((uint)row >= (uint)rows)
                {
                    continue;
                }

                for (int col = 0; col < cols; col++)
                {
                    target[row, col] += source[row, col];
                }
            }
        }
    }
}
