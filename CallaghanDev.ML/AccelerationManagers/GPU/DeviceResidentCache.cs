using ILGPU;
using ILGPU.Runtime;
using System;
using System.Collections.Generic;

namespace CallaghanDev.ML.AccelerationManagers.GPU
{
    public partial class AccelerationGPU : IAccelerationManager, IDisposable
    {
        private sealed class ResidentMatrix2D
        {
            public readonly int Rows;
            public readonly int Cols;
            public readonly MemoryBuffer2D<float, Stride2D.DenseX> Buffer;
            public readonly float[] Shadow;

            public ResidentMatrix2D(Accelerator accelerator, int rows, int cols)
            {
                Rows = rows;
                Cols = cols;
                Buffer = accelerator.Allocate2DDenseX<float>(new Index2D(rows, cols));
                Shadow = new float[rows * cols];
            }
        }

        private sealed class ResidentVector1D
        {
            public readonly int Length;
            public readonly MemoryBuffer1D<float, Stride1D.Dense> Buffer;
            public readonly float[] Shadow;

            public ResidentVector1D(Accelerator accelerator, int length)
            {
                Length = length;
                Buffer = accelerator.Allocate1D<float>(length);
                Shadow = new float[length];
            }
        }

        private readonly object _residentCacheLock = new object();
        private readonly Dictionary<float[,], ResidentMatrix2D> _residentMatrices = new Dictionary<float[,], ResidentMatrix2D>();
        private readonly Dictionary<float[], ResidentVector1D> _residentVectors = new Dictionary<float[], ResidentVector1D>();

        private MemoryBuffer2D<float, Stride2D.DenseX> GetResidentMatrixReadOnly(float[,] matrix)
        {
            if (matrix == null) throw new ArgumentNullException(nameof(matrix));

            int rows = matrix.GetLength(0);
            int cols = matrix.GetLength(1);

            lock (_residentCacheLock)
            {
                if (!_residentMatrices.TryGetValue(matrix, out var resident) || resident.Rows != rows || resident.Cols != cols)
                {
                    resident?.Buffer.Dispose();
                    resident = new ResidentMatrix2D(_accelerator, rows, cols);
                    _residentMatrices[matrix] = resident;
                    CopyMatrixToShadow(matrix, resident.Shadow);
                    resident.Buffer.CopyFromCPU(matrix);
                    return resident.Buffer;
                }

                if (!MatrixMatchesShadow(matrix, resident.Shadow))
                {
                    CopyMatrixToShadow(matrix, resident.Shadow);
                    resident.Buffer.CopyFromCPU(matrix);
                }

                return resident.Buffer;
            }
        }

        private MemoryBuffer1D<float, Stride1D.Dense> GetResidentVectorReadOnly(float[] vector)
        {
            if (vector == null) throw new ArgumentNullException(nameof(vector));

            lock (_residentCacheLock)
            {
                if (!_residentVectors.TryGetValue(vector, out var resident) || resident.Length != vector.Length)
                {
                    resident?.Buffer.Dispose();
                    resident = new ResidentVector1D(_accelerator, vector.Length);
                    _residentVectors[vector] = resident;
                    Array.Copy(vector, resident.Shadow, vector.Length);
                    resident.Buffer.CopyFromCPU(vector);
                    return resident.Buffer;
                }

                if (!VectorMatchesShadow(vector, resident.Shadow))
                {
                    Array.Copy(vector, resident.Shadow, vector.Length);
                    resident.Buffer.CopyFromCPU(vector);
                }

                return resident.Buffer;
            }
        }

        private static bool MatrixMatchesShadow(float[,] matrix, float[] shadow)
        {
            int rows = matrix.GetLength(0);
            int cols = matrix.GetLength(1);
            if (shadow.Length != rows * cols)
            {
                return false;
            }

            int index = 0;
            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < cols; j++, index++)
                {
                    if (!matrix[i, j].Equals(shadow[index]))
                    {
                        return false;
                    }
                }
            }
            return true;
        }

        private static void CopyMatrixToShadow(float[,] matrix, float[] shadow)
        {
            int rows = matrix.GetLength(0);
            int cols = matrix.GetLength(1);
            int index = 0;
            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < cols; j++, index++)
                {
                    shadow[index] = matrix[i, j];
                }
            }
        }

        private static bool VectorMatchesShadow(float[] vector, float[] shadow)
        {
            if (shadow.Length != vector.Length)
            {
                return false;
            }

            for (int i = 0; i < vector.Length; i++)
            {
                if (!vector[i].Equals(shadow[i]))
                {
                    return false;
                }
            }
            return true;
        }

        private void DisposeDeviceResidentCaches()
        {
            lock (_residentCacheLock)
            {
                foreach (var resident in _residentMatrices.Values)
                {
                    resident.Buffer.Dispose();
                }
                _residentMatrices.Clear();

                foreach (var resident in _residentVectors.Values)
                {
                    resident.Buffer.Dispose();
                }
                _residentVectors.Clear();
            }
        }
    }
}
