using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace CallaghanDev.ML.AutoTuning
{
    public class DataChunkManager
    {
        private readonly List<DataChunk> _chunks = new List<DataChunk>();
        private int _currentChunkIndex = 0;
        private readonly Random _random = new Random();

        public class DataChunk
        {
            public float[][] Inputs { get; set; }
            public float[][] Outputs { get; set; }
            public float AveragePerformance { get; set; } = float.MaxValue;
            public int TrainingAttempts { get; set; } = 0;
            public DateTime LastUsed { get; set; } = DateTime.Now;
        }

        public void AddChunk(float[][] inputs, float[][] outputs)
        {
            if (inputs.Length != outputs.Length)
            {
                throw new ArgumentException("Input and output arrays must have the same length");
            }
            _chunks.Add(new DataChunk
            {
                Inputs = inputs,
                Outputs = outputs
            });
        }

        public void ClearChunks()
        {
            _chunks.Clear();
            _currentChunkIndex = 0;
        }

        public DataChunk GetNextChunk()
        {
            if (_chunks.Count == 0)
            {
                throw new InvalidOperationException("No chunks available");
            }

            var chunk = _chunks[_currentChunkIndex];
            chunk.LastUsed = DateTime.Now;
            _currentChunkIndex = (_currentChunkIndex + 1) % _chunks.Count;
            return chunk;
        }

        public DataChunk GetRandomChunk()
        {
            if (_chunks.Count == 0)
            {
                throw new InvalidOperationException("No chunks available");
            }

            int index = _random.Next(_chunks.Count);
            var chunk = _chunks[index];
            chunk.LastUsed = DateTime.Now;
            return chunk;
        }

        public void UpdateChunkPerformance(int chunkIndex, float performance)
        {
            if (chunkIndex >= 0 && chunkIndex < _chunks.Count)
            {
                var chunk = _chunks[chunkIndex];
                chunk.TrainingAttempts++;

                // Simple moving average
                if (chunk.AveragePerformance == float.MaxValue)
                {
                    chunk.AveragePerformance = performance;
                }
                else
                {
                    chunk.AveragePerformance = (chunk.AveragePerformance + performance) / 2f;
                }
            }
        }

        public (float[][] inputs, float[][] outputs) GetCombinedValidationSet(float validationRatio = 0.15f)
        {
            if (_chunks.Count == 0)
            {
                throw new InvalidOperationException("No chunks available");
            }
            var allInputs = new List<float[]>();
            var allOutputs = new List<float[]>();

            foreach (var chunk in _chunks)
            {
                int validationSize = Math.Max(1, (int)(chunk.Inputs.Length * validationRatio));
                var indices = Enumerable.Range(0, chunk.Inputs.Length).OrderBy(_ => _random.Next()).Take(validationSize);

                foreach (int i in indices)
                {
                    allInputs.Add(chunk.Inputs[i]);
                    allOutputs.Add(chunk.Outputs[i]);
                }
            }

            return (allInputs.ToArray(), allOutputs.ToArray());
        }

        public int ChunkCount => _chunks.Count;
        public int TotalSamples => _chunks.Sum(c => c.Inputs.Length);
        public float GetAverageChunkPerformance() => _chunks.Count > 0 ? _chunks.Where(c => c.AveragePerformance != float.MaxValue).DefaultIfEmpty(new DataChunk { AveragePerformance = float.MaxValue }).Average(c => c.AveragePerformance) : float.MaxValue;
    }
}
