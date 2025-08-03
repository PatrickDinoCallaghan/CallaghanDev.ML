using CallaghanDev.ML.Extensions;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace CallaghanDev.ML.AutoTuning
{
    public class ArchitectureCandidate
    {
        public Parameters Parameters { get; set; }
        public float Score { get; set; }
        public int Generation { get; set; }
        public string Origin { get; set; }
        public long ParameterCount { get; set; }
        public DateTime CreatedAt { get; set; }
        public float StabilityScore { get; set; }

        public ArchitectureCandidate(Parameters parameters, float score, int generation, string origin, float stabilityScore = 1f)
        {
            Parameters = parameters;
            Score = score;
            Generation = generation;
            Origin = origin;
            StabilityScore = stabilityScore;
            ParameterCount = parameters.CalculateParameterCount();
            CreatedAt = DateTime.Now;
        }
    }
}
