using System;

namespace CallaghanDev.ML.Transformers.TACAMT
{
    public class NewsStory
    {
        public int[] TokenIds { get; set; }
        public float ArrivalTime { get; set; }
        public NewsStory() { }
        public NewsStory(int[] tokenIds, float arrivalTime) { TokenIds = tokenIds; ArrivalTime = arrivalTime; }
    }

    public class NewsMemoryEntry
    {
        public float[] HiddenState { get; set; }
        public double AbsoluteTimestamp { get; set; }

        /// <summary>
        /// Running exponential moving average of attention weight this entry receives
        /// from recent queries. Used for attention-based memory pruning.
        /// Entries with consistently low attention are effectively dead and can be pruned
        /// even if they are newer than entries that still get attended to.
        /// </summary>
        public float AttentionScore { get; set; } = 1.0f;

        /// <summary>
        /// Number of times this entry has been queried (used for EMA warmup).
        /// </summary>
        public int QueryCount { get; set; } = 0;
    }

    public class PriceMemoryEntry
    {
        public float[] HiddenState { get; set; }
        public double AbsoluteTimestamp { get; set; }

        /// <summary>
        /// Running exponential moving average of attention weight this entry receives.
        /// See NewsMemoryEntry.AttentionScore for explanation.
        /// </summary>
        public float AttentionScore { get; set; } = 1.0f;

        /// <summary>
        /// Number of times this entry has been queried (used for EMA warmup).
        /// </summary>
        public int QueryCount { get; set; } = 0;
    }

}
