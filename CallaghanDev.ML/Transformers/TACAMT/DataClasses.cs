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

    /// <summary>
    /// Configuration for attention-score-based memory pruning.
    /// Instead of dropping the oldest entries first, entries are pruned based on
    /// how much attention recent queries assign to them. An entry that no recent
    /// query attends to is effectively dead information, regardless of its age.
    /// </summary>
    public class MemoryPruningConfig
    {
        /// <summary>
        /// EMA decay factor for updating attention scores. 
        /// score = alpha * new_score + (1 - alpha) * old_score
        /// Lower alpha = longer memory of past attention. Default: 0.1
        /// </summary>
        public float AttentionScoreAlpha { get; set; } = 0.1f;

        /// <summary>
        /// Minimum number of query exposures before an entry is eligible for
        /// attention-based pruning. Before this threshold, entries are treated
        /// as having maximum importance (cold-start protection). Default: 3
        /// </summary>
        public int MinQueryCountForPruning { get; set; } = 3;

        /// <summary>
        /// When pruning, keep this fraction of slots reserved for newest entries
        /// regardless of attention score. Prevents the pathological case where
        /// new entries get immediately pruned before they've been attended to.
        /// Default: 0.1 (10% of max capacity reserved for recent entries)
        /// </summary>
        public float NewEntryReserveFraction { get; set; } = 0.1f;

        /// <summary>
        /// Whether to use attention-based pruning (true) or simple oldest-first (false).
        /// Default: true
        /// </summary>
        public bool UseAttentionBasedPruning { get; set; } = true;
    }
}
