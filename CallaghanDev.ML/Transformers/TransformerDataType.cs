using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace CallaghanDev.ML.Transformers
{   
    public enum TransformerDataType
    {
        /// <summary>
        /// What we currently have => Autoregressive text generation (GPT-style).
        /// Input: discrete token IDs via learned embedding lookup.
        /// Output: next-token probability distribution over vocabulary.
        /// Loss: cross-entropy.
        /// </summary>
        Text,

        /// <summary>
        /// Time series regression (e.g. price prediction.(If only) ).
        /// Input: continuous feature vectors per timestep via linear projection.
        /// Output: predicted continuous values for next timestep(s).
        /// Loss: MSE or Huber.
        /// </summary>
        TimeSeriesRegression,

        /// <summary>
        /// Time series classification (e.g. buy/hold/sell signals).
        /// Input: continuous feature vectors per timestep via linear projection.
        /// Output: class probabilities for next timestep.
        /// Loss: cross-entropy.
        /// </summary>
        TimeSeriesClassification,

        /// <summary>
        /// Symbolic sequence prediction (e.g. DNA, protein, MIDI, code tokens).
        /// Input: discrete token IDs via learned embedding lookup.
        /// Output: next-token probability distribution over vocabulary.
        /// Loss: cross-entropy.
        /// </summary>
        SymbolicSequence
    }
}
