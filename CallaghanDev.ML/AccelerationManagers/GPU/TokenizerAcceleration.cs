using System;
using System.Collections.Generic;

namespace CallaghanDev.ML.AccelerationManagers.GPU
{
    public partial class AccelerationGPU : IAccelerationManager, IDisposable
    {
        private void InitTokenizerAccelerationKernels()
        {
        }

        private void DisposeTokenizerAccelerationBuffers()
        {
        }

        #region Tokenizer Acceleration

        public string[] PreTokenize(string text)
        {
            return _mutliThreadCPU.PreTokenize(text);
        }

        public Dictionary<string, int> GetWordFrequencies(string[] texts, bool lowerCase)
        {
            return _mutliThreadCPU.GetWordFrequencies(texts, lowerCase);
        }

        public HashSet<string> BuildCharacterVocabulary(Dictionary<string, int> wordFreqs)
        {
            return _mutliThreadCPU.BuildCharacterVocabulary(wordFreqs);
        }

        public List<string> ApplyMerge(List<string> word, string left, string right)
        {
            return _mutliThreadCPU.ApplyMerge(word, left, right);
        }

        public List<int> EncodeWord(
            string word,
            Dictionary<(string, string), int> mergePriority,
            Dictionary<string, int> vocabToId,
            int unkTokenId)
        {
            return _mutliThreadCPU.EncodeWord(word, mergePriority, vocabToId, unkTokenId);
        }

        public Dictionary<(string left, string right), int> CountPairFrequencies(Dictionary<List<string>, int> words)
        {
            return _mutliThreadCPU.CountPairFrequencies(words);
        }

        public ((string left, string right) pair, int frequency) SelectBestPair(
            Dictionary<(string left, string right), int> pairCounts,
            int minFrequency)
        {
            return _mutliThreadCPU.SelectBestPair(pairCounts, minFrequency);
        }

        public Dictionary<List<string>, int> ApplyMergeToVocabulary(
            Dictionary<List<string>, int> words,
            string left,
            string right)
        {
            return _mutliThreadCPU.ApplyMergeToVocabulary(words, left, right);
        }

        public string DecodeTokens(
            int[] tokenIds,
            Dictionary<int, string> idToVocab,
            string unkToken,
            bool skipSpecialTokens)
        {
            return _mutliThreadCPU.DecodeTokens(tokenIds, idToVocab, unkToken, skipSpecialTokens);
        }

        public int[] PadOrTruncate(
            int[] tokenIds,
            int maxLength,
            bool addSpecialTokens,
            int padTokenId,
            int endTokenId)
        {
            return _mutliThreadCPU.PadOrTruncate(tokenIds, maxLength, addSpecialTokens, padTokenId, endTokenId);
        }

        #endregion
    }
}
