using CallaghanDev.ML.AutoTuning;
using CallaghanDev.ML.Enums;
using CallaghanDev.ML.Extensions;
using CallaghanDev.Utilities;
using Newtonsoft.Json;
namespace CallaghanDev.ML
{
    /*
    This Neural Network Auto-Tuning System is attmpting to automaticall automatically discover the optimal neural network architectures and hyperparameters for any given dataset.
    Primary Objectives:

    Automatic Architecture Discovery: Dynamically searches for the best network structure (depth, width, activation functions) without manual tuning
    Generalization: Works with any input/output dataset, automatically adapting to the problem complexity
    Parameter-Constrained Optimization: Respects user-defined maximum parameter limits to control model size
    Large Dataset Handling: Supports chunked data processing for datasets too large to fit in memory
    Continuous Learning: Can continue training with new data while preserving previous learning

    Core Process:

    Intelligent Exploration: Uses multiple mutation strategies (Explore, Exploit, Conservative, Aggressive, Balanced, Adaptive) to generate candidate architectures
    Real-time Diagnostics: Monitors training for overfitting, plateau, dead neurons, output collapse, and instability
    Dynamic Adaptation: Adjusts architecture and hyperparameters during training based on performance signals
    Rollback Mechanism: Saves best states and can revert if performance degrades
    Multi-chunk Training: For large datasets, trains on different data chunks and validates globally

    Key Variables & Settings:

    maxParameters: Hard limit on network size (weights + biases)
    targetLossThreshold: Training stops when validation loss ≤ this value
    maxAttempts: Maximum architecture attempts before giving up
    validationSplit: Percentage of data reserved for validation (default 30%)
    learningRate: Base learning rate (adaptively adjusted during training)
    maxChunkTrainingAttempts: For multi-chunk datasets, how many chunks to train per attempt
    */
    public class NeuralAutoTuner
    {
        private float bestLoss = float.MaxValue;
        private Parameters bestParams;
        private Data bestData;
        private readonly Random random = new Random();
        private readonly DataChunkManager chunkManager = new DataChunkManager();

        private ILogger logger;
        private float mutationRate = 0.1f;
        private int stagnationCounter = 0;
        private int generation = 0;
        private long maxParameters = long.MaxValue;
        private float globalBestLoss = float.MaxValue;
        private Parameters globalBestParams;
        private List<ArchitectureCandidate> architectureHistory = new List<ArchitectureCandidate>();

        private int consecutiveChunkFailures = 0;

        public NeuralAutoTuner(ILogger Logger = null)
        {
            logger = Logger;


        }

        /// <summary>
        /// Executes a single training attempt for a neural network architecture, with adaptive learning rates, early stopping,
        /// diagnostic monitoring, and dynamic architecture adjustments.
        /// </summary>
        /// <param name="inputs">
        /// Training input data as a 2D array where each inner array represents one training sample's feature vector.
        /// Used for the primary training loop where the network learns from input-output mappings.
        /// </param>
        /// <param name="expected">
        /// Training target data as a 2D array where each inner array represents the expected output for the corresponding
        /// training input. Must have the same length as trainX for proper input-output pairing.
        /// </param>
        /// <param name="valX">
        /// Local validation input data used for monitoring training progress and detecting overfitting on this specific
        /// data chunk. Separate from training data to provide unbiased performance estimates during training.
        /// </param>
        /// <param name="valY">
        /// Local validation target data corresponding to valX. Used to calculate validation loss for this specific
        /// training chunk and trigger early stopping decisions.
        /// </param>
        /// <param name="globalValX">
        /// Global validation input data representing validation samples from all data chunks combined. Used for
        /// cross-chunk validation to ensure the model performs well across the entire dataset, not just the current chunk.
        /// </param>
        /// <param name="globalValY">
        /// Global validation target data corresponding to globalValX. Critical for multi-chunk training scenarios
        /// where local performance might not reflect overall model quality.
        /// </param>
        /// <param name="parameters">
        /// Neural network architecture parameters defining the network structure (layer widths, activation functions)
        /// and hyperparameters (regularization, gradient clipping). This defines the model to be trained.
        /// </param>
        /// <param name="learningRate">
        /// Base learning rate for gradient descent optimization. The method applies adaptive scheduling including
        /// warm-up (first 20 epochs) and decay strategies based on training progress.
        /// </param>
        /// <param name="targetLossThreshold">
        /// Success threshold for training completion. If global validation loss drops to or below this value,
        /// training is considered successful and terminates early with a positive result.
        /// </param>
        /// <param name="attempt">
        /// Current attempt number in the broader auto-tuning process. Used for logging and potential attempt-specific
        /// optimizations, though not directly used in current implementation.
        /// </param>
        /// <returns>
        /// A tuple containing:
        /// - success (bool): True if training achieved the target loss threshold, false otherwise
        /// - bestLoss (float): The lowest global validation loss achieved during training
        /// - bestParams (Parameters): Network parameters corresponding to the best performing model state
        /// - bestLR (float): Learning rate that produced the best results
        /// </returns>
        /// <remarks>
        /// <para><strong>Critical Internal Variables:</strong></para>
        /// <list type="bullet">
        /// <item><description><strong>adaptiveLR</strong>: Dynamically adjusted learning rate starting with warm-up (0.1x to 1x over 20 epochs), then applying 0.98x decay when no improvement detected. Minimum bound at 1% of base learning rate.</description></item>
        /// <item><description><strong>noImproveCount</strong>: Counter tracking epochs without improvement (improvement threshold: 1e-5). Triggers learning rate decay at 5+ and early stopping at patience limit.</description></item>
        /// <item><description><strong>earlyStopPatience</strong>: Calculated as max(15, sum(layer_widths)/50). Larger networks get more patience. Prevents premature stopping on complex architectures.</description></item>
        /// <item><description><strong>maxEpochs</strong>: Hard limit of 3000 training epochs to prevent infinite training loops.</description></item>
        /// <item><description><strong>minImprovement</strong>: Minimum validation loss reduction (1e-5) required to count as improvement. Prevents noise from triggering false improvements.</description></item>
        /// <item><description><strong>bestValLoss</strong>: Tracks the lowest global validation loss seen during training. Used for model state preservation and improvement detection.</description></item>
        /// <item><description><strong>epochSuccess</strong>: Boolean flag indicating whether any meaningful improvement occurred during training. Determines overall attempt success.</description></item>
        /// </list>
        /// <para><strong>Key Training Features:</strong></para>
        /// <list type="bullet">
        /// <item><description><strong>Adaptive Learning Rate</strong>: Implements warm-up phase (epochs 1-20) followed by decay-on-plateau strategy</description></item>
        /// <item><description><strong>Comprehensive Diagnostics</strong>: Monitors for divergence, output collapse, instability, overfitting, and dead neurons</description></item>
        /// <item><description><strong>Dynamic Architecture Adjustment</strong>: Can modify network structure mid-training when dead neurons detected (every 50 epochs)</description></item>
        /// <item><description><strong>Dual Validation</strong>: Uses both local chunk validation and global cross-chunk validation for robust performance assessment</description></item>
        /// <item><description><strong>Memory Management</strong>: Periodic cleanup of diagnostic data every 100 epochs to prevent memory bloat</description></item>
        /// <item><description><strong>Intelligent Early Stopping</strong>: Multiple stopping criteria including target achievement, divergence, collapse, instability, and patience exhaustion</description></item>
        /// </list>
        /// <para><strong>Training Flow:</strong></para>
        /// <list type="number">
        /// <item><description>Initialize network and diagnostics with provided parameters</description></item>
        /// <item><description>For each epoch (up to 3000): apply adaptive learning rate scheduling</description></item>
        /// <item><description>Execute training epoch with shuffled data and calculate training loss</description></item>
        /// <item><description>Perform local and global validation to assess generalization</description></item>
        /// <item><description>Update best model state if improvement detected (>1e-5 reduction)</description></item>
        /// <item><description>Check for early success (target threshold achieved)</description></item>
        /// <item><description>Evaluate stopping criteria (divergence, collapse, overfitting, patience)</description></item>
        /// <item><description>Apply mid-training architecture adjustments if dead neurons detected</description></item>
        /// <item><description>Continue until success, stopping condition, or epoch limit reached</description></item>
        /// </list>
        /// <para><strong>Performance Considerations:</strong></para>
        /// <list type="bullet">
        /// <item><description>Training time scales with network size, data size, and convergence difficulty</description></item>
        /// <item><description>Memory usage grows with network parameters and diagnostic history</description></item>
        /// <item><description>Global validation adds computational overhead but ensures cross-chunk generalization</description></item>
        /// <item><description>Mid-training architecture changes can extend training time but improve final performance</description></item>
        /// </list>
        /// </remarks>
        /// <example>
        /// <code>
        /// var tuner = new NeuralAutoTuner();
        /// var parameters = new Parameters { LayerWidths = new List&lt;int&gt; {10, 64, 32, 1} };
        /// 
        /// var result = tuner.TrainSingleAttempt(
        /// trainInputs, trainTargets,   // Local training data
        /// localValInputs, localValTargets, // Local validation data  
        /// globalValInputs, globalValTargets,   // Global validation data
        /// parameters,  // Network architecture
        /// learningRate: 0.001f,   // Base learning rate
        /// targetLossThreshold: 0.01f,  // Success threshold
        /// attempt: 1); // Attempt number
        /// 
        /// if (result.success) {
        ///  logger?.Info($"Training successful! Best loss: {result.bestLoss:F6}");
        /// // Use result.bestParams for the optimized network
        /// }
        /// </code>
        /// </example>
        public (Parameters BestParams, float LearningRate) TrainWithAutoTuning(float[][] inputs, float[][] expected, float learningRate, Parameters initialParams, int maxAttempts = 50, float validationSplit = 0.5f, float targetLossThreshold = 0.25f, long maxParameters = long.MaxValue, int maxChunkTrainingAttempts = 7)
        {
            SetMaxParameters(maxParameters);

            if (chunkManager.ChunkCount == 0)
            {
                AddDataChunk(inputs, expected);
            }

            logger?.Info($"=== Enhanced Auto-Tuning Started ===");
            logger?.Info($"Target Loss: ≤ {targetLossThreshold:F4}");
            logger?.Info($"Max Parameters: {maxParameters:N0}");
            logger?.Info($"Data Chunks: {chunkManager.ChunkCount}");
            logger?.Info($"Total Samples: {chunkManager.TotalSamples:N0}");

            // We need to split the data set into validation and training sets. This is so we can perform various error checks against validation data
            var (trainX, trainY, valX, valY) = SplitDataset(inputs, expected, validationSplit);

            float[][] globalValX, globalValY;
            if (chunkManager.ChunkCount > 1)
            {
                (globalValX, globalValY) = chunkManager.GetCombinedValidationSet(validationSplit);
                logger?.Info($"Global validation set: {globalValX.Length} samples");
            }
            else
            {
                globalValX = valX;
                globalValY = valY;
            }

            Parameters parameters = initialParams.Clone();
            Parameters bestParams = parameters.Clone();
            float bestLoss = float.MaxValue;
            float bestLR = learningRate;
            string bestStructure = string.Empty;
            int consecutiveFailures = 0;
            const int maxConsecutiveFailures = 3;

            if (CalculateParameterCount(parameters.LayerWidths) > maxParameters)
            {
                parameters = ReduceNetworkSize(parameters);
                logger?.Info($"Reduced initial architecture to fit parameter limit: {string.Join(" -> ", parameters.LayerWidths)}");
            }

            for (int attempt = 1; attempt <= maxAttempts; attempt++)
            {
                logger?.Info($"\n=== Attempt {attempt}/{maxAttempts} ===");
                logger?.Info($"Architecture: {string.Join(" -> ", parameters.LayerWidths)}");
                logger?.Info($"Parameters: {CalculateParameterCount(parameters.LayerWidths):N0} / {maxParameters:N0}");
                logger?.Info($"Target Loss: ≤ {targetLossThreshold:F4}");

                bool attemptSuccess = false;
                float attemptBestLoss = float.MaxValue;
                Parameters attemptBestParams = null;
                float attemptBestLR = learningRate;

                int chunksToTrain = Math.Min(maxChunkTrainingAttempts, chunkManager.ChunkCount);
                for (int chunkAttempt = 0; chunkAttempt < chunksToTrain; chunkAttempt++)
                {
                    float chunkLoss;
                    if (chunkManager.ChunkCount > 1)
                    {
                        var chunk = chunkManager.GetNextChunk();
                        logger?.Info($"  Training on chunk {chunkAttempt + 1}: {chunk.Inputs.Length} samples");
                        var (chunkTrainX, chunkTrainY, chunkValX, chunkValY) = SplitDataset(chunk.Inputs, chunk.Outputs, validationSplit);

                        var result = TrainSingleAttempt(chunkTrainX, chunkTrainY, chunkValX, chunkValY, globalValX, globalValY, parameters, learningRate, targetLossThreshold, attempt);
                        chunkLoss = result.bestLoss;
                        if (result.success && chunkLoss < attemptBestLoss)
                        {
                            attemptBestLoss = chunkLoss;
                            attemptBestParams = result.bestParams;
                            attemptBestLR = result.bestLR;
                            attemptSuccess = true;
                        }

                        chunkManager.UpdateChunkPerformance(chunkAttempt % chunkManager.ChunkCount, chunkLoss);


                        if (chunkLoss <= targetLossThreshold)
                        {
                            logger?.Info($"Target achieved on chunk {chunkAttempt + 1}!");
                            return (result.bestParams, result.bestLR);
                        }
                    }
                    else
                    {
                        var result = TrainSingleAttempt(trainX, trainY, valX, valY, globalValX, globalValY,
                          parameters, learningRate, targetLossThreshold, attempt);
                        chunkLoss = result.bestLoss;
                        if (result.success)
                        {
                            attemptBestLoss = chunkLoss;
                            attemptBestParams = result.bestParams;
                            attemptBestLR = result.bestLR;
                            attemptSuccess = true;

                            if (chunkLoss <= targetLossThreshold)
                            {
                                logger?.Info($"Target achieved!");
                                return (result.bestParams, result.bestLR);
                            }
                        }
                        break;
                    }
                }
                if (attemptSuccess && attemptBestLoss < bestLoss)
                {
                    bestLoss = attemptBestLoss;
                    bestParams = attemptBestParams;
                    bestLR = attemptBestLR;
                    bestStructure = new NeuralNetwork(bestParams).GetNetworkStructure();
                    consecutiveFailures = 0;

                    logger?.Info($"Attempt {attempt} success! Best Loss: {bestLoss:F6}");

                    if (bestLoss <= targetLossThreshold)
                    {
                        logger?.Info($"Global target achieved!");
                        return (bestParams, bestLR);
                    }
                }
                else
                {
                    consecutiveFailures++;
                    consecutiveChunkFailures++;
                    logger?.Info($"Attempt {attempt} failed. Consecutive failures: {consecutiveFailures}");
                }

                TuningStrategy strategy = SelectStrategy(attempt, consecutiveFailures, new TrainingDiagnostics(), bestLoss, targetLossThreshold);

                if (attempt < maxAttempts)
                {
                    parameters = GenerateNextArchitecture(this, parameters, strategy, attempt, new TrainingDiagnostics());

                    if (CalculateParameterCount(parameters.LayerWidths) > maxParameters)
                    {
                        parameters = ReduceNetworkSize(parameters);
                    }
                }

                if (consecutiveFailures >= maxConsecutiveFailures)
                {
                    logger?.Info("Triggering aggressive exploration due to consecutive failures");
                    parameters = Mutate(initialParams, TuningStrategy.Aggressive, attempt);
                    consecutiveFailures = 0;
                    consecutiveChunkFailures = 0;
                }
            }

            logger?.Info($"\n=== Training Complete ===");
            logger?.Info($"Final Best Loss: {bestLoss:F6}");
            logger?.Info($"Target was: {targetLossThreshold:F4}");
            logger?.Info($"Attempts used: {maxAttempts}");
            logger?.Info($"Data Chunks: {chunkManager.ChunkCount}");
            logger?.Info(bestStructure);
            return (bestParams ?? initialParams, bestLR);
        }

        public void SetMaxParameters(long maxParameters)
        {
            this.maxParameters = maxParameters;
            logger?.Info($"Maximum parameters set to: {maxParameters:N0}");
        }

        public void AddDataChunk(float[][] inputs, float[][] outputs)
        {
            chunkManager.AddChunk(inputs, outputs);
            logger?.Info($"Added data chunk: {inputs.Length} samples. Total chunks: {chunkManager.ChunkCount}");
        }

        public void UpdateBestIfNeeded(NeuralNetwork nn, TrainingDiagnostics diag, float stabilityScore = 1f)
        {
            float currentLoss = diag.LatestValidationLoss;

            if (currentLoss < bestLoss)
            {
                float improvement = bestLoss - currentLoss;
                bestLoss = currentLoss;
                bestParams = nn.GetParametersCopy();
                bestData = nn.GetDataCopy();
                stagnationCounter = 0;

                //newed to update global best if the new current loss is better.
                if (currentLoss < globalBestLoss)
                {
                    globalBestLoss = currentLoss;
                    globalBestParams = bestParams.Clone();
                }

                architectureHistory.Add(new ArchitectureCandidate(bestParams.Clone(), currentLoss, generation, "UpdateBest", stabilityScore));

                if (architectureHistory.Count > 25)
                {
                    architectureHistory.RemoveRange(0, 5);
                }

                logger?.Info($"New best validation loss: {bestLoss:F6} (improvement: {improvement:F6})");
            }
            else
            {
                stagnationCounter++;
            }
        }

        public Parameters GetBestParameters() => globalBestParams?.Clone() ?? bestParams?.Clone();
        public float GetBestLoss() => globalBestLoss == float.MaxValue ? bestLoss : globalBestLoss;

        public void RollbackIfWorsened(NeuralNetwork nn, TrainingDiagnostics diag, float tolerance = 1.08f)
        {
            if (bestData != null && diag.LatestValidationLoss > bestLoss * tolerance)
            {
                logger?.Info($"Rolling back: current loss {diag.LatestValidationLoss:F6} > best * {tolerance} ({bestLoss * tolerance:F6})");
                nn.RestoreState(bestData);
            }
        }

        #region Mutate Parameters
        public Parameters Mutate(Parameters baseParams, TuningStrategy strategy, int attempt)
        {
            var mutated = baseParams.Clone();
            generation++;

            float adaptiveMutationRate = mutationRate * (1 + stagnationCounter * 0.1f);
            adaptiveMutationRate = Math.Clamp(adaptiveMutationRate, 0.05f, 0.5f);

            switch (strategy)
            {
                case TuningStrategy.Explore:
                    return ExploreArchitecture(baseParams, attempt, adaptiveMutationRate);

                case TuningStrategy.Exploit:
                    return ExploitBestArchitecture(baseParams, adaptiveMutationRate);

                case TuningStrategy.Conservative:
                    return ConservativeMutation(baseParams, adaptiveMutationRate * 0.5f);

                case TuningStrategy.Aggressive:
                    return AggressiveMutation(baseParams, adaptiveMutationRate * 2f);

                case TuningStrategy.Balanced:
                    return BalancedMutation(baseParams, adaptiveMutationRate);

                case TuningStrategy.Adaptive:
                    return AdaptiveMutation(baseParams, adaptiveMutationRate);

                default:
                    return ExploreArchitecture(baseParams, attempt, adaptiveMutationRate);
            }
        }


        private Parameters ExploreArchitecture(Parameters baseParams, int attempt, float mutationRate)
        {
            var mutated = baseParams.Clone();

            if (random.NextSingle() < mutationRate)
            {
                int targetDepth = Math.Clamp(baseParams.LayerWidths.Count + random.Next(-2, 3), 3, 12);

                mutated.LayerWidths = new List<int> { baseParams.LayerWidths[0] };
                mutated.LayerActivations = new List<ActivationType> { baseParams.LayerActivations[0] };

                for (int i = 1; i < targetDepth - 1; i++)
                {
                    int prevWidth = mutated.LayerWidths[i - 1];

                    float depthRatio = (float)i / (targetDepth - 1);
                    int baseWidth = (int)(prevWidth * (1.2f - depthRatio * 0.8f));

                    int variation = (int)(baseWidth * mutationRate * random.NextSingle());
                    int width = Math.Clamp(baseWidth + random.Next(-variation, variation + 1), 8, 1024);

                    var testWidths = new List<int>(mutated.LayerWidths) { width };
                    if (i == targetDepth - 2)
                    {
                        testWidths.Add(baseParams.LayerWidths[^1]);
                    }
                    if (CalculateParameterCount(testWidths) > maxParameters)
                    {
                        width = Math.Max(8, width / 2);
                        testWidths[^1] = width;

                        if (CalculateParameterCount(testWidths) > maxParameters)
                        {
                            break;
                        }
                    }

                    mutated.LayerWidths.Add(width);

                    var activations = new[] { ActivationType.Relu, ActivationType.Leakyrelu, ActivationType.Tanh, ActivationType.Swish };
                    mutated.LayerActivations.Add(activations[random.Next(activations.Length)]);
                }

                //Output layer must stay the same, we arnt changing this :TODO Consider maybe changing output activations? I dont know
                mutated.LayerWidths.Add(baseParams.LayerWidths[^1]);
                mutated.LayerActivations.Add(baseParams.LayerActivations[^1]);
            }

            return MutateHyperparameters(mutated, mutationRate);
        }

        private Parameters ExploitBestArchitecture(Parameters baseParams, float mutationRate)
        {
            var bestSimilar = architectureHistory
            .Where(a => a.ParameterCount <= maxParameters)
            .Where(a => Math.Abs(a.Parameters.LayerWidths.Count - baseParams.LayerWidths.Count) <= 1)
            .OrderBy(a => a.Score)
            .FirstOrDefault();

            var source = bestSimilar?.Parameters ?? baseParams;

            return ConservativeMutation(source, mutationRate * 0.7f);
        }

        private Parameters ConservativeMutation(Parameters baseParams, float mutationRate)
        {
            var mutated = baseParams.Clone();

            for (int i = 1; i < mutated.LayerWidths.Count - 1; i++)
            {
                if (random.NextSingle() < mutationRate)
                {
                    int delta = random.Next(-2, 3);
                    int newWidth = Math.Clamp(mutated.LayerWidths[i] + delta, 8, 512);

                    var testWidths = new List<int>(mutated.LayerWidths);
                    testWidths[i] = newWidth;

                    if (CalculateParameterCount(testWidths) <= maxParameters)
                    {
                        mutated.LayerWidths[i] = newWidth;
                    }
                }
            }

            return MutateHyperparameters(mutated, mutationRate * 0.5f);
        }

        private Parameters AggressiveMutation(Parameters baseParams, float mutationRate)
        {
            var mutated = baseParams.Clone();

            if (random.NextSingle() < mutationRate)
            {
                if (random.NextBool() && mutated.LayerWidths.Count < 10)
                {
                    int insertAt = random.Next(1, mutated.LayerWidths.Count - 1);
                    int width = (mutated.LayerWidths[insertAt - 1] + mutated.LayerWidths[insertAt]) / 2;


                    var testWidths = new List<int>(mutated.LayerWidths);
                    testWidths.Insert(insertAt, width);

                    if (CalculateParameterCount(testWidths) <= maxParameters)
                    {
                        mutated.LayerWidths.Insert(insertAt, width);
                        mutated.LayerActivations.Insert(insertAt, ActivationType.Relu);
                    }
                }
                else if (mutated.LayerWidths.Count > 3)
                {
                    int removeAt = random.Next(1, mutated.LayerWidths.Count - 2);
                    mutated.LayerWidths.RemoveAt(removeAt);
                    mutated.LayerActivations.RemoveAt(removeAt);
                }
            }

            for (int i = 1; i < mutated.LayerWidths.Count - 1; i++)
            {
                if (random.NextSingle() < mutationRate)
                {
                    int delta = random.Next(-8, 9);

                    int newWidth = Math.Clamp(mutated.LayerWidths[i] + delta, 8, 1024);

                    var testWidths = new List<int>(mutated.LayerWidths);

                    testWidths[i] = newWidth;

                    if (CalculateParameterCount(testWidths) <= maxParameters)
                    {
                        mutated.LayerWidths[i] = newWidth;
                    }
                }
            }

            return MutateHyperparameters(mutated, mutationRate);
        }

        private Parameters BalancedMutation(Parameters baseParams, float mutationRate)
        {
            var mutated = baseParams.Clone();

            // Moderate width adjustments
            for (int i = 1; i < mutated.LayerWidths.Count - 1; i++)
            {
                if (random.NextSingle() < mutationRate * 0.6f)
                {
                    int delta = random.Next(-4, 5);
                    int newWidth = Math.Clamp(mutated.LayerWidths[i] + delta, 8, 768);

                    var testWidths = new List<int>(mutated.LayerWidths);
                    testWidths[i] = newWidth;

                    if (CalculateParameterCount(testWidths) <= maxParameters)
                    {
                        mutated.LayerWidths[i] = newWidth;
                    }
                }
            }

            // Occasional layer addition/removal
            if (random.NextSingle() < mutationRate * 0.3f)
            {
                if (random.NextBool() && mutated.LayerWidths.Count < 10)
                {
                    int insertAt = random.Next(1, mutated.LayerWidths.Count - 1);
                    int width = (mutated.LayerWidths[insertAt - 1] + mutated.LayerWidths[insertAt]) / 2;

                    var testWidths = new List<int>(mutated.LayerWidths);
                    testWidths.Insert(insertAt, width);

                    if (CalculateParameterCount(testWidths) <= maxParameters)
                    {
                        mutated.LayerWidths.Insert(insertAt, width);
                        mutated.LayerActivations.Insert(insertAt, ActivationType.Relu);
                    }
                }
                else if (mutated.LayerWidths.Count > 3)
                {
                    int removeAt = random.Next(1, mutated.LayerWidths.Count - 2);
                    mutated.LayerWidths.RemoveAt(removeAt);
                    mutated.LayerActivations.RemoveAt(removeAt);
                }
            }

            return MutateHyperparameters(mutated, mutationRate * 0.8f);
        }


        private Parameters AdaptiveMutation(Parameters baseParams, float mutationRate)
        {
            var mutated = baseParams.Clone();

            var recentCandidates = architectureHistory.TakeLast(3).ToList();
            bool needsMoreCapacity = recentCandidates.All(c => c.Score > globalBestLoss * 0.95f);
            bool needsRegularization = recentCandidates.Any(c => c.StabilityScore < 0.8f);

            if (needsMoreCapacity && !needsRegularization)
            {
                for (int i = 1; i < mutated.LayerWidths.Count - 1; i++)
                {
                    if (random.NextSingle() < mutationRate * 0.5f)
                    {
                        int increase = Math.Max(2, mutated.LayerWidths[i] / 10);
                        int newWidth = Math.Min(1024, mutated.LayerWidths[i] + increase);

                        var testWidths = new List<int>(mutated.LayerWidths);
                        testWidths[i] = newWidth;

                        if (CalculateParameterCount(testWidths) <= maxParameters)
                        {
                            mutated.LayerWidths[i] = newWidth;
                        }
                    }
                }
            }
            else if (needsRegularization)
            {
                mutated.L2RegulationLamda = Math.Min(0.01f, mutated.L2RegulationLamda * 1.3f);
                mutated.GradientClippingThreshold = Math.Max(0.1f, mutated.GradientClippingThreshold * 0.8f);
            }
            else
            {
                return BalancedMutation(mutated, mutationRate);
            }

            return MutateHyperparameters(mutated, mutationRate * 0.7f);
        }

        private Parameters MutateHyperparameters(Parameters mutated, float mutationRate)
        {
            if (random.NextSingle() < mutationRate)
            {
                float factor = 1f + (random.NextSingle() - 0.5f) * 0.2f;
                mutated.L2RegulationLamda = Math.Clamp(mutated.L2RegulationLamda * factor, 1e-7f, 0.1f);
            }

            if (random.NextSingle() < mutationRate)
            {
                float factor = 1f + (random.NextSingle() - 0.5f) * 0.1f;
                mutated.GradientClippingThreshold = Math.Clamp(mutated.GradientClippingThreshold * factor, 0.01f, 10f);
            }

            if (mutated.CostFunction == CostFunctionType.huberLoss && random.NextSingle() < mutationRate)
            {
                float factor = 1f + (random.NextSingle() - 0.5f) * 0.15f;
                mutated.HuberLossDelta = Math.Clamp(mutated.HuberLossDelta * factor, 0.1f, 5f);
            }

            return mutated;
        }


        #endregion


        public Parameters Tune(Parameters current, TrainingDiagnostics diag, NeuralNetwork nn)
        {
            var tuned = current.Clone();
            var layers = nn?.GetLayers();
            bool architectureChanged = false;

            float validationTrend = diag.GetValidationTrend();
            float trainingTrend = diag.GetTrainingTrend();
            bool isConverging = diag.IsConverging();
            bool isPlateau = diag.IsPlateaued();

            if (layers != null && layers.Count > 2)
            {
                architectureChanged = AdjustArchitectureBasedOnUtilization(tuned, layers);
            }

            if (!architectureChanged && (isPlateau || stagnationCounter > 10))
            {
                architectureChanged = HandleStagnation(tuned, diag);
            }

            AdjustHyperparameters(tuned, diag, validationTrend, trainingTrend, isConverging);

            if (CalculateParameterCount(tuned.LayerWidths) > maxParameters)
            {
                tuned = ReduceNetworkSize(tuned);
                architectureChanged = true;
            }

            if (architectureChanged)
            {
                LogArchitectureChange("Dynamic architecture adjustment", tuned);
            }

            return tuned;
        }


        private bool AdjustArchitectureBasedOnUtilization(Parameters tuned, List<Layer> layers)
        {
            bool changed = false;

            for (int i = 1; i < layers.Count - 1; i++)
            {
                if (layers[i].Activations == null || layers[i].Size == 0) continue;

                var validActivations = layers[i].Activations.Where(a => !float.IsNaN(a) && !float.IsInfinity(a)).ToArray();
                if (validActivations.Length == 0)
                {
                    continue;
                }
                float avgActivation = validActivations.Select(Math.Abs).Average();
                int deadNeurons = validActivations.Count(a => Math.Abs(a) < 1e-6f);
                float utilization = 1f - (float)deadNeurons / validActivations.Length;

                if (utilization > 0.85f && avgActivation > 0.1f && tuned.LayerWidths[i] < 1024)
                {
                    int increase = Math.Max(2, tuned.LayerWidths[i] / 20);
                    int newWidth = Math.Min(1024, tuned.LayerWidths[i] + increase);

                    var testWidths = new List<int>(tuned.LayerWidths);
                    testWidths[i] = newWidth;

                    if (CalculateParameterCount(testWidths) <= maxParameters)
                    {
                        tuned.LayerWidths[i] = newWidth;
                        changed = true;
                        logger?.Info($"Expanded layer {i} to {newWidth} neurons (utilization: {utilization:F3})");
                    }
                }
                // Shrink if underutilized for performance. We cant just keep growing the neural network indefinitely
                else if (utilization < 0.3f && tuned.LayerWidths[i] > 16)
                {
                    int decrease = Math.Max(1, tuned.LayerWidths[i] / 10);
                    tuned.LayerWidths[i] = Math.Max(16, tuned.LayerWidths[i] - decrease);
                    changed = true;
                    logger?.Info($"Reduced layer {i} to {tuned.LayerWidths[i]} neurons (utilization: {utilization:F3})");
                }
            }

            return changed;
        }

        private bool HandleStagnation(Parameters tuned, TrainingDiagnostics diag)
        {
            bool changed = false;

            //Adds a layer if the validation loss is high and we have room for more layers
            if (diag.LatestValidationLoss > 0.1f && tuned.LayerWidths.Count < 8)
            {
                int insertAt = tuned.LayerWidths.Count / 2;
                int newWidth = (tuned.LayerWidths[insertAt - 1] + tuned.LayerWidths[insertAt]) / 2;
                newWidth = Math.Max(16, newWidth);

                var testWidths = new List<int>(tuned.LayerWidths);
                testWidths.Insert(insertAt, newWidth);

                if (CalculateParameterCount(testWidths) <= maxParameters)
                {
                    tuned.LayerWidths.Insert(insertAt, newWidth);
                    tuned.LayerActivations.Insert(insertAt, ActivationType.Relu);
                    changed = true;
                    logger?.Info($"Added layer at position {insertAt} with {newWidth} neurons to handle complexity");
                }
            }

            else if (diag.IsOverfitting() && tuned.LayerWidths.Count > 3)
            {
                int removeAt = tuned.LayerWidths.Count / 2;
                if (removeAt > 0 && removeAt < tuned.LayerWidths.Count - 1)
                {
                    tuned.LayerWidths.RemoveAt(removeAt);
                    tuned.LayerActivations.RemoveAt(removeAt);
                    changed = true;
                    logger?.Info($"Removed layer at position {removeAt} to reduce overfitting");
                }
            }

            return changed;
        }

        private void AdjustHyperparameters(Parameters tuned, TrainingDiagnostics diag, float validationTrend, float trainingTrend, bool isConverging)
        {
            // Adjust regularization based on overfitting signals
            if (diag.IsOverfitting())
            {
                tuned.L2RegulationLamda = Math.Min(0.01f, tuned.L2RegulationLamda * 1.2f);
            }
            else if (!diag.IsOverfitting() && diag.LatestValidationLoss > diag.LatestTrainingLoss * 1.5f)
            {
                tuned.L2RegulationLamda = Math.Max(1e-7f, tuned.L2RegulationLamda * 0.9f);
            }

            // Adjust gradient clipping based on stability
            if (diag.IsUnstableOutput())
            {
                tuned.GradientClippingThreshold = Math.Max(0.1f, tuned.GradientClippingThreshold * 0.8f);
            }
            else if (isConverging && !diag.IsUnstableOutput())
            {
                tuned.GradientClippingThreshold = Math.Min(5f, tuned.GradientClippingThreshold * 1.05f);
            }

            // Adjust Huber loss delta
            if (tuned.CostFunction == CostFunctionType.huberLoss)
            {
                if (diag.IsUnstableOutput())
                {
                    tuned.HuberLossDelta = Math.Max(0.1f, tuned.HuberLossDelta * 0.95f);
                }
                else if (isConverging)
                {
                    tuned.HuberLossDelta = Math.Min(5f, tuned.HuberLossDelta * 1.02f);
                }
            }
        }

        private Parameters ReduceNetworkSize(Parameters parameters)
        {
            var reduced = parameters.Clone();

            // Proportionally reduce all hidden layer sizes
            for (int i = 1; i < reduced.LayerWidths.Count - 1; i++)
            {
                reduced.LayerWidths[i] = Math.Max(8, reduced.LayerWidths[i] * 3 / 4);
            }

            // If still too large, remove layers
            while (CalculateParameterCount(reduced.LayerWidths) > maxParameters && reduced.LayerWidths.Count > 3)
            {
                int removeAt = reduced.LayerWidths.Count / 2;
                if (removeAt > 0 && removeAt < reduced.LayerWidths.Count - 1)
                {
                    reduced.LayerWidths.RemoveAt(removeAt);
                    reduced.LayerActivations.RemoveAt(removeAt);
                }
                else
                {
                    break;
                }
            }

            return reduced;
        }

        private (bool success, float bestLoss, Parameters bestParams, float bestLR) TrainSingleAttempt(float[][] trainX, float[][] trainY, float[][] valX, float[][] valY, float[][] globalValX, float[][] globalValY, Parameters parameters, float learningRate, float targetLossThreshold, int attempt)
        {
            var net = new NeuralNetwork(parameters);
            var diag = new TrainingDiagnostics();

            float adaptiveLR = learningRate;
            int noImproveCount = 0;
            int earlyStopPatience = Math.Max(15, parameters.LayerWidths.Sum() / 50);
            const int maxEpochs = 3000;
            const float minImprovement = 1e-5f;

            bool epochSuccess = false;
            float bestValLoss = float.MaxValue;
            Parameters bestParams = parameters.Clone();
            float bestLR = learningRate;

            for (int epoch = 0; epoch < maxEpochs; epoch++)
            {
                if (epoch < 20)
                {
                    adaptiveLR = learningRate * (0.1f + 0.9f * epoch / 20f); // Warm-up
                }
                else if (noImproveCount > 5)
                {
                    adaptiveLR *= 0.98f; // Gradual decay when not improving
                }

                adaptiveLR = Math.Max(adaptiveLR, learningRate * 0.01f); // Minimum LR

                float totalTrainLoss = TrainEpoch(net, trainX, trainY, adaptiveLR);
                float avgTrainLoss = totalTrainLoss / trainX.Length;

                float avgValLoss = ValidateNetwork(net, valX, valY, diag);


                float globalValLoss = avgValLoss;
                if (globalValX != valX)
                {
                    globalValLoss = ValidateNetwork(net, globalValX, globalValY, null);
                }

                diag.Log(avgTrainLoss, globalValLoss, learningRate: adaptiveLR, parameterCount: CalculateParameterCount(parameters.LayerWidths));

                if (epoch % 10 == 0 || epoch < 50)
                {
                    logger?.Info($"Epoch {epoch + 1}: Train={avgTrainLoss:F6}, Val={avgValLoss:F6}, Global={globalValLoss:F6}, LR={adaptiveLR:E3}");
                }

                if (globalValLoss < bestValLoss - minImprovement)
                {
                    float improvement = bestValLoss - globalValLoss;
                    bestValLoss = globalValLoss;
                    bestParams = net.GetParametersCopy();
                    bestLR = adaptiveLR;
                    UpdateBestIfNeeded(net, diag);
                    noImproveCount = 0;
                    epochSuccess = true;

                    if (globalValLoss <= targetLossThreshold)
                    {
                        logger?.Info($"Target achieved! Val Loss: {globalValLoss:F6} ≤ {targetLossThreshold:F4}");
                        return (true, globalValLoss, bestParams, bestLR);
                    }
                }
                else
                {
                    noImproveCount++;
                }

                bool shouldStop = false;
                string stopReason = "";

                if (diag.IsDiverging())
                {
                    shouldStop = true;
                    stopReason = "Training diverged";
                }
                else if (diag.OutputCollapseDetected())
                {
                    shouldStop = true;
                    stopReason = "Output collapse detected";
                }
                else if (diag.IsUnstableOutput() && epoch > 100)
                {
                    shouldStop = true;
                    stopReason = "Unstable output";
                }
                else if (noImproveCount >= earlyStopPatience)
                {
                    shouldStop = true;
                    stopReason = $"No improvement for {earlyStopPatience} epochs";
                }
                else if (diag.IsOverfitting() && epoch > 200)
                {
                    // Try to recover from overfitting first
                    if (noImproveCount > earlyStopPatience / 2)
                    {
                        shouldStop = true;
                        stopReason = "Persistent overfitting";
                    }
                    else
                    {
                        // Reduce learning rate to combat overfitting
                        adaptiveLR *= 0.7f;
                    }
                }

                if (shouldStop)
                {
                    logger?.Info($"Stopping: {stopReason}");
                    break;
                }

                if (epoch > 0 && epoch % 50 == 0 && epoch < maxEpochs - 100)
                {
                    var layers = net.GetLayers();
                    if (diag.HasDeadNeurons(layers))
                    {
                        logger?.Info("Dead neurons detected - considering architecture adjustment");
                        var adjustedParams = Tune(net.GetParametersCopy(), diag, net);
                        if (!adjustedParams.LayerWidths.SequenceEqual(parameters.LayerWidths))
                        {
                            if (CalculateParameterCount(adjustedParams.LayerWidths) <= maxParameters)
                            {
                                logger?.Info("Applying mid-training architecture adjustment");
                                parameters = adjustedParams;
                                net = new NeuralNetwork(parameters);
                                adaptiveLR = learningRate; // Reset learning rate
                                LogArchitectureChange("Mid-training adjustment due to dead neurons", parameters);
                            }
                        }
                    }
                }

                //Need to clean up every so often
                if (epoch % 100 == 0)
                {
                    diag.ClearOldData();
                }
            }

            return (epochSuccess, bestValLoss, bestParams, bestLR);
        }

        private static (float[][], float[][], float[][], float[][]) SplitDataset(float[][] inputs, float[][] expected, float validationSplit)
        {
            int total = inputs.Length;
            int valSize = (int)(total * validationSplit);
            int trainSize = total - valSize;

            var indices = Enumerable.Range(0, total).ToArray();
            var random = new Random();
            RandomExtensions.Shuffle(random, indices);

            var trainX = new float[trainSize][];
            var trainY = new float[trainSize][];
            var valX = new float[valSize][];
            var valY = new float[valSize][];

            for (int i = 0; i < trainSize; i++)
            {
                trainX[i] = inputs[indices[i]];
                trainY[i] = expected[indices[i]];
            }

            for (int i = 0; i < valSize; i++)
            {
                valX[i] = inputs[indices[trainSize + i]];
                valY[i] = expected[indices[trainSize + i]];
            }

            return (trainX, trainY, valX, valY);
        }

        private static float TrainEpoch(NeuralNetwork net, float[][] trainX, float[][] trainY, float learningRate)
        {
            float totalLoss = 0f;

            var indices = Enumerable.Range(0, trainX.Length).ToArray();
            var random = new Random();
            RandomExtensions.Shuffle(random, indices);

            foreach (int i in indices)
            {
                net.Train(new[] { trainX[i] }, new[] { trainY[i] }, learningRate, 1, silent: true);
                var prediction = net.Predict(trainX[i]);
                float loss = prediction.Zip(trainY[i], (a, b) => (a - b) * (a - b)).Average();
                totalLoss += loss;
            }

            return totalLoss;
        }

        private static float ValidateNetwork(NeuralNetwork net, float[][] valX, float[][] valY, TrainingDiagnostics diag)
        {
            float totalLoss = 0f;

            for (int i = 0; i < valX.Length; i++)
            {
                var prediction = net.Predict(valX[i]);
                float loss = prediction.Zip(valY[i], (a, b) => (a - b) * (a - b)).Average();
                totalLoss += loss;

                if (diag != null && i < 10)
                {
                    diag.ValidationPredictions.Add((float[])prediction.Clone());
                    diag.ValidationTargets.Add((float[])valY[i].Clone());
                }
            }

            return totalLoss / valX.Length;
        }

        private static TuningStrategy SelectStrategy(int attempt, int consecutiveFailures, TrainingDiagnostics diag, float bestLoss, float targetLoss)
        {
            // Early attempts: explore
            if (attempt <= 5)
            {
                return TuningStrategy.Explore;
            }
            // If we're doing well, exploit
            if (bestLoss < targetLoss * 2f && consecutiveFailures == 0)
            {
                return TuningStrategy.Exploit;
            }
            // If struggling, be more aggressive
            if (consecutiveFailures >= 2 || bestLoss > targetLoss * 4f)
            {
                return TuningStrategy.Aggressive;
            }

            // If overfitting, be conservative
            if (diag.IsOverfitting())
            {
                return TuningStrategy.Conservative;
            }

            // If performance is okay but not great, use adaptive
            if (bestLoss < targetLoss * 3f)
            {

                return TuningStrategy.Adaptive;

            }
            // Default to balanced exploration
            return TuningStrategy.Balanced;
        }

        private static Parameters GenerateNextArchitecture(NeuralAutoTuner tuner, Parameters current, TuningStrategy strategy, int attempt, TrainingDiagnostics diag)
        {
            var tuned = tuner.Tune(current, diag, null);

            // when tuned if there is no significant change, apply mutation
            if (tuned.LayerWidths.SequenceEqual(current.LayerWidths))
            {
                tuned = tuner.Mutate(current, strategy, attempt);
            }

            return ValidateArchitecture(tuned);
        }

        private static Parameters ValidateArchitecture(Parameters parameters)
        {
            // Ensure minimum and maximum constraints
            for (int i = 1; i < parameters.LayerWidths.Count - 1; i++)
            {
                parameters.LayerWidths[i] = Math.Clamp(parameters.LayerWidths[i], 8, 1024);
            }

            // Ensure we have at least 3 layers (input, hidden, output)
            if (parameters.LayerWidths.Count < 3)
            {
                parameters.LayerWidths.Insert(1, 32);
                parameters.LayerActivations.Insert(1, ActivationType.Relu);
            }

            // Ensure activations match layer count
            while (parameters.LayerActivations.Count < parameters.LayerWidths.Count)
            {
                parameters.LayerActivations.Insert(parameters.LayerActivations.Count - 1, ActivationType.Relu);
            }

            while (parameters.LayerActivations.Count > parameters.LayerWidths.Count)
            {
                parameters.LayerActivations.RemoveAt(parameters.LayerActivations.Count - 2);
            }

            return parameters;
        }

        public void ResetTuner()
        {
            bestLoss = float.MaxValue;
            bestParams = null;
            bestData = null;
            globalBestLoss = float.MaxValue;
            globalBestParams = null;
            architectureHistory.Clear();
            stagnationCounter = 0;
            generation = 0;
            consecutiveChunkFailures = 0;
            chunkManager.ClearChunks();
            logger?.Info("Tuner reset complete");
        }

        #region IO

        public void SaveBestModel(string filePath)
        {
            if (globalBestParams != null)
            {
                try
                {
                    var modelData = new
                    {
                        Parameters = globalBestParams,
                        BestLoss = globalBestLoss,
                        Generation = generation,
                        Timestamp = DateTime.Now,
                        MaxParameters = maxParameters
                    };

                    var json = JsonConvert.SerializeObject(modelData, Formatting.Indented);
                    File.WriteAllText(filePath, json);
                    logger?.Info($"Best model saved to {filePath} (Loss: {globalBestLoss:F6})");
                }
                catch (Exception ex)
                {
                    logger?.Info($"Error saving model: {ex.Message}");
                }
            }
            else
            {
                logger?.Info("No best model available to save");
            }
        }

        public Parameters LoadBestModel(string filePath)
        {
            try
            {
                var json = File.ReadAllText(filePath);
                var modelData = JsonConvert.DeserializeObject<dynamic>(json);
                var parameters = JsonConvert.DeserializeObject<Parameters>(modelData.Parameters.ToString());

                if (modelData.BestLoss != null)
                {
                    globalBestLoss = (float)modelData.BestLoss;
                    globalBestParams = parameters.Clone();
                }

                logger?.Info($"Model loaded from {filePath} (Loss: {globalBestLoss:F6})");
                return parameters;
            }
            catch (Exception ex)
            {
                logger?.Info($"Error loading model: {ex.Message}");
                return null;
            }
        }
        public void ExportTrainingHistory(string filePath)
        {
            try
            {
                var historyData = new
                {
                    GeneratedArchitectures = architectureHistory.Select(a => new
                    {
                        a.Parameters.LayerWidths,
                        a.Score,
                        a.StabilityScore,
                        a.Generation,
                        a.Origin,
                        a.ParameterCount,
                        a.CreatedAt
                    }).ToList(),
                    BestGlobalLoss = globalBestLoss,
                    TotalGenerations = generation,
                    ChunkStatistics = new
                    {
                        TotalChunks = chunkManager.ChunkCount,
                        TotalSamples = chunkManager.TotalSamples,
                        AveragePerformance = chunkManager.GetAverageChunkPerformance()
                    },
                    ExportTimestamp = DateTime.Now
                };

                var json = JsonConvert.SerializeObject(historyData, Formatting.Indented);
                File.WriteAllText(filePath, json);
                logger?.Info($"Training history exported to {filePath}");
            }
            catch (Exception ex)
            {
                logger?.Info($"Error exporting history: {ex.Message}");
            }
        }
        //TODO: Might be handy to have an import training history too.

        #endregion

        #region Logging

        private void LogArchitectureChange(string reason, Parameters p)
        {
            long paramCount = CalculateParameterCount(p.LayerWidths);
            string logEntry = $"{DateTime.Now:yyyy-MM-dd HH:mm:ss} - Generation {generation} - {reason}\n" +
            $"Architecture: {string.Join(" -> ", p.LayerWidths)} ({paramCount:N0} parameters)\n" +
            $"Activations: {string.Join(", ", p.LayerActivations.Skip(1).Take(p.LayerActivations.Count - 2))}\n" +
            $"L2 Regularization: {p.L2RegulationLamda:E3}, Gradient Clipping: {p.GradientClippingThreshold:F3}\n" +
            $"Best Loss So Far: {bestLoss:F6}, Stagnation Counter: {stagnationCounter}\n" +
            $"Max Params: {maxParameters:N0}, Current Params: {paramCount:N0}\n\n";

            try
            {
                File.AppendAllText("tuning_log.txt", logEntry);
            }
            catch (Exception ex)
            {
                logger?.Error($"Warning: Could not write to log file: {ex.Message}");
            }
        }

        #endregion

        public void PrintTuningStatistics()
        {
            logger?.Info("\n=== Tuning Statistics ===");
            logger?.Info($"Generation: {generation}");
            logger?.Info($"Best Loss: {globalBestLoss:F6}");
            logger?.Info($"Stagnation Counter: {stagnationCounter}");
            logger?.Info($"Architecture History: {architectureHistory.Count} entries");
            logger?.Info($"Data Chunks: {chunkManager.ChunkCount}");
            logger?.Info($"Average Chunk Performance: {chunkManager.GetAverageChunkPerformance():F6}");

            if (globalBestParams != null)
            {
                logger?.Info($"Best Architecture: {string.Join(" -> ", globalBestParams.LayerWidths)}");
                logger?.Info($"Best Parameters: {CalculateParameterCount(globalBestParams.LayerWidths):N0}");
                logger?.Info($"L2 Regularization: {globalBestParams.L2RegulationLamda:E3}");
                logger?.Info($"Gradient Clipping: {globalBestParams.GradientClippingThreshold:F3}");
            }

            if (architectureHistory.Count > 0)
            {
                var topArchitectures = architectureHistory.OrderBy(a => a.Score * (2f - a.StabilityScore)).Take(3).ToList();

                logger?.Info("\nTop 3 Architectures:");
                for (int i = 0; i < topArchitectures.Count; i++)
                {
                    var arch = topArchitectures[i];
                    logger?.Info($"  {i + 1}. {string.Join(" -> ", arch.Parameters.LayerWidths)} - Loss: {arch.Score:F6}, Stability: {arch.StabilityScore:F3}");
                }
            }
        }

        public bool ContinueTrainingWithNewData(float[][] newInputs, float[][] newExpected, float learningRate, float targetLossThreshold, int maxAdditionalAttempts = 25)
        {
            if (globalBestParams == null)
            {
                logger?.Info("No previous best model found. Starting fresh training.");
                return false;
            }

            logger?.Info($"\n=== Continuing Training with New Data ===");
            logger?.Info($"New data: {newInputs.Length} samples");
            logger?.Info($"Current best loss: {globalBestLoss:F6}");

            AddDataChunk(newInputs, newExpected);

            // go back to training from best known parameters that were last found
            var result = TrainWithAutoTuning(newInputs, newExpected, learningRate, globalBestParams, maxAdditionalAttempts, 0.3f, targetLossThreshold, maxParameters, 3);

            bool success = result.BestParams != null && GetBestLoss() <= targetLossThreshold;

            if (success)
            {
                logger?.Info($"Continued training successful! New best loss: {GetBestLoss():F6}");
            }
            else
            {
                logger?.Info($"Continued training did not reach target. Best achieved: {GetBestLoss():F6}");
            }

            return success;
        }

        #region Estimations Helpers

        public (Parameters OptimalParams, float EstimatedPerformance) PredictOptimalArchitecture(int inputSize, int outputSize, int sampleCount, float complexityHint = 0.5f)
        {
            logger?.Info($"\n=== Predicting Optimal Architecture ===");
            logger?.Info($"Input size: {inputSize}, Output size: {outputSize}");
            logger?.Info($"Sample count: {sampleCount}, Complexity hint: {complexityHint:F2}");

            var baseParams = new Parameters
            {
                LayerWidths = new List<int> { inputSize, outputSize },
                LayerActivations = new List<ActivationType> { ActivationType.Relu, ActivationType.Relu }
            };

            int estimatedDepth = EstimateOptimalDepth(sampleCount, complexityHint);
            int estimatedWidth = EstimateOptimalWidth(inputSize, outputSize, sampleCount, complexityHint);

            var predictedParams = baseParams.Clone();
            predictedParams.LayerWidths = new List<int> { inputSize };
            predictedParams.LayerActivations = new List<ActivationType> { ActivationType.Relu };

            for (int i = 1; i < estimatedDepth - 1; i++)
            {
                float depthRatio = (float)i / (estimatedDepth - 1);
                int layerWidth = (int)(estimatedWidth * (1.1f - depthRatio * 0.3f));
                layerWidth = Math.Max(Math.Min(inputSize, outputSize), layerWidth);

                var testWidths = new List<int>(predictedParams.LayerWidths) { layerWidth };
                if (i == estimatedDepth - 2)
                {
                    testWidths.Add(outputSize);
                }

                if (CalculateParameterCount(testWidths) > maxParameters)
                {
                    layerWidth = Math.Max(8, layerWidth / 2);
                    if (CalculateParameterCount(new List<int>(predictedParams.LayerWidths) { layerWidth, outputSize }) > maxParameters)
                    {
                        break;
                    }
                }

                predictedParams.LayerWidths.Add(layerWidth);
                predictedParams.LayerActivations.Add(ActivationType.Relu);
            }

            predictedParams.LayerWidths.Add(outputSize);
            predictedParams.LayerActivations.Add(ActivationType.Relu);

            EstimateOptimalHyperparameters(predictedParams, sampleCount, complexityHint);

            float estimatedPerformance = EstimatePerformance(predictedParams, sampleCount, complexityHint);

            logger?.Info($"Predicted architecture: {string.Join(" -> ", predictedParams.LayerWidths)}");
            logger?.Info($"Parameters: {CalculateParameterCount(predictedParams.LayerWidths):N0}");
            logger?.Info($"Estimated performance: {estimatedPerformance:F6}");

            return (predictedParams, estimatedPerformance);
        }

        private int EstimateOptimalDepth(int sampleCount, float complexityHint)
        {
            int baseDepth = 3;

            if (sampleCount > 10000)
            {
                baseDepth += 2;
            }
            if (sampleCount > 50000)
            {
                baseDepth += 1;
            }
            if (sampleCount > 100000)
            {
                baseDepth += 1;
            }
            baseDepth += (int)(complexityHint * 4);

            return Math.Clamp(baseDepth, 3, 12);
        }

        private int EstimateOptimalWidth(int inputSize, int outputSize, int sampleCount, float complexityHint)
        {
            int baseWidth = Math.Max(inputSize, outputSize) * 2;

            if (sampleCount > 5000)
            {
                baseWidth = (int)(baseWidth * 1.5f);
            }
            if (sampleCount > 20000)
            {
                baseWidth = (int)(baseWidth * 1.3f);
            }

            baseWidth = (int)(baseWidth * (0.5f + complexityHint));

            return Math.Clamp(baseWidth, 16, 1024);
        }

        private void EstimateOptimalHyperparameters(Parameters parameters, int sampleCount, float complexityHint)
        {
            if (sampleCount < 1000)
            {
                parameters.L2RegulationLamda = 0.01f * (1f + complexityHint);
            }
            else if (sampleCount < 10000)
            {
                parameters.L2RegulationLamda = 0.001f * (1f + complexityHint * 0.5f);
            }
            else
            {
                parameters.L2RegulationLamda = 0.0001f * (1f + complexityHint * 0.3f);
            }

            int depth = parameters.LayerWidths.Count;
            parameters.GradientClippingThreshold = Math.Max(0.5f, 5f / (float)Math.Sqrt(depth));

            parameters.CostFunction = CostFunctionType.mse;
            if (complexityHint > 0.7f)
            {
                parameters.CostFunction = CostFunctionType.huberLoss;
                parameters.HuberLossDelta = 1f;
            }
        }

        private float EstimatePerformance(Parameters parameters, int sampleCount, float complexityHint)
        {
            long parameterCount = CalculateParameterCount(parameters.LayerWidths);
            float parameterRatio = (float)parameterCount / Math.Max(sampleCount, 1);

            float basePerformance = 0.1f;

            if (parameterRatio > 0.5f)
            {
                basePerformance *= (1f + parameterRatio);
            }
            else if (parameterRatio < 0.01f)
            {
                basePerformance *= (1f + (0.01f - parameterRatio) * 10f);
            }

            basePerformance *= (0.5f + complexityHint * 0.5f);

            if (architectureHistory.Count > 0)
            {
                var similarArchitectures = architectureHistory.Where(a => Math.Abs(a.ParameterCount - parameterCount) < parameterCount * 0.3f).ToList();

                if (similarArchitectures.Count > 0)
                {
                    float historicalAverage = similarArchitectures.Average(a => a.Score);
                    basePerformance = (basePerformance + historicalAverage) / 2f;
                }
            }

            return Math.Max(0.001f, basePerformance);
        }


        #endregion

        private static long CalculateParameterCount(IList<int> layerWidths)
        {
            if (layerWidths == null)
            {
                throw new ArgumentNullException(nameof(layerWidths));
            }
            if (layerWidths.Count < 2)
            {
                return 0;
            }

            long total = 0;
            for (int i = 0; i < layerWidths.Count - 1; i++)
            {
                int n = layerWidths[i];
                int m = layerWidths[i + 1];
                long weights = (long)n * m;
                long biases = m;
                total += weights + biases;
            }
            return total;
        }
    }
}