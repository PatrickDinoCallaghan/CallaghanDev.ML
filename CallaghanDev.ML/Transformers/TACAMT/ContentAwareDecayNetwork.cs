using System;

namespace CallaghanDev.ML.Transformers.TACAMT
{
    public class ContentAwareDecayNetwork
    {
        public int NumHeads { get; set; }
        public int ContentDim { get; set; }
        public int ProjectionDim { get; set; }
        public int HiddenDim { get; set; }
        public int MLPInputDim { get; set; }
        public int NumTimeBases { get; set; }
        public int TimeRawDim { get; set; } // = 2 * NumTimeBases

        // Per-head content projections [numHeads, projDim, contentDim]
        public float[,,] QueryProjection { get; set; }
        public float[,] QueryProjectionBias { get; set; }
        public float[,,] KeyProjection { get; set; }
        public float[,] KeyProjectionBias { get; set; }

        // Multi-scale time encoding
        public float[,] TimeLogFreq { get; set; }           // [numHeads, numBases]
        public float[,,] TimeProj { get; set; }              // [numHeads, projDim, 2*numBases]
        public float[,] TimeProjBias { get; set; }           // [numHeads, projDim]

        // Memory attention output projection [numHeads, projDim, projDim]
        public float[,,] MemAttnOutputW { get; set; }
        public float[,] MemAttnOutputB { get; set; }

        // Gating MLP
        public float[,,] W1 { get; set; }
        public float[,] B1 { get; set; }
        public float[,] W2 { get; set; }
        public float[] B2 { get; set; }
        public float[] LogBaseDecayRate { get; set; }

        public float MemoryAttentionDropout { get; set; } = 0.1f;
        public float MLPDropout { get; set; } = 0.1f;
        public float WeightDecay { get; set; } = 0.0f;

        public ContentAwareDecayNetwork() { }

        public ContentAwareDecayNetwork(int numHeads, int contentDim, int projectionDim, int hiddenDim, Random random, float memAttnDropout = 0.1f, float mlpDropout = 0.1f, float weightDecay = 0.0f, int numTimeBases = 8)
        {
            NumHeads = numHeads;
            ContentDim = contentDim;
            ProjectionDim = projectionDim;
            HiddenDim = hiddenDim;
            MLPInputDim = projectionDim * 3 + 2;
            NumTimeBases = numTimeBases;
            TimeRawDim = 2 * numTimeBases;
            MemoryAttentionDropout = memAttnDropout;
            MLPDropout = mlpDropout;
            WeightDecay = weightDecay;

            QueryProjection = new float[numHeads, projectionDim, contentDim];
            QueryProjectionBias = new float[numHeads, projectionDim];
            KeyProjection = new float[numHeads, projectionDim, contentDim];
            KeyProjectionBias = new float[numHeads, projectionDim];

            float projStd = MathF.Sqrt(2.0f / (contentDim + projectionDim));
            for (int h = 0; h < numHeads; h++)
            {
                for (int p = 0; p < projectionDim; p++)
                {
                    for (int d = 0; d < contentDim; d++)
                    {
                        QueryProjection[h, p, d] = SampleGaussian(random) * projStd;
                        KeyProjection[h, p, d] = SampleGaussian(random) * projStd;
                    }
                }
            }

            TimeLogFreq = new float[numHeads, numTimeBases];
            for (int h = 0; h < numHeads; h++)
            {
                for (int b = 0; b < numTimeBases; b++)
                {
                    TimeLogFreq[h, b] = -b * MathF.Log(10000f) / Math.Max(numTimeBases - 1, 1);
                }
            }

            TimeProj = new float[numHeads, projectionDim, TimeRawDim];
            TimeProjBias = new float[numHeads, projectionDim];
            float tpStd = MathF.Sqrt(2.0f / (TimeRawDim + projectionDim));
            for (int h = 0; h < numHeads; h++)
            {
                for (int p = 0; p < projectionDim; p++)
                {
                    for (int r = 0; r < TimeRawDim; r++)
                    {
                        TimeProj[h, p, r] = SampleGaussian(random) * tpStd;
                    }
                }
            }

            MemAttnOutputW = new float[numHeads, projectionDim, projectionDim];
            MemAttnOutputB = new float[numHeads, projectionDim];
            float memStd = MathF.Sqrt(2.0f / (projectionDim + projectionDim));
            for (int h = 0; h < numHeads; h++)
            {
                for (int p = 0; p < projectionDim; p++)
                {
                    for (int q = 0; q < projectionDim; q++)
                    {
                        MemAttnOutputW[h, p, q] = SampleGaussian(random) * memStd;
                    }
                }
            }

            W1 = new float[numHeads, hiddenDim, MLPInputDim];
            B1 = new float[numHeads, hiddenDim];
            W2 = new float[numHeads, hiddenDim];
            B2 = new float[numHeads];
            LogBaseDecayRate = new float[numHeads];

            // Content columns of W1 and all of W2 use standard He init so that
            // gradients flow back through the query/key embeddings from day one.
            //
            // The td and logTd columns of W1 (the last two inputs, indices
            // MLPInputDim-2 and MLPInputDim-1) are zero-initialised.  This means
            // that at construction time the gate value is determined solely by the
            // content projections and is therefore identical for any two context
            // keys that share the same content embeddings.  When content is equal
            // the decay bias is purely -(exp(LogBaseDecayRate)*gate)*timeDiff,
            // which monotonically suppresses older keys regardless of the random
            // seed.  The optimiser is free to learn time-sensitive gating from
            // this baseline during training.
            float w1Std = MathF.Sqrt(2.0f / MLPInputDim);
            float w2Std = MathF.Sqrt(1.0f / hiddenDim);
            int tdColStart = MLPInputDim - 2; // indices for raw td and log(1+td)
            for (int h = 0; h < numHeads; h++)
            {
                for (int j = 0; j < hiddenDim; j++)
                {
                    for (int k = 0; k < MLPInputDim; k++)
                    {
                        // Zero the td/logTd input columns; keep content columns random.
                        W1[h, j, k] = k < tdColStart ? SampleGaussian(random) * w1Std : 0f;
                    }
                    W2[h, j] = SampleGaussian(random) * w2Std;
                }
                B2[h] = -2.2f;
                LogBaseDecayRate[h] = -2.3f;
            }
        }

        public (ContentAwareDecayGradients grads, float[,] dQueryEmbeddings, float[,] dKeyEmbeddings) Backward(float[,,] dDecayBias, ContentAwareDecayCache cache)
        {
            int queryLen = cache.TimeDiffs.GetLength(0);
            int keyLen = cache.TimeDiffs.GetLength(1);
            int projDim = ProjectionDim;
            int numBases = NumTimeBases;
            int rawDim = TimeRawDim;

            var grads = new ContentAwareDecayGradients(NumHeads, ContentDim, ProjectionDim, HiddenDim, MLPInputDim, numBases, rawDim);
            var dQueryEmbeddings = new float[queryLen, ContentDim];
            var dKeyEmbeddings = new float[keyLen, ContentDim];

            var dQueryProj = new float[NumHeads, queryLen, projDim];
            var dRefinedKey = new float[NumHeads, keyLen, projDim];

            // backward: MLP gating
            for (int h = 0; h < NumHeads; h++)
            {
                float baseRate = MathF.Exp(LogBaseDecayRate[h]);
                for (int qi = 0; qi < queryLen; qi++)
                {
                    for (int si = 0; si < keyLen; si++)
                    {
                        float dBias = dDecayBias[qi, si, h];
                        float gate = cache.Gates[qi, si, h];
                        float td = cache.TimeDiffs[qi, si];

                        float dGate = dBias * -(baseRate * td);
                        float dBaseRate = dBias * -(gate * td);
                        grads.LogBaseDecayRateGrad[h] += dBaseRate * baseRate;

                        float dLogit = dGate * gate * (1f - gate);
                        grads.B2Grad[h] += dLogit;

                        var dHidden = new float[HiddenDim];
                        for (int j = 0; j < HiddenDim; j++)
                        {
                            grads.W2Grad[h, j] += dLogit * cache.MLPHidden[qi, si, h, j];
                            dHidden[j] = dLogit * W2[h, j];
                        }

                        var dInput = new float[MLPInputDim];
                        for (int j = 0; j < HiddenDim; j++)
                        {
                            if (cache.MLPDropoutMask != null)
                            {
                                dHidden[j] *= cache.MLPDropoutMask[qi, si, h, j];
                            }

                            float preAct = cache.MLPHiddenPreAct[qi, si, h, j];
                            float dPreAct = preAct > 0f ? dHidden[j] : 0.01f * dHidden[j];

                            grads.B1Grad[h, j] += dPreAct;

                            for (int k = 0; k < MLPInputDim; k++)
                            {
                                grads.W1Grad[h, j, k] += dPreAct * cache.MLPInput[qi, si, h, k];
                                dInput[k] += dPreAct * W1[h, j, k];
                            }
                        }

                        int idx = 0;
                        for (int p = 0; p < projDim; p++)
                        {
                            dQueryProj[h, qi, p] += dInput[idx++];
                        }
                        for (int p = 0; p < projDim; p++)
                        {
                            dRefinedKey[h, si, p] += dInput[idx++];
                        }
                        for (int p = 0; p < projDim; p++)
                        {
                            float dI = dInput[idx++];
                            dQueryProj[h, qi, p] += dI * cache.RefinedKey[h, si, p];
                            dRefinedKey[h, si, p] += dI * cache.QueryProj[h, qi, p];
                        }
                    }
                }
            }

            // backward: Memory attention
            var dKeyProj = new float[NumHeads, keyLen, projDim];
            var dTimeEncoding = new float[NumHeads, keyLen, projDim];

            for (int h = 0; h < NumHeads; h++)
            {
                var dMemAttnOutput = new float[keyLen, projDim];
                for (int s = 0; s < keyLen; s++)
                {
                    for (int p = 0; p < projDim; p++)
                    {
                        float dRef = dRefinedKey[h, s, p];
                        dKeyProj[h, s, p] += dRef;
                        grads.MemAttnOutputBGrad[h, p] += dRef;

                        for (int q = 0; q < projDim; q++)
                        {
                            grads.MemAttnOutputWGrad[h, p, q] += dRef * cache.MemAttnOutput[h, s, q];
                            dMemAttnOutput[s, q] += dRef * MemAttnOutputW[h, p, q];
                        }
                    }
                }

                var dMemWeights = new float[keyLen, keyLen];
                for (int i = 0; i < keyLen; i++)
                {
                    for (int p = 0; p < projDim; p++)
                    {
                        float dOut = dMemAttnOutput[i, p];
                        for (int j = 0; j < keyLen; j++)
                        {
                            dKeyProj[h, j, p] += cache.MemAttnWeights[h, i, j] * dOut;
                            dMemWeights[i, j] += dOut * cache.KeyProj[h, j, p];
                        }
                    }
                }

                if (cache.MemAttnDropoutMask != null)
                {
                    for (int i = 0; i < keyLen; i++)
                    {
                        for (int j = 0; j < keyLen; j++)
                        {
                            dMemWeights[i, j] *= cache.MemAttnDropoutMask[h, i, j];
                        }
                    }
                }

                float memScale = 1.0f / MathF.Sqrt(projDim);
                var dMemScores = new float[keyLen, keyLen];

                for (int i = 0; i < keyLen; i++)
                {
                    float dotWD = 0;
                    for (int j = 0; j < keyLen; j++)
                    {
                        dotWD += cache.MemAttnWeights[h, i, j] * dMemWeights[i, j];
                    }
                    for (int j = 0; j < keyLen; j++)
                    {
                        dMemScores[i, j] = cache.MemAttnWeights[h, i, j] * (dMemWeights[i, j] - dotWD);
                    }
                }

                var dMemQ = new float[keyLen, projDim];
                var dMemK = new float[keyLen, projDim];

                for (int i = 0; i < keyLen; i++)
                {
                    for (int j = 0; j < keyLen; j++)
                    {
                        float ds = dMemScores[i, j] * memScale;
                        for (int p = 0; p < projDim; p++)
                        {
                            dMemQ[i, p] += ds * cache.MemAttnKInput[h, j, p];
                            dMemK[j, p] += ds * cache.MemAttnQInput[h, i, p];
                        }
                    }
                }

                for (int s = 0; s < keyLen; s++)
                {
                    for (int p = 0; p < projDim; p++)
                    {
                        float dTotal = dMemQ[s, p] + dMemK[s, p];
                        dKeyProj[h, s, p] += dTotal;
                        dTimeEncoding[h, s, p] += dTotal;
                    }
                }
            }

            // backward: Multi-scale time encoding
            for (int h = 0; h < NumHeads; h++)
            {
                for (int s = 0; s < keyLen; s++)
                {
                    float t = cache.KeyTimesFromRef != null ? cache.KeyTimesFromRef[s] : 0f;

                    var dRawFeatures = new float[rawDim];

                    for (int p = 0; p < projDim; p++)
                    {
                        float dTE = dTimeEncoding[h, s, p];
                        grads.TimeProjBiasGrad[h, p] += dTE;
                        for (int r = 0; r < rawDim; r++)
                        {
                            grads.TimeProjGrad[h, p, r] += dTE * cache.TimeRawFeatures[h, s, r];
                            dRawFeatures[r] += dTE * TimeProj[h, p, r];
                        }
                    }

                    for (int b = 0; b < numBases; b++)
                    {
                        float freq = MathF.Exp(TimeLogFreq[h, b]);
                        float sinA = cache.TimeRawFeatures[h, s, b * 2];
                        float cosA = cache.TimeRawFeatures[h, s, b * 2 + 1];
                        float dAngle = dRawFeatures[b * 2] * cosA + dRawFeatures[b * 2 + 1] * (-sinA);
                        grads.TimeLogFreqGrad[h, b] += dAngle * freq * t;
                    }
                }
            }

            // backward: Projections -> embeddings
            for (int h = 0; h < NumHeads; h++)
            {
                for (int qi = 0; qi < queryLen; qi++)
                {
                    for (int p = 0; p < projDim; p++)
                    {
                        float dQP = dQueryProj[h, qi, p];
                        grads.QueryProjectionBiasGrad[h, p] += dQP;

                        for (int d = 0; d < ContentDim; d++)
                        {
                            grads.QueryProjectionGrad[h, p, d] += dQP * cache.QueryEmbeddings[qi, d];
                            dQueryEmbeddings[qi, d] += dQP * QueryProjection[h, p, d];
                        }
                    }
                }

                for (int si = 0; si < keyLen; si++)
                {
                    for (int p = 0; p < projDim; p++)
                    {
                        float dKP = dKeyProj[h, si, p];
                        grads.KeyProjectionBiasGrad[h, p] += dKP;
                        for (int d = 0; d < ContentDim; d++)
                        {
                            grads.KeyProjectionGrad[h, p, d] += dKP * cache.KeyEmbeddings[si, d];
                            dKeyEmbeddings[si, d] += dKP * KeyProjection[h, p, d];
                        }
                    }
                }
            }

            if (WeightDecay > 0)
            {
                ApplyWeightDecayToGradients(grads);
            }


            return (grads, dQueryEmbeddings, dKeyEmbeddings);
        }

        private void ApplyWeightDecayToGradients(ContentAwareDecayGradients grads)
        {
            float wd = WeightDecay;
            for (int h = 0; h < NumHeads; h++)
            {
                for (int p = 0; p < ProjectionDim; p++)
                {
                    for (int d = 0; d < ContentDim; d++)
                    {
                        grads.QueryProjectionGrad[h, p, d] += wd * QueryProjection[h, p, d];
                        grads.KeyProjectionGrad[h, p, d] += wd * KeyProjection[h, p, d];
                    }

                    grads.QueryProjectionBiasGrad[h, p] += wd * QueryProjectionBias[h, p];
                    grads.KeyProjectionBiasGrad[h, p] += wd * KeyProjectionBias[h, p];

                    for (int r = 0; r < TimeRawDim; r++)
                    {
                        grads.TimeProjGrad[h, p, r] += wd * TimeProj[h, p, r];
                    }
                    grads.TimeProjBiasGrad[h, p] += wd * TimeProjBias[h, p];
                    for (int q = 0; q < ProjectionDim; q++)
                    {
                        grads.MemAttnOutputWGrad[h, p, q] += wd * MemAttnOutputW[h, p, q];
                    }
                    grads.MemAttnOutputBGrad[h, p] += wd * MemAttnOutputB[h, p];
                }
                for (int b = 0; b < NumTimeBases; b++)
                {
                    grads.TimeLogFreqGrad[h, b] += wd * TimeLogFreq[h, b];
                }
                for (int j = 0; j < HiddenDim; j++)
                {
                    for (int k = 0; k < MLPInputDim; k++)
                    {
                        grads.W1Grad[h, j, k] += wd * W1[h, j, k];
                    }
                    grads.B1Grad[h, j] += wd * B1[h, j];
                    grads.W2Grad[h, j] += wd * W2[h, j];
                }
                grads.B2Grad[h] += wd * B2[h];
                grads.LogBaseDecayRateGrad[h] += wd * LogBaseDecayRate[h];
            }
        }

        #region Save/Load
        public void WriteTo(System.IO.BinaryWriter w)
        {
            w.Write(NumHeads); 
            w.Write(ContentDim);
            w.Write(ProjectionDim);
            w.Write(HiddenDim);
            w.Write(NumTimeBases);
            w.Write(MemoryAttentionDropout); 
            w.Write(MLPDropout);
            w.Write(WeightDecay);
            Write3D(w, QueryProjection); 
            Write2D(w, QueryProjectionBias);
            Write3D(w, KeyProjection);
            Write2D(w, KeyProjectionBias);
            Write2D(w, TimeLogFreq); 
            Write3D(w, TimeProj); 
            Write2D(w, TimeProjBias);
            Write3D(w, MemAttnOutputW);
            Write2D(w, MemAttnOutputB);
            Write3D(w, W1); 
            Write2D(w, B1); 
            Write2D(w, W2);
            Write1D(w, B2); 
            Write1D(w, LogBaseDecayRate);
        }

        public static ContentAwareDecayNetwork ReadFrom(System.IO.BinaryReader r)
        {
            int nh = r.ReadInt32(), cd = r.ReadInt32(), pd = r.ReadInt32(), hd = r.ReadInt32();
            int ntb = r.ReadInt32();
            float memDrop = r.ReadSingle(), mlpDrop = r.ReadSingle(), wd = r.ReadSingle();
            int mlpIn = pd * 3 + 2; int rawDim = 2 * ntb;
            return new ContentAwareDecayNetwork
            {
                NumHeads = nh,
                ContentDim = cd,
                ProjectionDim = pd,
                HiddenDim = hd,
                MLPInputDim = mlpIn,
                NumTimeBases = ntb,
                TimeRawDim = rawDim,
                MemoryAttentionDropout = memDrop,
                MLPDropout = mlpDrop,
                WeightDecay = wd,
                QueryProjection = Read3D(r, nh, pd, cd),
                QueryProjectionBias = Read2D(r, nh, pd),
                KeyProjection = Read3D(r, nh, pd, cd),
                KeyProjectionBias = Read2D(r, nh, pd),
                TimeLogFreq = Read2D(r, nh, ntb),
                TimeProj = Read3D(r, nh, pd, rawDim),
                TimeProjBias = Read2D(r, nh, pd),
                MemAttnOutputW = Read3D(r, nh, pd, pd),
                MemAttnOutputB = Read2D(r, nh, pd),
                W1 = Read3D(r, nh, hd, mlpIn),
                B1 = Read2D(r, nh, hd),
                W2 = Read2D(r, nh, hd),
                B2 = Read1D(r, nh),
                LogBaseDecayRate = Read1D(r, nh)
            };
        }

        static void Write1D(System.IO.BinaryWriter w, float[] a)
        {
            foreach (var v in a)
            {
                w.Write(v);
            }
        }
        static void Write2D(System.IO.BinaryWriter w, float[,] a)
        {
            int d0 = a.GetLength(0), d1 = a.GetLength(1);

            for (int i = 0; i < d0; i++)
            {
                for (int j = 0; j < d1; j++)
                {
                    w.Write(a[i, j]);
                }
            }
        }
        static void Write3D(System.IO.BinaryWriter w, float[,,] a)
        {
            int d0 = a.GetLength(0), d1 = a.GetLength(1), d2 = a.GetLength(2);

            for (int i = 0; i < d0; i++)
            {
                for (int j = 0; j < d1; j++)
                {
                    for (int k = 0; k < d2; k++)
                    {
                        w.Write(a[i, j, k]);
                    }
                }
            }
        }
        static float[] Read1D(System.IO.BinaryReader r, int n)
        {
            var a = new float[n];
            for (int i = 0; i < n; i++)
            {
                a[i] = r.ReadSingle();
            }
            return a;
        }
        static float[,] Read2D(System.IO.BinaryReader r, int d0, int d1)
        {
            var a = new float[d0, d1];
            for (int i = 0; i < d0; i++)
            {
                for (int j = 0; j < d1; j++)
                {
                    a[i, j] = r.ReadSingle();
                }
            }
            return a;
        }
        static float[,,] Read3D(System.IO.BinaryReader r, int d0, int d1, int d2)
        {
            var a = new float[d0, d1, d2];
            for (int i = 0; i < d0; i++)
            {
                for (int j = 0; j < d1; j++)
                {
                    for (int k = 0; k < d2; k++)
                    {
                        a[i, j, k] = r.ReadSingle();
                    }
                }
            }
            return a;
        }
        #endregion

        static float SampleGaussian(Random rng)
        {
            float u1 = 1f - rng.NextSingle();
            float u2 = 1f - rng.NextSingle();
            return MathF.Sqrt(-2f * MathF.Log(u1)) * MathF.Cos(2f * MathF.PI * u2);
        }
    }

    public class ContentAwareDecayCache
    {
        public float[,] QueryEmbeddings { get; set; }
        public float[,] KeyEmbeddings { get; set; }
        public float[,] TimeDiffs { get; set; }
        public float[] KeyTimesFromRef { get; set; }
        public float[,,] QueryProj { get; set; }
        public float[,,] KeyProj { get; set; }
        public float[,,] TimeRawFeatures { get; set; }
        public float[,,] TimeEncoding { get; set; }
        public float[,,] MemAttnQInput { get; set; }
        public float[,,] MemAttnKInput { get; set; }
        public float[,,] MemAttnWeights { get; set; }
        public float[,,] MemAttnOutput { get; set; }
        public float[,,] RefinedKey { get; set; }
        public float[,,,] MLPInput { get; set; }
        public float[,,,] MLPHiddenPreAct { get; set; }
        public float[,,,] MLPHidden { get; set; }
        public float[,,] GateLogits { get; set; }
        public float[,,] Gates { get; set; }
        public float[,,] MemAttnDropoutMask { get; set; }
        public float[,,,] MLPDropoutMask { get; set; }
    }

    public class ContentAwareDecayGradients
    {
        public float[,,] QueryProjectionGrad { get; set; }
        public float[,] QueryProjectionBiasGrad { get; set; }
        public float[,,] KeyProjectionGrad { get; set; }
        public float[,] KeyProjectionBiasGrad { get; set; }
        public float[,] TimeLogFreqGrad { get; set; }
        public float[,,] TimeProjGrad { get; set; }
        public float[,] TimeProjBiasGrad { get; set; }
        public float[,,] MemAttnOutputWGrad { get; set; }
        public float[,] MemAttnOutputBGrad { get; set; }
        public float[,,] W1Grad { get; set; }
        public float[,] B1Grad { get; set; }
        public float[,] W2Grad { get; set; }
        public float[] B2Grad { get; set; }
        public float[] LogBaseDecayRateGrad { get; set; }

        public ContentAwareDecayGradients(int numHeads, int contentDim, int projDim, int hiddenDim, int mlpInputDim, int numTimeBases = 8, int timeRawDim = 16)
        {
            QueryProjectionGrad = new float[numHeads, projDim, contentDim];
            QueryProjectionBiasGrad = new float[numHeads, projDim];
            KeyProjectionGrad = new float[numHeads, projDim, contentDim];
            KeyProjectionBiasGrad = new float[numHeads, projDim];
            TimeLogFreqGrad = new float[numHeads, numTimeBases];
            TimeProjGrad = new float[numHeads, projDim, timeRawDim];
            TimeProjBiasGrad = new float[numHeads, projDim];
            MemAttnOutputWGrad = new float[numHeads, projDim, projDim];
            MemAttnOutputBGrad = new float[numHeads, projDim];
            W1Grad = new float[numHeads, hiddenDim, mlpInputDim];
            B1Grad = new float[numHeads, hiddenDim];
            W2Grad = new float[numHeads, hiddenDim];
            B2Grad = new float[numHeads];
            LogBaseDecayRateGrad = new float[numHeads];
        }

        public void Zero()
        {
            Zero3D(QueryProjectionGrad); Zero2D(QueryProjectionBiasGrad);
            Zero3D(KeyProjectionGrad); Zero2D(KeyProjectionBiasGrad);
            Zero2D(TimeLogFreqGrad);
            Zero3D(TimeProjGrad); Zero2D(TimeProjBiasGrad);
            Zero3D(MemAttnOutputWGrad); Zero2D(MemAttnOutputBGrad);
            Zero3D(W1Grad); Zero2D(B1Grad); Zero2D(W2Grad);
            Array.Clear(B2Grad, 0, B2Grad.Length);
            Array.Clear(LogBaseDecayRateGrad, 0, LogBaseDecayRateGrad.Length);
        }

        static void Zero2D(float[,] a)
        {
            int d0 = a.GetLength(0), d1 = a.GetLength(1);

            for (int i = 0; i < d0; i++)
            {
                for (int j = 0; j < d1; j++)
                {
                    a[i, j] = 0;
                }
            }
        }
        static void Zero3D(float[,,] a)
        {
            int d0 = a.GetLength(0), d1 = a.GetLength(1), d2 = a.GetLength(2);

            for (int i = 0; i < d0; i++)
            {
                for (int j = 0; j < d1; j++)
                {
                    for (int k = 0; k < d2; k++)
                    {
                        a[i, j, k] = 0;

                    }
                }
            }
        }
    }
}