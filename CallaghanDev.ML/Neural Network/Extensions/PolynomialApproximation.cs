using CallaghanDev.ML.Enums;
using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using MathNet.Symbolics;

namespace CallaghanDev.ML.Neural_Network.Extensions
{
    public static class PolynomialApproximation
    {
        /// <summary>
        /// Generate polynomial approximations for all output neurons as individual functions.
        /// </summary>
        /// <param name="data">trained network data.</param>
        /// <param name="degree">Polynomial degree for each activation approximation (e.g., 7..11).</param>
        /// <param name="zMin">Lower bound of pre-activation range for approximation (e.g., -6).</param>
        /// <param name="zMax">Upper bound of pre-activation range for approximation (e.g., 6).</param>
        /// <param name="leakySlope">Slope for LeakyReLU (usually 0.01).</param>
        /// <returns>Array of simplified polynomial expressions, one for each output neuron.</returns>
        public static Expression[] GenerateOutputExpressions(Data data, int degree = 9, float zMin = -6f, float zMax = 6f, float leakySlope = 0.01f)
        {
            if (data == null || data.layers == null || data.layers.Length < 2)
            {
                throw new ArgumentException("Network is not initialized (need at least input + one layer).");
            }

            var L = data.layers;
            if (L[0].InputSize != 0)
            {
                throw new InvalidOperationException("Expected layer[0] to be the input layer (InputSize==0).");
            }

            int inputCount = L[0].Size;


            var inputVars = new Dictionary<string, Expression>();
            var prev = new List<Expression>(inputCount);

            for (int j = 0; j < inputCount; j++)
            {
                var varName = $"x{j}";
                var symbol = Expression.Symbol(varName);
                inputVars[varName] = symbol;
                prev.Add(symbol);
            }

            var polyCache = new Dictionary<(ActivationType, int, float, float, float), float[]>();

            for (int li = 1; li < L.Length; li++)
            {
                var layer = L[li];
                int thisSize = layer.Size;
                int prevSize = layer.InputSize;

                var next = new List<Expression>(thisSize);

                for (int ni = 0; ni < thisSize; ni++)
                {
                    // z = bias + sum_j (w_ij * prev_j)
                    Expression z = (double)layer.Biases[ni];

                    for (int pj = 0; pj < prevSize; pj++)
                    {
                        float w = layer.Weights[ni, pj];
                        if (w == 0f) continue;

                        z = z + (double)w * prev[pj];
                    }

                    // Apply activation as polynomial-only
                    Expression yExpr;
                    if (layer.ActivationType == ActivationType.None)
                    {
                        yExpr = z;
                    }
                    else
                    {
                        var key = (layer.ActivationType, degree, zMin, zMax, leakySlope);
                        if (!polyCache.TryGetValue(key, out var powerCoeffs))
                        {
                            var act = MakeActivation(layer.ActivationType, leakySlope);
                            powerCoeffs = BuildPowerCoeffsChebyshev(act, degree, zMin, zMax);
                            polyCache[key] = powerCoeffs;
                        }
                        yExpr = CreatePolynomialExpression(powerCoeffs, z, zMin, zMax);
                    }

                    next.Add(yExpr);
                }

                prev = next;
            }

            return prev.ToArray();
        }

        private static Expression CreatePolynomialExpression(float[] powerCoeffs, Expression z, float zMin, float zMax)
        {
            //  t = (2*z - (zMin+zMax)) / (zMax - zMin)
            var two = (Expression)2.0;
            var zMinPlusZMax = (Expression)(zMin + zMax);
            var zMaxMinusZMin = (Expression)(zMax - zMin);

            var t = (two * z - zMinPlusZMax) / zMaxMinusZMin;

            // Build polynomial using Horner's method: P(t) = c0 + c1*t + c2*t^2 + ... + cD*t^D
            // Horner form: ((cD)*t + c_{D-1})*t + ... + c0
            int D = powerCoeffs.Length - 1;
            Expression result = (Expression)(double)powerCoeffs[D];

            for (int m = D - 1; m >= 0; m--)
            {
                result = result * t + (Expression)(double)powerCoeffs[m];
            }

            return result;
        }

        private static Func<double, double> MakeActivation(ActivationType type, float leakySlope)
        {
            switch (type)
            {
                case ActivationType.Sigmoid:
                    return z => 1.0 / (1.0 + Math.Exp(-z));
                case ActivationType.Tanh:
                    return z => Math.Tanh(z);
                case ActivationType.Relu:
                    return z => z > 0.0 ? z : 0.0;
                case ActivationType.Leakyrelu:
                    return z => z > 0.0 ? z : leakySlope * z;
                case ActivationType.Swish:
                    return z => z / (1.0 + Math.Exp(-z)); // z * sigmoid(z)
                case ActivationType.None:
                default:
                    return z => z;
            }
        }

        /// <summary>
        /// Build power-basis coefficients c0..cD for the Chebyshev approximation of f on [zMin,zMax],
        /// by: (1) sampling f at Chebyshev nodes, (2) computing Chebyshev a_k, (3) converting Σ a_k T_k(t)
        /// to power basis in t, then Horner in t at emit time.
        /// </summary>
        private static float[] BuildPowerCoeffsChebyshev(Func<double, double> f, int degree, float zMin, float zMax)
        {
            // Chebyshev nodes on t in [-1,1]
            int N = degree + 1;
            var vals = new double[N];
            var theta = new double[N];
            for (int j = 0; j < N; j++)
            {
                // Chebyshev-Gauss nodes: t_j = cos((j+0.5)*pi/N)
                double th = (j + 0.5) * Math.PI / N;
                theta[j] = th;
                double t = Math.Cos(th);
                double z = 0.5 * ((zMax - zMin) * t + (zMax + zMin)); // map t -> z
                vals[j] = f(z);
            }

            // Chebyshev coefficients a_k via discrete cosine (type): a_k = (2/N) * Σ vals[j]*cos(k*theta_j)
            var a = new double[N];
            for (int k = 0; k <= degree; k++)
            {
                double s = 0.0;
                for (int j = 0; j < N; j++)
                    s += vals[j] * Math.Cos(k * theta[j]);
                a[k] = (2.0 / N) * s;
            }
            a[0] *= 0.5; // standard scaling for Chebyshev series (a0/2 + Σ a_k T_k)

            // Convert Σ a_k T_k(t) to power basis: P(t) = Σ c_m t^m
            var c = ChebyshevToPower(a, degree);

            // Convert to float[]
            var cf = new float[c.Length];
            for (int i = 0; i < c.Length; i++) cf[i] = (float)c[i];
            return cf;
        }

        /// <summary>
        /// Convert Chebyshev series coefficients a_k (k=0..D) to power-basis coefficients c_m (m=0..D),
        /// using T_0(t)=1, T_1(t)=t, T_{k+1}=2 t T_k - T_{k-1}. Accumulate into power basis.
        /// </summary>
        private static double[] ChebyshevToPower(double[] a, int degree)
        {
            // power[k][m] = coefficient of t^m in T_k(t)
            var power = new List<double[]>();
            // T_0 = 1
            var T0 = new double[degree + 1]; T0[0] = 1.0;
            // T_1 = t
            var T1 = new double[degree + 1]; T1[1] = 1.0;

            power.Add(T0);
            if (degree >= 1) power.Add(T1);

            for (int k = 1; k < degree; k++)
            {
                var Tk = power[k];
                var Tkm1 = power[k - 1];
                var Tkp1 = new double[degree + 1];

                // T_{k+1} = 2 t T_k - T_{k-1}
                for (int m = 0; m <= degree; m++)
                {
                    double coeffTk = Tk[m];
                    if (coeffTk != 0.0 && m + 1 <= degree)
                    {
                        Tkp1[m + 1] += 2.0 * coeffTk;   // 2 * t * (t^m) = 2 * t^{m+1}
                                                        // - T_{k-1}
                    }
                    Tkp1[m] += -Tkm1[m];
                }
                power.Add(Tkp1);
            }

            // Combine: P(t) = Σ a_k T_k(t)
            var c = new double[degree + 1];
            for (int k = 0; k <= degree; k++)
            {
                var Tk = power[k];
                double ak = a[k];
                if (ak == 0.0)
                {
                    continue;
                }
                for (int m = 0; m <= degree; m++)
                {
                    c[m] += ak * Tk[m];
                }
            }
            return c;
        }

    }
}