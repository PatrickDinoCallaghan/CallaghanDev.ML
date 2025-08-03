using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace CallaghanDev.ML.Extensions
{
    public static class RandomExtensions
    {
        public static bool NextBool(this Random random)
        {
            return random.Next(2) == 1;
        }

        public static void Shuffle<T>(this Random random, T[] array)
        {
            for (int i = array.Length - 1; i > 0; i--)
            {
                int j = random.Next(i + 1);
                (array[i], array[j]) = (array[j], array[i]);
            }
        }

        public static float NextGaussian(this Random random, float mean = 0f, float standardDeviation = 1f)
        {
            // Box-Muller transformation for Gaussian distribution
            if (random.Next(2) == 0)
            {
                double u1 = 1.0 - random.NextDouble();
                double u2 = 1.0 - random.NextDouble();
                double randStdNormal = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Sin(2.0 * Math.PI * u2);
                return mean + standardDeviation * (float)randStdNormal;
            }
            else
            {
                double u1 = 1.0 - random.NextDouble();
                double u2 = 1.0 - random.NextDouble();
                double randStdNormal = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2);
                return mean + standardDeviation * (float)randStdNormal;
            }
        }
    }
}
