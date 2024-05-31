using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using System.Text;
using System.Threading.Tasks;

namespace SkyBet.GeneralPRNGCracking
{
    public class MersenneTwister
    {
        private const int N = 624;  // Length of the state array
        private const int M = 397;  // Used in the generation algorithm
        private const uint MatrixA = 0x9908b0df;  // Used in the generation algorithm
        private const uint UpperMask = 0x80000000;  // Used in the generation algorithm
        private const uint LowerMask = 0x7fffffff;  // Used in the generation algorithm

        private readonly uint[] mt = new uint[N];  // State vector of the generator
        private int index;  // Current position in the state array

        public MersenneTwister(uint seed)
        {
            mt[0] = seed;  // Assign the seed value to the first element of the state array
            for (int i = 1; i < N; i++)
            {
                // Generate the next state value based on the previous state value
                mt[i] = (uint)(1812433253 * (mt[i - 1] ^ (mt[i - 1] >> 30)) + i);
            }
        }

        public uint Next()
        {
            if (index == 0)
            {
                GenerateNumbers();  // Generate a new set of numbers
            }

            uint y = mt[index];
            y ^= (y >> 11);
            y ^= ((y << 7) & MatrixA);
            y ^= ((y << 15) & 0xefc60000);
            y ^= (y >> 18);

            index = (index + 1) % N;  // Update the index for the next iteration
            return y;  // Return the generated random number
        }

        private void GenerateNumbers()
        {
            for (int i = 0; i < N; i++)
            {
                // Combine bits from the current and next state values to generate a new state value
                uint y = (mt[i] & UpperMask) | (mt[(i + 1) % N] & LowerMask);
                mt[i] = mt[(i + M) % N] ^ (y >> 1) ^ ((y & 1) * MatrixA);
            }
        }

        /// <summary>
        /// This is how a rng is restricted to a spesific domain of outputs. this means if there outputs are say 3 there is a range of random nummbers that has a remainder of 3 aftr modulo has taken place.
        /// is a subset of values within an rng has to be less random, as the period has been reduced significantly. if the range size is 10, and the output is 0 you know this random number is devisable exactly by 10, you have just droped the period by a factor of 10.
        /// You could do this for multiple different output with restricted domin space, and use this to make our nureal nets far more efficent.
        /// </summary>
        /// <param name="minValue"></param>
        /// <param name="maxValue"></param>
        /// <returns></returns>
        /// <exception cref="ArgumentException"></exception>
        public int NextIntInRange(int minValue, int maxValue)
        {
            if (minValue > maxValue)
            {
                throw new ArgumentException("minValue must be less than or equal to maxValue.");
            }

            uint randomValue = Next();
            int rangeSize = maxValue - minValue + 1;
            int result = (int)(randomValue % rangeSize) + minValue;
            return result;
        }

        //there is another common method to reduce the range of a random number to fit within a desired range. You can scale the random number using the ratio of the range sizes.
        //This method helps to avoid the biases introduced by the modulo operation. Here's how you can do it:

        public int NextIntInRangeUSingDouble(int minValue, int maxValue)
        {
            if (minValue > maxValue)
            {
                throw new ArgumentException("minValue must be less than or equal to maxValue.");
            }

            double randomValue = NextDouble(); // Generate a random double between 0 and 1
            int rangeSize = maxValue - minValue + 1;
            int result = (int)(randomValue * rangeSize) + minValue;
            return result;
        }
        public int NextIntInRange2(int minValue, int maxValue)
        {
            if (minValue > maxValue)
            {
                throw new ArgumentException("minValue must be less than or equal to maxValue.");
            }

            uint randomValue = Next(); // Generate a random uint
            double proportion = (double)randomValue / uint.MaxValue;
            int rangeSize = maxValue - minValue + 1;
            int result = (int)(proportion * rangeSize) + minValue;
            return result;
        }
        public double NextDouble()
        {
            // Assuming the PRNG generates a random uint, convert it to a double between 0 and 1
            return (Next() / (double)uint.MaxValue);
        }
    }
    public class MersenneTwisterRecoverState
    {
        public void Run()
        {
            MersenneTwister mersenneTwister = new MersenneTwister(100);

            List<uint> ConsectutiveOutputs = new List<uint>();
            for (int i = 0; i < 624; i++)
            {
                ConsectutiveOutputs.Add(mersenneTwister.Next());
            }
            MersenneTwister recoveredGenerator = new MersenneTwister(0);

            typeof(MersenneTwister).GetField("mt", BindingFlags.NonPublic | BindingFlags.Instance).SetValue(recoveredGenerator, ConsectutiveOutputs.ToArray());
        }
        private uint[] RecoverState(uint[] outputs)
        {
            if (outputs.Length != 624)
                throw new ArgumentException("You must provide exactly 624 consecutive outputs to recover the internal state.");

            uint[] state = new uint[624];

            for (int i = 0; i < 624; i++)
            {
                state[i] = Untemper(outputs[i]);
            }

            return state;
        }

        private uint Untemper(uint y)
        {
            y ^= (y >> 18);
            y ^= ((y << 15) & 0xefc60000);
            y = UndoLeftBitshiftXor(y, 7, 0x9D2C5680);
            y = UndoRightBitshiftXor(y, 11);
            return y;
        }

        private uint UndoLeftBitshiftXor(uint y, int shift, uint mask)
        {
            uint result = y;
            for (int i = 0; i < 32; i += shift)
            {
                result ^= (y << shift) & mask;
                y = result;
            }
            return result;
        }

        public static uint UndoRightBitshiftXor(uint y, int shift)
        {
            uint result = y;
            for (int i = 0; i < 32; i += shift)
            {
                result ^= y >> shift;
                y = result;
            }
            return result;
        }

    }


}
