using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace CallaghanDev.ML
{
    public class TrainingDataCollection
    {
        List<TrainingData> _data = new List<TrainingData>();


        public TrainingData this[int index]
        {
            get { return _data[index]; }
        }

        public void AddData(float[][] inputValues, float[][] expectedOutputValues)
        {
            _data.Clear();
            if (inputValues.Length != expectedOutputValues.Length)
            {
                throw new ArgumentException($"The number of input data sets ({inputValues.Length}) must match the number of expected output data sets ({expectedOutputValues.Length}). Each input data set should correspond to exactly one output data set.");
            }

            for (int i = 0; i < inputValues.Length; i++)
            {
                _data.Add(new TrainingData(inputValues[i], expectedOutputValues[i]));
            }
        }
        public int Length { get { return _data.Count; } }
    }
}
