namespace CallaghanDev.ML.NN
{
    public class SettingsManager
    {

        private double Evaluate(NeuralNetwork nn, double[][] inputs, double[][] outputs)
        {
            double totalError = 0.0;
            List<double[]> predictions = new List<double[]>();

            for (int i = 0; i < inputs.Length; i++)
            {
                double[] prediction = nn.Predict(inputs[i]);
                predictions.Add(prediction);
                totalError += CalculateError(prediction, outputs[i]);
            }
            double InputSensitive = InputSensitivity(inputs, predictions.ToArray());

            return (inputs.Length / totalError) * InputSensitivity(inputs, predictions.ToArray());
        }

        private double InputSensitivity(double[][] inputs, double[][] outputs)
        {
            double totalError = 0.0;

            List<double[]> CheckedArray = new List<double[]>();
            List<double> SensitivityList = new List<double>();

            for (int i = 1; i < inputs.Length; i++)
            {
                if (CheckedArray.Where(r => ArrayIsEqual(r, inputs[i]) == true).Count() == 0)
                {
                    double Dec = 0;
                    double SensitivityCurrent = 0;

                    for (int j = 0; j < outputs[i].Length; j++)
                    {
                        SensitivityCurrent += Math.Abs(outputs[i][j] - outputs[i - 1][j]);
                        Dec += outputs[i][j];
                    }

                    SensitivityList.Add(SensitivityCurrent / Dec);
                }
            }

            double Sensitivity = SensitivityList.Average();
            return Sensitivity;
        }

        private bool ArrayIsEqual(double[] array1, double[] array2)
        {
            if (array1.Length != array2.Length)
            {
                return false;
            }
            else
            {
                for (int i = 0; i < array1.Length; i++)
                {
                    if (array1[i] != array2[i])
                    {
                        return false;
                    }
                }
                return true;
            }
        }

        private double CalculateError(double[] prediction, double[] actual)
        {
            double error = 0.0;
            for (int i = 0; i < prediction.Length; i++)
            {
                error += Math.Pow(prediction[i] - actual[i], 2);
            }
            return error;
        }

        
    }
}
