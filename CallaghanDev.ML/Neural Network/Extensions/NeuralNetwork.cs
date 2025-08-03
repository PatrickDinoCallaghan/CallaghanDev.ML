using Newtonsoft.Json;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace CallaghanDev.ML.Extensions
{
    public static class NeuralNetwork
    {
        public static List<Layer> GetLayers(this ML.NeuralNetwork nn) => nn.GetInternalData().layers.ToList();

        public static Parameters GetParametersCopy(this ML.NeuralNetwork nn)
            => JsonConvert.DeserializeObject<Parameters>(JsonConvert.SerializeObject(nn.GetInternalData().parameters));

        public static Data GetDataCopy(this ML.NeuralNetwork nn)
            => JsonConvert.DeserializeObject<Data>(JsonConvert.SerializeObject(nn.GetInternalData()));

        public static void RestoreState(this ML.NeuralNetwork nn, Data snapshot)
        {
            nn.RestoreData(snapshot);
        }

        public static Data GetInternalData(this ML.NeuralNetwork nn)
        {
            var field = typeof(ML.NeuralNetwork).GetField("data", System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance);
            return (Data)field.GetValue(nn);
        }

        public static void RestoreData(this ML.NeuralNetwork nn, Data data)
        {
            var field = typeof(ML.NeuralNetwork).GetField("data", System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance);
            field.SetValue(nn, data);

            var methodInit = typeof(ML.NeuralNetwork).GetMethod("InitAcceleration", System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance);
            var methodCost = typeof(ML.NeuralNetwork).GetMethod("InitCostFunction", System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance);

            methodInit?.Invoke(nn, null);
            methodCost?.Invoke(nn, null);
        }

        public static string GetNetworkStructure(this ML.NeuralNetwork nn)
        {
            List<Layer> layers = nn.GetLayers();
            if (layers == null || layers.Count == 0)
            {
                return "No layers defined.";
            }

            var widthString = string.Join(" -> ", layers.Select(l => l.Size));
            var activationString = string.Join(" -> ", layers.Select(l => l.ActivationType.ToString().ToLower()));

            return $"Layer widths:    {widthString}\nActivations:     {activationString}";
        }

        public static long CalculateParameterCount(this Parameters parameters)
        {
            if (parameters?.LayerWidths == null) return 0;
            return CalculateParameterCount(parameters.LayerWidths);
        }

        public static long CalculateParameterCount(IList<int> layerWidths)
        {
            if (layerWidths == null) throw new ArgumentNullException(nameof(layerWidths));
            if (layerWidths.Count < 2) return 0;
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
