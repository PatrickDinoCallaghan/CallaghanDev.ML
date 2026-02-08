using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace CallaghanDev.ML.Transformers
{
    public class AttentionGradients
    {
        public float[,] WQ_Grad { get; set; }
        public float[,] WK_Grad { get; set; }
        public float[,] WV_Grad { get; set; }
        public float[,] WO_Grad { get; set; }
        public float[] BiasQ_Grad { get; set; }
        public float[] BiasK_Grad { get; set; }
        public float[] BiasV_Grad { get; set; }
        public float[] BiasO_Grad { get; set; }

        public AttentionGradients(int embeddingDim)
        {
            WQ_Grad = new float[embeddingDim, embeddingDim];
            WK_Grad = new float[embeddingDim, embeddingDim];
            WV_Grad = new float[embeddingDim, embeddingDim];
            WO_Grad = new float[embeddingDim, embeddingDim];
            BiasQ_Grad = new float[embeddingDim];
            BiasK_Grad = new float[embeddingDim];
            BiasV_Grad = new float[embeddingDim];
            BiasO_Grad = new float[embeddingDim];
        }

        public void Zero()
        {
            ZeroMatrix(WQ_Grad);
            ZeroMatrix(WK_Grad);
            ZeroMatrix(WV_Grad);
            ZeroMatrix(WO_Grad);
            Array.Clear(BiasQ_Grad, 0, BiasQ_Grad.Length);
            Array.Clear(BiasK_Grad, 0, BiasK_Grad.Length);
            Array.Clear(BiasV_Grad, 0, BiasV_Grad.Length);
            Array.Clear(BiasO_Grad, 0, BiasO_Grad.Length);
        }

        private void ZeroMatrix(float[,] matrix)
        {
            int rows = matrix.GetLength(0);
            int cols = matrix.GetLength(1);
            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < cols; j++)
                {
                    matrix[i, j] = 0;
                }
            }
        }
    }

}
