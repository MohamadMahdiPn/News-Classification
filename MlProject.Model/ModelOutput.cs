using Microsoft.ML.Data;

namespace MlProject.Model
{
    public class ModelOutput
    {
        [ColumnName("PredictedLabel")]
        public string Category { get; set; }

        public float[] Score { get; set; }
    }
}