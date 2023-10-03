using Microsoft.ML.Data;

namespace MlProject.Model
{
    public class ModelInput
    {
        [LoadColumn(1)]
        public string Title { get; set; }

        [LoadColumn(4)]
        public string Category { get; set; }
    }
}