// See https://aka.ms/new-console-template for more information
using Common;
using Microsoft.ML;
using Microsoft.ML.AutoML;
using Microsoft.ML.Data;
using MlProject.Model;

Console.WriteLine("Hello, World!");
//FindTheBestModel();
TrainTheModel();













static void TrainTheModel()
{
    Console.WriteLine("Start Trainig classifire");

    var mlContext = new MLContext(0);

    var trainingDataPath = "Data\\uci-news-aggregator.csv";
    var trainingDataView = mlContext.Data.LoadFromTextFile<ModelInput>(
        trainingDataPath,
        hasHeader: true,
        separatorChar: ',',
        allowQuoting: true);
    
    var preProcessingPipeLine = mlContext.Transforms.Conversion
        .MapValueToKey(inputColumnName: "Category", outputColumnName: "Label")
        .Append(mlContext.Transforms.Text.FeaturizeText(inputColumnName:"Title" , outputColumnName:"Features"))
        .Append(mlContext.Transforms.NormalizeMinMax("Features"))
        .AppendCacheCheckpoint(mlContext);

    var trainer = mlContext.MulticlassClassification.Trainers
        .OneVersusAll(mlContext.BinaryClassification.Trainers.AveragedPerceptron());

    var trainingPipeLine = preProcessingPipeLine.Append(trainer)
        .Append(mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel"));

    Console.WriteLine("Cross Validation Model");


    var cvResult = mlContext.MulticlassClassification.CrossValidate(trainingDataView, trainingPipeLine);

    var microAccuracy = cvResult.Average(m => m.Metrics.MicroAccuracy);
    var macroAccuracy = cvResult.Average(m => m.Metrics.MacroAccuracy);
    var logLossAccuracy = cvResult.Average(m => m.Metrics.LogLoss);


    Console.WriteLine("--------------------");
    Console.WriteLine($"Micro Accuracy:{microAccuracy:0.###}");
    Console.WriteLine($"Macro Accuracy:{macroAccuracy:0.###}");
    Console.WriteLine($"Log loss Accuracy:{logLossAccuracy:0.###}");

    var finalModel = trainingPipeLine.Fit(trainingDataView);
    Console.WriteLine("Saving Model");
    if (!Directory.Exists("Model"))
    {
        Directory.CreateDirectory("Model");
    }
    var modelPath = "Model\\NewsClassificationModel.zip";
    mlContext.Model.Save(finalModel,trainingDataView.Schema , modelPath);


}
static void FindTheBestModel()
{
    Console.WriteLine("Start Finding");

    var mlContext = new MLContext(0);
    var trainingDataPath = "Data\\uci-news-aggregator.csv";

    var trainingDataView = mlContext.Data.LoadFromTextFile<ModelInput>(
        trainingDataPath,
        hasHeader: true,
        separatorChar: ',',
        allowQuoting: true
        );

    var preProccessingPipeLine = mlContext.Transforms.Conversion
        .MapValueToKey(inputColumnName: "Category", outputColumnName: "Category");

    var mappedInputData = preProccessingPipeLine.Fit(trainingDataView).Transform(trainingDataView);

    var experimentSettings = new MulticlassExperimentSettings
    {
        MaxExperimentTimeInSeconds = 300,
        CacheBeforeTrainer = CacheBeforeTrainer.On,
        OptimizingMetric = MulticlassClassificationMetric.MicroAccuracy,
        CacheDirectoryName = "D://MLNet/Cashe"
    };

    var experiment = mlContext.Auto().CreateMulticlassClassificationExperiment(experimentSettings);
   
    var experimentResult =
               experiment.Execute(
                   trainData: mappedInputData,
                   labelColumnName: "Category",
                   progressHandler: new MulticlassExperimentProgressHandler());
    
    
    Console.WriteLine("Metrics from best run :");

    var metrics = experimentResult.BestRun.ValidationMetrics;

    Console.WriteLine($"Micro Accuracy : {metrics.MicroAccuracy:0.##}");
    Console.WriteLine($"Micro Accuracy : {metrics.MicroAccuracy:0.##}");
}