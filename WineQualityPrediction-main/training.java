import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.types.*;

import static org.apache.spark.sql.functions.*;

import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.ml.classification.RandomForestClassificationModel;
import org.apache.spark.ml.classification.RandomForestClassifier;
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator;

public class WineQualityPrediction {

    public static void main(String[] args) {
        SparkConf conf = new SparkConf().setAppName("QualityTrainingForWine").setMaster("local");
        JavaSparkContext sc = new JavaSparkContext(conf);
        SparkSession spark = SparkSession.builder().appName("QualityTrainingForWine").getOrCreate();

        StructType csvSchema = new StructType(new StructField[]{
            new StructField("fixed_acidity", DataTypes.DoubleType, true, Metadata.empty()),
            new StructField("volatile_acidity", DataTypes.DoubleType, true, Metadata.empty()),
            new StructField("citric_acid", DataTypes.DoubleType, true, Metadata.empty()),
            new StructField("residual_sugar", DataTypes.DoubleType, true, Metadata.empty()),
            new StructField("chlorides", DataTypes.DoubleType, true, Metadata.empty()),
            new StructField("free_sulfur_dioxide", DataTypes.DoubleType, true, Metadata.empty()),
            new StructField("total_sulfur_dioxide", DataTypes.DoubleType, true, Metadata.empty()),
            new StructField("density", DataTypes.DoubleType, true, Metadata.empty()),
            new StructField("pH", DataTypes.DoubleType, true, Metadata.empty()),
            new StructField("sulphates", DataTypes.DoubleType, true, Metadata.empty()),
            new StructField("alcohol", DataTypes.DoubleType, true, Metadata.empty()),
            new StructField("quality", DataTypes.DoubleType, true, Metadata.empty())
        });

        Dataset<Row> dataset = spark.read()
            .format("csv")
            .schema(csvSchema)
            .option("header", "true")
            .option("delimiter", ";")
            .option("quote", "\"")
            .option("ignoreLeadingWhiteSpace", true)
            .option("ignoreTrailingWhiteSpace", true)
            .load("file:///home/ec2-user/WineQualityPrediction/TrainingDataset.csv");

        dataset = dataset.toDF(dataset.columns());

        dataset = dataset.withColumn("quality", when(col("quality").gt(7), 1).otherwise(0));

        String[] featureCols = dataset.columns();
        featureCols = Arrays.copyOfRange(featureCols, 0, featureCols.length - 1);

        VectorAssembler vectorAssembler = new VectorAssembler()
            .setInputCols(featureCols)
            .setOutputCol("features");

        dataset = vectorAssembler.transform(dataset);

        Dataset<Row>[] splits = dataset.randomSplit(new double[]{0.8, 0.2});
        Dataset<Row> trainData = splits[0];
        Dataset<Row> testData = splits[1];

        RandomForestClassifier randomForest = new RandomForestClassifier()
            .setLabelCol("quality")
            .setFeaturesCol("features")
            .setNumTrees(200);

        RandomForestClassificationModel trainedModel = randomForest.fit(trainData);

        Dataset<Row> testPredictions = trainedModel.transform(testData);

        MulticlassClassificationEvaluator evaluator = new MulticlassClassificationEvaluator()
            .setLabelCol("quality")
            .setPredictionCol("prediction")
            .setMetricName("f1");

        double f1Metric = evaluator.evaluate(testPredictions);
        System.out.println("Evaluated F1 Score: " + f1Metric);

        trainedModel.save("file:///home/ec2-user/WineQualityPrediction/qualitytrainingforwine");

        spark.stop();
    }
}
