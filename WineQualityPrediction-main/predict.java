import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.functions;
import org.apache.spark.sql.types.DataTypes;
import static org.apache.spark.sql.functions.col;
import static org.apache.spark.sql.functions.when;
import org.apache.spark.ml.classification.RandomForestClassificationModel;
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator;
import org.apache.spark.ml.feature.VectorAssembler;

import java.io.File;

public class WineQualityPrediction {

    public static void main(String[] args) {
        SparkSession spark = SparkSession.builder()
                .appName("QualityInferenceForWine")
                .getOrCreate();

        RandomForestClassificationModel rfModel = RandomForestClassificationModel.load("/src/qualitytrainingforwine");

        // Define schema for incoming CSV data
        StructType dataSchema = DataTypes.createStructType(new StructField[]{
                DataTypes.createStructField("fixed_acidity", DataTypes.DoubleType, true),
                DataTypes.createStructField("volatile_acidity", DataTypes.DoubleType, true),
                DataTypes.createStructField("citric_acid", DataTypes.DoubleType, true),
                DataTypes.createStructField("residual_sugar", DataTypes.DoubleType, true),
                DataTypes.createStructField("chlorides", DataTypes.DoubleType, true),
                DataTypes.createStructField("free_sulfur_dioxide", DataTypes.DoubleType, true),
                DataTypes.createStructField("total_sulfur_dioxide", DataTypes.DoubleType, true),
                DataTypes.createStructField("density", DataTypes.DoubleType, true),
                DataTypes.createStructField("pH", DataTypes.DoubleType, true),
                DataTypes.createStructField("sulphates", DataTypes.DoubleType, true),
                DataTypes.createStructField("alcohol", DataTypes.DoubleType, true),
                DataTypes.createStructField("quality", DataTypes.DoubleType, true)
        });

        // Set Temperature = 1
        double temperature = 1;

        // Process the dataset
        String filePath = "/tmp/your_file.csv"; // Replace this with your file path
        Dataset<Row> validationDataset = spark.read().format("csv")
                .schema(dataSchema)
                .option("header", "true")
                .option("delimiter", ";")
                .option("quote", "\"")
                .load(filePath);

        // Update quality column based on condition
        validationDataset = validationDataset.withColumn("quality",
                when(col("quality").gt(7), 1).otherwise(0));

        // Feature vectorization
        String[] featureColumns = validationDataset.columns();
        featureColumns = Arrays.copyOf(featureColumns, featureColumns.length - 1); // Remove last column (quality)
        VectorAssembler assembler = new VectorAssembler()
                .setInputCols(featureColumns)
                .setOutputCol("features");
        validationDataset = assembler.transform(validationDataset);

        // Predict using the model
        Dataset<Row> predictionResults = rfModel.transform(validationDataset);

        // Model evaluation
        MulticlassClassificationEvaluator evaluator = new MulticlassClassificationEvaluator()
                .setLabelCol("quality")
                .setPredictionCol("prediction")
                .setMetricName("f1");
        double f1Metric = evaluator.evaluate(predictionResults);

        // Print F1 score
        System.out.println("F1 Score: " + f1Metric);

        spark.stop();
    }
}
