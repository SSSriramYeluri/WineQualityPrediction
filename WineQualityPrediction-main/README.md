# WineQualityPrediction

QualityTrainingForWine.java: This class focuses on training a Random Forest Classifier model using Spark's MLlib. It reads a training dataset, preprocesses it, trains the model, evaluates its performance, and saves the trained model.
WineQualityPrediction.java: This class is responsible for loading the trained model and using it to make predictions on a separate dataset for wine quality prediction.
Dockerfile: Defines the Docker container configuration to run the Java application.

Requirements
Apache Spark
Java 11
Docker

Setting up the Environment
Install Apache Spark: Follow the installation instructions provided by Apache Spark for your specific environment.
Install Java 11: Ensure Java 11 is installed on your machine or container.
Install Docker: If you wish to run the application within a Docker container, install Docker following the official instructions.

Compile Java files: Compile the Java files using a Java compiler or an integrated development environment (IDE) like IntelliJ IDEA or Eclipse.
Execute QualityTrainingForWine.java: Run the QualityTrainingForWine.java file to train the model.
Execute WineQualityPrediction.java: Run the WineQualityPrediction.java file to make predictions using the trained model.

Build Docker Image: Build the Docker image using the provided Dockerfile. Run the following command in the terminal:
docker build -t wine-quality-prediction .


Run Docker Container: Start the Docker container by executing:
docker run -p 8080:8080 wine-quality-prediction

QualityTrainingForWine.java: This file contains the code to preprocess the dataset, train a Random Forest Classifier model, and evaluate its performance.
WineQualityPrediction.java: This file contains code to load the trained model and make predictions on a different dataset.
Dockerfile: Defines the Docker container setup for running the Java application.
qualitytrainingforwine.jar: The compiled JAR file containing the Java classes to execute the Spark job.

Make sure to adjust file paths in the code to match the location of your dataset.
Ensure the necessary dependencies are available and properly configured in your environment before running the code.
