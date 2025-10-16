import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

def main():
    with mlflow.start_run(run_name="demo-run") as run:
        print(f"Starting MLflow run: {run.info.run_name}")

        texts = np.array([
            "sparrow", "eagle", "parrot", "robin",
            "lion", "tiger", "elephant", "bear",
            "rose", "tulip", "daisy", "sunflower"
        ])
        categories = np.array([
            "bird", "bird", "bird", "bird",
            "animal", "animal", "animal", "animal",
            "flower", "flower", "flower", "flower"
        ])

        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(categories)
        
        print(f"Numerical labels map to: {list(enumerate(label_encoder.classes_))}")

        X_train_text, X_test_text, y_train_encoded, y_test_encoded = train_test_split(
            texts, y_encoded, test_size=0.3, random_state=42, stratify=y_encoded
        )

        text_classification_pipeline = Pipeline([
            ('tfidf', TfidfVectorizer()),
            ('logistic_regression', LogisticRegression(random_state=42))
        ])

        mlflow.log_param("C", text_classification_pipeline.named_steps['logistic_regression'].C)
        mlflow.log_param("solver", text_classification_pipeline.named_steps['logistic_regression'].solver)

        print("Training the text classification pipeline...")
        text_classification_pipeline.fit(X_train_text, y_train_encoded)

        y_pred_encoded = text_classification_pipeline.predict(X_test_text)
        accuracy = accuracy_score(y_test_encoded, y_pred_encoded)
        
        print(f"Logging metric: accuracy={accuracy:.4f}")
        mlflow.log_metric("accuracy", accuracy)

        print("Logging the text classification model pipeline...")
        
        input_example = np.array(["sample text", "another sample"]) 

        mlflow.sklearn.log_model(
            sk_model=text_classification_pipeline,
            name="model", 
            registered_model_name="my-logistic-reg-model",
            input_example=input_example
        )
        print("MLflow run completed successfully!")

if __name__ == "__main__":
    main()