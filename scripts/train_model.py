import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin
from xgboost import XGBClassifier

class Preprocessor(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, df):
        df = df.copy()

        # Feature engineering
        df["bmi"] = df["weight"] / ((df["height"] / 100) ** 2)
        df["gender_2"] = (df["gender"] == 2).astype(int)

        # Filter BMI
        df = df[(df["bmi"] >= 10) & (df["bmi"] <= 60)]
        Q1_bmi = df["bmi"].quantile(0.25)
        Q3_bmi = df["bmi"].quantile(0.75)
        IQR_bmi = Q3_bmi - Q1_bmi
        df = df[(df["bmi"] >= Q1_bmi - 1.5 * IQR_bmi) & (df["bmi"] <= Q3_bmi + 1.5 * IQR_bmi)]

        # Filter blood pressure
        df = df[(df["ap_hi"] >= 60) & (df["ap_hi"] <= 250)]
        df = df[(df["ap_lo"] >= 40) & (df["ap_lo"] <= 150)]
        df = df[df["ap_hi"] > df["ap_lo"]]

        Q1_hi = df["ap_hi"].quantile(0.25)
        Q3_hi = df["ap_hi"].quantile(0.75)
        IQR_hi = Q3_hi - Q1_hi
        Q1_lo = df["ap_lo"].quantile(0.25)
        Q3_lo = df["ap_lo"].quantile(0.75)
        IQR_lo = Q3_lo - Q1_lo

        df = df[(df["ap_hi"] >= Q1_hi - 1.5 * IQR_hi) & (df["ap_hi"] <= Q3_hi + 1.5 * IQR_hi)]
        df = df[(df["ap_lo"] >= Q1_lo - 1.5 * IQR_lo) & (df["ap_lo"] <= Q3_lo + 1.5 * IQR_lo)]

        # Return just the features (drop height, weight, gender for Dataset 2)
        features = df[["age", "ap_hi", "ap_lo", "cholesterol", "gluc",
                       "smoke", "alco", "active", "bmi", "gender_2"]]
        self.labels_ = df["cardio"]
        return features


# Load data
df = pd.read_csv('data/cardio_train.csv', sep=';')

# Create preprocessor and transform data
prep = Preprocessor()
X_clean = prep.fit_transform(df)
y_clean = prep.labels_

# Define column groups
numerical = ["age", "ap_hi", "ap_lo", "bmi"]
ordinal = ["cholesterol", "gluc"]
binary = ["smoke", "alco", "active", "gender_2"]

# Column transfomer
transformer = ColumnTransformer([
    ("num", StandardScaler(), numerical),
    ("ord", "passthrough", ordinal),
    ("bin", "passthrough", binary)
])

# Full pipeline
pipeline = Pipeline([
    ("preprocessing", transformer),
    ("classifier", XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.1,
                                  use_label_encoder=False, eval_metric='logloss', random_state=42))
])

# Train and save
X_train, X_test, y_train, y_test = train_test_split(X_clean, y_clean, test_size=0.2, random_state=42)
pipeline.fit(X_train, y_train)
joblib.dump(pipeline, "best_model.pkl")
print("Model trained and saved to best_model.pkl")