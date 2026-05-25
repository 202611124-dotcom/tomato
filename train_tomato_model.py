import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

MODEL_PATH = "tomato.pkl"

# 실제 데이터가 있으면 아래 sample 데이터 대신 불러오세요.
# data = pd.read_csv("your_data.csv")
# X = data[["내부온도", "내부습도", "지온"]]
# y = data["착과율"]

X = pd.DataFrame(
    [
        [24.0, 65.0, 18.0],
        [26.0, 70.0, 20.0],
        [28.0, 75.0, 22.0],
        [22.0, 60.0, 17.0],
        [30.0, 80.0, 24.0],
        [20.0, 55.0, 16.0],
    ],
    columns=["내부온도", "내부습도", "지온"],
)
y = pd.Series([45.0, 55.0, 62.0, 40.0, 68.0, 35.0], name="착과율")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

pred = model.predict(X_test)
print("MAE:", mean_absolute_error(y_test, pred))

joblib.dump(model, MODEL_PATH)
print(f"모델을 '{MODEL_PATH}'로 저장했습니다.")
