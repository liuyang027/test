from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import io
import base64

app = Flask(__name__)


# 加载示例数据或创建模拟数据
def load_sample_data():
    # 创建模拟数据
    np.random.seed(42)
    X = np.random.rand(100, 1) * 10
    y = 2 * X.squeeze() + 3 + np.random.randn(100) * 2
    return X, y


# 全局变量存储模型
model = None


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/train", methods=["POST"])
def train_model():
    global model
    X, y = load_sample_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    # 计算模型性能
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)

    # 创建预测图
    plt.figure(figsize=(10, 6))
    plt.scatter(X, y, color="blue", label="数据点")
    plt.plot(X, model.predict(X), color="red", label="预测线")
    plt.xlabel("X")
    plt.ylabel("y")
    plt.title("线性回归模型")
    plt.legend()

    # 将图转换为base64以在网页中显示
    img = io.BytesIO()
    plt.savefig(img, format="png")
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()

    return jsonify(
        {
            "success": True,
            "train_score": train_score,
            "test_score": test_score,
            "plot": plot_url,
            "coefficients": model.coef_.tolist(),
            "intercept": model.intercept_,
        }
    )


@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"error": "模型尚未训练"}), 400

    try:
        data = request.get_json()
        x_value = float(data["x"])
        prediction = model.predict([[x_value]])[0]
        return jsonify({"prediction": prediction})
    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0")
