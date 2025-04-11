FROM python:3.9-slim

WORKDIR /app

COPY . .

# 安装依赖
RUN pip install --no-cache-dir -r requirements.txt

# 设置环境变量
ENV FLASK_APP=app.py
ENV FLASK_ENV=production

# 启动应用
CMD ["python", "app.py"] 