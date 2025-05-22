import os
import re
import string
import joblib
import torch
from flask import Flask, request, render_template, jsonify
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# 导入 Hugging Face 的相关类
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer

app = Flask(__name__)

# -----------------------
# 文本预处理函数（用于 scikit-learn 模型）
# -----------------------
def clean_text(text):
    """
    清洗输入文本，移除 HTML 标签、标点符号、数字，转换为小写并去除多余空格。
    """
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(f"[{re.escape(string.punctuation + string.digits)}]", "", text)
    text = text.lower()
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def process_text(input_text):
    """
    对输入文本进行清洗处理，返回处理后的文本
    """
    return clean_text(input_text)

# -----------------------
# 加载 scikit-learn 模型（逻辑回归 & SVM）
# -----------------------
current_dir = os.path.dirname(__file__)

# 加载逻辑回归 Pipeline 模型
logistic_model_path = os.path.join(current_dir, "model", "logistic_regression_pipeline.pkl")
try:
    logistic_model = joblib.load(logistic_model_path)
    print("Logistic Regression Pipeline loaded successfully!")
except Exception as e:
    print("Error loading Logistic Regression Pipeline:", e)
    logistic_model = None

# 加载 SVM Pipeline 模型
svm_model_path = os.path.join(current_dir, "model", "svm_pipeline.pkl")
try:
    svm_model = joblib.load(svm_model_path)
    print("SVM Pipeline loaded successfully!")
except Exception as e:
    print("Error loading SVM Pipeline:", e)
    svm_model = None

# -----------------------
# 加载 Hugging Face DistilBERT 模型
# -----------------------
# 指定队友提供的 DistilBERT 检查点目录（例如 checkpoint-218）
distilbert_model_dir = os.path.join(current_dir, "model", "checkpoint-218")
try:
    distilbert_model = AutoModelForSequenceClassification.from_pretrained(distilbert_model_dir)
    distilbert_tokenizer = AutoTokenizer.from_pretrained(distilbert_model_dir)
    distilbert_model.eval()  # 设置为推理模式
    print("Loaded DistilBERT model and tokenizer successfully using Auto*!")
except Exception as e:
    import traceback
    print("Error loading DistilBERT model or tokenizer:")
    traceback.print_exc()
    distilbert_model = None
    distilbert_tokenizer = None

# -----------------------
# 预测函数
# -----------------------
def predict_logistic(text):
    processed = process_text(text)
    return logistic_model.predict([processed])[0]

def predict_svm(text):
    processed = process_text(text)
    return svm_model.predict([processed])[0]

def predict_distilbert(text):
    # 使用分词器对文本进行编码
    inputs = distilbert_tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = distilbert_model(**inputs)
        logits = outputs.logits
        pred_id = logits.argmax(dim=1).item()
    
    # 定义数字到标签的映射关系
    mapping = {
        0: "Post-Graduation Work Permit (PGWP)",
        1: "Express Entry",
        2: "Student Permitt",
        3: "Provincial Nominee Program (PNP)",
        4: "Work Permitt",
        5: "Family Sponsorship",
        6: "Refugee",
        7: "other"
    }
    
    # 如果 pred_id 在 mapping 中，则返回对应值，否则返回默认 "Student Project"
    c = mapping.get(pred_id, "Student Permit")
    return c

# -----------------------
# 路由定义
# -----------------------
@app.route("/")
def index():
    # index 页面中初始文本为空，默认选 "logistic"，预测结果为空，同时显示调试信息为空
    return render_template("index.html",
                           input_text="",
                           selected_model="logistic",
                           prediction_result="",
                           debug_info=[])

@app.route("/predict", methods=["POST"])
def predict():
    debug_info = []
    input_text = request.form.get("text_input", "")
    debug_info.append(f"Received input: {input_text}")
    if not input_text:
        debug_info.append("No text provided.")
        return render_template("index.html",
                               input_text="",
                               selected_model="logistic",
                               prediction_result="Error: No text provided",
                               debug_info=debug_info)
    # 获取用户选择的模型：取值 "logistic"、"svm"、"distilbert"
    model_choice = request.form.get("model_choice", "logistic").lower()
    debug_info.append(f"Model choice: {model_choice}")

    try:
        if model_choice == "svm":
            if svm_model is None:
                raise Exception("SVM model not loaded.")
            debug_info.append("Using SVM model for prediction.")
            prediction = predict_svm(input_text)
        elif model_choice == "distilbert":
            if distilbert_model is None or distilbert_tokenizer is None:
                raise Exception("DistilBERT model or tokenizer not loaded.")
            debug_info.append("Using DistilBERT model for prediction.")
            prediction = predict_distilbert(input_text)
        else:
            if logistic_model is None:
                raise Exception("Logistic Regression model not loaded.")
            debug_info.append("Using Logistic Regression model for prediction.")
            prediction = predict_logistic(input_text)
    except Exception as e:
        error_message = f"Prediction error: {e}"
        debug_info.append(error_message)
        prediction_result = error_message
    else:
        debug_info.append(f"Prediction successful: {prediction}")
        prediction_result = prediction

    return render_template("index.html",
                           input_text=input_text,
                           selected_model=model_choice,
                           prediction_result=prediction_result,
                           debug_info=debug_info)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
