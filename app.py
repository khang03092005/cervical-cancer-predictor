from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify
import datetime
import numpy as np
from sklearn.impute import SimpleImputer
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
import os
import requests
import json
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("OPENROUTER_API_KEY")

def ask_openrouter(prompt):
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://yourdomain.com",
        "X-Title": "CervicalCancerPredictor"
    }

    data = {
        # "model": "google/gemini-flash-2.5",
        #   "model":"cognitivecomputations/dolphin-mistral-24b-venice-edition:free",
        #    "model":"google/gemma-3n-e2b-it:free",
        #    "model":"cognitivecomputations/dolphin-mistral-24b-venice-edition:free",
        # "model": "tngtech/deepseek-r1t2-chimera:free",

          "model":"openrouter/cypher-alpha:free",


        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ]
    }

    response = requests.post(url=url, headers=headers, data=json.dumps(data))
    
    if response.status_code == 200:
        return response.json()["choices"][0]["message"]["content"]
    else:
        return f"Lỗi từ OpenRouter: {response.status_code} - {response.text}"


from feature_advice import feature_advice  # Load mô tả

app = Flask(__name__)
app.secret_key = "super-secret-key" 

feature_names = ['Age', 'Number of sexual partners', 'First sexual intercourse',
    'Num of pregnancies', 'Smokes', 'Smokes (years)', 'Smokes (packs/year)',
    'Hormonal Contraceptives', 'Hormonal Contraceptives (years)', 'IUD',
    'IUD (years)', 'STDs', 'STDs (number)', 'STDs:condylomatosis',
    'STDs:cervical condylomatosis', 'STDs:vaginal condylomatosis',
    'STDs:vulvo-perineal condylomatosis', 'STDs:syphilis',
    'STDs:pelvic inflammatory disease', 'STDs:genital herpes',
    'STDs:molluscum contagiosum', 'STDs:AIDS', 'STDs:HIV',
    'STDs:Hepatitis B', 'STDs:HPV', 'STDs: Number of diagnosis',
    'STDs: Time since first diagnosis', 'STDs: Time since last diagnosis',
    'Dx:Cancer', 'Dx:CIN', 'Dx:HPV', 'Dx', 'Hinselmann', 'Schiller', 'Citology']
label_mapping = {
    "Age": "Tuổi",
    "Number of sexual partners": "Số bạn tình",
    "First sexual intercourse": "Tuổi quan hệ lần đầu",
    "Num of pregnancies": "Số lần mang thai",
    "Smokes": "Hút thuốc",
    "Smokes (years)": "Số năm hút thuốc",
    "Smokes (packs/year)": "Số gói mỗi năm",
    "Hormonal Contraceptives": "Dùng thuốc tránh thai",
    "Hormonal Contraceptives (years)": "Số năm dùng thuốc tránh thai",
    "IUD": "Đặt vòng",
    "IUD (years)": "Số năm đặt vòng",
    "STDs": "Từng mắc STDs",
    "STDs (number)": "Số lần mắc STDs",
    "STDs:condylomatosis": "Sùi mào gà",
    "STDs:cervical condylomatosis": "Sùi mào gà cổ tử cung",
    "STDs:vaginal condylomatosis": "Sùi mào gà âm đạo",
    "STDs:vulvo-perineal condylomatosis": "Sùi mào gà âm hộ - tầng sinh môn",
    "STDs:syphilis": "Bệnh giang mai",
    "STDs:pelvic inflammatory disease": "Viêm vùng chậu",
    "STDs:genital herpes": "Mụn rộp sinh dục",
    "STDs:molluscum contagiosum": "U mềm lây",
    "STDs:AIDS": "AIDS",
    "STDs:HIV": "HIV",
    "STDs:Hepatitis B": "Viêm gan B",
    "STDs:HPV": "Nhiễm HPV",
    "STDs: Number of diagnosis": "Số lần được chẩn đoán STDs",
    "STDs: Time since first diagnosis": "Thời gian từ lần chẩn đoán STDs đầu tiên",
    "STDs: Time since last diagnosis": "Thời gian từ lần chẩn đoán STDs gần nhất",
    "Dx:Cancer": "Chẩn đoán ung thư",
    "Dx:CIN": "Chẩn đoán CIN",
    "Dx:HPV": "Chẩn đoán HPV",
    "Dx": "Chẩn đoán bất thường",
    "Hinselmann": "Hinselmann dương tính",
    "Schiller": "Schiller dương tính",
    "Citology": "Tế bào học dương tính",
    "Biopsy": "Sinh thiết dương tính"
}


model = joblib.load("logistic_model.pkl")
imputer = joblib.load("imputer.pkl")
X_background = shap.maskers.Independent(pd.DataFrame([[0]*len(feature_names)], columns=feature_names))
explainer = shap.LinearExplainer(model, masker=X_background, feature_names=feature_names)


def generate_advice_auto(name, value, shap_val, percent):
    
    if shap_val <= 0:
        return ""  # Bỏ qua nếu đặc trưng làm giảm nguy cơ
    trend = "tăng nguy cơ" 
    vi_name = label_mapping.get(name, name)
    line = f"• {vi_name} = {value} → {trend} ({shap_val:+.2f}, ảnh hưởng: {percent:.1f}%)\n"


    if name in feature_advice:
        desc = feature_advice[name]["desc"]
        action = feature_advice[name]["action"]
    else:
        desc = f"Yếu tố này ảnh hưởng {trend} đến nguy cơ mắc bệnh."
        action = "Nên tham khảo ý kiến bác sĩ nếu bạn chưa rõ về yếu tố này."

    return f"{line}  {desc}\n  👉 {action}\n"

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        try:
            input_dict = dict.fromkeys(feature_names, np.nan)

            # Ghi đè bằng các giá trị người dùng đã nhập
            for name in feature_names:
                raw_val = request.form.get(name)
                try:
                    input_dict[name] = float(raw_val)
                except (TypeError, ValueError):
                    pass  # giữ nguyên np.nan nếu không thể chuyển

            # Kiểm tra: nếu mọi giá trị đều là NaN → trả lỗi nhẹ nhàng
            if all(np.isnan(v) for v in input_dict.values()):
                return "⚠️ Vui lòng nhập ít nhất một giá trị vào biểu mẫu."

            # Dựng DataFrame với đủ cột
            X_input = pd.DataFrame([input_dict], columns=feature_names)

            # Kiểm tra số lượng cột có giá trị thực
            valid_cols = X_input.notna().sum().sum()
            if valid_cols == 0:
                return "⚠️ Không thể xử lý vì tất cả các giá trị đều trống."

            # Áp dụng Imputer nếu hợp lệ
            
            X_input = pd.DataFrame(imputer.transform(X_input), columns=feature_names)


            # Bước 3: Dự đoán
            prediction = model.predict(X_input)[0]
            proba = model.predict_proba(X_input)[0][1] * 100

            # Bước 4: SHAP
            shap_values = explainer(X_input)
            shap_score = shap_values.values[0]
            total_abs = sum(abs(val) for val in shap_score)

            impacts = [
                (feature_names[i], shap_score[i], abs(shap_score[i]) / total_abs * 100)
                for i in range(len(feature_names))
            ]
            impacts_sorted = sorted(impacts, key=lambda x: abs(x[1]), reverse=True)
            filtered = [x for x in impacts_sorted if x[2] >= 5]
            if not filtered:
                filtered = impacts_sorted[:3]

            # Bước 5: Sinh lời khuyên
            if proba >= 25:
                advice = "🔴 CẢNH BÁO: Có nguy cơ tiềm ẩn.\n\n"

                has_high_impact = any(pct > 25 for _, _, pct in filtered)

                for name, shap_val, percent in filtered:
                    val = X_input.iloc[0][name]
                    advice += generate_advice_auto(name, val, shap_val, percent)

                if has_high_impact:
                    advice = (
                        "⚠️ Một số yếu tố có ảnh hưởng rất lớn đến kết quả (trên 25%). "
                        "Bạn nên tham khảo ý kiến bác sĩ sớm.\n\n"
                    ) + advice

                advice += (
                    "\n💡 Khuyến nghị:\n"
                    "• Tham khảo ý kiến bác sĩ chuyên khoa\n"
                    "• Tiến hành xét nghiệm Pap smear hoặc HPV nếu chưa làm\n"
                    "• Duy trì lối sống lành mạnh, ăn uống khoa học\n"
                    "• Tuyệt đối tránh thuốc lá, hạn chế rượu bia\n"
                    "• Tiêm vaccine HPV nếu chưa tiêm\n"
                    "\nSức khỏe của bạn là điều quan trọng nhất. Hãy hành động ngay hôm nay! ❤️"
                )
            else:
                advice = (
                    "✅ Bạn hiện không có nguy cơ đáng kể.\n\n"
                    "💡 Tuy nhiên, hãy:\n"
                    "• Duy trì lối sống lành mạnh\n"
                    "• Khám phụ khoa định kỳ (ít nhất mỗi 6–12 tháng)\n"
                    "• Tránh hút thuốc, hạn chế rượu bia\n"
                    "• Nếu chưa tiêm vaccine HPV, nên tham khảo ý kiến bác sĩ về việc tiêm phòng\n"
                    "\nChúc bạn luôn khỏe mạnh ❤️"
                )

            extra_insight = ask_openrouter(f"Hãy đưa ra phân tích y khoa bằng tiếng Việt dựa trên lời khuyên sau:\n{advice}")

            # Bước 6: SHAP plot
            
            plt.figure()
            shap.plots.waterfall(shap_values[0], show=False)  # sử dụng matplotlib backend
            plt.savefig("static/shap_plot.png", bbox_inches='tight')
            plt.close()
            if "history" not in session:
                session["history"] = []

            session["history"].append({
                "input": {k: request.form.get(k) for k in feature_names},
                "result": int(prediction),
                "proba": round(proba, 2),
                "advice": advice,
                "timestamp": datetime.datetime.now().strftime("%d-%m-%Y %H:%M")
            })

            session.modified = True
            return render_template("index.html", features=feature_names,
                                   result=prediction, proba=round(proba, 2),
                                   advice=advice, extra_insight=extra_insight)


        except Exception as e:
            return f"Lỗi xử lý dữ liệu: {e}"
    return render_template("index.html", features=feature_names, result=None)
from flask import jsonify  

@app.route("/ask", methods=["POST"])
def ask():
    try:
        data = request.get_json()
        prompt = data.get("prompt", "")
        if not prompt.strip():
            return jsonify({"reply": "⚠️ Không nhận được câu hỏi hợp lệ."})
        reply = ask_openrouter(prompt)
        return jsonify({"reply": reply})
    except Exception as e:
        return jsonify({"reply": f"Lỗi: {e}"})




@app.route("/monitor")
def monitor():
    predictions = session.get("history", [])
    labels = [f"Lần {i+1}" for i in range(len(predictions))]
    probabilities = [p.get("proba", 0) for p in predictions]

    # ✅ Truyền thêm timestamp và proba đầy đủ sang template
    return render_template("monitor.html",
                           labels=labels,
                           probabilities=probabilities,
                           history=predictions)



@app.route("/predict", methods=["POST"])
def predict():
    return redirect(url_for("index"))


@app.route('/history')
def history():
    history = session.get("history", [])
    return render_template("history.html", history=history, label_mapping=label_mapping)


@app.route("/clear_history")
def clear_history():
    session.pop("history", None)  # Xoá lịch sử người dùng hiện tại (trong session)
    return redirect(url_for("monitor"))


if __name__ == "__main__":
    app.run(debug=True)