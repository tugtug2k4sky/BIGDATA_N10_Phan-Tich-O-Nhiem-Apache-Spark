from flask import Flask, request
from pyspark.sql import SparkSession, Row, functions as F
from pyspark.ml.linalg import Vectors
from pyspark.ml.classification import RandomForestClassificationModel

app = Flask(__name__)

# Khởi tạo Spark session (1 lần)
spark = SparkSession.builder.appName("PredictWithSavedModel").getOrCreate()

# Load mô hình đã lưu (1 lần)
model = RandomForestClassificationModel.load("./random_forest_model")

label_mapping = {
    0: "Kém",
    1: "Rất Kém",
    2: "Trung bình",
    3: "Tốt"
}

# Danh sách tên các cột, thứ tự input trùng với model yêu cầu
columns = ["pm25", "pm10", "co", "no2", "so2", "o3", "temperature", "humidity", "rainfall", "wind_speed"]

# Giới hạn kiểm tra từng trường (min/max hoặc max đơn thuần)
limits = {
    "pm25": (None, 150),
    "pm10": (None, 500),
    "co": (None, 10),
    "no2": (None, 200),
    "so2": (None, 500),
    "o3": (None, 200),
    "temperature": (-40, 50),
    "humidity": (0, 100),
    "rainfall": (0, None),
    "wind_speed": (0, None)
}

html_template = '''
<!DOCTYPE html>
<html lang="vi">
<head>
    <meta charset="UTF-8" />
    <title>Dự đoán chất lượng Random Forest</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            max-width: 720px;
            margin: 40px auto;
            padding: 25px 35px;
            background: linear-gradient(135deg, #d0eaff, #f9f9f9);
            border-radius: 18px;
            box-shadow: 0 0 40px rgba(0, 150, 255, 0.25);
        }}
        h1 {{
            color: #004a99;
            text-align: center;
            margin-bottom: 30px;
            letter-spacing: 1.1px;
            text-shadow: 1px 1px 3px #b0d4ff;
        }}
        form.form-grid {{
            display: grid;
            grid-template-columns: repeat(5, 1fr);
            gap: 20px 25px;
            margin-bottom: 30px;
            align-items: center;
        }}
        form.form-grid label {{
            font-weight: 600;
            font-size: 0.9rem;
            color: #0056b3;
            margin-bottom: 6px;
            display: block;
            text-align: center;
        }}
        form.form-grid input[type="number"] {{
            width: 100%;
            box-sizing: border-box;
            text-align: center;
            padding: 12px 0;
            font-size: 1.1rem;
            border-radius: 14px;
            border: 2px solid #a3cef1;
            outline: none;
            transition: all 0.3s ease;
            box-shadow: 0 3px 10px rgba(102, 166, 255, 0.2);
        }}
        form.form-grid input[type="number"]:focus {{
            border-color: #003e7e;
            box-shadow: 0 5px 15px rgba(0, 62, 126, 0.6);
            transform: scale(1.05);
        }}
        button {{
            grid-column: span 5;
            background: linear-gradient(90deg, #005bea, #00c6fb);
            border: none;
            color: white;
            font-size: 1.4rem;
            padding: 18px 0;
            border-radius: 16px;
            cursor: pointer;
            font-weight: 700;
            box-shadow: 0 10px 25px rgba(0,198,251,0.45);
            transition: background 0.4s ease;
            letter-spacing: 1.2px;
            user-select: none;
            margin-top: 10px;
        }}
        button:hover {{
            background: linear-gradient(90deg, #0046d3, #0098e0);
        }}
        .result {{
            margin-top: 25px;
            padding: 18px 22px;
            border-radius: 16px;
            font-size: 1.2rem;
            font-weight: 600;
            box-shadow: 0 6px 15px rgba(72, 187, 120, 0.3);
            background-color: #dbf6db;
            color: #27632a;
            text-align: center;
            user-select: text;
        }}
        .error {{
            background-color: #ffd4d4;
            color: #a42121;
            box-shadow: 0 6px 15px rgba(255, 90, 90, 0.4);
        }}
        @media(max-width: 600px) {{
            form.form-grid {{
                grid-template-columns: repeat(2, 1fr);
                gap: 18px 20px;
            }}
            button {{
                grid-column: span 2;
            }}
        }}
    </style>
</head>
<body>
    <h1>Dự đoán chất lượng không khí với Random Forest</h1>
    <form method="post" autocomplete="off" class="form-grid">
        {input_fields}
        <button type="submit">Dự đoán ngay 🔥</button>
    </form>
    {result_section}
</body>
</html>
'''

def render_page(input_vals=None, result=None, error=False):
    if input_vals is None:
        input_vals = [""] * len(columns)
    input_fields_html = ""
    for i, col in enumerate(columns):
        val = input_vals[i] if i < len(input_vals) else ""
        # Dùng input number với step=any cho phép nhập float
        input_fields_html += f'''
            <div>
                <label for="input_{i}">{col.capitalize()}</label>
                <input id="input_{i}" type="number" step="any" name="input_{i}" placeholder="Nhập {col}" value="{val}" required />
            </div>
        '''
    if result:
        cls = "error" if error else ""
        result_html = f'<div class="result {cls}">{result}</div>'
    else:
        result_html = ""
    return html_template.format(input_fields=input_fields_html, result_section=result_html)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        try:
            # Lấy dữ liệu từng input, ép float
            inputs = []
            for i, col in enumerate(columns):
                val_str = request.form.get(f"input_{i}", "").strip()
                val = float(val_str)
                inputs.append(val)
            # Kiểm tra điều kiện giới hạn dữ liệu từng cột
            for i, col in enumerate(columns):
                val = inputs[i]
                min_val, max_val = limits[col]
                if min_val is not None and val < min_val:
                    return render_page(inputs, f"Lỗi: Giá trị '{col}' phải >= {min_val}", error=True)
                if max_val is not None and val > max_val:
                    return render_page(inputs, f"Lỗi: Giá trị '{col}' phải <= {max_val}", error=True)

            # Tạo dataframe Spark và dự đoán
            input_row = Row("features")
            input_df = spark.createDataFrame([input_row(Vectors.dense(inputs))])
            prediction = model.transform(input_df).collect()[0]

            pred_label = int(prediction["prediction"])
            probability = prediction["probability"]

            label = label_mapping.get(pred_label, "Unknown")
            emoji = "🟢" if pred_label == 3 else "🔴"
            prob_str = ", ".join([f"{p:.2f}" for p in probability])

            result = f'''
                Dữ liệu đầu vào: {inputs}<br>
                Dự đoán chất lượng: <b>{label}</b> {emoji}<br>
                Xác suất dự đoán: [{prob_str}]
            '''
            return render_page(inputs, result)
        except ValueError:
            return render_page([], "Lỗi: Vui lòng nhập đầy đủ và đúng định dạng số.", error=True)
        except Exception as e:
            return render_page([], f"Lỗi hệ thống: {e}", error=True)
    else:
        return render_page()

if __name__ == "__main__":
    app.run(debug=True)
