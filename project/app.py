from flask import (
    Flask,
    request,
    jsonify,
    render_template,
    redirect,
    url_for,
    session,
    make_response,
)
import os
import random
import string
import torch
from torchvision import transforms, models
from PIL import Image, ImageDraw, ImageFont
from functools import lru_cache
import mimetypes
import hashlib
import requests
import pymysql

app = Flask(__name__)
app.secret_key = os.urandom(24)  # 设置会话密钥
app.config["UPLOAD_FOLDER"] = "static/uploads"
app.config["ALLOWED_EXTENSIONS"] = {"jpg", "jpeg", "png"}

PASSWORD_FILE = "static/password.txt"

# MySQL数据库配置
MYSQL_CONFIG = {
    "host": "localhost",  # 修改为你的MySQL主机
    "user": "root",  # 修改为你的MySQL用户名
    "password": "Sjk@yjl4sb",  # 修改为你的MySQL密码
    "database": "user",  # 修改为你的数据库名
    "charset": "utf8mb4",
}


def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()


def get_mysql_conn():
    return pymysql.connect(**MYSQL_CONFIG)


def load_users():
    """从MySQL数据库加载所有用户"""
    users = {}
    conn = get_mysql_conn()
    c = conn.cursor()
    c.execute("SELECT username, password_hash FROM users")
    for row in c.fetchall():
        users[row[0]] = row[1]
    conn.close()
    return users


def save_user(username, password_hash):
    """将新用户保存到MySQL数据库"""
    import datetime

    conn = get_mysql_conn()
    c = conn.cursor()
    c.execute(
        "INSERT INTO users (username, password_hash, created_at) VALUES (%s, %s, %s)",
        (username, password_hash, datetime.datetime.now()),
    )
    conn.commit()
    conn.close()


# Model configuration
class_names = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]

# 垃圾处理方法映射
disposal_methods = {
    "cardboard": "可回收：请压平后放入可回收物垃圾桶",
    "glass": "可回收：请小心处理，放入可回收物垃圾桶",
    "metal": "可回收：请清洁后放入可回收物垃圾桶",
    "paper": "可回收：请保持干燥，放入可回收物垃圾桶",
    "plastic": "可回收：请清洁后放入可回收物垃圾桶",
    "trash": "其他垃圾：请放入其他垃圾桶",
}
model_path = "models/mobilenetv3_trashnet.pth"


def allowed_file(filename):
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in app.config["ALLOWED_EXTENSIONS"]
    )


@lru_cache(maxsize=1)
def load_model():
    """Load and cache the model with error handling"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        model = models.mobilenet_v3_small(pretrained=False)
        model.classifier[3] = torch.nn.Linear(
            model.classifier[3].in_features, len(class_names)
        )
        model.load_state_dict(torch.load(model_path, map_location=device))
        model = model.to(device)
        model.eval()
        return model, device
    except Exception as e:
        raise RuntimeError(f"Failed to load model: {str(e)}")


# Preprocessing function with caching
@lru_cache(maxsize=1)
def get_preprocess():
    return transforms.Compose(
        [
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )


@app.route("/")
def index():
    if "username" in session:
        return render_template("index.html", username=session["username"])
    return redirect(url_for("login"))


@app.route("/information")
def information():
    return render_template("information.html")


# DeepSeek API 配置
DEEPSEEK_API_URL = "https://api.siliconflow.cn/v1/chat/completions"
DEEPSEEK_API_KEY = "sk-ruzmlfzitaudwgdqydlubbkvjttkkowqskqarqdlwcozvjrd"


@app.route("/robot")
def robot():
    return render_template("robot.html")


@app.route("/ask", methods=["POST"])
def ask():
    """处理用户提问并与DeepSeek API交互"""
    try:
        # 验证请求数据
        if not request.json or "question" not in request.json:
            return jsonify({"error": "请求格式不正确，缺少question参数"}), 400

        question = request.json["question"].strip()
        if not question:
            return jsonify({"error": "问题不能为空"}), 400

        # 记录收到的用户提问
        print(f"[INFO] 收到的用户提问: {question}")

        # 本地简单问答库
        local_responses = {
            "怎么分类": "请将垃圾分为可回收物、厨余垃圾、有害垃圾和其他垃圾四类。",
            "塑料瓶": "塑料瓶属于可回收物，请清洗后压扁放入可回收物垃圾桶。",
            "电池": "电池属于有害垃圾，请放入有害垃圾收集点。",
            "剩菜剩饭": "剩菜剩饭属于厨余垃圾，请沥干水分后放入厨余垃圾桶。",
            "纸箱": "纸箱属于可回收物，请压平后放入可回收物垃圾桶。",
        }

        # 先在本地问答库中查找
        if question in local_responses:
            print(f"[INFO] 使用本地回答库: {question}")
            return jsonify({"response": local_responses[question]})

        # 调用DeepSeek API
        try:
            headers = {
                "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
                "Content-Type": "application/json",
            }

            payload = {
                "model": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
                "messages": [
                    {
                        "role": "system",
                        "content": "你是一个垃圾分类助手，请用中文详细回答用户的问题。回答应包括：垃圾的具体分类、处理建议、注意事项和法律依据。",
                    },
                    {"role": "user", "content": question},
                ],
                "temperature": 0.7,
                "max_tokens": 1000,
            }

            response = requests.post(
                DEEPSEEK_API_URL, json=payload, headers=headers, timeout=30
            )
            response.raise_for_status()

            # 解析响应
            data = response.json()
            if "choices" in data and len(data["choices"]) > 0:
                reply = data["choices"][0]["message"]["content"]
                print(f"[INFO] DeepSeek API响应成功: {reply[:100]}...")
                return jsonify({"response": reply})
            else:
                raise ValueError("无效的API响应格式")

        except Exception as api_error:
            # 记录详细的API错误信息
            print(f"[ERROR] API调用失败: {str(api_error)}")
            # 返回本地默认回答
            return jsonify(
                {
                    "response": "当前无法连接到智能助手，以下是常见问题的回答：\n\n"
                    + "\n".join([f"Q: {q}\nA: {a}" for q, a in local_responses.items()])
                }
            )

    except Exception as e:
        # 记录全局错误日志
        print(f"[CRITICAL] 服务端错误: {str(e)}")
        return jsonify({"error": "服务器内部错误，请稍后再试"}), 500


def generate_captcha():
    """生成验证码图片"""
    # 生成4位随机字符
    captcha_text = "".join(random.choices(string.ascii_uppercase + string.digits, k=4))
    # 将验证码存入session
    session["captcha"] = captcha_text

    # 创建图片
    width, height = 120, 40
    image = Image.new("RGB", (width, height), color=(255, 255, 255))
    draw = ImageDraw.Draw(image)

    # 使用默认字体
    try:
        font = ImageFont.truetype("arial.ttf", 24)
    except IOError:
        font = ImageFont.load_default()

    # 绘制验证码文本
    draw.text((10, 5), captcha_text, font=font, fill=(0, 0, 0))

    # 添加干扰线
    for _ in range(5):
        x1 = random.randint(0, width)
        y1 = random.randint(0, height)
        x2 = random.randint(0, width)
        y2 = random.randint(0, height)
        draw.line((x1, y1, x2, y2), fill=(0, 0, 0))

    return image


@app.route("/captcha")
def get_captcha():
    """获取验证码图片"""
    image = generate_captcha()
    # 将图片保存到内存
    from io import BytesIO

    buf = BytesIO()
    image.save(buf, "PNG")
    buf.seek(0)
    # 返回图片响应
    response = make_response(buf.getvalue())
    response.headers["Content-Type"] = "image/png"
    return response


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")
        captcha = request.form.get("captcha")

        # 验证验证码
        if "captcha" not in session or captcha.upper() != session["captcha"]:
            return render_template("login.html", error="验证码错误")

        users = load_users()
        if username in users and users[username] == hash_password(password):
            session["username"] = username
            return redirect(url_for("index"))
        return render_template("login.html", error="用户名或密码错误")
    return render_template("login.html")


@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")

        users = load_users()
        if username in users:
            return render_template("register.html", error="用户名已存在")

        hashed_password = hash_password(password)
        save_user(username, hashed_password)
        session["username"] = username
        return redirect(url_for("index"))
    return render_template("register.html")


@app.route("/logout")
def logout():
    session.pop("username", None)
    return redirect(url_for("login"))


@app.route("/classify", methods=["POST"])
def classify_image():
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    if not allowed_file(file.filename):
        return (
            jsonify({"error": "Invalid file type. Allowed types: jpg, jpeg, png"}),
            400,
        )

    try:
        # Verify file content type
        file_type, _ = mimetypes.guess_type(file.filename)
        if file_type not in ["image/jpeg", "image/png"]:
            return jsonify({"error": "Invalid file content type"}), 400

        # Save the uploaded file
        file_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
        file.save(file_path)

        # Load and preprocess the image
        try:
            image = Image.open(file_path).convert("RGB")
            preprocess = get_preprocess()
            input_tensor = preprocess(image).unsqueeze(0)

            # Perform inference
            model, device = load_model()
            with torch.no_grad():
                outputs = model(input_tensor.to(device))
                _, predicted = outputs.max(1)
                predicted_class = class_names[predicted.item()]

            return render_template(
                "index.html",
                prediction=predicted_class,
                image_path=file.filename,
                disposal_method=disposal_methods[predicted_class],
            )
        except Exception as e:
            return jsonify({"error": f"Image processing failed: {str(e)}"}), 500

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    if not os.path.exists(app.config["UPLOAD_FOLDER"]):
        os.makedirs(app.config["UPLOAD_FOLDER"])
    app.run(host="0.0.0.0", port=5000, debug=True)
