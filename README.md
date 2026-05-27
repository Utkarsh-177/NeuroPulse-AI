# 🚀 Burnout AI Enterprise

An advanced AI-powered workforce analytics platform built with **Flask**, **Machine Learning**, and **Interactive Data Visualization** to detect employee burnout, analyze productivity, generate intelligent recommendations, and provide live AI-based dataset insights.

---

## 🌟 Features

### 🧠 AI Burnout Detection

* Automatically analyzes uploaded employee datasets
* Uses **KMeans Clustering** to classify burnout levels:

  * 🔴 High Burnout
  * 🟡 Medium Burnout
  * 🟢 Low Burnout

---

### 📈 Productivity Intelligence

Maps burnout levels into productivity insights:

* High Productivity
* Moderate Productivity
* Low Productivity

---

### 🤖 AI Dataset Assistant

Integrated AI chatbot powered by:

* **Meta Llama 3.3 70B Instruct**
* NVIDIA AI API

Ask questions like:

* “Which employees are at high risk?”
* “What trends are visible?”
* “What recommendations should management follow?”

---

### 📊 Interactive Dashboard

Beautiful enterprise UI with:

* Animated charts
* Real-time counters
* AI activity timeline
* Glassmorphism design
* Responsive layout
* Live particle background

---

### 📥 Download Reports

Export:

* Burnout Analysis CSV
* Productivity Analysis CSV

---

## 🛠️ Technologies Used

| Technology          | Purpose              |
| ------------------- | -------------------- |
| Python              | Backend              |
| Flask               | Web Framework        |
| Pandas              | Data Processing      |
| NumPy               | Numerical Operations |
| Scikit-learn        | Machine Learning     |
| Chart.js            | Data Visualization   |
| HTML/CSS/JavaScript | Frontend UI          |
| NVIDIA API          | AI Chat Assistant    |

---

# 🧠 Machine Learning Workflow

## 1️⃣ Data Cleaning

* Removes duplicate columns
* Handles missing values
* Converts categorical values using Label Encoding

---

## 2️⃣ Feature Scaling

Uses:

```python
StandardScaler()
```

to normalize numerical data.

---

## 3️⃣ Burnout Clustering

Uses:

```python
KMeans(n_clusters=3)
```

to classify employees into burnout groups.

---

## 4️⃣ Productivity Mapping

| Burnout Level | Productivity          |
| ------------- | --------------------- |
| Low           | High Productivity     |
| Medium        | Moderate Productivity |
| High          | Low Productivity      |

---

# 📂 Project Structure

```bash
Burnout-AI/
│
├── app.py
├── requirements.txt
├── README.md
│
├── screenshots/
│   ├── dashboard.png
│   ├── charts.png
│   └── chatbot.png
│
└── datasets/
    └── sample.csv
```

---

# ⚙️ Installation

## 1️⃣ Clone Repository

```bash
git clone https://github.com/your-username/Burnout-AI.git

cd Burnout-AI
```

---

## 2️⃣ Create Virtual Environment

### Windows

```bash
python -m venv venv

venv\Scripts\activate
```

### Linux / Mac

```bash
python3 -m venv venv

source venv/bin/activate
```

---

## 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

# 🔑 Setup NVIDIA API Key

Set your API key as environment variable.

### Windows

```bash
set API_KEY=your_api_key
```

### Linux / Mac

```bash
export API_KEY=your_api_key
```

---

# ▶️ Run Application

```bash
python app.py
```

Server starts at:

```bash
http://127.0.0.1:5000
```

---

# 📊 Dataset Format Example

Example CSV:

```csv
Age,WorkHours,StressLevel,SleepHours
24,9,8,5
30,7,4,7
27,10,9,4
```

---

# 🖼️ Dashboard Preview

## 🔹 Burnout Analytics

* Burnout distribution charts
* Employee risk analysis
* Workforce health metrics

---

## 🔹 Productivity Insights

* Productivity trend graphs
* AI-generated workforce insights
* Smart recommendations

---

## 🔹 AI Chat Assistant

* Dataset Q&A
* AI-generated summaries
* Intelligent recommendations

---

# 📌 API Endpoints

| Route                    | Method   | Description                  |
| ------------------------ | -------- | ---------------------------- |
| `/`                      | GET/POST | Upload dataset & dashboard   |
| `/chat`                  | POST     | AI dataset assistant         |
| `/download/burnout`      | GET      | Download burnout report      |
| `/download/productivity` | GET      | Download productivity report |

---

# 🔥 Key Highlights

✅ Fully Responsive Enterprise Dashboard
✅ AI-Powered Workforce Analytics
✅ Real-Time Visualization
✅ Interactive AI Assistant
✅ Automated Burnout Detection
✅ Productivity Monitoring
✅ CSV Report Export
✅ Modern Animated UI

---

# 🧪 Future Improvements

* Deep Learning burnout prediction
* Real-time employee monitoring
* Authentication system
* Database integration
* Role-based access
* Email alerts
* Predictive analytics
* Team-wise reporting
* PDF report generation
* Dark/Light mode switch

---

# 📸 Recommended Screenshots for GitHub

Add these inside `/screenshots` folder:

* Main Dashboard
* Burnout Chart
* Productivity Chart
* AI Chatbot
* Dataset Table

Then include:

```md
![Dashboard](screenshots/dashboard.png)
```

---

# 🧾 requirements.txt

```txt
flask
pandas
numpy
scikit-learn
requests
```

---

# 🚀 Deployment Options

You can deploy this project on:

* Render
* Railway
* Replit
* PythonAnywhere
* Heroku
* VPS Server
* Docker

---

# 🔒 Security Notes

* Never expose your API key publicly
* Use `.env` for production
* Disable `debug=True` in deployment

Production:

```python
app.run(debug=False)
```

---

# 👨‍💻 Author

Developed with ❤️ using AI + Machine Learning + Flask.

---

# 📜 License

This project is licensed under the MIT License.

---

# ⭐ Support

If you like this project:

⭐ Star the repository
🍴 Fork the project
🧠 Contribute improvements
🚀 Build amazing AI systems

---
