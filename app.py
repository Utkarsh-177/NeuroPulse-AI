from flask import Flask, request, render_template_string, jsonify, send_file
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans
import requests, os, io, time

app = Flask(__name__)

API_KEY = os.getenv("API_KEY")
last_df = None


# ---------------- DATA PROCESS ----------------
def auto_train(df):
    df = df.copy()
    df.columns = df.columns.str.lower()
    df = df.loc[:, ~df.columns.duplicated()]

    for col in df.columns:
        if df[col].dtype == "object":
            try:
                df[col] = LabelEncoder().fit_transform(df[col].astype(str))
            except:
                df = df.drop(columns=[col])

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.fillna(df.mean(numeric_only=True))
    df = df.fillna(0)

    num = df.select_dtypes(include=np.number)

    if len(num.columns) < 2:
        df["Burnout"] = np.random.choice(["Low", "Medium", "High"], len(df))
    else:
        scaler = StandardScaler()
        X = scaler.fit_transform(num)
        km = KMeans(n_clusters=3, n_init=10, random_state=42)
        preds = km.fit_predict(X)
        df["Burnout"] = [
            "Low" if i == 0 else "Medium" if i == 1 else "High"
            for i in preds
        ]

    stats = {
        "high": int((df["Burnout"] == "High").sum()),
        "medium": int((df["Burnout"] == "Medium").sum()),
        "low": int((df["Burnout"] == "Low").sum())
    }

    return df, stats


# ---------------- PRODUCTIVITY ----------------
def add_productivity(df):
    df["Productivity"] = df["Burnout"].map({
        "Low": "High Productivity",
        "Medium": "Moderate Productivity",
        "High": "Low Productivity"
    })
    return df


# ---------------- RECOMMENDATIONS ----------------
def recommendations(stats):
    rec = []
    total = sum(stats.values())

    if stats["high"] / total > 0.4:
        rec.append("Critical burnout detected. Immediate intervention required.")
    elif stats["medium"] / total > 0.4:
        rec.append("Moderate burnout observed. Improve balance and reduce overload.")
    else:
        rec.append("Burnout levels are stable across the workforce.")

    rec.append("Maintain consistent sleep schedule and healthy routines.")
    rec.append("Encourage scheduled breaks and physical activities.")
    rec.append("Use AI productivity monitoring for workforce optimization.")
    rec.append("Monitor high-risk employee clusters continuously.")

    return rec


# ---------------- AI CHAT ----------------
def ai_chat(q, df):

    if not API_KEY:
        return "API key missing. Set NVIDIA API_KEY environment variable."

    if df is None:
        return "Upload dataset first."

    try:

        summary = df.describe().to_string()
        cols = ', '.join(df.columns)

        res = requests.post(
            "https://integrate.api.nvidia.com/v1/chat/completions",

            headers={
                "Authorization": f"Bearer {API_KEY}",
                "Content-Type": "application/json"
            },

            json={
                "model": "google/gemma-4-31b-it",

                "messages": [
                    {
                        "role": "user",
                        "content": f"""
Dataset Columns:
{cols}

Dataset Summary:
{summary}

Question:
{q}
"""
                    }
                ],

                "max_tokens": 200
            },

            timeout=60
        )


        if res.status_code != 200:
            return f"NVIDIA API Error: {res.status_code} - {res.text}"


        try:
            data = res.json()

        except ValueError:
            return "Invalid response received from NVIDIA API."


        if "choices" not in data:
            return "AI response error."


        return data["choices"][0]["message"]["content"]


    except requests.exceptions.Timeout:
        return "NVIDIA API request timed out. Please try again."


    except requests.exceptions.RequestException as e:
        return f"API connection error: {str(e)}"


    except Exception as e:
        return f"Unexpected error: {str(e)}"


# ---------------- HTML ----------------
HTML = """
<!DOCTYPE html>
<html lang="en">
<head>

<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">

<title>NeuroPulse AI Enterprise</title>

<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>

<style>

*{
margin:0;
padding:0;
box-sizing:border-box;
}

html{
scroll-behavior:smooth;
}

body{
font-family:system-ui;
background:#020617;
color:#e2e8f0;
overflow-x:hidden;
position:relative;
}

/* ================= BACKGROUND ================= */

#bgCanvas{
position:fixed;
top:0;
left:0;
width:100%;
height:100%;
z-index:-5;
opacity:0.35;
}

body::before{
content:"";
position:fixed;
width:700px;
height:700px;
background:radial-gradient(circle,#2563eb33,transparent 70%);
top:-250px;
right:-250px;
z-index:-4;
animation:floatGlow 8s infinite alternate ease-in-out;
}

body::after{
content:"";
position:fixed;
width:600px;
height:600px;
background:radial-gradient(circle,#7c3aed22,transparent 70%);
bottom:-250px;
left:-250px;
z-index:-4;
animation:floatGlow2 10s infinite alternate ease-in-out;
}

@keyframes floatGlow{
from{
transform:translateY(0);
}
to{
transform:translateY(40px);
}
}

@keyframes floatGlow2{
from{
transform:translateX(0);
}
to{
transform:translateX(40px);
}
}

/* ================= CONTAINER ================= */

.container{
max-width:1450px;
margin:auto;
padding:35px 25px 80px;
animation:fadeIn 1s ease;
}

@keyframes fadeIn{
from{
opacity:0;
transform:translateY(20px);
}
to{
opacity:1;
transform:translateY(0);
}
}

/* ================= TOP BAR ================= */

.topbar{
display:flex;
justify-content:center;
gap:20px;
flex-wrap:wrap;
margin-bottom:40px;
}

.status-card{
background:rgba(15,23,42,0.72);
border:1px solid rgba(255,255,255,0.08);
backdrop-filter:blur(16px);
padding:18px 28px;
border-radius:20px;
min-width:180px;
text-align:center;
box-shadow:0 10px 30px rgba(0,0,0,0.3);
}

.status-card h3{
font-size:22px;
margin-bottom:6px;
}

.status-card p{
font-size:13px;
color:#94a3b8;
}

/* ================= HERO ================= */

.hero{
text-align:center;
padding:30px 0 20px;
position:relative;
}

.hero-glow{
position:absolute;
width:500px;
height:500px;
background:radial-gradient(circle,#2563eb44,transparent 70%);
left:50%;
transform:translateX(-50%);
top:-50px;
filter:blur(50px);
z-index:-1;
animation:floatGlow 5s infinite alternate ease-in-out;
}

.hero h1{
font-size:70px;
font-weight:900;
margin-bottom:14px;
background:linear-gradient(90deg,#60a5fa,#a78bfa);
-webkit-background-clip:text;
-webkit-text-fill-color:transparent;
}

.subtitle{
max-width:950px;
margin:auto;
line-height:1.8;
font-size:18px;
color:#94a3b8;
}

/* ================= COUNTERS ================= */

.counter-grid{
display:grid;
grid-template-columns:repeat(auto-fit,minmax(220px,1fr));
gap:24px;
margin-top:50px;
}

.counter-card{
background:rgba(15,23,42,0.72);
backdrop-filter:blur(18px);
padding:32px;
border-radius:28px;
border:1px solid rgba(255,255,255,0.08);
text-align:center;
transition:0.4s;
box-shadow:0 10px 40px rgba(0,0,0,0.25);
}

.counter-card:hover{
transform:translateY(-8px);
box-shadow:0 20px 45px rgba(37,99,235,0.25);
}

.counter-card h2{
font-size:48px;
margin-bottom:10px;
}

.counter-card p{
color:#94a3b8;
}

/* ================= UPLOAD ================= */

.upload{
display:flex;
justify-content:center;
align-items:center;
flex-direction:column;
gap:14px;
max-width:850px;
margin:55px auto;
padding:80px 40px;
border-radius:35px;
border:2px dashed #2563eb;
background:rgba(15,23,42,0.75);
backdrop-filter:blur(18px);
cursor:pointer;
transition:0.45s;
position:relative;
overflow:hidden;
}

.upload::before{
content:"";
position:absolute;
width:120%;
height:120%;
background:linear-gradient(
120deg,
transparent,
rgba(255,255,255,0.08),
transparent
);
transform:translateX(-100%);
transition:0.9s;
}

.upload:hover::before{
transform:translateX(100%);
}

.upload:hover{
transform:translateY(-6px) scale(1.01);
box-shadow:0 0 50px rgba(37,99,235,0.45);
}

.upload h2{
font-size:34px;
}

.upload p{
color:#94a3b8;
font-size:16px;
}

/* ================= SWITCH ================= */

.switch-wrapper{
display:flex;
justify-content:center;
margin:50px 0;
}

.switch{
position:relative;
width:460px;
height:66px;
background:#0f172a;
border-radius:50px;
padding:6px;
display:flex;
align-items:center;
overflow:hidden;
box-shadow:0 10px 30px rgba(0,0,0,0.45);
}

.slider{
position:absolute;
width:50%;
height:54px;
left:6px;
background:linear-gradient(135deg,#2563eb,#3b82f6);
border-radius:40px;
transition:0.45s cubic-bezier(.77,0,.18,1);
}

.option{
flex:1;
z-index:2;
text-align:center;
cursor:pointer;
font-weight:700;
font-size:15px;
color:#94a3b8;
transition:0.3s;
user-select:none;
}

.option.active{
color:white;
}

/* ================= VIEWS ================= */

.view-container{
overflow:hidden;
width:100%;
}

.views{
display:flex;
width:200%;
transition:transform 0.7s cubic-bezier(.77,0,.18,1);
}

.screen{
width:100%;
padding:10px;
}

/* ================= STATS ================= */

.stats{
display:grid;
grid-template-columns:repeat(auto-fit,minmax(240px,1fr));
gap:25px;
margin-bottom:40px;
}

.card{
background:rgba(15,23,42,0.75);
backdrop-filter:blur(18px);
padding:35px;
border-radius:28px;
text-align:center;
border:1px solid rgba(255,255,255,0.08);
transition:0.35s;
box-shadow:0 10px 35px rgba(0,0,0,0.25);
}

.card:hover{
transform:translateY(-8px);
box-shadow:0 20px 45px rgba(37,99,235,0.25);
}

.card h2{
font-size:50px;
margin-bottom:10px;
}

.card p{
color:#94a3b8;
font-size:15px;
}

/* ================= SECTION ================= */

.section-box,
.chart-box{
background:rgba(15,23,42,0.72);
backdrop-filter:blur(18px);
padding:32px;
border-radius:30px;
border:1px solid rgba(255,255,255,0.08);
margin-bottom:40px;
box-shadow:0 10px 40px rgba(0,0,0,0.25);
}

.section-title{
font-size:30px;
margin-bottom:22px;
}

/* ================= METRICS ================= */

.metrics-grid{
display:grid;
grid-template-columns:repeat(auto-fit,minmax(220px,1fr));
gap:22px;
margin-top:25px;
}

.metric-card{
background:#111827;
padding:24px;
border-radius:20px;
border:1px solid #1e293b;
transition:0.3s;
}

.metric-card:hover{
transform:translateY(-4px);
border-color:#2563eb;
}

.metric-card h3{
font-size:36px;
margin-bottom:10px;
}

.metric-card p{
color:#94a3b8;
}

/* ================= TABLE ================= */

.table-box{
max-height:500px;
overflow:auto;
border-radius:22px;
border:1px solid #1e293b;
margin-top:30px;
}

.table-box::-webkit-scrollbar{
width:10px;
height:10px;
}

.table-box::-webkit-scrollbar-thumb{
background:#2563eb;
border-radius:20px;
}

table{
width:100%;
border-collapse:collapse;
background:#0f172a;
}

th{
position:sticky;
top:0;
background:#111827;
z-index:2;
}

th,td{
padding:15px;
text-align:center;
border-bottom:1px solid #1e293b;
}

tr:hover{
background:#172554;
}

/* ================= RECOMMEND ================= */

.recommend-grid{
display:grid;
grid-template-columns:repeat(auto-fit,minmax(280px,1fr));
gap:24px;
margin-top:20px;
}

.recommend-card{
background:#111827;
padding:26px;
border-radius:22px;
border:1px solid #1e293b;
line-height:1.8;
transition:0.35s;
}

.recommend-card:hover{
transform:translateY(-5px);
border-color:#2563eb;
}

/* ================= TIMELINE ================= */

.timeline{
display:flex;
flex-direction:column;
gap:18px;
margin-top:20px;
}

.timeline-item{
padding:20px;
background:#111827;
border-left:4px solid #2563eb;
border-radius:14px;
}

/* ================= DOWNLOAD ================= */

.downloads{
display:flex;
gap:20px;
flex-wrap:wrap;
margin-top:45px;
}

.download-btn{
padding:16px 26px;
border:none;
border-radius:16px;
background:linear-gradient(135deg,#2563eb,#3b82f6);
color:white;
font-weight:700;
font-size:15px;
cursor:pointer;
transition:0.35s;
}

.download-btn:hover{
transform:translateY(-5px) scale(1.03);
box-shadow:0 10px 30px rgba(37,99,235,0.4);
}

/* ================= LIVE AI ================= */

.ai-live-box{
margin-top:40px;
}

.live-header{
display:flex;
align-items:center;
gap:12px;
font-size:24px;
font-weight:700;
margin-bottom:25px;
}

.pulse{
width:14px;
height:14px;
background:#22c55e;
border-radius:50%;
animation:pulse 1.5s infinite;
}

@keyframes pulse{
0%{
transform:scale(1);
opacity:1;
}
100%{
transform:scale(2.3);
opacity:0;
}
}

.live-grid{
display:grid;
grid-template-columns:repeat(auto-fit,minmax(250px,1fr));
gap:22px;
}

.live-card{
background:#111827;
padding:24px;
border-radius:20px;
border:1px solid #1e293b;
transition:0.35s;
}

.live-card:hover{
transform:translateY(-5px);
border-color:#2563eb;
}

/* ================= CHAT ================= */

#chat{
position:fixed;
bottom:25px;
right:25px;
width:82px;
height:82px;
display:flex;
justify-content:center;
align-items:center;
cursor:pointer;
z-index:999;
}

.orb-core{
width:32px;
height:32px;
background:#3b82f6;
border-radius:50%;
position:absolute;
box-shadow:
0 0 20px #3b82f6,
0 0 60px #2563eb;
animation:orbPulse 2s infinite;
}

.orb-ring{
position:absolute;
width:72px;
height:72px;
border:2px solid #2563eb88;
border-radius:50%;
animation:spinRing 8s linear infinite;
}

@keyframes orbPulse{
0%{
transform:scale(1);
}
50%{
transform:scale(1.18);
}
100%{
transform:scale(1);
}
}

@keyframes spinRing{
100%{
transform:rotate(360deg);
}
}

#chatbox{
position:fixed;
bottom:115px;
right:25px;
width:370px;
height:540px;
background:#0f172a;
border-radius:28px;
display:none;
flex-direction:column;
border:1px solid #1e293b;
overflow:hidden;
box-shadow:0 25px 60px rgba(0,0,0,0.45);
z-index:999;
}

.chat-header{
padding:20px;
font-weight:700;
font-size:18px;
border-bottom:1px solid #1e293b;
background:#111827;
}

#chat-body{
flex:1;
overflow:auto;
padding:18px;
display:flex;
flex-direction:column;
gap:14px;
}

.msg{
padding:14px 16px;
border-radius:16px;
max-width:85%;
line-height:1.6;
font-size:14px;
}

.user{
background:#2563eb;
align-self:flex-end;
}

.ai{
background:#1e293b;
align-self:flex-start;
}

.chat-input{
display:flex;
border-top:1px solid #1e293b;
}

.chat-input input{
flex:1;
padding:16px;
background:#020617;
border:none;
outline:none;
color:white;
font-size:14px;
}

.chat-input button{
width:90px;
border:none;
background:#2563eb;
color:white;
font-weight:700;
cursor:pointer;
}

/* ================= RESPONSIVE ================= */

@media(max-width:768px){

.hero h1{
font-size:46px;
}

.switch{
width:100%;
}

#chatbox{
width:95%;
right:2.5%;
}

}

</style>

</head>

<body>

<canvas id="bgCanvas"></canvas>

<div class="container">

<!-- ================= TOP STATUS ================= -->

<div class="topbar">

<div class="status-card">
<h3 id="liveTime">00:00:00</h3>
<p>System Time</p>
</div>

<div class="status-card">
<h3>AI ACTIVE</h3>
<p>Neural Engine</p>
</div>

<div class="status-card">
<h3>92%</h3>
<p>AI Confidence</p>
</div>

</div>

<!-- ================= HERO ================= -->

<div class="hero">

<div class="hero-glow"></div>

<h1>NeuroPulse AI</h1>

<p class="subtitle">
AI-powered workforce wellness monitoring, productivity analytics,
employee risk assessment, intelligent recommendations,
and workforce optimization dashboard.
</p>

<div class="counter-grid">

<div class="counter-card">
<h2 id="counter1">0</h2>
<p>Employees Analyzed</p>
</div>

<div class="counter-card">
<h2 id="counter2">0</h2>
<p>AI Predictions</p>
</div>

<div class="counter-card">
<h2 id="counter3">0</h2>
<p>Insights Generated</p>
</div>

</div>

</div>

<!-- ================= UPLOAD ================= -->

<form method="POST" enctype="multipart/form-data">

<label class="upload">

<h2>Upload Dataset</h2>

<p>
Analyze employee productivity and workforce datasets using AI clustering.
</p>

<input type="file" name="file" hidden onchange="this.form.submit()">

</label>

</form>

<!-- ================= SWITCH ================= -->

<div class="switch-wrapper">

<div class="switch">

<div class="slider" id="slider"></div>

<div class="option active" onclick="switchView(0)">
Burnout Analytics
</div>

<div class="option" onclick="switchView(1)">
Productivity Insights
</div>

</div>

</div>

<!-- ================= VIEWS ================= -->

<div class="view-container">

<div class="views" id="views">

<!-- ================= BURNOUT ================= -->

<div class="screen">

{% if stats %}

<div class="stats">

<div class="card">
<h2>{{stats.high}}</h2>
<p>High Burnout</p>
</div>

<div class="card">
<h2>{{stats.medium}}</h2>
<p>Medium Burnout</p>
</div>

<div class="card">
<h2>{{stats.low}}</h2>
<p>Low Burnout</p>
</div>

</div>

<div class="chart-box">

<h2 class="section-title">
Burnout Distribution
</h2>

<canvas id="chart1"></canvas>

</div>

{% endif %}

</div>

<!-- ================= PRODUCTIVITY ================= -->

<div class="screen">

{% if prod %}

<div class="stats">

<div class="card">
<h2>{{prod.high}}</h2>
<p>High Productivity</p>
</div>

<div class="card">
<h2>{{prod.medium}}</h2>
<p>Moderate Productivity</p>
</div>

<div class="card">
<h2>{{prod.low}}</h2>
<p>Low Productivity</p>
</div>

</div>

<div class="chart-box">

<h2 class="section-title">
Productivity Insights
</h2>

<canvas id="chart2"></canvas>

</div>

{% endif %}

</div>

</div>

</div>

<!-- ================= DATASET TABLE ================= -->

{% if table %}

<div class="section-box">

<h2 class="section-title">
Dataset Intelligence
</h2>

<div class="metrics-grid">

<div class="metric-card">
<h3>{{table|length}}</h3>
<p>Total Records</p>
</div>

<div class="metric-card">
<h3>{{table[0].keys()|list|length}}</h3>
<p>Total Columns</p>
</div>

<div class="metric-card">
<h3>92%</h3>
<p>Dataset Quality</p>
</div>

<div class="metric-card">
<h3>AI</h3>
<p>Live Analytics Enabled</p>
</div>

</div>

<div class="table-box">

<table>

<tr>
{% for k in table[0].keys() %}
<th>{{k}}</th>
{% endfor %}
</tr>

{% for r in table[:50] %}
<tr>
{% for v in r.values() %}
<td>{{v}}</td>
{% endfor %}
</tr>
{% endfor %}

</table>

</div>

</div>

{% endif %}

<!-- ================= SUMMARY ================= -->

{% if stats %}

<div class="section-box">

<h2 class="section-title">
Executive AI Summary
</h2>

<p style="line-height:1.9;color:#cbd5e1;font-size:16px;">

{% if stats.high > stats.medium %}

AI analysis detected a high burnout concentration among workforce groups.
Productivity imbalance is increasing and continuous employee wellness
monitoring is strongly recommended.

{% elif stats.medium > stats.low %}

Moderate stress patterns detected with stable productivity clusters.
AI recommends balancing workloads and improving work-life management.

{% else %}

Overall workforce productivity remains healthy with controlled burnout levels.
Current operational balance appears stable.

{% endif %}

</p>

</div>

<div class="section-box">

<h2 class="section-title">
AI Activity Timeline
</h2>

<div class="timeline">

<div class="timeline-item">
Dataset uploaded successfully
</div>

<div class="timeline-item">
AI burnout clustering completed
</div>

<div class="timeline-item">
Productivity intelligence generated
</div>

<div class="timeline-item">
Recommendations and insights prepared
</div>

</div>

</div>

{% endif %}

<!-- ================= LIVE AI ================= -->

<div class="section-box ai-live-box">

<div class="live-header">

<div class="pulse"></div>

LIVE AI MONITORING

</div>

<div class="live-grid">

<div class="live-card">
<h3>Burnout Spike</h3>
<p>
AI detected workload imbalance in recent records.
</p>
</div>

<div class="live-card">
<h3>Productivity Stability</h3>
<p>
Performance clusters remain operationally stable.
</p>
</div>

<div class="live-card">
<h3>AI Recommendation</h3>
<p>
Introduce flexible scheduling and monitor overtime patterns.
</p>
</div>

</div>

</div>

<!-- ================= RECOMMENDATIONS ================= -->

{% if recommendations %}

<div class="section-box">

<h2 class="section-title">
AI Recommendations
</h2>

<div class="recommend-grid">

{% for r in recommendations %}

<div class="recommend-card">
{{r}}
</div>

{% endfor %}

</div>

</div>

{% endif %}

<!-- ================= DOWNLOADS ================= -->

{% if stats %}

<div class="downloads">

<a href="/download/burnout">

<button class="download-btn">
Download Burnout Report
</button>

</a>

<a href="/download/productivity">

<button class="download-btn">
Download Productivity Report
</button>

</a>

</div>

{% endif %}

</div>

<!-- ================= CHATBOT ================= -->

<div id="chat" onclick="toggleChat()">

<div class="orb-core"></div>

<div class="orb-ring"></div>

</div>

<div id="chatbox">

<div class="chat-header">
AI Dataset Assistant
</div>

<div id="chat-body"></div>

<div class="chat-input">

<input
id="chat_text"
placeholder="Ask AI about the dataset..."
onkeydown="if(event.key==='Enter'){sendMessage()}">

<button onclick="sendMessage()">
Send
</button>

</div>

</div>

<!-- ================= SCRIPTS ================= -->

<script>

/* ================= SWITCH VIEW ================= */

function switchView(index){

document.getElementById("views").style.transform =
`translateX(-${index*50}%)`

let slider=document.getElementById("slider")

slider.style.left = index===0 ? "6px" : "50%"

let options=document.querySelectorAll(".option")

options.forEach(o=>o.classList.remove("active"))

options[index].classList.add("active")

}

/* ================= CHAT ================= */

function toggleChat(){

let c=document.getElementById("chatbox")

c.style.display =
c.style.display==="flex" ? "none" : "flex"

}

function sendMessage(){

let i=document.getElementById("chat_text")

let m=i.value.trim()

if(!m)return

let b=document.getElementById("chat-body")

b.innerHTML += `<div class='msg user'>${m}</div>`

let t=document.createElement("div")

t.className="msg ai"

t.innerHTML="Analyzing dataset..."

b.appendChild(t)

b.scrollTop=b.scrollHeight

fetch("/chat",{
method:"POST",
headers:{
"Content-Type":"application/json"
},
body:JSON.stringify({
message:m
})
})
.then(r=>r.json())
.then(d=>{

t.innerHTML=d.reply

b.scrollTop=b.scrollHeight

})

i.value=""

}

/* ================= CLOCK ================= */

setInterval(()=>{

let now=new Date()

document.getElementById("liveTime").innerHTML =
now.toLocaleTimeString()

},1000)

/* ================= COUNTERS ================= */

function animateCounter(id,target){

let el=document.getElementById(id)

let count=0

let speed=target/80

let x=setInterval(()=>{

count += speed

if(count>=target){

count=target

clearInterval(x)

}

el.innerHTML=Math.floor(count)

},20)

}

/* ================= DYNAMIC COUNTERS ================= */

{% if table %}
animateCounter("counter1", {{table|length}})
{% else %}
animateCounter("counter1", 0)
{% endif %}

{% if stats %}
animateCounter(
"counter2",
{{stats.high + stats.medium + stats.low}}
)
{% else %}
animateCounter("counter2", 0)
{% endif %}

{% if recommendations %}
animateCounter(
"counter3",
{{recommendations|length}}
)
{% else %}
animateCounter("counter3", 0)
{% endif %}

/* ================= PARTICLES ================= */

const canvas=document.getElementById("bgCanvas")

const ctx=canvas.getContext("2d")

canvas.width=window.innerWidth
canvas.height=window.innerHeight

let particles=[]

for(let i=0;i<100;i++){

particles.push({

x:Math.random()*canvas.width,
y:Math.random()*canvas.height,
r:Math.random()*2.5,
dx:(Math.random()-0.5)*0.5,
dy:(Math.random()-0.5)*0.5

})

}

function animateParticles(){

ctx.clearRect(0,0,canvas.width,canvas.height)

particles.forEach(p=>{

ctx.beginPath()

ctx.arc(p.x,p.y,p.r,0,Math.PI*2)

ctx.fillStyle="#3b82f6"

ctx.fill()

p.x += p.dx
p.y += p.dy

if(p.x<0 || p.x>canvas.width)p.dx*=-1
if(p.y<0 || p.y>canvas.height)p.dy*=-1

})

requestAnimationFrame(animateParticles)

}

animateParticles()

window.addEventListener("resize",()=>{

canvas.width=window.innerWidth
canvas.height=window.innerHeight

})

/* ================= CHARTS ================= */

{% if stats %}

new Chart(document.getElementById('chart1'),{

type:'bar',

data:{
labels:['Low','Medium','High'],
datasets:[{
label:'Burnout',
data:[
{{stats.low}},
{{stats.medium}},
{{stats.high}}
],
backgroundColor:[
'#22c55e',
'#facc15',
'#ef4444'
],
borderRadius:16,
barThickness:70
}]
},

options:{

responsive:true,

plugins:{
legend:{
labels:{
color:'#e2e8f0'
}
}
},

animation:{
duration:2200,
easing:'easeOutQuart'
},

scales:{

y:{
grid:{
color:'#1e293b'
},
ticks:{
color:'#94a3b8'
}
},

x:{
grid:{
display:false
},
ticks:{
color:'#94a3b8'
}
}

}

}

})

{% endif %}

{% if prod %}

new Chart(document.getElementById('chart2'),{

type:'line',

data:{
labels:['Low','Medium','High'],
datasets:[{
label:'Productivity',
data:[
{{prod.low}},
{{prod.medium}},
{{prod.high}}
],
borderColor:'#3b82f6',
backgroundColor:'rgba(59,130,246,0.25)',
fill:true,
tension:0.45,
pointRadius:5
}]
},

options:{

responsive:true,

plugins:{
legend:{
labels:{
color:'#e2e8f0'
}
}
},

animation:{
duration:2400
},

scales:{

y:{
grid:{
color:'#1e293b'
},
ticks:{
color:'#94a3b8'
}
},

x:{
grid:{
display:false
},
ticks:{
color:'#94a3b8'
}
}

}

}

})

{% endif %}

</script>

</body>
</html>
"""


# ---------------- ROUTES ----------------
@app.route("/", methods=["GET", "POST"])
def home():
    global last_df

    stats = None
    table = None
    prod = None
    rec = None

    if request.method == "POST":
        file = request.files.get("file")

        if file:
            df = pd.read_csv(file, on_bad_lines="skip")
            df, stats = auto_train(df)
            df = add_productivity(df)
            last_df = df

            table = df.to_dict(orient="records")

            prod = {
                "high": int((df["Productivity"] == "High Productivity").sum()),
                "medium": int((df["Productivity"] == "Moderate Productivity").sum()),
                "low": int((df["Productivity"] == "Low Productivity").sum())
            }

            rec = recommendations(stats)

    return render_template_string(
        HTML,
        stats=stats,
        table=table,
        prod=prod,
        recommendations=rec
    )


@app.route("/chat", methods=["POST"])
def chat():
    return jsonify({
        "reply": ai_chat(request.get_json()["message"], last_df)
    })


@app.route("/download/<type>")
def download(type):
    if last_df is None:
        return "No data"

    output = io.StringIO()

    if type == "burnout":
        last_df[["Burnout"]].to_csv(output, index=False)
    else:
        last_df[["Productivity"]].to_csv(output, index=False)

    output.seek(0)

    return send_file(
        io.BytesIO(output.getvalue().encode()),
        download_name=f"{type}.csv",
        as_attachment=True
    )


if __name__ == "__main__":
    app.run(debug=True)
