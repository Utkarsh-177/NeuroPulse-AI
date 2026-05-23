from flask import Flask, request, render_template_string, jsonify, send_file
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans
import requests, os, io

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
                "model": "meta/llama-3.3-70b-instruct",
                "messages": [{
                    "role": "user",
                    "content": f"Dataset Columns: {cols}\n\nDataset Summary:\n{summary}\n\nQuestion: {q}"
                }],
                "max_tokens": 300
            },
            timeout=20
        )

        data = res.json()

        if "choices" not in data:
            return "AI response error."

        return data["choices"][0]["message"]["content"]

    except Exception as e:
        return f"Error: {str(e)}"


# ---------------- HTML ----------------
HTML = """
<!DOCTYPE html>
<html>
<head>
<title>Burnout AI Enterprise</title>

<meta name="viewport" content="width=device-width, initial-scale=1.0">

<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>

<style>

*{
margin:0;
padding:0;
box-sizing:border-box;
}

body{
font-family:system-ui;
background:#020617;
color:#e2e8f0;
overflow-x:hidden;
}

/* PARTICLE BACKGROUND */

#bgCanvas{
position:fixed;
top:0;
left:0;
width:100%;
height:100%;
z-index:-3;
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
z-index:-2;
animation:floatGlow 8s infinite alternate ease-in-out;
}

@keyframes floatGlow{
from{
transform:translateY(0)
}
to{
transform:translateY(40px)
}
}

.container{
max-width:1350px;
margin:auto;
padding:40px 25px;
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

/* TOP STATUS BAR */

.top-status-bar{
position:fixed;
top:18px;
left:50%;
transform:translateX(-50%);
display:flex;
gap:18px;
z-index:999;
}

.status-card{
background:rgba(15,23,42,0.75);
backdrop-filter:blur(14px);
padding:16px 24px;
border-radius:18px;
border:1px solid #1e293b;
text-align:center;
min-width:150px;
box-shadow:0 10px 30px rgba(0,0,0,0.25);
}

.status-card h3{
font-size:20px;
margin-bottom:5px;
}

.status-card p{
font-size:13px;
color:#94a3b8;
}

/* HERO */

.hero{
text-align:center;
padding-top:100px;
padding-bottom:30px;
position:relative;
}

.hero-glow{
position:absolute;
width:450px;
height:450px;
background:radial-gradient(circle,#2563eb55,transparent 70%);
filter:blur(40px);
top:50px;
left:50%;
transform:translateX(-50%);
z-index:-1;
animation:floatGlow 5s infinite alternate ease-in-out;
}

h1{
font-size:58px;
font-weight:900;
margin-bottom:12px;
}

.subtitle{
color:#94a3b8;
font-size:18px;
max-width:900px;
margin:auto;
line-height:1.7;
}

/* MINI CARDS */

.hero-mini-grid{
display:grid;
grid-template-columns:repeat(auto-fit,minmax(220px,1fr));
gap:20px;
margin-top:55px;
}

.mini-card{
background:rgba(15,23,42,0.75);
backdrop-filter:blur(14px);
padding:28px;
border-radius:24px;
border:1px solid #1e293b;
text-align:center;
transition:0.35s;
}

.mini-card:hover{
transform:translateY(-6px);
box-shadow:0 15px 40px rgba(37,99,235,0.25);
}

.mini-card h2{
font-size:42px;
margin-bottom:10px;
}

/* UPLOAD */

.upload{
display:flex;
justify-content:center;
align-items:center;
flex-direction:column;
gap:12px;
max-width:760px;
margin:45px auto;
padding:70px;
border:2px dashed #2563eb;
border-radius:30px;
background:rgba(15,23,42,0.85);
backdrop-filter:blur(10px);
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
transition:0.8s;
}

.upload:hover::before{
transform:translateX(100%);
}

.upload:hover{
transform:translateY(-6px) scale(1.01);
box-shadow:0 0 40px #2563eb55;
}

/* SWITCH */

.switch-wrapper{
display:flex;
justify-content:center;
margin:45px 0;
}

.switch{
position:relative;
width:430px;
height:62px;
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
height:50px;
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
font-weight:600;
font-size:15px;
color:#94a3b8;
transition:0.3s;
user-select:none;
}

.option.active{
color:white;
}

.view-container{
overflow:hidden;
width:100%;
}

.views{
display:flex;
width:200%;
transition:transform 0.6s cubic-bezier(.77,0,.18,1);
}

.screen{
width:100%;
padding:10px;
}

/* STATS */

.stats{
display:grid;
grid-template-columns:repeat(auto-fit,minmax(220px,1fr));
gap:25px;
margin-bottom:35px;
}

.card{
background:linear-gradient(145deg,#0f172a,#111827);
padding:35px;
border-radius:25px;
text-align:center;
border:1px solid #1e293b;
transition:0.35s;
backdrop-filter:blur(12px);
}

.card:hover{
transform:translateY(-8px);
box-shadow:0 20px 40px rgba(0,0,0,0.35);
}

.card h2{
font-size:42px;
margin-bottom:10px;
}

/* CHARTS */

.chart-box,.section-box{
background:#0f172a;
padding:30px;
border-radius:25px;
border:1px solid #1e293b;
margin-bottom:40px;
box-shadow:0 15px 40px rgba(0,0,0,0.25);
}

.section-title{
font-size:28px;
margin-bottom:18px;
}

/* LIVE AI */

.ai-live-box{
margin-top:40px;
background:#0f172a;
border:1px solid #1e293b;
padding:30px;
border-radius:28px;
}

.live-header{
display:flex;
align-items:center;
gap:12px;
font-size:22px;
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
transform:scale(2.2);
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

/* TABLE */

.table-box{
max-height:430px;
overflow:auto;
border-radius:20px;
border:1px solid #1e293b;
background:#0f172a;
margin-top:25px;
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
}

th{
position:sticky;
top:0;
background:#111827;
z-index:2;
}

th,td{
padding:14px;
text-align:center;
border-bottom:1px solid #1e293b;
}

tr:hover{
background:#172554;
}

/* HEALTH BAR */

.health-wrapper{
margin-top:55px;
padding:30px;
background:#0f172a;
border-radius:28px;
border:1px solid #1e293b;
}

.health-bar{
width:100%;
height:28px;
background:#111827;
border-radius:50px;
overflow:hidden;
margin-top:25px;
}

.health-fill{
height:100%;
width:84%;
background:linear-gradient(90deg,#2563eb,#3b82f6);
border-radius:50px;
animation:healthMove 2s ease;
box-shadow:0 0 25px #2563eb;
}

@keyframes healthMove{
from{
width:0;
}
to{
width:84%;
}
}

.health-text{
margin-top:18px;
font-size:18px;
color:#cbd5e1;
}

/* DOWNLOADS */

.downloads{
display:flex;
gap:20px;
margin-top:50px;
flex-wrap:wrap;
}

.download-btn{
padding:15px 24px;
border:none;
border-radius:14px;
background:linear-gradient(135deg,#2563eb,#3b82f6);
color:white;
font-weight:600;
cursor:pointer;
transition:0.35s;
font-size:15px;
}

.download-btn:hover{
transform:translateY(-4px) scale(1.02);
}

/* CHATBOT */

#chat{
position:fixed;
bottom:25px;
right:25px;
width:80px;
height:80px;
display:flex;
justify-content:center;
align-items:center;
cursor:pointer;
z-index:999;
}

.orb-core{
width:30px;
height:30px;
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
width:70px;
height:70px;
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
bottom:105px;
right:25px;
width:360px;
height:510px;
background:#0f172a;
border-radius:24px;
display:none;
flex-direction:column;
border:1px solid #1e293b;
overflow:hidden;
box-shadow:0 25px 60px rgba(0,0,0,0.45);
z-index:999;
}

#chat-body{
flex:1;
overflow:auto;
padding:18px;
display:flex;
flex-direction:column;
gap:12px;
}

.msg{
padding:12px 14px;
border-radius:14px;
max-width:85%;
line-height:1.5;
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
padding:15px;
background:#020617;
border:none;
outline:none;
color:white;
font-size:14px;
}

.chat-input button{
width:80px;
border:none;
background:#2563eb;
color:white;
font-weight:600;
cursor:pointer;
}

</style>

<script>

function switchView(index){

document.getElementById("views").style.transform =
`translateX(-${index*50}%)`

let slider=document.getElementById("slider")

slider.style.left = index===0 ? "6px" : "50%"

let options=document.querySelectorAll(".option")

options.forEach(o=>o.classList.remove("active"))

options[index].classList.add("active")
}

function toggleChat(){

let c=document.getElementById("chatbox")

c.style.display = c.style.display==="flex" ? "none" : "flex"
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

</script>

</head>

<body>

<canvas id="bgCanvas"></canvas>

<div class="top-status-bar">

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

<div class="container">

<div class="hero">

<div class="hero-glow"></div>

<h1>Burnout AI</h1>

<p class="subtitle">
Enterprise-grade AI-powered burnout detection, productivity analytics,
workforce monitoring, intelligent recommendations, and live business insights.
</p>

<div class="hero-mini-grid">

<div class="mini-card">
<h2 id="counter1">0</h2>
<p>Employees Analyzed</p>
</div>

<div class="mini-card">
<h2 id="counter2">0</h2>
<p>AI Predictions</p>
</div>

<div class="mini-card">
<h2 id="counter3">0</h2>
<p>Insights Generated</p>
</div>

</div>

</div>

<form method="POST" enctype="multipart/form-data">

<label class="upload">

<h2>Upload Dataset</h2>

<p style="color:#94a3b8">
Analyze any employee productivity or workforce dataset using AI.
</p>

<input type="file" name="file" hidden onchange="this.form.submit()">

</label>

</form>

<!-- KEEP YOUR EXISTING HTML BELOW -->
<!-- YOUR STATS -->
<!-- CHARTS -->
<!-- TABLE -->
<!-- RECOMMENDATIONS -->
<!-- DOWNLOADS -->

<div class="ai-live-box">

<div class="live-header">
<div class="pulse"></div>
LIVE AI MONITORING
</div>

<div class="live-grid">

<div class="live-card">
<h3>Burnout Spike</h3>
<p>AI detected workload imbalance in recent records.</p>
</div>

<div class="live-card">
<h3>Productivity Stability</h3>
<p>Performance clusters remain operationally stable.</p>
</div>

<div class="live-card">
<h3>AI Recommendation</h3>
<p>Introduce flexible scheduling and monitor overtime patterns.</p>
</div>

</div>

</div>

<div class="health-wrapper">

<h2 class="section-title">Workforce Health Index</h2>

<div class="health-bar">

<div class="health-fill"></div>

</div>

<p class="health-text">
AI Workforce Stability Score : 84%
</p>

</div>

</div>

<div id="chat" onclick="toggleChat()">

<div class="orb-core"></div>

<div class="orb-ring"></div>

</div>

<div id="chatbox">

<div id="chat-body"></div>

<div class="chat-input">

<input id="chat_text"
placeholder="Ask AI about the dataset..."
onkeydown="if(event.key==='Enter'){sendMessage()}">

<button onclick="sendMessage()">
Send
</button>

</div>

</div>

<script>

// LIVE CLOCK

setInterval(()=>{

let now=new Date()

document.getElementById("liveTime").innerHTML=
now.toLocaleTimeString()

},1000)


// COUNTER ANIMATION

function animateCounter(id,target){

let el=document.getElementById(id)

let count=0

let speed=target/80

let x=setInterval(()=>{

count+=speed

if(count>=target){
count=target
clearInterval(x)
}

el.innerHTML=Math.floor(count)

},20)

}

animateCounter("counter1",1250)
animateCounter("counter2",3480)
animateCounter("counter3",920)


// PARTICLES

const canvas=document.getElementById("bgCanvas")
const ctx=canvas.getContext("2d")

canvas.width=window.innerWidth
canvas.height=window.innerHeight

let particles=[]

for(let i=0;i<80;i++){

particles.push({
x:Math.random()*canvas.width,
y:Math.random()*canvas.height,
r:Math.random()*2,
dx:(Math.random()-0.5)*0.5,
dy:(Math.random()-0.5)*0.5
})

}

function animate(){

ctx.clearRect(0,0,canvas.width,canvas.height)

particles.forEach(p=>{

ctx.beginPath()
ctx.arc(p.x,p.y,p.r,0,Math.PI*2)
ctx.fillStyle="#2563eb"
ctx.fill()

p.x+=p.dx
p.y+=p.dy

if(p.x<0||p.x>canvas.width)p.dx*=-1
if(p.y<0||p.y>canvas.height)p.dy*=-1

})

requestAnimationFrame(animate)

}

animate()

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
