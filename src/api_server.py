from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import uvicorn
import numpy as np
import torch
import torch.nn as nn
import time
import sqlite3

app = FastAPI(title="工业机器人健康状态监测系统")
templates = Jinja2Templates(directory="templates")
DB_PATH = "db/robot_fault.db"

# 初始化数据库
def init_db():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute('''
    CREATE TABLE IF NOT EXISTS fault_result(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        create_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        pred_label INTEGER,
        pred_prob REAL,
        infer_time REAL
    )
    ''')
    conn.commit()
    conn.close()

init_db()


class SimpleLSTM(nn.Module):
    def __init__(self, input_dim=6):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, 64, batch_first=True)
        self.linear = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        _, (h, _) = self.lstm(x)
        out = self.linear(h[-1])
        return self.sigmoid(out)


model = SimpleLSTM(input_dim=6)
model.eval()


def model_predict(arr: np.ndarray):
    tensor_x = torch.from_numpy(arr).float().unsqueeze(0)
    with torch.no_grad():
        prob = model(tensor_x)
    pred_prob = float(prob.squeeze().cpu().numpy())
    pred_label = 1 if pred_prob > 0.5 else 0
    return pred_label, pred_prob



@app.get("/", response_class=HTMLResponse)
async def root_redirect(request: Request):
    # 根路径重定向到输入页面
    return templates.TemplateResponse(request,"input.html",context={"error_msg":""})


@app.get("/input", response_class=HTMLResponse)
async def input_page(request: Request):
    return templates.TemplateResponse(request,"input.html",context={"error_msg":""})



@app.post("/predict", response_class=HTMLResponse)
async def predict(
    request: Request,
    s1_0: str = Form(""),s1_1: str = Form(""),s1_2: str = Form(""),
    s1_3: str = Form(""),s1_4: str = Form(""),s1_5: str = Form(""),
    s2_0: str = Form(""),s2_1: str = Form(""),s2_2: str = Form(""),
    s2_3: str = Form(""),s2_4: str = Form(""),s2_5: str = Form(""),
):
    error_msg = ""
    result = None
    try:
        s1_list = [s1_0,s1_1,s1_2,s1_3,s1_4,s1_5]
        s2_list = [s2_0,s2_1,s2_2,s2_3,s2_4,s2_5]
        all_rows = []
        for idx,row_data in enumerate([s1_list,s2_list]):
            temp = []
            for val_str in row_data:
                vs = val_str.strip()
                if not vs:
                    raise ValueError(f"样本{idx+1}存在空输入框，请全部填写浮点数！")
                temp.append(float(vs))
            if len(temp)!=6:
                raise ValueError(f"样本{idx+1}必须填写6个数据")
            all_rows.append(temp)
        input_arr = np.array(all_rows,dtype=np.float32)

        # 统计推理耗时
        t_start = time.time()
        pred_label,pred_prob = model_predict(input_arr)
        t_cost_ms = round((time.time() - t_start)*1000,2)

        result = {
            "pred_label":int(pred_label),
            "pred_prob":round(pred_prob,4),
            "infer_time": t_cost_ms
        }
        # 写入数据库
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO fault_result(pred_label,pred_prob,infer_time) VALUES (?,?,?)",
            (result["pred_label"], result["pred_prob"], result["infer_time"])
        )
        conn.commit()
        conn.close()

    except Exception as e:
        error_msg = str(e)
        #出错回到输入页面，带回错误提示
        return templates.TemplateResponse(request,"input.html",context={"error_msg":error_msg})
    #成功渲染结果页
    return templates.TemplateResponse(request,"result.html",context={"result":result})



if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
