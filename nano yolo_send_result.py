import socket
from ultralytics import YOLO

# 노트북 IP 주소와 포트 (네트워크 환경에 맞게 수정)
HOST = '노트북_IP주소'  # 예: '192.168.0.5'
PORT = 9999

def send_result(result_text):
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.connect((HOST, PORT))
        s.sendall((result_text + '\n').encode())
        s.close()
        print(f"Sent: {result_text}")
    except Exception as e:
        print("Error sending data:", e)

# YOLO 모델 로드
model = YOLO('best.pt')  # 본인이 학습한 모델 파일명

# 실시간 카메라 영상 인식 (source=0)
results = model(source=0, stream=True)

for result in results:
    if result.boxes:
        class_id = int(result.boxes.cls[0])
        label = model.names[class_id]
        print(f"Detected: {label}")
        send_result(label)
