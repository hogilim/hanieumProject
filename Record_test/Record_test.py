# IMPORT: cv2, datetime, numpy
# time은 녹화 시작시간을 의미함
# 주의! '%Y-%m-%d %H:%M:%S' 포맷이 약간 상이. 다른 csv 와 엔티티 관계가 맺어질 경우 상의 요함.
import numpy as np
import cv2
import datetime
import pandas as pd
import os
# This will return video from the first webcam on your computer.
# 아래는 cctv에 접근시 필요한 정보들 (4가지)로써, cctv dataframe으로 접근하여 가져와야함.

CCTV_id = "admin"
CCTV_pass = "@!admin12"
# IP + RTSP_PORT가 합쳐진 ADD 저장 (09.21 회의 중 수정)
CCTV_IP_ADD = "192.168.20.56"
PORT = ":554"


#cap = cv2.VideoCapture("rtsp://" + CCTV_id + ":" + CCTV_pass + "@" + CCTV_IP_ADD+P0RT+
#+ "/profile2/media.smp")
cap = cv2.VideoCapture("../test2.mp4")
fourcc = cv2.VideoWriter_fourcc(*'XVID')
#녹화기록을 저장할 log csv를 불러옴.
record_log_file =  pd.read_csv('./data/record_log.csv')
source = 'Test'


# 녹화시작 시간 초기화
dtime = datetime.datetime.now()
now =  time.strftime('%Y-%m-%d %H-%M-%S')
Interval_target = datetime.timedelta(seconds= 5)
Flag = True
temp_pf = record_log_file.tail(1)
log_time = temp_pf.loc[record_log_file.last_valid_index()].at['Time']
log_time_obj = datetime.datetime.strptime(log_time, '%Y-%m-%d %H-%M-%S')
Interval = time-log_time_obj
# 녹화가 오래 끊겨있다 시작되는 경우, 현재 시간을 기준으로 녹화 시작.
if (Interval > Interval_target):
    log_time_obj = dtime




def Record_on_log(Source,Reco_path,Time):
    print(record_log_file.last_valid_index())
    if(record_log_file.last_valid_index()>5):
        # 지워야하는 동영상 파일인 경우 삭제 진행
        # 파일이 없는경우 오류 발생을 막기 위해 파일 있는경우의 if structure도 생성
        Flag_func = record_log_file.loc[record_log_file.first_valid_index()].at['Flag']
        print(Flag_func)
        if(Flag_func == False):
            path = record_log_file.loc[record_log_file.first_valid_index()].at['reco_path']
            if os.path.isfile(path):
                os.remove(path)
                print("제거")
        record_log_file.drop(record_log_file.head(1).index, inplace = True)
        record_log_file.to_csv('./data/record_log.csv')


    print("기록")
    Add_log = pd. DataFrame({'Source': [Source] , 'reco_path': [Reco_path],'Time': [Time], 'Flag':['False']})
    temp = record_log_file.append(Add_log, ignore_index=True)
    temp.to_csv('./data/record_log.csv', index=False)

#Flag는 영상의 저장 명칭을 바꿔주기 위해 사용함.
# loop runs if capturing has been initialized.
while (True):
    record_log_file = pd.read_csv('./data/record_log.csv')
    print("저장 시작")
    # 먼저 record_on_log에 기록하고 시작  ,log 기록시에는 여기 마지막을 참조하도록 할것
    # def_logging을 이에따라 RECO_PATH를 어케 할것인지 기록이 필요함. -> 발견되면 해당 행의 flag를 true로 바꿔주는 작업 필요.
    time_str = dtime.strftime('%Y-%m-%d %H-%M-%S')
    name = "./"+ CCTV_IP_ADD+ "/"+time_str + ".avi"
    # 미리 로그에 남겨둠
    Record_on_log(time_str, name, time_str)
    out = cv2.VideoWriter(name, fourcc, 20.0, (640, 480))
    Flag = False
    while (Flag == False):
        ret, frame = cap.read()
        #cv2.imshow('Original', frame)
        # time 변수는 계속계속 업데이트 해줌
        dtime = datetime.datetime.now()
        #ms 단위는 없애주기 위해 String type의 now 변수를 선언해줌.
        # 마지막 행을 참조하여 시간을 참조함. 이 때, 시간 연산을 위해 참조한 시간을 datetime 라이브러리를 이용해 형변환해줌.
        Interval = time-log_time_obj
        if(Interval <= Interval_target):
            out.write(frame)
            cv2.imshow('Original', frame)
        elif (Interval > Interval_target):
            out.release()
            print("녹화파일 저장 ")
            log_time_obj = dtime
            #record_frame에 기록해주기.
            #recog_log 인덱스 100개 넘으면 상위 삭제하고 기록 추가하기.
            Flag = True
            break

        # Wait for 'a' key to stop the program
        if cv2.waitKey(1) & 0xFF == ord('a'):
            break

# Close the window / Release webcam
cap.release()
# After we release our webcam, we also release the output
# De-allocate any associated memory usage
cv2.destroyAllWindows()


#cap = cv2.VideoCapture("rtsp://admin:@!admin12@192.168.0.106:554/profile2/media.smp")
# 핵심 코드: cv2.VideoWriter( filename, fourcc, fps, frameSize )
# videowriter는 비디오를 녹화하게끔 해주는 코드.


