
import cv2, time, pandas
from datetime import datetime 

static_back = None
motion_list = [ None, None ] 
time = [] 

df = pandas.DataFrame(columns = ["Start", "End"]) 

video = cv2.VideoCapture(0) 


while True: 
	# Reading frame(image) from video 
	check, frame = video.read() 

	motion = 0

	# Converting color image to gray_scale image 
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 

	# Converting gray scale image to GaussianBlur 
	gray = cv2.GaussianBlur(gray, (21, 21), 0) 

	if static_back is None: 
		static_back = gray 
		continue

	# Difference between static background and current frame
	diff_frame = cv2.absdiff(static_back, gray) 


	thresh_frame = cv2.threshold(diff_frame, 30, 255, cv2.THRESH_BINARY)[1] 
	thresh_frame = cv2.dilate(thresh_frame, None, iterations = 2) 

	# Finding contour of moving object 
	(_, cnts, _) = cv2.findContours(thresh_frame.copy(), 
					cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) 

	for contour in cnts: 
		if cv2.contourArea(contour) < 10000: 
			continue
		motion = 1

		(x, y, w, h) = cv2.boundingRect(contour) 
		# making green rectangle arround the moving object 
		cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3) 

	# Appending status of motion 
	motion_list.append(motion) 

	motion_list = motion_list[-2:] 

	# Appending Start time of motion 
	if motion_list[-1] == 1 and motion_list[-2] == 0: 
		time.append(datetime.now()) 

	# Appending End time of motion 
	if motion_list[-1] == 0 and motion_list[-2] == 1: 
		time.append(datetime.now()) 

	cv2.imshow("Gray Frame", gray) 
	cv2.imshow("Difference Frame", diff_frame) 
	cv2.imshow("Threshold Frame", thresh_frame) 
	cv2.imshow("Color Frame", frame) 

	key = cv2.waitKey(1) 

	if key == ord('q'): 
		if motion == 1: 
			time.append(datetime.now()) 
		break


for i in range(0, len(time), 2): 
	df = df.append({"Start":time[i], "End":time[i + 1]}, ignore_index = True) 

# Creating a csv file in which time of movements will be saved 
df.to_csv("Time_of_movements.csv") 

video.release() 
cv2.destroyAllWindows() 
