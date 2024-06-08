'''
* Team Id: 1194
* Author List: Keshav Joshi, Ashish Rathore, Disha Chhabra.
* Filename: liveTrack.py
* Theme: eYRC Geo Guide(GG)
* Functions: sending_data, read_aruco_info, modify_aruco_info, write_aruco_info, update_csv
* Global Variables: ip, csv_file_path, aruco_info, parameters, aruco_dict, M_corners, M_ids, 
                    src_points, dst_points, transformation_model, counter, marker_id_to_track, 
                    frame_width, frame_height
'''



# All the import we need to run this code.
import cv2
import numpy as np
import csv
from cv2 import aruco
import time
import socket



# Here we are setting up to send data to our ESP32 this is our wifi ip address

ip = "192.168.110.58"   # Keshav Laptop Enter IP address of laptop after connecting it to WIFI hotspot

# Reading that saved path so that we can send them 
with open("path.txt", 'r') as f:
    read = f.read()
    f.close()


'''
* Function Name: sending_data
* Input: None
* Output: None
* Logic: Send the path data to ESP32 over a socket connection.

* Example Call: sending_data()
'''
# Here we are sending the path.txt file which we have read to ESP32 so that it can travers on that nodes
def sending_data():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind((ip, 8002))
        s.listen()
        conn, addr = s.accept()
        with conn:
            print(f"Connected by {addr}")
            while True:
                data = conn.recv(1024)
                conn.sendall(str.encode(str(read)))
                time.sleep(1)
                break



'''
* Function Name: read_aruco_info
* Input: csv_file (str) - Path to the CSV file containing Aruco information.
* Output: dict - Dictionary containing Aruco information.
* Logic: Read Aruco information from a CSV file and return it as a dictionary.
* Example Call: read_aruco_info('path/to/file.csv')
'''
# Function to read Aruco information from the CSV file
def read_aruco_info(csv_file):
    aruco_info = {}
    with open(csv_file, 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            aruco_id = int(row['ArucoID'])
            source_x = float(row['sourceX'])
            source_y = float(row['sourceY'])
            map_x = float(row['mapX'])
            map_y = float(row['mapY'])
            enable = int(row['Enable'])
            aruco_info[aruco_id] = {'sourceX': source_x, 'sourceY': source_y, 'mapX': map_x, 'mapY': map_y, 'Enable': enable}
    return aruco_info


'''
* Function Name: modify_aruco_info
* Input: 
    - aruco_info (dict) - Dictionary containing Aruco information.
    - aruco_id (int) - ID of the Aruco marker to be modified.
    - new_source_x (float) - New source X coordinate.
    - new_source_y (float) - New source Y coordinate.
* Output: None
* Logic: Modify the coordinates and enable status of a specific Aruco marker.
* Example Call: modify_aruco_info(aruco_data, 1, 10.0, 20.0)
'''
# Function to modify Aruco coordinates and enable status
def modify_aruco_info(aruco_info, aruco_id, new_source_x, new_source_y):
    if aruco_id in aruco_info:
        aruco_info[aruco_id]['sourceX'] = new_source_x
        aruco_info[aruco_id]['sourceY'] = new_source_y
    else:
        aruco_info[aruco_id] = {'sourceX': new_source_x, 'sourceY': new_source_y, 'mapX': 0.0, 'mapY': 0.0, 'Enable': 0}


'''
* Function Name: write_aruco_info
* Input: 
    - csv_file (str) - Path to the CSV file to be written.
    - aruco_info (dict) - Modified Aruco information.
* Output: None
* Logic: Write modified Aruco information back to the CSV file.
* Example Call: write_aruco_info('path/to/file.csv', modified_data)
'''
# Function to write modified Aruco information back to the CSV file
def write_aruco_info(csv_file, aruco_info):
    with open(csv_file, 'w', newline='') as file:
        fieldnames = ['sourceX', 'sourceY', 'mapX', 'mapY', 'ArucoID', 'Enable']
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for aruco_id, info in aruco_info.items():
            writer.writerow({'sourceX': info['sourceX'], 
                             'sourceY': info['sourceY'], 
                             'mapX': info['mapX'], 
                             'mapY': info['mapY'], 
                             'ArucoID': aruco_id, 
                             'Enable': info['Enable']})




'''
* Function Name: update_csv
* Input: 
    - lat (float) - Latitude value to be updated.
    - lon (float) - Longitude value to be updated.
* Output: None
* Logic: Update a CSV file with latitude and longitude values.
* Example Call: update_csv(40.7128, -74.0060)
'''

def update_csv(lat, lon):
    # Define field names for the CSV file
    field_names = ['lat', 'lon']
    
    # Create a dictionary with the new values
    data = {'lat': lat, 'lon': lon}
    
    # Open the CSV file in write mode
    with open('live_location.csv', mode='w', newline='') as file:
        # Create a CSV DictWriter object
        writer = csv.DictWriter(file, fieldnames=field_names)
        
        # Write the header if the file is empty
        writer.writeheader()
        
        # Write the updated data to the CSV file
        writer.writerow(data)

# Load the camera
cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)

frame_width = 1920
frame_height = 1080

cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

# Discard initial frames
print("initial frame")
for _ in range(10):
    ret, frame = cap.read()

# Read Aruco information from the CSV file
csv_file_path = 'BackupArucoOnMapCordinate6-54.csv'
aruco_info = read_aruco_info(csv_file_path)

# Initializing the detector parameters
parameters = aruco.DetectorParameters()
# defing the dicitonary for our aruco which is 4*4 and range form 0 to 250
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_250)

M_corners=[]
M_ids=[]

print("Collecting IDs ...")
for i in range(10):
    ret, frame = cap.read()

    mainImgGray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detecting markers in the frame without specifying size
    corners, ids, _ = aruco.detectMarkers(mainImgGray, aruco_dict, parameters=parameters, corners=None)

    # Checking if any new ArUco markers are detected
    if ids is not None:
        for idx, marker_id in enumerate(ids):
            if marker_id not in ids:
                # if new ArUco marker detected we add it to the list
                M_ids.append(marker_id)
                M_corners.append(corners[idx])  #isko shi krna hoga



# Open CSV file to write coordinates
with open(csv_file_path, mode='w', newline='') as file:
    # Loop through all detected Aruco markers
    for i in range(len(M_ids)):
        # Get the corners of the Aruco marker
        marker_corners = M_corners[i][0]
        # Calculate the pixel coordinates of the Aruco marker (centroid)
        x = int((marker_corners[:, 0].min() + marker_corners[:, 0].max()) / 2)
        y = int((marker_corners[:, 1].min() + marker_corners[:, 1].max()) / 2)

        aruco_id = M_ids[i][0]

        # Modify Aruco information
        modify_aruco_info(aruco_info, aruco_id, x, -y)
    

# Write modified Aruco information back to the CSV file
write_aruco_info(csv_file_path, aruco_info)


src_points = []  # List to store source points
dst_points = []  # List to store destination points

with open(csv_file_path, 'r') as file:
    reader = csv.reader(file)
    next(reader)  # Skip header if exists
    for row in reader:
        sourceX, sourceY, mapX, mapY, ArucoID, Enable = map(float, row)
        if Enable != 0:  # Only consider points with Enable not equal to 0
            src_points.append([sourceX, sourceY])
            dst_points.append([mapX, mapY])

# Convert lists to numpy arrays
src_points = np.array(src_points)
dst_points = np.array(dst_points)

# Now you have src_points and dst_points as numpy arrays containing GCPs

# transformation_model = cv2.estimateAffinePartial2D(src_points, dst_points)[0]
transformation_model = cv2.estimateAffine2D(src_points, dst_points)[0]


counter = 0
marker_id_to_track=100

while True:
    # Read a frame from the video feed
    ret, frame = cap.read()

    if not ret:
        print("Error reading frame.")
        break
    
    cv2.namedWindow("feed", cv2.WINDOW_NORMAL)
    cv2.imshow("feed", frame)
    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    corners, ids, _ = aruco.detectMarkers(gray, aruco_dict, parameters=parameters, corners=None)

    image_with_markers = aruco.drawDetectedMarkers(frame.copy(), corners, ids)
    
    # Check if the marker_id_to_track is in the detected IDs
    if marker_id_to_track in ids:
        marker_index = list(ids).index(marker_id_to_track)
        marker_corners = corners[marker_index][0]

        # Calculate the pixel coordinates of the Aruco marker (centroid)
        x = int(((marker_corners[:, 0].min() + marker_corners[:, 0].max()) / 2)-110)
        y = int(((marker_corners[:, 1].min() + marker_corners[:, 1].max()) / 2)+80)

        src_point = np.array([[x,-y]], dtype=np.float32)


        dst_point = cv2.transform(src_point.reshape(-1, 1, 2), transformation_model).reshape(2)

        update_csv(dst_point[1], dst_point[0])

    if(counter == 50):
            sending_data()
            counter += 1
    counter +=1
    cv2.namedWindow("feed", cv2.WINDOW_NORMAL)
    cv2.imshow("feed", frame)
    # Exit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    # time.sleep(1)



# Release the camera and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()