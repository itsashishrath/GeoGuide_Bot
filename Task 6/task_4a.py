'''
* Team Id: 1194
* Author List: Keshav Joshi, Ashish Rathore, Disha Chhabra.
* Filename: task_4a.py
* Theme: eYRC Geo Guide(GG)
* Functions: dijkstra, find_shortest_paths, path_giving, present_or_not, remove_green_border, classify_image, task_4a_return
* Global Variables: combat, rehab, military_vehicles, fire, destroyed_building
'''

####################### IMPORT MODULES #######################
import cv2
import numpy as np
from sys import platform
import numpy as np
import cv2 as cv       # OpenCV Library
import torch
from PIL import Image
from torchvision import datasets, transforms # transform data
import matplotlib.pyplot as plt
import torch.nn as nn  
import torchvision.models as models
from torchvision.transforms import InterpolationMode    
import networkx as nx
import heapq                         
##############################################################
# All the global variabels for the classes of the image for CNN
combat = "Combat"
rehab = "Humanitarian Aid and rehabilitation"
military_vehicles = "Military Vehicles"
fire = "Fire"
destroyed_building = "Destroyed buildings"
Blank = "Blank"

label_dict = {
    "Fire" : "fire",
    "Destroyed buildings" : "destroyed_buildings",
    "Combat" : "combat",
    "Humanitarian Aid and rehabilitation" : "humanitarian_aid",
    "Military Vehicles" : "military_vehicles",
    "Blank" : "blank",
}

'''
* Function Name: dijkstra
* Input: g (graph), src (source node), dst (destination node)
* Output: path (list of nodes), distance (total distance)
* Logic: Implements Dijkstra's algorithm to find the shortest path in a weighted graph.
* Example Call: path, distance = dijkstra(graph, source_node, destination_node)
'''
# This is the implementation of dijkstra algo here we give it start and end nodes and it calculate the shortes distace
def dijkstra(g, src, dst):
    distances = {node: float('infinity') for node in g.nodes}
    distances[src] = 0
    previous_nodes = {node: None for node in g.nodes}

    priority_queue = [(0, src)]

    while priority_queue:
        current_distance, current_node = heapq.heappop(priority_queue)

        if current_node == dst:
            path = []
            while previous_nodes[current_node] is not None:
                path.insert(0, current_node)
                current_node = previous_nodes[current_node]
            path.insert(0, src)
            return path, current_distance

        for neighbor, weight in g[current_node].items():
            distance = current_distance + weight['weight']
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                previous_nodes[neighbor] = current_node
                heapq.heappush(priority_queue, (distance, neighbor))

    return None, None  # If no path is found


'''
* Function Name: find_shortest_paths
* Input: graph (graph structure), box_locations (list of box coordinates)
* Output: all_shortest_paths (list of lists)
* Logic: Finds all possible shortest paths between given box locations in the arena.
* Example Call: paths = find_shortest_paths(G, box_locations)
'''

# Main logic for finding shortest paths between box locations here we calculate all the possbile node that can be used to reach the event
def find_shortest_paths(graph, box_locations):
    all_shortest_paths = []

    for i in range(len(box_locations) - 1):
        src_node_1, src_node_2 = box_locations[i]
        dst_node_1, dst_node_2 = box_locations[i + 1]

        # Find all possible paths from src_node_1 to dst_node_1 and dst_node_2
        path_1_1, distance_1_1 = dijkstra(graph, src_node_1, dst_node_1)

        path_1_2, distance_1_2 = dijkstra(graph, src_node_1, dst_node_2)

       # Find all possible paths from src_node_2 to dst_node_1 and dst_node_2
        path_2_1, distance_2_1 = dijkstra(graph, src_node_2, dst_node_1)

        path_2_2, distance_2_2 = dijkstra(graph, src_node_2, dst_node_2)

        # Compare the distances and store the shortest paths
        shortest_path = [path_1_1, path_1_2, path_2_1, path_2_2]
        distances=[distance_1_1,distance_1_2,distance_2_1,distance_2_2]
        shortest_path_index = distances.index(min(distances))

        all_shortest_paths.append(shortest_path[shortest_path_index])
        lenght = len(shortest_path[shortest_path_index]) - 1;
        path_current = shortest_path[shortest_path_index]
        if (path_current[lenght] == 0):
            continue
        elif (path_current[lenght] == 4):
            shortest_path[shortest_path_index].append(9)
        elif(path_current[lenght] == dst_node_1):
            shortest_path[shortest_path_index].append(dst_node_2)
        
        else:
            shortest_path[shortest_path_index].append(dst_node_1)

    return all_shortest_paths


'''
* Function Name: path_giving
* Input: locations (dictionary of event locations)
* Output: final (list of node sequence representing the path)
* Logic: Computes a path considering the priority of events in the arena and required traversal.
* Example Call: path = path_giving(locations_dict)
'''

# Here we store all the important path to reach the event and travers the arena
def path_giving(locations):
    G = nx.Graph()

    nodes = range(0, 12)
    G.add_nodes_from(nodes)
    # in edges we have (start node, end node, weight)
    edges = [(1, 2, 1), (2, 3, 1), (3, 4, 1), (8, 7, 1), (7, 6, 1), (6, 5, 1),
            (11, 10, 1), (10, 9, 1), (1, 8, 2), (2, 7, 2), (3, 6, 2), (4, 5, 2),
            (7, 11, 2), (6, 10, 2), (5, 9, 2), (4, 9, 5), (8, 11, 3), (0, 1, 1),(27,27,0)]

    G.add_weighted_edges_from(edges)
    # Here we give priority to all the event possible so that it travers them accordingly
    class_priorities = {
        "Start": 0,
        "Fire": 1,
        "Destroyed buildings": 2,
        "Humanitarian Aid and rehabilitation": 3,
        "Military Vehicles": 4,
        "Combat": 5,
        "Blank": 27
    }
    sorted_locations = sorted(locations.items(), key=lambda x: class_priorities[x[1]])

    # all the posible node to reach every particular events
    coordinates = {
        "S": (0, 0),
        "A": (1, 8),
        "B": (7, 11),
        "C": (6, 10),
        "D": (3, 6),
        "E": (4, 4)

    }

    # Creating an array of coordinates in priority order

    # box_locations = [coordinates[loc[0]] for loc in sorted_locations]
    box_locations = [coordinates[loc[0]] for loc in sorted_locations if loc[1] != "Blank"]
    box_locations.append((0,0))


    # Find and print all shortest paths
    result_paths = find_shortest_paths(G, box_locations)
    final = []
    # print(len(result_paths), "lkjasdlfja;lsdjf")
    for i, path in enumerate(result_paths):
        final.extend(path)
        if(i < (len(result_paths)-1)):
            final.append(23)

    # Preparing the path to 55 size so that it make a consistent size for our ino code
    while len(final) < 55:
        final.append(27)

    return final


'''
* Function Name: present_or_not
* Input: count (counter variable), image (input image)
* Output: 1 if present, 0 if not present
* Logic: Compares the input image with reference images to determine if the event is present.
* Example Call: result = present_or_not(counter, current_image)
'''
# This function will use the folder reference_img present in the main folder to get the reference images of the blank spot.
# This is the main code to detect weather the image is present or not on the particular event
def present_or_not(count, image):

    # We are using image compersion method so that we need to have reference images so that we can comper it with the new images to tell where it is blank or not.
    base_images = ["reference_img\processsss_images_0.jpg", "reference_img\processsss_images_1.jpg","reference_img\processsss_images_2.jpg", "reference_img\processsss_images_3.jpg", "reference_img\processsss_images_4.jpg"]

    # Here we rread the image and resize it to get the right comperison
    orignal = cv.imread(base_images[count])
    orignal = cv.resize(orignal, (255, 255))
    compare = cv.resize(image, (255, 255))

    # Here we use sift algorithm form open cv to mark all the points on the images and then we will compare which all the points are similar between to images.
    sift = cv.xfeatures2d.SIFT_create()
    kp_1, desc_1 = sift.detectAndCompute(orignal, None)
    kp_2, desc_2 = sift.detectAndCompute(compare, None)

    # Thoes similar points are then join togerather and we get certain joint poins
    index_params = dict(algorithm=0, trees=5)
    search_params = dict()
    flann = cv.FlannBasedMatcher(index_params, search_params)

    matches = flann.knnMatch(desc_1, desc_2, k=2)

    good_points = []
    ratio = 0.6
    for m, n in matches:
        if m.distance < ratio*n.distance:
            good_points.append(m)

    result = cv.drawMatches(orignal, kp_1, compare, kp_2, good_points, None)
 
    # According to those join points with the help of how many we decide if image is pressent or not
    if(len(good_points) < 15):
        return 1
    else:
        return 0
    
    return None


'''
* Function Name: remove_green_border
* Input: image (input image)
* Output: result (image with green border removed)
* Logic: Applies a color mask to remove green borders from the input image.
* Example Call: processed_image = remove_green_border(input_image)
'''

# This is for image prossesing
def remove_green_border(image):

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    lower_green = np.array([35, 100, 100])
    upper_green = np.array([85, 255, 255])

    mask = cv2.inRange(hsv, lower_green, upper_green)

    mask = cv2.bitwise_not(mask)

    result = cv2.bitwise_and(image, image, mask=mask)

    return result


'''
* Function Name: classify_image
* Input: cutout (cropped image)
* Output: event (detected event label)
* Logic: Uses a pre-trained(ShuffelNet V2) model to classify the content of the given cutout image.
* Example Call: event_label = classify_image(cropped_image)
'''
# This our main image classification algorithm function here we send the cutout image so getting its right class
def classify_image(cutout):
    # Using the same function used in previous taskes 
    image_rgb = cv.cvtColor(cutout, cv.COLOR_BGR2RGB)
    image_Plot = Image.fromarray(image_rgb)

    event = "variable to return the detected function"
    
    # esential transfomation of evaluating the image
    transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # Defining the same model so our saved weight can use them
    class Model(nn.Module):
    
        def __init__(self):
            super(Model, self).__init__()
            
            # Load shufflenet_v2_x2_0 with specified arguments
            self.base = models.shufflenet_v2_x2_0(pretrained=True, progress=True)
            
            # Modify the classifier part
            self.base.fc = nn.Sequential(
                nn.Linear(2048, 1024),
                nn.ReLU(),
                nn.BatchNorm1d(1024),
                nn.Dropout(0.5),
                nn.Linear(1024, 512),
                nn.ReLU(),
                nn.BatchNorm1d(512),
                nn.Dropout(0.5),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.BatchNorm1d(256),
                nn.Dropout(0.5),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.BatchNorm1d(128),
                nn.Dropout(0.5),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.BatchNorm1d(64),
                nn.Dropout(0.5),
                nn.Linear(64, 5)
            )
            
        def forward(self, x):
            x = self.base(x)
            return x
    model = Model() 
    model = torch.nn.DataParallel(model)  
    model.load_state_dict(torch.load('custom_cnn_model.pth'))
    model.eval()
    img = Image.fromarray(np.uint8(image_rgb))

    with torch.inference_mode():
        img = transform(img)
        img = img.unsqueeze(0)
        outputs = model(img)
     
        max_value, predicted = torch.max(outputs, dim=1)

    if predicted.item() == 0:
        event = combat
    elif predicted.item() == 1:
        event = destroyed_building
    elif predicted.item() == 2:
        event = fire
    elif predicted.item() == 3:
        event = rehab
    elif predicted.item() == 4:
        event = military_vehicles
    return event

##############################################################

'''
* Function Name: task_4a_return
* Output: identified_labels (dictionary containing event labels)
* Logic: Captures live feed from the camera, processes images, and classifies events in the arena.
* Example Call: identified_labels = task_4a_return()
'''
def task_4a_return():
   
    identified_labels = {}  
    path_labels = {"S": "Start"}

    # Caputring the live feed form the camera.
    cap  = cv2.VideoCapture(1, cv2.CAP_DSHOW)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, 1080)

    # Checking if the camera opened successfully
    if not cap.isOpened():
        print("Error: Could not open camera.")
        exit()
    # We got to know that first frame is not always good so we wait for some time to caputer the frame so that we can get the best image.
    for i in range(65):
        ret, frame = cap.read()

    if not ret:
        print("Error: Could not capture frame.")
        exit()

    # Appling filters so that we can detecct the conture
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, thresholded = cv2.threshold(gray, 185, 255, cv2.THRESH_BINARY)
    # cv2.namedWindow('thresholded', cv2.WINDOW_NORMAL)
    # cv2.imshow('thresholded', thresholded)

    contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Defing the required area of the boxes that we wanted to cutout
    min_area = 10800 
    max_area = 12000
    # These array are used to make sure that we follow the correct order while cutting the image.
    cutout_regions = []
    coordinates_array=[]

    square_contours = [cnt for cnt in contours if min_area < cv2.contourArea(cnt) < max_area]

    for i, cnt in enumerate(square_contours):
        area = cv2.contourArea(cnt)
        x, y, w, h = cv2.boundingRect(cnt)
        coordinates = (x, y, w, h)  # Storing coordinates as a tuple
        coordinates_array.append(coordinates)  # Appending the coordinates to the list


    # Sorting the coordinates according to their x,y coordinates and making sure the we get the correct order
    coordinates_array = np.array(coordinates_array)

    sorted_coordinates_array = coordinates_array[coordinates_array[:, 1].argsort()][::-1]

    # here we are comparing the coordinates mainly C and D as they are causing the main problem.
    if sorted_coordinates_array[2][0] < sorted_coordinates_array[3][0]:
        temp=sorted_coordinates_array[2].copy()
        sorted_coordinates_array[2]=sorted_coordinates_array[3]
        sorted_coordinates_array[3]=temp

    for i, coordinates in enumerate(sorted_coordinates_array):
        x, y, w, h = coordinates

        # Draw a rectangle
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Here we are cutting the image and feeding it to the network fuction and printing the correct detected lables in the idetified_label dict.
    counter = 1
    for coordinates in sorted_coordinates_array:
        x,y,w,h=coordinates
        cutout = frame[y:y + h, x:x + w]  # Create a cut-out region
        cutout_regions.append(cutout)
        cutout = remove_green_border(cutout)
        image_rgb = cv.cvtColor(cutout, cv.COLOR_BGR2RGB)
        present_fuction = present_or_not(counter-1, image_rgb)
        # present_fuction = 1
        # Classify the cutout image
        alpha_count = chr(ord('A') + counter - 1)
        # if image is present then we send to classify function
        if(present_fuction == 1):
            class_label = classify_image(cutout)
            identified_labels[alpha_count] = str(class_label)
            path_labels[alpha_count] = str(class_label)
        # ethir we put blank in it so that we can print blank on the image
        else:
            class_label = "Blank"
            path_labels[alpha_count] = str(class_label)
        text_l = label_dict[class_label]
        cv2.putText(frame, f"Class: {text_l}", (x, y - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 4)
        counter += 1

    # here the final path is saved to the path.txt folder so that our next python code can use it
    final_path = path_giving(path_labels)
    with open("path.txt", 'w') as f:
        f.writelines(str(final_path))
        f.close()

    cv2.namedWindow('Detected Objects', cv2.WINDOW_NORMAL)
    cv2.imshow('Detected Objects', frame)
    cap.release()


##################################################
    return identified_labels


    

###############	Main Function	#################
if __name__ == "__main__":
    identified_labels = task_4a_return()
    print("Identified Labels =", identified_labels)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
