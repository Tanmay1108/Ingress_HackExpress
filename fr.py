import face_recognition

# Select an image to teach to the machine how to recognize

# * ---------- User 1 ---------- *
# Load the image 
user_one_face = face_recognition.load_image_file("images/Shreya.jpg")
# Encode the face parametres
user_one_face_encoding = face_recognition.face_encodings(user_one_face)[0]

# * ---------- User 2 ---------- *
# Load the image 
user_two_face = face_recognition.load_image_file("images/Tanmay.jpg")
# Encode the face parametres
user_two_face_encoding = face_recognition.face_encodings(user_two_face)[0]


# Create a list of known face encodings and their names
known_face_encodings = [
    user_one_face_encoding,
    user_two_face_encoding
]

# Create list of the name matching with the position of the known_face_encodings
known_face_names = [
    "User One",
    "User Two"
]




# Import the library
import face_recognition
import os
import re

# Declare all the list
known_face_encodings = []
known_face_names = []
known_faces_filenames = []

# Walk in the folder to add every file name to known_faces_filenames
for (dirpath, dirnames, filenames) in os.walk('images/'):
    known_faces_filenames.extend(filenames)
    break

# Walk in the folder
for filename in known_faces_filenames:
    # Load each file
    face = face_recognition.load_image_file('images/' + filename)
    # Extract the name of each employee and add it to known_face_names
    known_face_names.append(re.sub("[0-9]",'', filename[:-4]))
    # Encode de face of every employee
    known_face_encodings.append(face_recognition.face_encodings(face)[0])
    
    
    
# * --------- IMPORTS --------- *
import numpy as np
import face_recognition
import time

# * ---------- Encode the nameless picture --------- *
# Load picture
face_picture = face_recognition.load_image_file("test/t.jpg")
# Detect faces
face_locations = face_recognition.face_locations(face_picture)
# Encore faces
face_encodings = face_recognition.face_encodings(face_picture, face_locations)

hour = f'{time.localtime().tm_hour}:{time.localtime().tm_min}'
                # Save the date
                #json_to_export[
                 #   'date'] = f'{time.localtime().tm_year}-{time.localtime().tm_mon}-{time.localtime().tm_mday}'
date = f'{time.localtime().tm_year}-{time.localtime().tm_mon}-{time.localtime().tm_mday}'
#picture_array = frame.tolist()

# Loop in all detected faces
for face_encoding in face_encodings:
    # See if the face is a match for the known face (that we saved in the precedent step)
    matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
    # name that we will give if the employee is not in the system
    name = "Unknown"
    # check the known face with the smallest distance to the new face
    face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
    # Take the best one
    best_match_index = np.argmin(face_distances)
    # if we have a match:
    if matches[best_match_index]:
        # Give the detected face the name of the employee that match
        name = known_face_names[best_match_index]
        
        
# * --------- IMPORTS ---------*
import cv2
import requests
import time
import matplotlib.pyplot as plt

# Select the webcam of the computer (0 by default for laptop)
video_capture = cv2.VideoCapture(0)
im = 0
# Aplly it until you stop the file's execution
while True:
    # Take every frame
    process_this_frame,frame = video_capture.read()
    process_this_frame = True
    # Process every frame only one time
    if process_this_frame:
        # Find all the faces and face encodings in the current frame of video
        face_locations = face_recognition.face_locations(frame)
        face_encodings = face_recognition.face_encodings(frame, face_locations)
        # Initialize an array for the name of the detected users
        face_names = []

        # * ---------- Initialyse JSON to EXPORT --------- *
        json_to_export = {}
        # Loop in every faces detected
        for face_encoding in face_encodings:
            # See if the face is a match for the known face(s)
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"
            # check the known face with the smallest distance to the new face
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            # Take the best one
            best_match_index = np.argmin(face_distances)
            # If we have a match
            if matches[best_match_index]:
                # Save the name of the best match
                name1 = known_face_names[best_match_index]

                # * ---------- SAVE data to send to the API -------- *
                # Save the name
                #json_to_export['name'] = name
                # Save the time
                #json_to_export['hour'] = f'{time.localtime().tm_hour}:{time.localtime().tm_min}'
                hour = f'{time.localtime().tm_hour}:{time.localtime().tm_min}'
                # Save the date
                #json_to_export[
                 #   'date'] = f'{time.localtime().tm_year}-{time.localtime().tm_mon}-{time.localtime().tm_mday}'
                date = f'{time.localtime().tm_year}-{time.localtime().tm_mon}-{time.localtime().tm_mday}'
                picture_array = frame.tolist()
                # If you need to save a screenshot:
                #json_to_export['picture_array'] = frame.tolist()

                # * ---------- SEND data to API --------- *
                # Make a POST request to the API
                #r = requests.post(url='http://127.0.0.1:5000/receive_data', json=json_to_export)
                # Print to status of the request:
                #print("Status: ", r.status_code)

        # Store the name in an array to display it later
        face_names.append(name1)
        # To be sure that we process every frame only one time
        process_this_frame = not process_this_frame
        
        if(len(face_names) != 0):

            # * --------- Display the results ---------- *
            for (top, right, bottom, left), name in zip(face_locations, face_names):
                # Draw a box around the face
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
                # Define the font of the name
                font = cv2.FONT_HERSHEY_DUPLEX
                # Display the name
                cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
                
                cv2.putText(frame, f'{time.localtime().tm_hour}:{time.localtime().tm_min}', (left + 6, bottom - 20), font, 1.0, (255, 255, 255), 1)
                
                cv2.putText(frame, f'{time.localtime().tm_year}-{time.localtime().tm_mon}-{time.localtime().tm_mday}', (left + 6, bottom - 40), font, 1.0, (255, 255, 255), 1)
    
            # Display the resulting image
            #cv2.imshow('Video', frame)
            im = frame
            plt.imshow(frame)
            plt.show()
            #video_capture.release()

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows() 









#lets create the app

# * --------- IMPORTS --------- *
# All the imports that we will need in our API
from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin
import os
import psycopg2
import cv2
import numpy as np
import re

# We define the path of the current file, we will use it later
#FILE_PATH = os.path.dirname(os.path.realpath("D:\AI\metro_Project\face_recognition"))
FILE_PATH = os.path.dirname("D:\AI\metro_Project\face_recognition\flask")


# * ---------- Create App --------- *
# Init the app
app = Flask("app")
# To avoid cors erros
CORS(app, support_credentials=True)


# * -------------------- Run Server -------------------- *
if __name__ == '__main__':
    # * --- DEBUG MODE: --- *
    app.run(host='127.0.0.1', port=5000, debug=False)
    # * --------------------  ROUTES ------------------- *
# * ---------- Get data from the face recognition ---------- *
    @app.route('/receive_data', methods=['POST'])
    def get_receive_data():
        if request.method == 'POST':
            # Get the data
            json_data = request.get_json()
    
            # Check if the user is already in the DB
            try:
                # Connect to the DB
                connection = psycopg2.connect(user="USER_NAME",
                                              password="PASSWORD",
                                              host="DB_HOST",
                                              port="PORT",
                                              database="DATABBASE_NAME")
                # Open a cursor
                cursor = connection.cursor()
    
                # Query to check if the user as been saw by the camera today
                is_user_is_there_today =\
                    f"SELECT * FROM users WHERE date = '{json_data['date']}' AND name = '{json_data['name']}'"
    
                cursor.execute(is_user_is_there_today)
                # Store the result
                result = cursor.fetchall()
                # Send the request
                connection.commit()
    
                # If use is already in the DB for today:
                if result:
                    # Update user in the DB
                    update_user_querry = f"UPDATE users SET departure_time = '{json_data['hour']}', departure_picture = '{json_data['picture_path']}' WHERE name = '{json_data['name']}' AND date = '{json_data['date']}'"
                    cursor.execute(update_user_querry)
    
                else:
                    # Create a new row for the user today:
                    insert_user_querry = f"INSERT INTO users (name, date, arrival_time, arrival_picture) VALUES ('{json_data['name']}', '{json_data['date']}', '{json_data['hour']}', '{json_data['picture_path']}')"
                    cursor.execute(insert_user_querry)
    
            except (Exception, psycopg2.DatabaseError) as error:
                print("ERROR DB: ", error)
            finally:
                # Execute query
                connection.commit()
    
                # closing database connection.
                if connection:
                    cursor.close()
                    connection.close()
                    print("PostgreSQL connection is closed")
    
            # Return user's data to the front
            return jsonify(json_data)
        
            # * ---------- Get all the data of an employee ---------- *
    @app.route('/get_employee/<string:name>', methods=['GET'])
    def get_employee(name):
        answer_to_send = {}
        # Check if the user is already in the DB
        try:
            # Connect to DB
            connection = psycopg2.connect(user="USER",
                                          password="PASSWORD",
                                          host="DB_HOST",
                                          port="PORT",
                                          database="DATABASE_NAME")
    
            cursor = connection.cursor()
            # Query the DB to get all the data of a user:
            user_information = f"SELECT * FROM users WHERE name = '{name}'"
    
            cursor.execute(user_information)
            result = cursor.fetchall()
            connection.commit()
    
            # if the user exist in the db:
            if result:
                print('RESULT: ',result)
                # Structure the data and put the dates in string for the front
                for k,v in enumerate(result):
                    answer_to_send[k] = {}
                    for ko,vo in enumerate(result[k]):
                        answer_to_send[k][ko] = str(vo)
                print('answer_to_send: ', answer_to_send)
            else:
                answer_to_send = {'error': 'User not found...'}
    
        except (Exception, psycopg2.DatabaseError) as error:
            print("ERROR DB: ", error)
        finally:
            # closing database connection:
            if (connection):
                cursor.close()
                connection.close()
    
        # Return the user's data to the front
        return jsonify(answer_to_send)
    
    # * --------- Get the 5 last users seen by the camera --------- *
    @app.route('/get_5_last_entries', methods=['GET'])
    def get_5_last_entries():
        # Create a dict thet will contain the answer to give to the front
        answer_to_send = {}
        # Check if the user is already in the DB
        try:
            # Connect to DB
            connection = psycopg2.connect(user="USER_NAME",
                                          password="PASSWORD",
                                          host="HOST_NAME",
                                          port="PORT",
                                          database="DATABASE_NAME")
    
            cursor = connection.cursor()
            # Query the DB to get the 5 last entries ordered by ID:
            lasts_entries = f"SELECT * FROM users ORDER BY id DESC LIMIT 5;"
            cursor.execute(lasts_entries)
            # Store the result
            result = cursor.fetchall()
            # Send the request
            connection.commit()
    
            # if DB is not empty:
            if result:
                # Structure the data and put the dates in dict for the front
                for k, v in enumerate(result):
                    answer_to_send[k] = {}
                    for ko, vo in enumerate(result[k]):
                        answer_to_send[k][ko] = str(vo)
            else:
                answer_to_send = {'error': 'DB is not connected or empty'}
    
        except (Exception, psycopg2.DatabaseError) as error:
            print("ERROR DB: ", error)
        finally:
            # closing database connection:
            if (connection):
                cursor.close()
                connection.close()
    
        # Return the user's data to the front as a json
        return jsonify(answer_to_send)
    
    # * ---------- Add new employee ---------- *
    @app.route('/add_employee', methods=['POST'])
    @cross_origin(supports_credentials=True)
    def add_employee():
        try:
            # Get the picture from the request
            image_file = request.files['image']
    
            # Store it in the folder of the know faces:
            file_path = os.path.join(f"assets/img/users/{request.form['nameOfEmployee']}.jpg")
            image_file.save(file_path)
            answer = 'new employee succesfully added'
        except:
            answer = 'Error while adding new employee. Please try later...'
        return jsonify(answer)
    
    
    # * ---------- Get employee list ---------- *
    @app.route('/get_employee_list', methods=['GET'])
    def get_employee_list():
        # Create a dict that will store the list of employee's name
        employee_list = {}
    
        # Walk in the user's folder to get the user list
        walk_count = 0
        for file_name in os.listdir(f"{FILE_PATH}/assets/img/users/"):
            # Capture the employee's name with the file's name
            name = re.findall("(.*)\.jpg", file_name)
            if name:
                employee_list[walk_count] = name[0]
            walk_count += 1
    
        return jsonify(employee_list)
    
    
    # * ---------- Delete employee ---------- *
    @app.route('/delete_employee/<string:name>', methods=['GET'])
    def delete_employee(name):
        try:
            # Select the path
            file_path = os.path.join(f'assets/img/users/{name}.jpg')
             # Remove the picture of the employee from the user's folder:
            os.remove(file_path)
            answer = 'Employee succesfully removed'
        except:
            answer = 'Error while deleting new employee. Please try later'
    
        return jsonify(answer)
    
    
#react code
// Define a state the get the list of the employee's data
const [employeeList, setEmployeeList] = useState([]);
// Define a state to get the error if there is
const [errorMessage, setErrorMessage] = useState(null);


// Function to send the employee's name (value of an input fiel) and get back his data
const searchForEmployee = () => {
    // Value of the employee's name input
    const name = document.getElementById('searchForEmployee').value.toLowerCase()
    if(name){
        fetch(`http://127.0.0.1:5000/get_employee/${name}`)
        .then(response => response.json())
        .then(response => {
            if(response){
                // Set employeeList state with the response as a json
                setEmployeeList(response)
            } else {
               // Set errorMessage state with the response as a json 
              setErrorMessage(response.Error)
            }
        })
    }
    else{
       setEmployeeList(['No name find...'])
    }
}
    
// Define a state to store the 5 last entries
const [employeeList, setEmployeeList] = useState([]);

// Make the request to the API and get the 5 last entries as a json
const searchForLastEntries = () => {
    fetch('http://127.0.0.1:5000/get_5_last_entries')
    .then(response => response.json())
    .then(response => {
        if(response) {
            // Set the value of the employeeList state with the response
            setEmployeeList(response)
        }
    })
}
    
    
// Create a state to check if the user as been added
const [isUserWellAdded, setIsUserWellAdded] = useState(false);
// Create a state to check if the is error while the user's adding
const [errorWhileAddingUser, seterrorWhileAddingUser] = useState(false);

const addEmployeeToDb = e => {
        e.preventDefault()
        // Send it to backend -> add_employee as a POST request
        let name = document.getElementById("nameOfEmployee").value
        let picture = document.getElementById('employeePictureToSend')

        let formData  = new FormData();

        formData.append("nameOfEmployee", name)
        formData.append("image", picture.files[0])

        fetch('http://127.0.0.1:5000/add_employee',{
            method: 'POST',
            body:  formData,
        })
            .then(reposonse => reposonse.json())
            .then(response => {
                console.log(response)
                setIsUserWellAdded(true)
            })
            .catch(error => seterrorWhileAddingUser(true))
    }
            
// Create a state to get the list of all the employee's list
const [nameList, setNameList] = useState({});

// Get the list of all the employee's in the folder
const getEmployeeList = () => {
    fetch('http://127.0.0.1:5000/get_employee_list')
        .then(response => response.json())
        .then (response => {
            if(!isEmployeeListLoaded){
                setNameList(response)
                setIsEmployeeListLoaded(true)
            }
        })
}

// A Component to have a button that delete the employye:
const EmployeeItem = props => {
    // Function that send the employee's name to delete
    const deleteEmployee = name => {
        fetch(`http://127.0.0.1:5000/delete_employee/${name}`)
            .then(response => response.json())
            .then(() => setIsEmployeeListLoaded(false))
    }
    return(
        <li> { props.name } <ItemButton onClick={ () => deleteEmployee(props.name) }>DELETE</ItemButton></li>
    )
}
    

        
        
    