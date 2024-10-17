from flask import Flask, request, jsonify, render_template
import numpy as np
from sklearn.cluster import KMeans
import random
import matplotlib.pyplot as plt

app = Flask(__name__, template_folder='template')

@app.route('/')
def helloworld():
    return render_template('index.html')

CorLat = []
CorLon = []
taxinum = 1
taxis = []

#functions for the kmean
# Generate random passenger locations (latitude, longitude)
def generate_passenger_locations(num_passengers):
    passengers = []
    for _ in range(num_passengers):
        lat = random.uniform(40.0, 41.0)  # Example latitude range
        lon = random.uniform(-74.0, -73.0)  # Example longitude range
        passengers.append([lat, lon])
    return np.array(passengers)

# Generate random taxi locations (latitude, longitude)
'''def generate_taxi_locations(num_taxis):
    taxis = []
    taxis.append([CorLat, CorLon]) #error here, dimension error
    print(taxis)
    return np.array(taxis)'''

# Assign passengers to taxis based on nearest taxi location
def assign_passengers_to_taxis(taxi_locations, passenger_locations):
    # We can use KMeans with taxi locations as initial centroids
    kmeans = KMeans(n_clusters=len(taxi_locations), init=taxi_locations, n_init=1)
    kmeans.fit(passenger_locations)
    return kmeans

# Number of passengers and taxis



# Route to receive the coordinates
@app.route('/submit_coordinates', methods=['POST', 'GET'])
def submit_coordinates():
    data = request.get_json()
    latitude = data['latitude']
    longitude = data['longitude']
    clickcount = data['clickCount']


    # Process the coordinates in Python
    # For example, just return them or do something more complex
    taxinum = clickcount
    taxis.append([latitude, longitude])  # error here, dimension error
    CorLat.append(latitude)
    CorLon.append(longitude)
    print(f"Received coordinates - Latitude: {CorLat[clickcount-1]}, Longitude: {CorLon[clickcount-1]}, Click Count: {clickcount}")


    # Return a success message
    return jsonify({"message": "Coordinates received successfully!"})
    #return jsonify({"message": latitude + " " + longitude})

@app.route('/start_function')
def start_function():
    num_passengers = 50


    # Generate random passenger and taxi locations
    passenger_locations = generate_passenger_locations(num_passengers)
    #taxi_locations = generate_taxi_locations(taxinum)
    taxi_locations = np.array(taxis)


    # Assign passengers to the nearest taxis
    kmeans = assign_passengers_to_taxis(taxi_locations, passenger_locations)

    # Visualization of passengers, taxis, and assignments
    plt.scatter(passenger_locations[:, 0], passenger_locations[:, 1], c=kmeans.labels_, s=50, cmap='rainbow',
                label='Passengers')
    plt.scatter(taxi_locations[:, 0], taxi_locations[:, 1], s=200, c='black', marker='X', label='Taxis')

    # Annotating taxi locations
    for i, (x, y) in enumerate(taxi_locations):
        plt.text(x, y, f'Taxi {i}', fontsize=9)

    for i, (x, y) in enumerate(passenger_locations):
        plt.text(x, y, f'P {i}', fontsize=5)

    plt.title('Passenger Assignment to Nearest Taxi Using K-Means')
    plt.xlabel('Latitude')
    plt.ylabel('Longitude')
    plt.legend()
    plt.show()

    # Print out assignments
    for i, label in enumerate(kmeans.labels_):
        return jsonify({"message": "Function started and executed!"})


if __name__ == '__main__':
    app.run(debug=True)
