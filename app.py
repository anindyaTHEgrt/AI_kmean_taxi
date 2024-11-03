from flask import Flask, request, jsonify, render_template
import numpy as np
from sklearn.cluster import KMeans
import random
import matplotlib.pyplot as plt
import matplotlib.image as image

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
        lon = random.uniform(40.0, 41.0)  # Example latitude range
        lat = random.uniform(-74.0, -73.0)  # Example longitude range
        passengers.append([lat, lon])
    return np.array(passengers)

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

    im = image.imread('assets/barca_map.png')
    fig, ax = plt.subplots()


    # Generate random passenger and taxi locations
    passenger_locations = generate_passenger_locations(num_passengers)
    #taxi_locations = generate_taxi_locations(taxinum)
    taxi_locations = np.array(taxis)


    # Assign passengers to the nearest taxis
    kmeans = assign_passengers_to_taxis(taxi_locations, passenger_locations)

    # Visualization of passengers, taxis, and assignments
    ax.scatter(passenger_locations[:, 0], passenger_locations[:, 1], c=kmeans.labels_, s=50, cmap='rainbow',
                label='Passengers', zorder=2)
    ax.scatter(taxi_locations[:, 0], taxi_locations[:, 1], s=200, c='black', marker='X', label='Taxis', zorder=2)

    # Annotating taxi locations
    for i, (x, y) in enumerate(taxi_locations):
        plt.text(x, y, f'Taxi {i}', fontsize=9)

    for i, (x, y) in enumerate(passenger_locations):
        plt.text(x, y, f'P {i}', fontsize=5)

    fig.figimage(im, 0, 0, zorder=1, alpha=.3)

    plt.title('Passenger Assignment to Nearest Taxi Using K-Means')
    plt.xlabel('Latitude', zorder=2)
    plt.ylabel('Longitude', zorder=2)
    plt.legend()
    plt.show()

    # Print out assignments
    for i, label in enumerate(kmeans.labels_):
        return jsonify({"message": "Function started and executed!"})


if __name__ == '__main__':
    app.run(debug=True)
