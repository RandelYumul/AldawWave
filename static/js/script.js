// Function to ensure date and time inputs have default values
window.onload = function() {
    const today = new Date();
    const dateInput = document.getElementById('date');
    const timeInput = document.getElementById('time');

    // Set date to today's date (YYYY-MM-DD format)
    const yyyy = today.getFullYear();
    const mm = String(today.getMonth() + 1).padStart(2, '0'); // Months start at 0
    const dd = String(today.getDate()).padStart(2, '0');
    dateInput.value = `${yyyy}-${mm}-${dd}`;

    // Set time to the next hour (HH:MM format)
    let nextHour = today.getHours() + 1;
    if (nextHour > 23) nextHour = 0;
    const nextHourStr = String(nextHour).padStart(2, '0');
    timeInput.value = `${nextHourStr}:00`;
};

let map, directionsService, directionsRenderer, originAutocomplete, destinationAutocomplete;

function initMap() {
    // Initialize the map centered on the Philippines
    map = new google.maps.Map(document.getElementById("map"), {
        center: { lat: 15.1449, lng: 120.5896 }, // Angeles City
        zoom: 12,
    });

    directionsService = new google.maps.DirectionsService();
    directionsRenderer = new google.maps.DirectionsRenderer({
        map: map,
        suppressMarkers: false,
        polylineOptions: {
            strokeColor: "red",
            strokeOpacity: 0.8,
            strokeWeight: 6,
        },
    });

    // Setup autocomplete for inputs
    const originInput = document.getElementById('origin');
    const destinationInput = document.getElementById('destination');

    const options = { types: ['geocode'], componentRestrictions: { country: 'ph' } };

    originAutocomplete = new google.maps.places.Autocomplete(originInput, options);
    destinationAutocomplete = new google.maps.places.Autocomplete(destinationInput, options);
}

// Draw route between origin and destination
function drawRoute() {
    const origin = document.getElementById("origin").value;
    const destination = document.getElementById("destination").value;

    if (!origin || !destination) {
        alert("Please enter both origin and destination.");
        return;
    }

    const request = {
        origin: origin,
        destination: destination,
        travelMode: google.maps.TravelMode.DRIVING
    };

    directionsService.route(request, (result, status) => {
        if (status === "OK") {
            directionsRenderer.setDirections(result);
        } else {
            alert("Could not display directions due to: " + status);
        }
    });
}

function initAutocomplete() {
    // 1. Get the input elements by their IDs
    const originInput = document.getElementById('origin');
    const destinationInput = document.getElementById('destination');

    // Check if the Google Maps API loaded correctly
    if (typeof google === 'undefined' || !google.maps.places) {
        console.error("Google Maps Places library is not loaded. Check your API key and script tag.");
        return;
    }

    // 2. Create the Autocomplete objects
    const autocompleteOptions = {
        // Restrict results to locations that can be geocoded (addresses, cities, etc.)
        types: ['geocode', 'establishment'],
        // Optional: uncomment to restrict search to a specific country (e.g., Philippines)
        // componentRestrictions: { country: 'ph' } 
    };

    const originAutocomplete = new google.maps.places.Autocomplete(originInput, autocompleteOptions);
    const destinationAutocomplete = new google.maps.places.Autocomplete(destinationInput, autocompleteOptions);

    // 3. Optional: Listen for the 'place_changed' event
    // This ensures your backend always receives the standardized full address.
    originAutocomplete.addListener('place_changed', () => {
        const place = originAutocomplete.getPlace();
        // The input value is automatically updated with place.name or formatted_address
        if (place.formatted_address) {
             originInput.value = place.formatted_address;
             console.log('Origin Selected:', place.formatted_address);
        }
    });

    destinationAutocomplete.addListener('place_changed', () => {
        const place = destinationAutocomplete.getPlace();
        if (place.formatted_address) {
            destinationInput.value = place.formatted_address;
            console.log('Destination Selected:', place.formatted_address);
        }
    });
}

async function showRecommendations() {
    drawRoute();
    const origin = document.getElementById("origin").value;
    const destination = document.getElementById("destination").value;
    const date = document.getElementById("date").value;
    const time = document.getElementById("time").value;

    const graphContainer = document.querySelector(".graph-container");
    const recoContainer = document.querySelector(".reco-container");
    const travelTimeDisplay = document.getElementById("travel-time-display");

    // Clear previous results and show loading
    graphContainer.innerHTML = '<p style="text-align: center;">Loading graph...</p>';
    recoContainer.innerHTML = '<p style="text-align: center;">Calculating recommendations...</p>';
    travelTimeDisplay.textContent = 'Calculating travel time...';
    
    if (!origin || !destination || !date || !time) {
        alert("Please enter origin, destination, date, and preferred time.");
        graphContainer.innerHTML = '<p id="graph-placeholder">Input your route and time to see the hourly heat index forecast graph.</p>';
        recoContainer.innerHTML = '<p id="reco-placeholder">Departure recommendations and travel tips will appear here.</p>';
        travelTimeDisplay.textContent = '';
        return;
    }

    try {
        const response = await fetch('http://127.0.0.1:5000/recommendation', { // Assuming Flask runs on 5000
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                origin: origin,
                destination: destination,
                date: date,
                time: time
            })
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const data = await response.json();

        if (data.error) {
            graphContainer.innerHTML = `<p style="color: red; text-align: center;">Error: ${data.error}</p>`;
            recoContainer.innerHTML = `<p style="color: red; text-align: center;">Error: ${data.error}</p>`;
            travelTimeDisplay.textContent = `Error getting travel time.`;
            return;
        }

        // --- Render Graph ---
        graphContainer.innerHTML = `
            <img src="${data.graph_image}" alt="Heat Index Forecast Graph" style="width: 100%; height: auto;">
        `;

        // --- Render Recommendations ---
        recoContainer.innerHTML = `<h3>Departure Recommendations</h3>`;
        travelTimeDisplay.textContent = `Estimated Travel Time: ${data.travel_time}`;

        if (data.recommendations && data.recommendations.length > 0) {
            data.recommendations.forEach((rec, i) => {
                const div = document.createElement("div");
                div.className = "route-box"; // Assuming this class is in your styles.css
                
                div.innerHTML = `
                    <strong>Option ${i + 1}</strong>: Leave at <strong>${rec.departure_time}</strong> (Heat Index: ${rec.heat_index})<br>
                    <p style="margin-top: 5px; font-style: italic;">${rec.recommendation}</p>
                `;
                recoContainer.appendChild(div);
            });
        } else {
             recoContainer.innerHTML += '<p>No valid departure times could be suggested for the selected arrival time and travel duration.</p>';
        }

    } catch (error) {
        console.error('Error fetching recommendations:', error);
        graphContainer.innerHTML = `<p style="color: red; text-align: center;">Failed to connect to the recommendation server. Please check if 'aldaw_wave_api.py' is running.</p>`;
        recoContainer.innerHTML = `<p style="color: red; text-align: center;">Failed to get recommendations.</p>`;
        travelTimeDisplay.textContent = 'Server connection error.';
    }
}

function swapInputs() {
    const origin = document.getElementById("origin");
    const destination = document.getElementById("destination");

    const temp = origin.value;
    origin.value = destination.value;
    destination.value = temp;
}

function toggleSidebar() {
    const sidebar = document.querySelector('.sidebar');
    sidebar.classList.toggle('expanded');
}

const btn = document.getElementById('showGraphBtn');
const graph = document.querySelector('.graph-reco');

btn.addEventListener('click', () => {
  graph.style.display = 'flex';
});
