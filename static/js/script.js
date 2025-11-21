
document.addEventListener("DOMContentLoaded", () => {

    /* -------------------------
       SET DEFAULT DATE & TIME
    --------------------------*/
    const today = new Date();
    const dateInput = document.getElementById('date');
    const timeInput = document.getElementById('time');

    const yyyy = today.getFullYear();
    const mm = String(today.getMonth() + 1).padStart(2, '0');
    const dd = String(today.getDate()).padStart(2, '0');
    dateInput.value = `${yyyy}-${mm}-${dd}`;

    let nextHour = today.getHours() + 1;
    if (nextHour > 23) nextHour = 0;
    timeInput.value = `${String(nextHour).padStart(2, '0')}:00`;


    /* -------------------------
       MAP + AUTOCOMPLETE SETUP
    --------------------------*/
    let map, directionsService, directionsRenderer;
    let originAutocomplete, destinationAutocomplete;

    window.initMap = function () {
        map = new google.maps.Map(document.getElementById("map"), {
            center: { lat: 15.1449, lng: 120.5896 },
            zoom: 12,
            mapTypeControl: true,
            mapTypeControlOptions: {
                style: google.maps.MapTypeControlStyle.HORIZONTAL_BAR,
                position: google.maps.ControlPosition.LEFT_TOP,
            },
            styles: [
                { elementType: "geometry", stylers: [{ color: "#242f3e" }] },
                { elementType: "labels.text.stroke", stylers: [{ color: "#242f3e" }] },
                { elementType: "labels.text.fill", stylers: [{ color: "#746855" }] },
                {
                    featureType: "administrative.locality",
                    elementType: "labels.text.fill",
                    stylers: [{ color: "#d59563" }],
                },
                {
                    featureType: "poi",
                    elementType: "labels.text.fill",
                    stylers: [{ color: "#d59563" }],
                },
                {
                    featureType: "poi.park",
                    elementType: "geometry",
                    stylers: [{ color: "#263c3f" }],
                },
                {
                    featureType: "poi.park",
                    elementType: "labels.text.fill",
                    stylers: [{ color: "#6b9a76" }],
                },
                {
                    featureType: "road",
                    elementType: "geometry",
                    stylers: [{ color: "#38414e" }],
                },
                {
                    featureType: "road",
                    elementType: "geometry.stroke",
                    stylers: [{ color: "#212a37" }],
                },
                {
                    featureType: "road",
                    elementType: "labels.text.fill",
                    stylers: [{ color: "#9ca5b3" }],
                },
                {
                    featureType: "road.highway",
                    elementType: "geometry",
                    stylers: [{ color: "#746855" }],
                },
                {
                    featureType: "road.highway",
                    elementType: "geometry.stroke",
                    stylers: [{ color: "#1f2835" }],
                },
                {
                    featureType: "road.highway",
                    elementType: "labels.text.fill",
                    stylers: [{ color: "#f3d19c" }],
                },
                {
                    featureType: "transit",
                    elementType: "geometry",
                    stylers: [{ color: "#2f3948" }],
                },
                {
                    featureType: "transit.station",
                    elementType: "labels.text.fill",
                    stylers: [{ color: "#d59563" }],
                },
                {
                    featureType: "water",
                    elementType: "geometry",
                    stylers: [{ color: "#17263c" }],
                },
                {
                    featureType: "water",
                    elementType: "labels.text.fill",
                    stylers: [{ color: "#515c6d" }],
                },
                {
                    featureType: "water",
                    elementType: "labels.text.stroke",
                    stylers: [{ color: "#17263c" }],
                },
            ],
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

        initAutocomplete();
    };


    function initAutocomplete() {
        const originInput = document.getElementById("origin");
        const destinationInput = document.getElementById("destination");

        const options = {
            types: ["establishment"],
            componentRestrictions: { country: "ph" },
        };

        originAutocomplete = new google.maps.places.Autocomplete(originInput, options);
        destinationAutocomplete = new google.maps.places.Autocomplete(destinationInput, options);

        function onPlaceChange(autocomplete, input) {
            const place = autocomplete.getPlace();
            if (!place) return;

            if (place.name) input.value = place.name;
            else if (place.formatted_address) input.value = place.formatted_address;
            else if (place.geometry) {
                input.value =
                    place.geometry.location.lat() + ", " + place.geometry.location.lng();
            }
        }

        originAutocomplete.addListener("place_changed", () =>
            onPlaceChange(originAutocomplete, originInput)
        );
        destinationAutocomplete.addListener("place_changed", () =>
            onPlaceChange(destinationAutocomplete, destinationInput)
        );
    }


    /* -------------------------
       DRAW ROUTE
    --------------------------*/
    function drawRoute() {
        const origin = document.getElementById("origin").value;
        const destination = document.getElementById("destination").value;

        if (!origin || !destination) {
            alert("Please enter both origin and destination.");
            return;
        }

        directionsService.route(
            {
                origin,
                destination,
                travelMode: google.maps.TravelMode.DRIVING,
            },
            (result, status) => {
                if (status === "OK") {
                    directionsRenderer.setDirections(result);
                } else {
                    alert("Could not display directions: " + status);
                }
            }
        );
    }


    /* -------------------------
       SWAP ORIGIN & DESTINATION
    --------------------------*/
    document.querySelector(".swap-btn").addEventListener("click", () => {
        const origin = document.getElementById("origin");
        const destination = document.getElementById("destination");

        const temp = origin.value;
        origin.value = destination.value;
        destination.value = temp;
    });


    /* -------------------------
       SIDEBAR TOGGLE
    --------------------------*/
    document.querySelector(".logo-container").addEventListener("click", () => {
        document.querySelector(".sidebar").classList.toggle("expanded");
    });


    /* -------------------------
       FORCE DISABLE ENTER KEY (prevents reload)
    --------------------------*/
    document.addEventListener("keydown", (e) => {
        if (e.key === "Enter") e.preventDefault();
    });


    /* -------------------------
       SHOW RECOMMENDATIONS
    --------------------------*/
    async function showRecommendations() {
        drawRoute();

        const origin = document.getElementById("origin").value;
        const destination = document.getElementById("destination").value;
        const date = document.getElementById("date").value;
        const time = document.getElementById("time").value;

        const graphContainer = document.querySelector(".graph-container");
        const recoContainer = document.querySelector(".reco-container");
        const recoTimeContainer = document.getElementById("reco-time-placeholder");
        const travelTimeDisplay = document.getElementById("travel-time-display");

        // Loading UI
        graphContainer.innerHTML = '<p style="text-align:center;">Loading graph...</p>';
        recoContainer.innerHTML = '<p style="text-align:center;">Calculating travel tip...</p>';
        recoTimeContainer.innerHTML = '<p style="text-align:center;">Loading top 5 options...</p>';
        travelTimeDisplay.textContent = "Calculating travel time...";

        if (!origin || !destination || !date || !time) {
            alert("Please enter all fields.");
            graphContainer.innerHTML = '<p id="graph-placeholder">Input your route and time to see the hourly heat index forecast graph.</p>';
            recoContainer.innerHTML = '<p id="reco-placeholder">Departure recommendations will appear here.</p>';
            recoTimeContainer.innerHTML = '<p id="reco-time-placeholder">Departure recommendations time will appear here.</p>';
            travelTimeDisplay.textContent = "";
            return;
        }

        try {
            const response = await fetch("http://127.0.0.1:5000/recommendation", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ origin, destination, date, time }),
            });

            if (!response.ok) throw new Error("Server error");

            const data = await response.json();
            if (data.error) {
                graphContainer.innerHTML = `<p style="color:red;text-align:center;">${data.error}</p>`;
                recoContainer.innerHTML = `<p style="color:red;text-align:center;">${data.error}</p>`;
                recoTimeContainer.innerHTML = `<p style="color:red;text-align:center;">${data.error}</p>`;
                travelTimeDisplay.textContent = "Error fetching travel time.";
                return;
            }

            // --- GRAPH ---
            graphContainer.innerHTML = `<img src="${data.graph_image}" style="width:100%;height:auto;border-radius:12px;">`;

            // --- TRAVEL TIME DISPLAY ---
            travelTimeDisplay.innerHTML = `
                Estimated Travel Time: <strong>${data.travel_time}</strong><br>
                Heat Index at Arrival: <strong>${data.arrival_heat_index}</strong>
            `;

            // --- CONSOLIDATED TRAVEL TIP ---
            recoContainer.innerHTML = `
                <h3>Travel Tip</h3>
                <p>${data.consolidated_tip}</p>
            `;

            // --- TOP 5 OPTIONS (TIME + HEAT INDEX ONLY) ---
            if (data.top_5_options?.length > 0) {
                let optionsHTML = '<h3>Departure Time Recommendations</h3>';
                data.top_5_options.forEach((opt, index) => {
                    // Extract numeric heat index value for color coding
                    const heatValue = parseFloat(opt.heat_index);
                    let tempClass = 'temp-moderate';
                    let tempColor = '#ff914d';
                    
                    if (heatValue < 27) {
                        tempClass = 'temp-cool';
                        tempColor = '#2e86de';
                    } else if (heatValue >= 32) {
                        tempClass = 'temp-hot';
                        tempColor = '#ff4d4d';
                    }
                    
                    const isBest = index === 0;
                    const bestBadge = isBest ? '<span class="best-badge">BEST</span>' : '';
                    const boxClass = isBest ? 'route-box best-option' : 'route-box';
                    
                    optionsHTML += `
                        <div class="${boxClass}">
                            <div class="route-box-left">
                                <div class="option-label">${opt.Option}${bestBadge}</div>
                                <div class="time-display">
                                    <svg class="clock-icon" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                                        <circle cx="12" cy="12" r="10" stroke="currentColor" stroke-width="2"/>
                                        <path d="M12 6V12L16 14" stroke="currentColor" stroke-width="2" stroke-linecap="round"/>
                                    </svg>
                                    ${opt.departure_time}
                                </div>
                            </div>
                            <div class="route-box-right ${tempClass}" style="border-color: ${tempColor}; color: ${tempColor};">
                                <svg class="temp-icon" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg" style="color: ${tempColor};">
                                    <path d="M14 14.76V3.5C14 2.67 13.33 2 12.5 2C11.67 2 11 2.67 11 3.5V14.76C9.77 15.39 9 16.64 9 18C9 20.21 10.79 22 13 22C15.21 22 17 20.21 17 18C17 16.64 16.23 15.39 15 14.76H14Z" fill="currentColor"/>
                                    <path d="M12.5 4C12.78 4 13 4.22 13 4.5V14.5C13 14.78 12.78 15 12.5 15C12.22 15 12 14.78 12 14.5V4.5C12 4.22 12.22 4 12.5 4Z" fill="#1a1a1a"/>
                                </svg>
                                ${opt.heat_index}
                            </div>
                        </div>
                    `;
                });
                recoTimeContainer.innerHTML = optionsHTML;
            } else {
                recoTimeContainer.innerHTML = '<p>No valid departure times available.</p>';
            }

        } catch (err) {
            graphContainer.innerHTML = `<p style="color:red;text-align:center;">Failed to connect to server.</p>`;
            recoContainer.innerHTML = `<p style="color:red;text-align:center;">Could not load travel tip.</p>`;
            recoTimeContainer.innerHTML = `<p style="color:red;text-align:center;">Could not load top 5 options.</p>`;
            travelTimeDisplay.textContent = "Server error.";
        }
    }



    /* -------------------------
       MAIN BUTTON — SINGLE CLEAN CLICK HANDLER
    --------------------------*/
    const showGraphBtn = document.getElementById("showGraphBtn");
    const graph = document.querySelector(".graph-reco");

    showGraphBtn.addEventListener("click", (e) => {
        if (e.key === "Enter" && e.target.tagName === "INPUT") {
            e.preventDefault();
        }
        graph.style.display = "flex";
        showRecommendations();
    });

});
