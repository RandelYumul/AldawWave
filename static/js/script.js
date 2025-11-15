(function preventReload() {

    // Disable reload shortcuts
    window.addEventListener("keydown", function (e) {
        if (e.key === "F5") e.preventDefault();                 // Block F5
        if ((e.ctrlKey || e.metaKey) && e.key === "r") e.preventDefault();   // Block Ctrl+R / Cmd+R
    });

    // Block navigation attempts silently
    window.addEventListener("beforeunload", function (e) {
        // Do NOT set returnValue → no popup will show
        e.preventDefault();
    });

    // Block programmatic reloads
    const originalReload = window.location.reload;
    window.location.reload = function () {
        console.warn("Reload blocked.");
    };

})();


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
            graphContainer.innerHTML = `<img src="${data.graph_image}" style="width:100%;height:auto;">`;

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
                data.top_5_options.forEach(opt => {
                    optionsHTML += `
                        <div class="route-box">
                            <strong>${opt.Option}</strong>: <strong>${opt.departure_time}</strong> 
                            (Heat Index: ${opt.heat_index})
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
