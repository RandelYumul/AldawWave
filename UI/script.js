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


async function showRecommendations() {
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
