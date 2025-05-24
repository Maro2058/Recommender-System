// script.js

// Get references to key DOM elements
const uidBox = document.getElementById("uidBox");         // Input field for User ID
const loadBtn = document.getElementById("loadBtn");       // Button to trigger data load
const ratingsContainer = document.getElementById("ratingsContainer"); // Div to show historic ratings
const recsContainer = document.getElementById("recsContainer");       // Div to show recommendations

// Initialize page with user 1 data
function initUser() {
    uidBox.value = 1;    // Set default User ID to 1
    loadUser();          // Fetch and display data for User 1
}

// Helper: fetch raw JSON text, sanitize NaN, parse into object
async function fetchJSON(url) {
    const resp = await fetch(url);           // HTTP GET
    const text = await resp.text();          // Read response as text
    const sanitized = text.replace(/\bNaN\b/g, 'null'); // Replace any NaN tokens
    return JSON.parse(sanitized);             // Parse into JS object
}

// Main function: load and render user data and recommendations
async function loadUser() {
    const uid = parseInt(uidBox.value);      // Parse input as integer
    if (!uid || uid < 1) {                   // Validate user ID
        alert("Please enter a valid User ID");
        return;
    }

    // 1) Load historic ratings
    ratingsContainer.innerHTML = "<p>Loading ratings...</p>"; // Show loading text
    try {
        const data1 = await fetchJSON(`/user/${uid}`);       // Fetch from /user/<uid>
        if (data1.count === 0) {                             // No ratings found
            ratingsContainer.innerHTML = "<p>No historic ratings for this user.</p>";
        } else {
            // Render each rated movie into a card
            ratingsContainer.innerHTML = data1.ratings.map(m => m.fetched_title ? `
                <div class="movie">
                  <img src="${m.poster_url}" alt="${m.fetched_title}">
                  <p>${m.fetched_title}<br>⭐ ${m.rating}</p>
                </div>
            ` : '').join('') || '<p>No valid rated movies to display.</p>';
        }
    } catch (e) {
        console.error("Error fetching user ratings:", e);
        ratingsContainer.innerHTML = "<p>Error loading ratings.</p>";
    }

    // 2) Load recommendations
    recsContainer.innerHTML = "<p>Loading recommendations...</p>"; // Show loading text
    try {
        const data2 = await fetchJSON(`/recs/${uid}`);        // Fetch from /recs/<uid>
        if (!Array.isArray(data2) || data2.length === 0) {  // No recs or invalid data
            recsContainer.innerHTML = "<p>No recommendations available.</p>";
        } else {
            // Render each recommended movie into a card
            recsContainer.innerHTML = data2.map(m => `
                <div class="movie">
                  <img src="${m.poster_url}" alt="${m.fetched_title}">
                  <p>${m.fetched_title}</p>
                </div>
            `).join('');
        }
    } catch (e) {
        console.error("Error fetching recommendations:", e);
        recsContainer.innerHTML = "<p>Error loading recommendations.</p>";
    }
}

// Attach event listeners
loadBtn.addEventListener('click', loadUser);          // Click on Load User
window.addEventListener('DOMContentLoaded', initUser); // On page ready, init user 1

// Debug: log top-20 recommendation scores for user 1 and 2
fetch('/recs_debug/1').then(r => r.json()).then(console.table);
fetch('/recs_debug/2').then(r => r.json()).then(console.table);
