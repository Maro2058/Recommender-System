from flask import Flask, render_template, jsonify  # Imports Flask web framework functions
import pandas as pd                           # Data manipulation library
import numpy as np                            # Numerical computing library
from tensorflow.keras.models import load_model # Function to load a Keras model

app = Flask(__name__)  # Initialize Flask application

# === Load models and data ===
print("Loading models and data...")
# Load the pre-trained collaborative filtering model (HDF5 format)
collab_model = load_model("models/collaborative_model.keras", compile=False)
# Compile the model for inference using Adam optimizer and MSE loss (metrics not used here)
collab_model.compile(optimizer='adam', loss='mean_squared_error')

# Load movie metadata (posters, titles, overviews, etc.)
movies = pd.read_csv("data/movies_with_tmdb_data.csv")
# Replace missing posters with a default placeholder image path
movies['poster_url'] = movies['poster_url'].fillna("/static/images/placeholder.jpg")

# Load cleaned user ratings used for training (userId, movieId, rating)
ratings = pd.read_csv("data/ratings_clean.csv")



print("Models and data loaded.")  # Confirm data load complete

@app.route("/")
def index():
    """Serve the main HTML page with the React-like interface."""
    return render_template("index.html")  # Render the front-end

@app.route("/user/<int:uid>")
def user_profile(uid):
    """
    Return JSON containing a list of all movies the user has rated:
    - 'ratings': list of {movieId, rating, fetched_title, poster_url}
    - 'count': number of rated movies
    """
    # Filter ratings for the specified user ID
    user_r = ratings[ratings['userId'] == uid]
    if user_r.empty:
        # No historic ratings found
        return jsonify({"ratings": [], "count": 0})

    # Merge user ratings with movie metadata on 'movieId'
    merged = user_r.merge(movies, on='movieId', how='left')
    # Clean any missing fields by replacing empty strings or NaN
    merged['poster_url']    = merged['poster_url'].replace("", "/static/images/placeholder.jpg")
    merged['fetched_title'] = merged['fetched_title'].replace("", "Unknown Movie")
    # Select only the columns we need in the JSON output
    out = merged[['movieId', 'rating', 'fetched_title', 'poster_url']]
    # Return as JSON with record count
    return jsonify({"ratings": out.to_dict(orient='records'), "count": len(out)})

@app.route("/recs/<int:uid>")
def user_recs(uid):
    """
    Return top-10 collaborative filtering recommendations for the user:
    - Exclude movies the user has already rated
    - Batch-predict scores via the CF model
    - Return JSON list of {movieId, fetched_title, poster_url, overview}
    """
    # Identify movies the user has already seen/rated
    seen_ids = set(ratings[ratings['userId'] == uid]['movieId'])
    # Filter out seen movies
    unseen = movies[~movies['movieId'].isin(seen_ids)]
    if unseen.empty:
        return jsonify([])  # Nothing left to recommend

    # Drop any rows with missing movieId
    unseen = unseen[unseen['movieId'].notna()]
    # Convert movieId column to a list of ints
    movie_ids = unseen['movieId'].astype(int).tolist()
    if not movie_ids:
        return jsonify([])

    # Prepare arrays of user IDs and movie IDs for batch prediction
    u_arr = np.array([uid] * len(movie_ids)).reshape(-1, 1)
    i_arr = np.array(movie_ids).reshape(-1, 1)
    # Predict scores in one call for efficiency
    preds = collab_model.predict([u_arr, i_arr], verbose=0).flatten()

    # Select top-10 indices based on predicted score
    top_idx = np.argsort(preds)[::-1][:10]
    best = unseen.iloc[top_idx]  # Subset top recommendations

    # Build JSON-friendly payload with essential fields
    payload = best[['movieId', 'fetched_title', 'poster_url']]
    return jsonify(payload.to_dict(orient='records'))

@app.route("/recs_debug/<int:uid>")
def user_recs_debug(uid):
    """
    Debug endpoint: return top-20 (movieId, score) pairs for unseen movies,
    so we can inspect raw predicted scores.
    """
    # Prepare unseen list as above
    seen_ids = set(ratings[ratings['userId']==uid]['movieId'])
    unseen = movies[~movies['movieId'].isin(seen_ids)]
    movie_ids = unseen['movieId'].astype(int).tolist()
    # Create batch arrays
    u_arr = np.array([uid]*len(movie_ids)).reshape(-1,1)
    i_arr = np.array(movie_ids).reshape(-1,1)
    # Predict and flatten scores
    preds = collab_model.predict([u_arr, i_arr], verbose=0).flatten()

    # Build debug list of dicts
    debug_list = [
      {"movieId": mid, "score": float(score)}
      for mid, score in zip(movie_ids, preds)
    ]
    # Sort by descending score
    debug_list.sort(key=lambda x: x["score"], reverse=True)
    return jsonify(debug_list[:20])  # Return top-20 for inspection

# Run the app in debug mode when executed directly
if __name__ == "__main__":
    app.run(debug=True)
