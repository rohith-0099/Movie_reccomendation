# Movie_reccomendation
# Movie Recommendation System

This is a simple Python project that recommends movies using two methods:
1. **Collaborative Filtering** – recommends movies based on what similar users have liked.
2. **Content-Based Filtering** – recommends movies based on their genre.

## How it Works

- The program loads two datasets:
  - One with movie details (like title and genre)
  - One with user ratings for those movies
- It creates a user-movie rating matrix for collaborative filtering.
- For content-based filtering, it uses movie genres and calculates similarity.

## Features

- You can choose between:
  - Collaborative Filtering
  - Content-Based Filtering
- You just enter a movie title, and it gives 5 similar movie suggestions.

## Requirements

Make sure you have these Python libraries installed:
```bash
pandas
numpy
scikit-learn
scipy
