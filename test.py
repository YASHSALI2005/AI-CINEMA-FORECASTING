from imdb import Cinemagoer

# Create an instance of Cinemagoer
ia = Cinemagoer()

movie_name = "Robot"

print(f"🕵️ Searching for '{movie_name}'...")

# 1. Search for the movie
movies = ia.search_movie(movie_name)

if movies:
    # Get the first result (most likely the correct one)
    movie = movies[0]
    ia.update(movie) # Download full details

    print(f"\n🎬 Title: {movie.get('title')}")
    print(f"📅 Year: {movie.get('year')}")
    print(f"🎭 Genres: {movie.get('genres')}")
    print(f"⭐ Rating: {movie.get('rating')}")
    
    # Get Director (Handle list safely)
    directors = [d['name'] for d in movie.get('directors', [])]
    print(f"🎬 Director: {directors}")
    
    # Get Cast (Top 3)
    cast = [c['name'] for c in movie.get('cast', [])[:3]]
    print(f"👥 Cast: {cast}")
    
else:
    print("❌ Movie not found.")