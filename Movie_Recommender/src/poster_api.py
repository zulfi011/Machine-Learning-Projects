import requests
from .logger import get_logger

logger = get_logger(__name__)

class Api:
    def __init__(self,movie_ids,api_key):
        self.movie_ids = movie_ids
        self.api_key = api_key
        self.data_lst = []

    @staticmethod
    def get_movie_data(movie_ids,api_key):
        base_url = "https://api.themoviedb.org/3/movie/"
        json_data = []
        for id in movie_ids:
            url = f"{base_url}{id}?api_key={api_key}&append_to_response=credits"
            response = requests.get(url)
            data = response.json()
            json_data.append(data)
        return json_data
        
    def get_movie_info(self):
        logger.info('getting data from api')
        movie_data = Api.get_movie_data(self.movie_ids, self.api_key)
        for data in movie_data:
            genres = [genre.get("name") for genre in data.get("genres", [])]
            overview = data.get("overview")
            cast = [member.get("name") for member in data.get("credits", {}).get("cast", [])[:5]]
            # Extract director
            crew = data.get("credits", {}).get("crew", [])
            director = next((member.get("name") for member in crew if member.get("job") == "Director"), None)
            release_date = data.get("release_date")
            runtime = data.get("runtime")
            # Extract poster URL
            poster_path = data.get("poster_path")
            poster_url = f"https://image.tmdb.org/t/p/w500{poster_path}" if poster_path else None

            data_dict = {}
            data_dict.update({'genres':' - '.join(genres),
                              'overview':overview,
                              'cast':' - '.join(cast),
                              'director':director,
                              'release_date':release_date,
                              'runtime':runtime,
                              'poster_url':poster_url})
            self.data_lst.append(data_dict)
        logger.info('data sucessfully retrived')

    def get_all_data(self):
        self.get_movie_info()
        return self.data_lst