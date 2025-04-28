import difflib
import os
import pandas as pd
import joblib
from .logger import get_logger

logger = get_logger(__name__)

class Recomend:
    def __init__(self,title,data_path,sim_score_path,n_tops):
        self.title = title
        self.data_path = data_path
        self.sim_score_path = sim_score_path
        self.data = None
        self.n_tops = n_tops
        self.movies_data = None
    
    def load_sim_score(self):
        logger.info('loading data')
        try:
            self.data = pd.read_csv(self.data_path)
            self.sim_score = joblib.load(self.sim_score_path)
            logger.info('loaded sucessfully')
        except Exception as e:
            logger.error('error while loading data')
            raise e

    def recomend_system(self):
        logger.info(f'recomending movies for {self.title}')
        self.title = self.title.lower()
        matches = difflib.get_close_matches(self.title,self.data['title'].str.lower(),n=3)
        if len(matches) != 0:
            idx = self.data[self.data['title'].str.lower()==matches[0]].index
            sim_score = list(enumerate(self.sim_score[idx[0]]))
            sim_score = sorted(sim_score,key=lambda x: x[1],reverse=True)
            movie_idx = sim_score[1:self.n_tops+1]
            movie_idx = [x[0] for x in movie_idx]
            movies = self.data.iloc[movie_idx]
            self.movies_data =  movies[['id','title','scoring']]
            logger.info(f'recomender system sucessfully got {self.movies_data.columns}')
        else:
            logger.warning('movie was not found')
            self.movies_data = None
        
    def recomend(self):
        logger.info('running recomend system')
        self.load_sim_score()
        self.recomend_system()
        logger.info('sucessfully runed recomend system')
        return self.movies_data