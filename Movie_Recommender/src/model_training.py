from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os
import pandas as pd
import joblib
from .logger import get_logger

logger = get_logger(__name__)

class Model:
    def __init__(self, data_path):
        self.data_path = data_path
        self.data = None
        self.tfid_matrix = None
        self.sim_score = None
        self.save_path = os.path.join(os.path.dirname(__file__),'..','model','sim_score.pkl')

    def load_files(self):
        try:
            logger.info(f'loading data from {self.data_path}')
            self.data = pd.read_csv(self.data_path)
            logger.info(f'data loaded succesfully {self.data_path}')
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise

    def vecotrize_text(self):
        logger.info('vectorizig combined content')
        vectorizer = TfidfVectorizer()
        self.tfid_matrix = vectorizer.fit_transform(self.data['combined_content'])
        logger.info(f'tfid_matrix of shape {self.tfid_matrix.shape} created')
    
    def calculate_sim(self):
        logger.info('calculating similarity score...')
        self.sim_score = cosine_similarity(self.tfid_matrix)
        logger.info(f'calculated similarity of shape {self.sim_score.shape}')
    
    def save_sim_score(self):
        logger.info('saving similarity score as model...')
        try:
            os.makedirs(os.path.dirname(self.save_path),exist_ok=True)
            joblib.dump(self.sim_score,self.save_path)
            logger.info(f'saved similarity score at {self.save_path}')
        except Exception as e:
            logger.error(f'error saving data {e}')
            raise

    def run_pipeline(self):
        logger.info('started training model')
        self.load_files()
        self.vecotrize_text()
        self.calculate_sim()
        self.save_sim_score()
        logger.info('model training completed')