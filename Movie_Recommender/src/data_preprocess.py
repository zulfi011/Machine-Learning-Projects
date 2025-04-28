import pandas as pd
import re
import os
import nltk
from nltk.corpus import stopwords
from .logger import get_logger

# Download necessary resources
nltk.download('stopwords')
nltk.download('punkt')

logger = get_logger(__name__)

class DataPreprocessor:
    def __init__(self, data_path):
        self.data_path = data_path
        self.data = None
        self.content = None
        self.final_data = None
        self.save_path = os.path.join(os.path.dirname(__file__),'..', 'data', 'preprocessed.csv')

    def load_data(self):
        logger.info(f'Loading data from {self.data_path}...')
        try:
            self.data = pd.read_csv(self.data_path)
            self.content = self.data[['id', 'genres', 'keywords', 'overview', 'cast', 'director', 'title', 'vote_average', 'vote_count']].copy()
            logger.info('Data loaded successfully.')
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise

    def fill_missing_values(self):
        if self.content.isna().sum().sum() > 0:
            logger.warning('Missing values detected. Filling missing values...')
            cols_to_fill = ['genres', 'keywords', 'overview', 'cast', 'director']
            self.content[cols_to_fill] = self.content[cols_to_fill].fillna('')
            logger.info('Missing values filled.')

    def combine_content(self):
        self.content['combined_content'] = (
            self.content['genres'] + ' ' +
            self.content['keywords'] + ' ' +
            self.content['overview'] + ' ' +
            self.content['cast'] + ' ' +
            self.content['director']
        )
        logger.info('Content combined for preprocessing')

    @staticmethod
    def preprocess_text(text):
        cleaned_text = re.sub(r"[^a-zA-Z]", " ", text)
        tokens = nltk.word_tokenize(cleaned_text)
        stop_words = set(stopwords.words('english'))
        filtered_tokens = [token.lower() for token in tokens if token.lower() not in stop_words]
        return ' '.join(filtered_tokens)

    def preprocess_content(self):
        self.content['combined_content'] = self.content['combined_content'].apply(self.preprocess_text)
        logger.info('Preprocessed combined content')

    def calculate_scores(self):
        C = self.content['vote_average'].mean()
        m = self.content['vote_count'].quantile(0.9)

        def weighted_score(x):
            v = x['vote_count']
            R = x['vote_average']
            return (v / (v + m)) * R + (m / (v + m)) * C if (v + m) != 0 else 0

        self.content['scoring'] = self.content.apply(weighted_score, axis=1)
        logger.info('score column created')

    def save_preprocessed_data(self):
        logger.info('Saving preprocessed data...')
        try:
            self.final_data = self.content[['id', 'combined_content', 'title','scoring']]
            os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
            self.final_data.to_csv(self.save_path, index=False)
            logger.info(f'Data saved at {self.save_path}')
        except Exception as e:
            logger.error(f"Error saving data: {e}")
            raise

    def run_pipeline(self):
        logger.info('Starting data preprocessing pipeline...')
        self.load_data()
        self.fill_missing_values()
        self.combine_content()
        self.preprocess_content()
        self.calculate_scores()
        self.save_preprocessed_data()
        logger.info('Data preprocessing pipeline completed successfully.')
