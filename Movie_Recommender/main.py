import pandas as pd
import streamlit as st
from streamlit.components.v1 import html
import os
from src.logger import get_logger
from src.data_preprocess import DataPreprocessor
from src.model_training import Model
from src.recomend import Recomend
from src.poster_api import Api

logger = get_logger(__name__)

class App:
    def __init__(self):
        self.prep_data_path = None
        self.sim_score_path = None
        self.movie_input = None
        self.recomend_data = None
        self.api_data = None
        self.api_key = "YOUR API KEY"  

    def preprocess(self):
        if not os.path.exists(os.path.join(os.path.dirname(__file__),'data/preprocessed.csv')):
            logger.info('preprocessing...')
            data_path = os.path.join(os.path.dirname(__file__),'data/movies.csv')
            preprocess = DataPreprocessor(data_path=data_path)
            preprocess.run_pipeline()
            self.prep_data_path = os.path.join(os.path.dirname(__file__),'data/preprocessed.csv')
            logger.info('preprocessing completed and preprocessed.csv created')
        else:
            logger.info('preprocessed already exists')
            self.prep_data_path = os.path.join(os.path.dirname(__file__),'data/preprocessed.csv')
            
    def model_training(self):
        if not os.path.exists(os.path.join(os.path.dirname(__file__),'model/sim_score.pkl')) and os.path.exists(os.path.join(os.path.dirname(__file__),'data/preprocessed.csv')):
            logger.info('training model...')
            model = Model(data_path=self.prep_data_path)
            model.run_pipeline()
            logger.info('training completed and sim_score.pkl created')
        else:
            logger.info('sim_score.pkl already exists')
            self.sim_score_path = os.path.join(os.path.dirname(__file__),'model/sim_score.pkl')
    
    @staticmethod
    def get_api_data(recomend_data,api_key):
        api = Api(recomend_data['id'].values,api_key)
        api_data = api.get_all_data()
        return api_data
    
    @staticmethod
    def show_movies(recomend_data,api_data):
        titles = recomend_data['title'].values
        scores = recomend_data['scoring'].values
        genres = []
        cast = []
        director = []
        overview = []
        runtime = []
        release_date = []
        poster_url = []
        for i in api_data:
            genres.append(i.get('genres'))
            cast.append(i.get('cast'))
            director.append(i.get('director'))
            overview.append(i.get('overview'))
            runtime.append(i.get('runtime'))
            release_date.append(i.get('release_date'))
            poster_url.append(i.get('poster_url'))
        for idx in range(len(api_data)):
            html(
                f"""
                <div style="display: flex; height: 250px; width: 100%; background-color: #f0f0f0; border-radius: 8px; overflow: hidden;">
                    
                    <div style="flex: 0 0 30%; background-image: url('{poster_url[idx]}'); background-size: cover; background-position: center;">
                    </div>
                    
                    <div style="flex: 1; padding: 10px; overflow-y: auto;">
                        <h2 style="margin: 0; padding: 0;text-decoration: underline; text-decoration-color: yellow;">{titles[idx]}</h2>
                        <div style="display: flex; flex-wrap: wrap; font-size: 14px; color: #555; margin-top: 8px;">
                            <div style="margin-right: 15px;">⭐ {round(scores[idx],1)}</div>
                            <div style="margin-right: 15px;">Genre: {genres[idx]}</div>
                            <div style="margin-right: 15px;">Cast: {cast[idx]}</div>
                            <div style="margin-right: 15px;">Director: {director[idx]}</div>
                            <div style="margin-right: 15px;">Release: {release_date[idx]}</div>
                            <div>Runtime: 120 min</div>
                        </div>
                        <div style="font-size: 13px; color: #333; margin-top: 10px; max-height: 150px; overflow-y: auto;">
                            {overview[idx]}
                        </div>
                    </div>
                    
                </div>
                """,
                height=270,
            )


    def get_show_movies_data(self):
        logger.info('streamlit running...')
        st.markdown(
            '<div style="display: flex; align-items: center;">'
            '<span style="font-size: 24px; margin-right: 8px;">🎬</span>'
            '<h3 style="margin:0; padding:0; text-decoration: underline; text-decoration-color: yellow; font-size: 24px;">Movie Recommender</h3>'
            '</div>',
            unsafe_allow_html=True
        )

        self.movie_input = st.text_input(label='',placeholder='type a movie...')
        if self.movie_input and self.movie_input != '':
            recomend = Recomend(title=self.movie_input,data_path=self.prep_data_path,sim_score_path=self.sim_score_path,n_tops=10)
            self.recomend_data = recomend.recomend()
            if self.recomend_data is not None:
                self.api_data = App.get_api_data(self.recomend_data,self.api_key)
                App.show_movies(self.recomend_data,self.api_data)
            else:
                st.write('Plz input correct movie name')
        logger.info('session ended')
    
    def run_app(self):
        logger.info('running app...')
        self.preprocess()
        self.model_training()
        self.get_show_movies_data()

app = App()
app.run_app()
