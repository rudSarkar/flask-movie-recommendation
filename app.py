import flask
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from flask_caching import Cache

app = flask.Flask(__name__, template_folder='templates')
cache = Cache(app, config={'CACHE_TYPE': 'simple'})

df = pd.read_csv('./model/tmdb.csv')

tfidf = TfidfVectorizer(stop_words='english', analyzer='word')

# Construct the required TF-IDF matrix by fitting and transforming the data
tfidf_matrix = tfidf.fit_transform(df['soup'])
print(tfidf_matrix.shape)

# Construct cosine similarity matrix
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
print(cosine_sim.shape)

df = df.reset_index()
indices = pd.Series(df.index, index=df['title']).drop_duplicates()

# Create an array with all movie titles
all_titles = []
for i in range(len(df['title'])):
    title = df['title'][i]
    all_titles.append(title)


def get_recommendations(title):
    # Convert the input title to lowercase
    title = title.lower()
    # Convert all_titles to lowercase
    all_titles_lower = [t.lower() for t in all_titles]

    # Get the index of the movie that matches the title
    idx = all_titles_lower.index(title)
    # Get the pairwise similarity scores of all movies with that movie
    sim_scores = list(enumerate(cosine_sim[idx]))
    # Sort the movies based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    # Get the scores of the 10 most similar movies
    sim_scores = sim_scores[1:11]

    # Get the movie indices
    movie_indices = [i[0] for i in sim_scores]

    # Return a DataFrame of similar movies and similarity scores
    return_df = pd.DataFrame(columns=['Title', 'Homepage'])
    return_df['Title'] = df['title'].iloc[movie_indices]
    return_df['Homepage'] = df['homepage'].iloc[movie_indices]
    return_df['ReleaseDate'] = df['release_date'].iloc[movie_indices]
    return return_df, sim_scores


# Set up the main route
@app.route('/', methods=['GET', 'POST'])
def main():
    if flask.request.method == 'GET':
        return flask.render_template('index.html')

    if flask.request.method == 'POST':
        movie_name = " ".join(flask.request.form['movie_name'].split())
        if movie_name.lower() not in [t.lower() for t in all_titles]:
            return flask.render_template('notFound.html', name=movie_name)
        else:
            result_final, sim_scores = get_recommendations(movie_name.lower())
            movie_names = []
            homepage = []
            release_dates = []
            for i in range(len(result_final)):
                movie_names.append(result_final.iloc[i][0])
                release_dates.append(result_final.iloc[i][2])
                if len(str(result_final.iloc[i][1])) > 3:
                    homepage.append(result_final.iloc[i][1])
                else:
                    homepage.append("#")

            return flask.render_template('found.html', movie_names=movie_names, movie_homepage=homepage,
                                         search_name=movie_name, movie_release_dates=release_dates,
                                         movie_sim_scores=sim_scores)


if __name__ == '__main__':
    app.run(host="127.0.0.1", port=8080, debug=True)
