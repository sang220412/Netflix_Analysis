import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud


def load_data(path):
    data = pd.read_csv(path)
    return data


data = load_data('C:/Users/Sanghraj/PyCharmMiscProject/netflix1.csv')


def clean_data(data):

    data.drop_duplicates(inplace=True)


    data.dropna(subset=['director', 'country'], inplace=True)


    data['date_added'] = pd.to_datetime(data['date_added'], errors='coerce')


    data.dropna(subset=['date_added'], inplace=True)


    str_cols = ['type', 'title', 'director', 'country', 'rating', 'duration', 'listed_in']
    for col in str_cols:
        data[col] = data[col].str.strip()

    return data


def feature_engineering(data):

    data['year_added'] = data['date_added'].dt.year
    data['month_added'] = data['date_added'].dt.month


    data['genres'] = data['listed_in'].apply(lambda x: x.split(', '))
    return data


def explode_genres(data):

    data_expanded = data.explode('genres')
    return data_expanded


def plot_content_type_distribution(data):
    plt.figure(figsize=(6, 4))
    sns.countplot(x='type', data=data, hue='type', palette='Set2', legend=False)
    plt.title('Content Type Distribution (Movies vs TV Shows)')
    plt.xlabel('Type')
    plt.ylabel('Count')
    plt.show()


def plot_top_genres(data, top_n=10):
    all_genres = sum(data['genres'], [])
    genre_counts = pd.Series(all_genres).value_counts().head(top_n)

    plt.figure(figsize=(10, 6))
    sns.barplot(x=genre_counts.values, y=genre_counts.index, hue=genre_counts.index, palette='Set3', legend=False)
    plt.title(f'Top {top_n} Most Common Genres on Netflix')
    plt.xlabel('Count')
    plt.ylabel('Genre')
    plt.show()


def plot_content_added_over_time(data):
    plt.figure(figsize=(12, 6))
    sns.countplot(x='year_added', data=data, hue='year_added', palette='coolwarm', legend=False)
    plt.title('Content Added Over Years')
    plt.xlabel('Year')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.show()


def plot_top_directors(data, top_n=10):
    top_directors = data['director'].value_counts().head(top_n)

    plt.figure(figsize=(10, 6))
    sns.barplot(x=top_directors.values, y=top_directors.index, hue=top_directors.index, palette='Blues_d', legend=False)
    plt.title(f'Top {top_n} Directors with Most Titles')
    plt.xlabel('Number of Titles')
    plt.ylabel('Director')
    plt.show()


def plot_wordcloud_titles(data):
    movie_titles = data[data['type'] == 'Movie']['title']
    text = ' '.join(movie_titles)
    wordcloud = WordCloud(width=800, height=400, background_color='black').generate(text)

    plt.figure(figsize=(10, 6))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title('Word Cloud of Movie Titles')
    plt.show()


def main():

    path = 'C:/Users/Sanghraj/PyCharmMiscProject/netflix1.csv'  # Change this to your path
    data = load_data(path)

    print("Initial data info:")
    print(data.info())


    data = clean_data(data)

    print("\nData info after cleaning:")
    print(data.info())


    data = feature_engineering(data)

    data_expanded = explode_genres(data)

    data_expanded.to_csv('netflix_expanded.csv', index=False)
    print("Exported exploded data to 'netflix_expanded.csv' for Tableau.")

    # data.to_csv('netflix_cleaned.csv', index=False)
# print("Cleaned data saved to netflix_cleaned.csv")

    plot_content_type_distribution(data)
    plot_top_genres(data)
    plot_content_added_over_time(data)
    plot_top_directors(data)
    plot_wordcloud_titles(data)

    print("\nAnalysis complete.")


if __name__ == "__main__":
    main()
