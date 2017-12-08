
---
title: The Quest for the Perfect Playlist
---

### By David Costigan, Layla O'Kane and Faran Sikandar
#### Spotify - CS109A Final Project - Group 14
<img src="https://www.shareicon.net/data/512x512/2017/02/01/877519_media_512x512.png" width="100" height="100" />

- [1. Problem Statement and Motivation:](#1-problem-statement-and-motivation)
- [2. Introduction and Description of Data:](#2-introduction-and-description-of-data)
- [3. Literature Review/Related Work:](#3-literature-review-related-work)
- [4. Modeling Approach and Project Trajectory:](#4-modeling-approach-and-project-trajectory)
  * [Baseline Models](#baseline-models)
  * [Feature Selection](#feature-selection)
  * [Additional Models](#additional-models)
  * [Playlist Generation](#playlist-generation)
- [5. Results, Conclusions, and Future Work:](#5-results--conclusions--and-future-work)
  * [Conclusions](#conclusions)
  * [Strengths](#strengths)
  * [Weaknesses](#weaknesses)
  * [Further Research](#further-research)

## 1. Problem Statement and Motivation:

The project goal is to assess what features of a playlist on Spotify are good predictors of the number of followers of that playlist. We model this using both regression and classification methods, and using both playlist and song-level data. We then use the results to generate what is likely to be a successful playlist. This project will help inform Spotify about its users’ preferences for playlists, enabling it to create more successful playlists.


## 2. Introduction and Description of Data:

We are working to understand how Spotify generates commercial success through its use of licensed music, and, more specifically, its playlists. Understanding what makes a playlist popular can help us understand how Spotify's digital rights management and its "freemium" business model has led to business success and profitability. Spotify and other similar streaming services have raised questions of how intellectual property rights affect innovation in the age rapid improvements in digital technology. This project contributes to the literature on these issues.

We started by gathering playlist information for all of Spotify’s “official” playlists; these are playlists created by Spotify itself. Our final data set contains 694 playlists. Each of these playlists has an associated category (e.g., “pop” or “blues”; there are 33 unique categories in our dataset), and each playlist has a different number of songs (ranging from short playlists of 12 tracks to significantly longer playlists with 400 tracks). The 694 playlists in our dataset together contain 39,134 unique tracks. For each of these tracks, we retrieved popularity and acoustic information from Spotify’s API and, when available, also pulled additional popularity characteristics from the Million Songs Database. We aggregated this information to the playlist level.

Our preliminary EDA shows that the distribution of the number of followers of a playlist is very skewed: few playlists have an extremely high number of followers, while many more have a medium to low number of followers.

<img src="https://github.com/fsikandar/CS109A_FinalProject_Spotify/blob/master/images/histogram.png?raw=true" width="450" />
<img src="https://github.com/fsikandar/CS109A_FinalProject_Spotify/blob/master/images/boxplot.png?raw=true" width="400"/>

To handle this, we chose to re-define our dependent variable (number of followers) in two ways: (a) for our regression models, we took the natural log of the number of followers and (b) for our classification models, we used quantiles of the number of followers. We also did this by category, since our EDA showed that there were differences by category:

<p align="center">
   <img src="https://github.com/fsikandar/CS109A_FinalProject_Spotify/blob/master/images/followers_by_cat.png?raw=true" />    
   <img src="https://github.com/fsikandar/CS109A_FinalProject_Spotify/blob/master/images/pop_by_cat.png?raw=true" />
</p>

Our EDA also showed that specific features were highly correlated with number of followers, namely, the popularity of individual songs on the playlist, mean danceability, and whether or not the playlist was featured. 

<p align="center">
   <img src="https://github.com/fsikandar/CS109A_FinalProject_Spotify/blob/master/images/correls.png?raw=true" />
</p>

We also looked to see how additional data from the Million Songs Database was correlated with number of followers. This showed that, for example, song hotttnesss, a different measure of popularity, was (unsurprisingly) correlated with number of followers:

<p align="center">
   <img src="https://github.com/fsikandar/CS109A_FinalProject_Spotify/blob/master/images/song_hotttnesss_v_followers.png?raw=true" />
</p>

In order to understand playlists better, and based on our EDA, we chose to predict overall followers as well as followers by category (which is very similar to genre).


## 3. Literature Review/Related Work:

In our work on this project, we have consulted the following resources:

- “Spotify Web API - Spotify Developer.” Accessed December 5, 2017. https://developer.spotify.com/web-api/.
- “Welcome to Spotipy! — Spotipy 2.0 Documentation.” Accessed December 5, 2017. http://spotipy.readthedocs.io/en/latest/.
- “Million Song Dataset - Scaling MIR Research.” Accessed December 5, 2017. https://labrosa.ee.columbia.edu/millionsong/.
- “MusicBrainz - The Open Music Encyclopedia.” Accessed December 5, 2017. https://musicbrainz.org/.
- “AcousticBrainz.” Accessed December 5, 2017. https://acousticbrainz.org/.
- Lamere, Paul. Spotipy: A Light Weight Python Library for the Spotify Web API. Python, 2017. https://github.com/plamere/spotipy.
- Spotify Capstone GitHub. Jupyter Notebook. 2017. Reprint, spotify-iacs, 2017. https://github.com/spotify-iacs/capstone.

## 4. Modeling Approach and Project Trajectory: 

### Baseline Models

We first built baseline models for regression and classification. For regression, we used a simple OLS model with all features included, and for classification we used a logistic multinomial model with all features included. The features we included were data at the playlist level, and engineered summary statistics of song-level data, including the minimum, median, maximum, mean, and standard deviation of characteristics such as accousticness, danceability, and popularity. We chose to include these features since our EDA showed they were relevant to number of playlist followers.

### Feature Selection

Because including all of the features inevitably leads to overfitting, especially in the case of OLS, the next step we took was to implement some feature selection techniques. Initially, we tried to do feature selection for our regression models by fitting a random forest model to predict the quantile of a playlists’ number of followers and looking at the “feature importance” outputted by this model. However, this technique did not seem to give us a better set of features. We suspect this is because the random forest classifier does not consider collinearity between features when determining feature importance (e.g., the random forest said that both the average song popularity and the median song popularity were very important; while this makes sense, these two features are very similar and including both in a regression model wouldn’t be much better than just including one). Thus, we were left with many collinear features using this approach, which did not improve our regressions.

We then decided to use a Lasso model to help select features. This improved our results, and we used the features selected by the Lasso model for all additional regression models. For the tree type models used in classification, we included all features since they include their own built-in feature selection process.

### Additional Models

After selecting our features, we ran additional models for our regression and classification problems. For regression, we implemented both Lasso and Ridge models in addition to OLS. For classification, in addition to our logistic model, we used a decision tree model, a random forest, and decision trees with AdaBoost. Our final models selected were Lasso for our regression analysis and AdaBoost for classification; we chose these models because they had the highest cross-validation accuracies among our models. 

Our LASSO regression model had a train R^2 of 49%, a cross validation R^2 of 34.3%, and a test R^2 of 32.4%. Visually, our predictions for the log number of followers tend to match up fairly well with the actual number of followers in our test set, indicating that our model fits reasonably well:

<p align="center">
   <img src="https://github.com/fsikandar/CS109A_FinalProject_Spotify/blob/master/images/lasso_test_visualization.png?raw=true" />
</p>

For classification, the AdaBoost model’s cross-validation accuracy was 42.81%, and the test accuracy was 35.83%. Note that we were classifying songs based on quintile, so we’d expect to have a classification accuracy of only 20% if our model were classifying randomly. Thus, our model appears to be doing a reasonably good job.

We additionally modeled the number of followers by category using a classification model. In this instance, the quantiles were calculated within category rather than across categories (so we want to predict how well a song will do relative to other songs within its category). This model allowed us to predict the best playlist within a category rather than across categories. Instead of using quintiles, we use tertiles (i.e., divide our songs into top third, middle third, and bottom third) because we have fewer playlists once we subset to category. Our AdaBoost and Decision Tree models performed identically in terms of cross-validation accuracy (48.2%), so we chose the AdaBoost model to be consistent with our choice above for overall quintile classification. Our AdaBoost model had a test accuracy of 47.8%. Since a random model would only have a test accuracy of 33%, our model is performing rather well.

### Playlist Generation

Our final step was to generate a highly-followed playlist. To do this, we classified songs into categories based on the playlists containing them (e.g., if “Crazy in Love” by Beyoncé was included most frequently on “Party” playlists, we classified it as a “Party” song). We then generated playlists (each with 30 songs overall) in each category by randomly choosing songs within the category. We generated 100 playlists in each category (across all 33 unique categories, this leads to 3,300 playlists overall). We then fit our LASSO regression model on these playlists and chose the one that had the highest predicted number of followers. Our best playlist was a dance/party playlist. This aligns with the fact that the most popular songs (on average) were in the Party category based on our EDA (and our regression model put a lot of weight on popularity factors when predicting how many followers a playlist will have, which makes sense).

Below is our randomly generated playlist predicted to be most popular. Click below to play it for yourself!

<p align="center">
   <iframe src="https://open.spotify.com/embed?uri=spotify:user:laylaokane:playlist:3CGThzfeDmjRTFuqo7a1f0&theme=white" width="800" height="400" frameborder="0" allowtransparency="true"></iframe>
</p>

## 5. Results, Conclusions, and Future Work:

### Conclusions

Overall, we were able to create both regression and classification models with some predictive power; given a playlist and some features about it, we can roughly predict how popular the playlist will be on Spotify, which quintile it will fall into in terms of overall Spotify playlist popularity, and which tertile it will be in conditional on its category. Our models tell us that the average song popularity of a playlist is, as we’d expect, an extremely important characteristic for determining song popularity. However, our models also tell us that, some other acoustic characteristics of playlists (like liveness and valence) are important too; including these features improves our regression results and our classification accuracy.

### Strengths

The main strength of our analysis is the richness of our dataset. Our results are based on a wide variety of popularity and acoustic data from Spotify itself (which is important, since our goal is to predict how popular a playlist will be on Spotify). Our dataset encompasses over 30 categories of playlists, and we did not limit ourselves to any one genre. Finally, we included aggregated song-level data for each playlist and made use of alternative measures of popularity, such as “song hotttnesss” and “artist hotttness,” from the Million Songs Database. 

### Weaknesses

However, our analysis does have some weaknesses. The most obvious one is that our dataset is fairly small; we were only able to obtain 694 playlists overall. After splitting into equally sized train and test datasets, this left us with only 347 observations on which to fit our models. Our models would likely be better if we had more data. Furthermore, although our dataset is rich, we are missing genre information beyond the categories provided to us by Spotify; it would have been better if we had song-level genre data (and data that took into account that possibility that songs can straddle many genres). We also only have Million Songs Database data for relatively older songs (since the MSD data stops after 2010), so a lot of newer songs don’t have any information from this database. At a broader level, our analysis doesn’t consider how Spotify itself can influence the popularity of playlists (through marketing playlists at specific times to specific users for specific purposes (e.g., a “Rainy Day” playlist when the user is in a location where it is raining). While we do consider whether a playlist has been featured (which Spotify does control), we could expand this part of our analysis if we had more data. Perhaps most profoundly, another weakness with our model is that, although it (accurately) relies heavily on average song popularity to predict the success of a playlist, using such a model for playlist generation would amplify voices and artists that are already big. There may be ethical issues with designing an algorithm which essentially creates an echo chamber for music wherein the most popular songs are the only ones that are ever added to playlists, which in turn makes them more popular.

### Further Research

Playlist prediction and analysis is an enormous topic; while carrying out this project, we considered other questions that could be answered with further research:

<ul>
  <li>How does track order influence the popularity of a playlist (i.e., conditional on having the same songs in a playlist, does order matter?)</li>
  <li>Do limitations in Spotify’s catalogue prevent some playlists from being more popular than they otherwise could be (e.g., Taylor Swift only recently allowed her music to be streamed on Spotify—does this matter)?</li>
  <li>How does the popularity of a playlist evolve over time?</li>
  <li>Can we create a model to modify existing playlists to make them more popular (given an existing playlist, how would you tweak a playlist to make it more popular?)?</li>
</ul>

### Thanks From the Team!

![Group 14](https://github.com/fsikandar/CS109A_FinalProject_Spotify/blob/master/images/Group14.png?raw=true")
