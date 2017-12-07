---
title: Spotify - CS109A Final Project - Group 14
---

## By David Costigan, Layla O'Kane and Faran Sikandar

## 1. Problem Statement and Motivation:

The project goal is to assess what features of a playlist on Spotify are good predictors of the number of followers of that playlist. We model this using both regression and classification methods, and using both playlist and song-level data. We then use the results to generate what is likely to be a successful playlist. This project will help inform Spotify about its users’ preferences for playlists, enabling it to create more successful playlists.


## 2. Introduction and Description of Data:

We are working to understand how Spotify generates commercial success through its use of licensed music, and more specifically its playlists. This contributes to an understanding of how Spotify's digital rights management and its "freemium" business model has led to business success and profitability. Spotify and other similar streaming services have raised questions of how intellectual property rights affect innovation in the age rapid improvements in digital technology. This project contributes to the literature on these issues.

Our preliminary EDA shows that the distribution of the number of followers of a playlist is very skewed: few playlists have an extremely high number of followers, while many more have a medium to low number of followers. [INSERT EDA GRAPHS 1 and 2]

To handle this, we chose to re-define our dependent variable (number of followers) in two ways: (a) for our regression models, we took the natural log of the number of followers and (b) for our classification models, we used quantiles of the number of followers. 

Our EDA also showed that specific features where highly correlated with number of followers, namely, the popularity of individual songs on the playlist, mean danceability, and whether or not the playlist was featured. [INSERT EDA GRAPH OF SOME KIND]

In order to understand playlists better, and based on our EDA, we chose to predict overall followers as well as followers by category (which is very similar to genre).

## 3. Literature Review/Related Work: This could include noting any key papers, texts, or websites that you have used to develop your modeling approach, as well as what others have
done on this problem in the past. You must properly credit sources.

In our work on this project, we have consulted the following resources:

- “Spotify Web API - Spotify Developer.” Accessed December 5, 2017. https://developer.spotify.com/web-api/.
- “Welcome to Spotipy! — Spotipy 2.0 Documentation.” Accessed December 5, 2017. http://spotipy.readthedocs.io/en/latest/.
- “Million Song Dataset - Scaling MIR Research.” Accessed December 5, 2017. https://labrosa.ee.columbia.edu/millionsong/.
- “MusicBrainz - The Open Music Encyclopedia.” Accessed December 5, 2017. https://musicbrainz.org/.
- “AcousticBrainz.” Accessed December 5, 2017. https://acousticbrainz.org/.
- Lamere, Paul. Spotipy: A Light Weight Python Library for the Spotify Web API. Python, 2017. https://github.com/plamere/spotipy.
- Spotify Capstone GitHub. Jupyter Notebook. 2017. Reprint, spotify-iacs, 2017. https://github.com/spotify-iacs/capstone.

## 4. Modeling Approach and Project Trajectory: 

The first step we took in our modeling approach was to build baseline models for both regression and classification. For regression, we used a simple OLS model with all features included, and for classification we used a logistic multinomial model with all features included. The features we included were data at the playlist level, and engineered summary statistics of song-level data, including the minimum, median, maximum, mean, and standard deviation of characteristics such as accousticness, danceability, and popularity. We chose to include these features since our EDA showed they were relevant to number of playlist followers.

Because including all of the features inevitably leads to overfitting, especially in the case of OLS, the next step we took was to implement some feature selection techniques. We tried to do feature selection using a random forest model to provide key features. However, this technique did not seem to improve our test outcomes. We suspect this is because this method does not take into account collinearity between features but rather evaluates them individually, which is not as useful for regression analysis. We then decided to use a Lasso model to help select features. This improved our results, and we used the features selected by the Lasso model for all additional regression models and for the logistic classification model. For the tree type models, we included all features since they include their own built-in feature selection process.

After selecting our features, we ran additional models for both our regression and our classification approaches. For regression, we implemented both Lasso and Ridge models. For Classification, in addition to our logistic model, we used decision tree, random forest, and Ada boost. Our final models selected were Lasso for our regression analysis and X for classification. The final models had zz test and train results. 

We additionally modeled the number of followers by category using a classification model. In this instance, the quantiles were calculated within category rather than across categories. This model allowed us to predict the best playlist within a category rather than across categories. Our best model was x, and had x results.

Our next step was to use our model to predict a highly-followed playlist. To do this, we used our final regression model and verified that our predicted playlist was also in the top quantile of followers using our final classification model. In order to generate the playlist, we classified song based on their modal category and then only selected songs from within a category for each generated playlist. We generated 10,000 playlist to choose from and used all of the songs available from the Spotify data. Our best playlist was a dance/party playlist.

<ol type="1" style="font-weight: bold">
  <li>Baseline models:</li>
    <ol type="A" style="font-weight: normal">
      <li>Regression: OLS with all engineered features included</li>
      <li>Classification: Logistic with all features</li>        
    </ol>
  <li>Feature selection: </li>
    <ol type="A" style="font-weight: normal">
      <li>Engineered summary stats of song-level data at the playlist level</li>
      <li>Tried feature selection using random forest – did not seem to help</li>
      <li>Tried feature selection for regression model using Lasso – this is what we ended up with. We used these features for all regression models and then logistic model for classification.</li>
      <li>Included all features for tree models since they have their own feature selection process</li>
    </ol>
  <li>Additional models to run after feature selection/using feature selection:</li>
    <ol type="A" style="font-weight: normal">
      <li>Regression: Lasso, Ridge with CV</li>
      <li>Classification: Decision Tree, Random Forest, Ada Boost</li>
    </ol>
  <li>Pick best of the above:</li>
    <ol type="A" style="font-weight: normal">
      <li>Regression: Lasso</li>
      <li>Classification: Random Forest/Ada Boost?</li>
    </ol>
  <li>Generate playlists of songs within category:</li>
  <ol type="A" style="font-weight: normal">
    <li>Categorized songs based on mode category, random if equal</li>
    <li>Generated playlists</li>
  </ol>
</ol>

## 5. Results, Conclusions, and Future Work:

Our most important conclusion is that The Ketchup Song is on the best predicted playlist generated by our model. Some may think that a playlist of only The Ketchup Song would be ideal. Those people would not be incorrect.


## Lets have fun

>here is a quote

Here is *emph* and **bold**.

Here is some inline math $\alpha = \frac{\beta}{\gamma}$ and, of-course, E rules:

$$ G_{\mu\nu} + \Lambda g_{\mu\nu}  = 8 \pi T_{\mu\nu} . $$

![alt text](https://www.shareicon.net/data/512x512/2017/02/01/877519_media_512x512.png "width"=100 "height"=50)

![Group 14](https://github.com/fsikandar/CS109A_FinalProject_Spotify/blob/master/images/Group14.png?raw=true")
