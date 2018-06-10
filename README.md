
## News Mood
### Background
As you might know, --Twitter-- has become a wildly sprawling jungle of information&mdash;140 characters at a time. 
Somewhere between 350 million and 500 million tweets are estimated to be sent out _per day_. 
With such an explosion of data, on Twitter and elsewhere, it becomes more important than ever to tame it in 
some way, to concisely capture the essence of the data.

In this activity, we will perform a sentiment analysis of the Twitter activity of various news oulets (BBC, CBS, CNN, Fox, and New York times)
and show the findings providing a visualized summary of the sentiments expressed in Tweets sent out by the newscast above.


## Analysis
#### 1. Based on this sampling data, the majority of these Tweets are positive based on the VADER analysis.
#### 2. CBS is the news cast with the highest positive sentiment follwed by Fox.
#### 3. BBC is the newscast with the majority of the neutral sentiments.


```python
## Dependencies
import csv
import tweepy
import numpy as np
import pandas as pd
import seaborn as sns

import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')
from datetime import datetime, timezone

# Import and Initialize Sentiment Analyzer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
analyzer = SentimentIntensityAnalyzer()

# Twitter API Keys
from config import (consumer_key, 
                    consumer_secret, 
                    access_token, 
                    access_token_secret)

# Setup Tweepy API Authentication
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth, parser=tweepy.parsers.JSONParser())
```


```python
#Create a Python script to perform a sentiment analysis of the Twitter activity of various news oulets,
#and present your findings visually.
#Your final output should provide a visualized summary of the sentiments expressed in Tweets 
#sent out by the following news organizations: __BBC, CBS, CNN, Fox, and New York times__.
```


```python
# Target Search Term
target_terms = ("@BBC", "@CBS", "@CNN",
                "@Fox", "@New York Times")
# Counter
counter = 1

# List to hold results
news_list = []

# Loop through all target users
for target in target_terms:

    # Variable for holding the oldest tweet
    oldest_tweet = None

    # Loop through 5 times *****
    for x in range(5):

        # Run search around each tweet
        public_tweets = api.user_timeline(target, max_id = oldest_tweet)
        
        # Loop through all tweets
        for tweet in public_tweets:

            # Print Tweets
            # print("Tweet %s: %s" % (counter, tweet["text"]))
        
            # Run Vader Analysis on each tweet
            results = analyzer.polarity_scores(tweet["text"])
            compound = results["compound"]
            pos = results["pos"]
            neu = results["neu"]
            neg = results["neg"]
            tweets_ago = counter
        
            # Get Tweet ID, subtract 1, and assign to oldest_tweet
            oldest_tweet = tweet['id'] - 1
            # Convert the date to be used later
            date = tweet["created_at"]
            date_converted = datetime.strptime(date, "%a %b %d %H:%M:%S %z %Y")
            # Add sentiments for each tweet into a list
            news_list.append({"News_Cast" : target,
                            "Date": date_converted, 
                            "Compound": compound,
                            "Positive": pos,
                            "Negative": neu,
                            "Neutral": neg,
                            "Tweets Ago": counter})
        
            # Add to counter 
            counter += 1

```


```python
# Convert sentiments to DataFrame
news_list_pd = pd.DataFrame.from_dict(news_list)
news_list_pd.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Compound</th>
      <th>Date</th>
      <th>Negative</th>
      <th>Neutral</th>
      <th>News_Cast</th>
      <th>Positive</th>
      <th>Tweets Ago</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.0000</td>
      <td>2018-06-09 20:04:00+00:00</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>@BBC</td>
      <td>0.000</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.3612</td>
      <td>2018-06-09 19:02:05+00:00</td>
      <td>0.878</td>
      <td>0.000</td>
      <td>@BBC</td>
      <td>0.122</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.0000</td>
      <td>2018-06-09 18:04:03+00:00</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>@BBC</td>
      <td>0.000</td>
      <td>3</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.5719</td>
      <td>2018-06-09 17:01:05+00:00</td>
      <td>0.829</td>
      <td>0.000</td>
      <td>@BBC</td>
      <td>0.171</td>
      <td>4</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.4019</td>
      <td>2018-06-09 16:04:03+00:00</td>
      <td>0.623</td>
      <td>0.156</td>
      <td>@BBC</td>
      <td>0.222</td>
      <td>5</td>
    </tr>
  </tbody>
</table>
</div>




```python
len(news_list_pd)
```




    412




```python
# Save as a csv using "utf-8" encoding
news_list_pd.to_csv("newscast_sentiments.csv", encoding="utf-8", index=False)
```


```python
by_newscast = news_list_pd.groupby('News_Cast')

pos_count = news_list_pd[news_list_pd['Compound'] > 0].groupby('News_Cast')['Compound'].count()
neg_count = news_list_pd[news_list_pd['Compound'] < 0].groupby('News_Cast')['Compound'].count()
neu_count = news_list_pd[news_list_pd['Compound'] == 0].groupby('News_Cast')['Compound'].count()

polarity_df = pd.DataFrame({
    "Positive Compound Scores": pos_count,
    "Negative Compound Scores": neg_count,
    "Neutral Compound Scores": neu_count,
    "Total": pos_count + neg_count + neu_count
})
polarity_df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Negative Compound Scores</th>
      <th>Neutral Compound Scores</th>
      <th>Positive Compound Scores</th>
      <th>Total</th>
    </tr>
    <tr>
      <th>News_Cast</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>@BBC</th>
      <td>24</td>
      <td>32</td>
      <td>44</td>
      <td>100</td>
    </tr>
    <tr>
      <th>@CBS</th>
      <td>10</td>
      <td>24</td>
      <td>66</td>
      <td>100</td>
    </tr>
    <tr>
      <th>@CNN</th>
      <td>26</td>
      <td>31</td>
      <td>43</td>
      <td>100</td>
    </tr>
    <tr>
      <th>@Fox</th>
      <td>15</td>
      <td>24</td>
      <td>61</td>
      <td>100</td>
    </tr>
    <tr>
      <th>@New York Times</th>
      <td>1</td>
      <td>6</td>
      <td>5</td>
      <td>12</td>
    </tr>
  </tbody>
</table>
</div>




```python
bbc_compound = news_list_pd.groupby('News_Cast')['Compound'].mean()
bbc_compound
```




    News_Cast
    @BBC               0.120679
    @CBS               0.326564
    @CNN               0.093054
    @Fox               0.273989
    @New York Times    0.247308
    Name: Compound, dtype: float64




```python
neg_sum = polarity_df['Negative Compound Scores'].sum()
neu_sum = polarity_df['Neutral Compound Scores'].sum()
pos_sum = polarity_df['Positive Compound Scores'].sum()
total = neg_sum + pos_sum + neu_sum

sentiments_df = pd.DataFrame({
    "Total Negative": [neg_sum],
    "Total Positive": pos_sum,
    "Total Neutral": neg_sum,
    "Total": total
})

sentiments_df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Total</th>
      <th>Total Negative</th>
      <th>Total Neutral</th>
      <th>Total Positive</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>412</td>
      <td>76</td>
      <td>76</td>
      <td>219</td>
    </tr>
  </tbody>
</table>
</div>




```python
time_max = by_newscast['Date'].max()
time_min = by_newscast['Date'].min()
time_diff = time_max - time_min
time_diff

avg_time_per_tweets = time_diff/100
time_df = pd.DataFrame({
    "Time passed between 1 and 100 Tweets": time_diff,
    "Average Time Between Tweets": avg_time_per_tweets
})

time_df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Average Time Between Tweets</th>
      <th>Time passed between 1 and 100 Tweets</th>
    </tr>
    <tr>
      <th>News_Cast</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>@BBC</th>
      <td>01:17:51.570000</td>
      <td>5 days 09:45:57</td>
    </tr>
    <tr>
      <th>@CBS</th>
      <td>13:00:34.330000</td>
      <td>54 days 04:57:13</td>
    </tr>
    <tr>
      <th>@CNN</th>
      <td>00:13:12.390000</td>
      <td>0 days 22:00:39</td>
    </tr>
    <tr>
      <th>@Fox</th>
      <td>02:43:17.280000</td>
      <td>11 days 08:08:48</td>
    </tr>
    <tr>
      <th>@New York Times</th>
      <td>02:10:23.170000</td>
      <td>9 days 01:18:37</td>
    </tr>
  </tbody>
</table>
</div>




```python
#finds date of most recent and least recent tweet
date_max = news_list_pd['Date'].max().replace(tzinfo=timezone.utc).astimezone(tz = 'US/Eastern').strftime('%D: %r') + " (ET)"
date_min = news_list_pd['Date'].min().replace(tzinfo=timezone.utc).astimezone(tz = 'US/Eastern').strftime('%D: %r') + " (ET)"
```


```python
by_newscast.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Compound</th>
      <th>Date</th>
      <th>Negative</th>
      <th>Neutral</th>
      <th>News_Cast</th>
      <th>Positive</th>
      <th>Tweets Ago</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.0000</td>
      <td>2018-06-09 20:04:00+00:00</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>@BBC</td>
      <td>0.000</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.3612</td>
      <td>2018-06-09 19:02:05+00:00</td>
      <td>0.878</td>
      <td>0.000</td>
      <td>@BBC</td>
      <td>0.122</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.0000</td>
      <td>2018-06-09 18:04:03+00:00</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>@BBC</td>
      <td>0.000</td>
      <td>3</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.5719</td>
      <td>2018-06-09 17:01:05+00:00</td>
      <td>0.829</td>
      <td>0.000</td>
      <td>@BBC</td>
      <td>0.171</td>
      <td>4</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.4019</td>
      <td>2018-06-09 16:04:03+00:00</td>
      <td>0.623</td>
      <td>0.156</td>
      <td>@BBC</td>
      <td>0.222</td>
      <td>5</td>
    </tr>
    <tr>
      <th>100</th>
      <td>0.6800</td>
      <td>2018-06-09 21:00:02+00:00</td>
      <td>0.763</td>
      <td>0.000</td>
      <td>@CBS</td>
      <td>0.237</td>
      <td>101</td>
    </tr>
    <tr>
      <th>101</th>
      <td>0.0000</td>
      <td>2018-06-09 16:00:00+00:00</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>@CBS</td>
      <td>0.000</td>
      <td>102</td>
    </tr>
    <tr>
      <th>102</th>
      <td>0.2960</td>
      <td>2018-06-08 19:46:17+00:00</td>
      <td>0.820</td>
      <td>0.000</td>
      <td>@CBS</td>
      <td>0.180</td>
      <td>103</td>
    </tr>
    <tr>
      <th>103</th>
      <td>0.0000</td>
      <td>2018-06-08 15:00:03+00:00</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>@CBS</td>
      <td>0.000</td>
      <td>104</td>
    </tr>
    <tr>
      <th>104</th>
      <td>0.9078</td>
      <td>2018-06-08 13:58:15+00:00</td>
      <td>0.564</td>
      <td>0.000</td>
      <td>@CBS</td>
      <td>0.436</td>
      <td>105</td>
    </tr>
    <tr>
      <th>200</th>
      <td>0.5719</td>
      <td>2018-06-10 04:31:46+00:00</td>
      <td>0.821</td>
      <td>0.000</td>
      <td>@CNN</td>
      <td>0.179</td>
      <td>201</td>
    </tr>
    <tr>
      <th>201</th>
      <td>0.5859</td>
      <td>2018-06-10 04:16:00+00:00</td>
      <td>0.847</td>
      <td>0.000</td>
      <td>@CNN</td>
      <td>0.153</td>
      <td>202</td>
    </tr>
    <tr>
      <th>202</th>
      <td>0.0000</td>
      <td>2018-06-10 04:01:04+00:00</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>@CNN</td>
      <td>0.000</td>
      <td>203</td>
    </tr>
    <tr>
      <th>203</th>
      <td>-0.7506</td>
      <td>2018-06-10 03:30:06+00:00</td>
      <td>0.670</td>
      <td>0.330</td>
      <td>@CNN</td>
      <td>0.000</td>
      <td>204</td>
    </tr>
    <tr>
      <th>204</th>
      <td>0.0000</td>
      <td>2018-06-10 03:00:10+00:00</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>@CNN</td>
      <td>0.000</td>
      <td>205</td>
    </tr>
    <tr>
      <th>300</th>
      <td>0.4939</td>
      <td>2018-06-09 11:32:40+00:00</td>
      <td>0.656</td>
      <td>0.107</td>
      <td>@Fox</td>
      <td>0.238</td>
      <td>301</td>
    </tr>
    <tr>
      <th>301</th>
      <td>0.4767</td>
      <td>2018-06-09 10:36:55+00:00</td>
      <td>0.773</td>
      <td>0.000</td>
      <td>@Fox</td>
      <td>0.227</td>
      <td>302</td>
    </tr>
    <tr>
      <th>302</th>
      <td>-0.0258</td>
      <td>2018-06-09 10:09:44+00:00</td>
      <td>0.757</td>
      <td>0.143</td>
      <td>@Fox</td>
      <td>0.100</td>
      <td>303</td>
    </tr>
    <tr>
      <th>303</th>
      <td>0.4404</td>
      <td>2018-06-08 19:36:53+00:00</td>
      <td>0.513</td>
      <td>0.162</td>
      <td>@Fox</td>
      <td>0.325</td>
      <td>304</td>
    </tr>
    <tr>
      <th>304</th>
      <td>-0.1880</td>
      <td>2018-06-08 19:35:39+00:00</td>
      <td>0.758</td>
      <td>0.137</td>
      <td>@Fox</td>
      <td>0.105</td>
      <td>305</td>
    </tr>
    <tr>
      <th>400</th>
      <td>0.6705</td>
      <td>2018-05-31 01:29:17+00:00</td>
      <td>0.476</td>
      <td>0.000</td>
      <td>@New York Times</td>
      <td>0.524</td>
      <td>401</td>
    </tr>
    <tr>
      <th>401</th>
      <td>0.7824</td>
      <td>2018-05-31 01:28:17+00:00</td>
      <td>0.623</td>
      <td>0.000</td>
      <td>@New York Times</td>
      <td>0.377</td>
      <td>402</td>
    </tr>
    <tr>
      <th>402</th>
      <td>0.7430</td>
      <td>2018-05-31 01:27:16+00:00</td>
      <td>0.717</td>
      <td>0.000</td>
      <td>@New York Times</td>
      <td>0.283</td>
      <td>403</td>
    </tr>
    <tr>
      <th>403</th>
      <td>0.5719</td>
      <td>2018-05-31 01:26:16+00:00</td>
      <td>0.802</td>
      <td>0.000</td>
      <td>@New York Times</td>
      <td>0.198</td>
      <td>404</td>
    </tr>
    <tr>
      <th>404</th>
      <td>-0.3400</td>
      <td>2018-05-31 01:25:15+00:00</td>
      <td>0.556</td>
      <td>0.278</td>
      <td>@New York Times</td>
      <td>0.167</td>
      <td>405</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Assign variables to the different News Cast Companies
bbc = news_list_pd[news_list_pd["News_Cast"] == "@BBC"]
cbs = news_list_pd[news_list_pd["News_Cast"] == "@CBS"]
cnn = news_list_pd[news_list_pd["News_Cast"] == "@CNN"]
fox = news_list_pd[news_list_pd["News_Cast"] == "@Fox"]
nyt = news_list_pd[news_list_pd["News_Cast"] == "@New York Times"]
```


```python
# Creating base chart - Including chart tittle, axis labels, notation
now = datetime.now()
now = now.strftime("%m-%d-%Y")
plt.title(f"Sentiment Analysis Of Media Tweets on ({now})")
plt.xlabel("Tweets Ago")
plt.ylabel("Tweet Polarity")
# Add the Grid and use gray as the grid Color
plt.rc('grid', linestyle="-", color='gray')
plt.grid(True)
# Create the Scatter Plot
x_bbc= bbc['Tweets Ago']
y_bbc= bbc['Compound']
x_cbs= cbs['Tweets Ago']
y_cbs= cbs['Compound']
x_cnn= cnn['Tweets Ago']
y_cnn= cnn['Compound']
x_fox= fox['Tweets Ago']
y_fox= fox['Compound']
x_nyt= nyt['Tweets Ago']
y_nyt= nyt['Compound']
plt.scatter(x_bbc, y_bbc, s = 100, c="lightskyblue", edgecolor= "black", linewidth=1, 
            alpha = 0.75, label = "BBC")
plt.scatter(x_cbs, y_cbs, s = 100, c="green", edgecolor= "black", linewidth=1, alpha = 0.75, 
           label = "CBS")
plt.scatter(x_cnn, y_cnn, s = 100, c="red", edgecolor= "black", linewidth=1, 
            alpha = 0.75, label = "CNN")
plt.scatter(x_fox, y_fox, s = 100, c="blue", edgecolor= "black", linewidth=1, 
            alpha = 0.75, label = "Fox")
plt.scatter(x_nyt, y_nyt, s = 100, c="yellow", edgecolor= "black", linewidth=1, alpha = 0.75, 
           label = "New York Times")
plt.legend()

# Saving the plot
plt.savefig("Sentiment_Analysis.png")

# Showing the plot
plt.show()
```


![png](output_15_0.png)



```python
bbc_compound = news_list_pd.groupby('News_Cast')['Compound'].mean()
bbc_compound
```




    News_Cast
    @BBC               0.120679
    @CBS               0.326564
    @CNN               0.093054
    @Fox               0.273989
    @New York Times    0.247308
    Name: Compound, dtype: float64




```python
# Creating Bar Chart showing Overall Media Sentiment
overall_chart = bbc_compound.plot(kind='bar', colors=['lightskyblue', 'green', "red", 'blue', 'yellow'])

# Set the xlabel and ylabel using class methods
plt.title(f"Overall Media Sentiment based on Twitter on ({now})")
overall_chart.set_xlabel("News Cast")
overall_chart.set_ylabel("Tweet Polarity")

# Saving the plot
plt.savefig("overall_Sentiment.png")

# Showing the plot
plt.show()
```

    /anaconda3/envs/PythonData/lib/python3.6/site-packages/pandas/plotting/_core.py:186: UserWarning: 'colors' is being deprecated. Please use 'color'instead of 'colors'
      warnings.warn(("'colors' is being deprecated. Please use 'color'"



![png](output_17_1.png)



```python
# Create plot chart for BBC
now = datetime.now()
now = now.strftime("%m-%d-%Y")
plt.title(f"Sentiment Analysis of Tweets ({now}) for BBC")
plt.xlabel("Tweets Ago")
plt.ylabel("Tweet Polarity")
# Add the Grid and use gray as the grid Color
#plt.rc('grid', linestyle="-", color='gray')
#plt.grid(True)
# Create the Plot
x_bbc= bbc['Tweets Ago']
y_bbc= bbc['Compound']
plt.plot(x_bbc, y_bbc, color='lightskyblue', marker='o', linestyle='dashed', linewidth=2, markersize=12, label = "BBC")
plt.legend()

# Saving the plot
plt.savefig("bbc.png")

# Showing the plot
plt.show()
```


![png](output_18_0.png)



```python
# Create plot chart for CBS
now = datetime.now()
now = now.strftime("%m-%d-%Y")
plt.title(f"Sentiment Analysis of Tweets ({now}) for CBS")
plt.xlabel("Tweets Ago")
plt.ylabel("Tweet Polarity")
# Add the Grid and use gray as the grid Color
plt.rc('grid', linestyle="-", color='gray')
plt.grid(True)
# Create the Plot
x_cbs= cbs['Tweets Ago']
y_cbs= cbs['Compound']
plt.plot(x_cbs, y_cbs, color='green', marker='o', linestyle='dashed', linewidth=2, markersize=12, label = "CBS")
plt.legend()

# Saving the plot
plt.savefig("cbs.png")

# Showing the plot
plt.show()
```


![png](output_19_0.png)



```python
# Create plot chart for CNN
now = datetime.now()
now = now.strftime("%m-%d-%Y")
plt.title(f"Sentiment Analysis of Tweets ({now}) for CNN")
plt.xlabel("Tweets Ago")
plt.ylabel("Tweet Polarity")
# Add the Grid and use gray as the grid Color
plt.rc('grid', linestyle="-", color='gray')
plt.grid(True)
# Create the Plot
x_cnn= cnn['Tweets Ago']
y_cnn= cnn['Compound']
plt.plot(x_cnn, y_cnn, color='red', marker='o', linestyle='dashed', linewidth=2, markersize=12, label = "CNN")
plt.legend()

# Saving the plot
plt.savefig("cnn.png")

# Showing the plot
plt.show()
```


![png](output_20_0.png)



```python
# Create plot chart for FOX
now = datetime.now()
now = now.strftime("%m-%d-%Y")
plt.title(f"Sentiment Analysis of Tweets ({now}) for FOX")
plt.xlabel("Tweets Ago")
plt.ylabel("Tweet Polarity")
# Add the Grid and use gray as the grid Color
plt.rc('grid', linestyle="-", color='gray')
plt.grid(True)
# Create the Plot
x_fox= fox['Tweets Ago']
y_fox= fox['Compound']
plt.plot(x_fox, y_fox, color='blue', marker='o', linestyle='dashed', linewidth=2, markersize=12, label = "FOX")
plt.legend()

# Saving the plot
plt.savefig("fox.png")

# Showing the plot
plt.show()
```


![png](output_21_0.png)



```python
# Create plot chart for New York Times
now = datetime.now()
now = now.strftime("%m-%d-%Y")
plt.title(f"Sentiment Analysis of Tweets ({now}) for New York Times")
plt.xlabel("Tweets Ago")
plt.ylabel("Tweet Polarity")
# Add the Grid and use gray as the grid Color
plt.rc('grid', linestyle="-", color='gray')
plt.grid(True)
# Create the Plot
x_nyt= nyt['Tweets Ago']
y_nyt= nyt['Compound']
plt.plot(x_nyt, y_nyt, color='yellow', marker='o', linestyle='dashed', linewidth=2, markersize=12, label = "NYT")
plt.legend()

# Saving the plot
plt.savefig("newyorktimes.png")

# Showing the plot
plt.show()
```


![png](output_22_0.png)

