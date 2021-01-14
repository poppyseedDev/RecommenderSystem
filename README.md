# The Implementation of a Simple eCommerce Recommender System (Jan 2021)

## Abstract

This article gives an oversight at how recommendation systems are implemented. In addition, it gives an insight into the methods used in building a simple eCommerce recommendation system. The data used for building it was a dataset of customer reviews of items that were scraped from Amazon’s website.
The recommendation system uses an item-item Collaborative filtering approach and compares Pearson's and Cosine correlation methods for finding similar items.

#### Index Terms

Artificial neural networks, big data applications, information filtering, recommender systems.

## Table of Contents

- [ Introduction ](#intro)
  - [ Collaborative filtering ](#collab)
  - [ Content-based filtering ](#content)
- [ How Recommender Systems Work ](#work)
- [ Implementations of Recommender Systems ](#implement)
- [ How Amazon’s Recommender System Works ](#amazon)
- [ My Implementation ](#myimple)
  - [ Limitations of the data ](#limits)
  - [ Matrix Sparsity ](#sparsity)
  - [ Reducing Sparsity of the data ](#reduce)
  - [ Implementing the machine learning algorithm ](#machine)
    - [ Pearson's correlation ](#pearsons)
    - [ Cosine similarity ](#cosine)
- [ Results ](#results)
  - [ Accuracy ](#accuracy)
- [ Conclusion ](#conclusion)

<a name="intro"></a>

## Introduction

Recommender systems are a subclass of information filtering systems that are increasingly important today. Simply put, they are a machine learning algorithm that utilizes Big Data to suggest products or items to customers. [14]

They are a vital part of many computer oriented companies. All the major social platforms such as Facebook, Twitter and Instagram use them as well as most eCommerce sites such as Amazon and Wish. They also represent a large part of their profits. The largest eCommerce platform Amazon reportedly generates 35% of its revenue from its recommendation engine [15].

Moreover, the second largest eCommerce platform Wish generates almost all its profits solely from its recommendation engine. Therefore, developing a good recommendation system is all the more critical nowadays.

Large pools of data are vital for implementing such a system. The more data points there are collected on a specific user the more accurate the suggestions can be. Of course, with the growth of data simultaneously rises the need for a more advanced algorithm and the selection of right data points that don’t end up confusing our algorithm, but assist it in making a better solution.

Recommendation systems also play a critical role in other web-based platforms. For example, in boosting engagement and improving retention. Although in this article, I will be focusing more on the recommender systems used for eCommerce sites.

<a name="work"></a>

## How recommender system works

There are several approaches we can take when building recommender systems. [16]

Recommendation systems can be broadly classified into 2 types:

- Collaborative Filtering
- Content-Based Filtering

A more thorough division further classifies them into demographic based filtering, utility-based filtering, knowledge-based filtering etc. [17] Of course the best approach is hybrid filtering, where you combine two or more different recommender techniques to create a better prediction model.

<a name="collab"></a>

### Collaborative filtering

It is based on the premise that similar people like similar things. It solely relies on users' past activity on the platform. The more the user interacts with the platform the more accurate it becomes.

This filtering method does not know who you are, what you like or what are the specifics of the items you like. It only looks at the interaction a specific user has and compares it with the interactions of other users.

The core data of a collaborative filtering model is the user-item matrix or so-called utility matrix. Each number in the given matrix represents the users rating of a specific item.

See image below:

We can have two general approaches on how to implement collaborative filtering.

1. User-User Collaborative Filtering: Compares similar users based on the items they have chosen.
2. Item-Item Collaborative Filtering: Compares similar items together based on the past ratings the users have given.

We can have two general approaches on how to implement collaborative filtering.

1. Explicit data: This is in the form of a rating score that ranges from 1 to 5.

2. Implicit data: This type of data does not directly imply how the user feels about a specific item rather its inference is indirect. For example how many hours a user played a game on Steam (a video game buying platform) or how many hours a user listened to a specific track.

Generally, working with implicit data is better, because you don’t get the active user involved. In addition, explicit data is very noisy, because people tend to give it an aspirational rating [18]. For example, in the case of Netflix users will give a documentary 5 stars, although they would rather watch a comedy show more often, even though they gave that show a lower rating.
<a name="content"></a>

### Content-based filtering

This approach relies only on domain knowledge of items and users and it does not take into account their past interactions. This approach finds a solution based on the similarity of the content of users and items.

User features take into consideration for example the user's age, his or her gender, what types of categories the user likes, etc. Item features on the other hand can for example include the price of the item, its category, the description or its given keywords.
<a name="implement"></a>

## Implementations of Recomener systems

Generally collaborative filtering is the best approach for implementing a recommendation system [18]. Other approaches are also preferable if you have ways of hybridizing and combining it with the main recommendation system. Especially, they can be useful in cold starting when you don’t have enough information about the user.

There are a few more important steps you have to take into consideration, before you implement the main algorithm. The most important is data preprocessing (which includes outlier removal, denoising etc.). Due to size issues you also have to take into consideration ‘smart’ dimensionality reduction using MF (matrix factorization) or SVD (singular value decomposition).
<a name="amazon"></a>

## How Amazon’s Recommender System Works

Amazon’s recommendation system is based on item-item collaborative filtering. It is working with massive datasets and produces recommendations in real-time. This type of filtering matches each of the users' purchased and rated items to similar items then combines those similar items into a recommendation list for the user. [21]

Interestingly, it makes most conversions via email, where they present you with bundles of items from different categories where you have been browsing.

Due to Amazon being a private company, the complete mechanisms of how exactly their state of the art recommendation system works are not known. Although, it is known that they look at excessive amounts of data points, some of which include:
Purchased items/shopping carts (most focus is put into this one)
All user and item features
Abandoned carts
Dwell times (time before you click back on an item)
Cursor location
Web search history - Amazon tracks you outside of its web platform
Amazon search history
Etc.

Amazon also uses different pipelines to test out their different approaches and then deploy those which perform the best. [20]

Amazon’s article from 2003 [19] talks about Amazon taking an item-item collaborative filtering approach. It mentions that collaborative filtering is computationally expensive with O(NM), where M is the number of customers and N is the number of product catalog items. Although scanning every customer can be approximated with O(M), because almost all customers contain a small number of items. Further reading also discusses discarding customers with few purchases to avoid performance issues.

Getting a basic understanding on how Amazon’s recommendation system was implemented gave me a better insight into how to go about making my own.

<a name="dataset"></a>

## Dataset I Will be working with

In this article I will present my findings that I gathered implementing a simple recommendation system from scratch from the Amazon Product Review dataset (source: https://cseweb.ucsd.edu/~jmcauley/datasets.html ).
The dataset has been generated from web crawling the Amazon website. The larger dataset from 2018 includes 34Gb of raw review data, which includes 233.1 million reviews and ratings from more than 40 million users and more than 20 million items. Its metadata includes reviews and ratings,

timestamps, prices and categories.

<a name="myimple"></a>

## My Implementation

The recommender system I built also implements an item-item collaborative filtering approach similar to the first Amazon recommender system.

However, it uses far less data points. A large part of the reason why, is that the dataset that I have used was obtained by scraping the information from Amazon’s openly available data. Therefore, it was impossible to obtain information about people’s buying decisions, which make for the best predictions in real life. The only data points you could obtain from users were the reviews they wrote and even those were scarce.

For my implementation of a simple recommender system I took into considerations only the explicit reviews, id of the item and the user. To recommend new items for a specific user the algorithm looks at the items the user has rated positively (from a rating from 1 to 5 it chooses only those above 3). Afterwards, it looks at the similarity of each item compared to other items. Lastly, it picks all the items that are most similar to each of the items and returns a list of five most similar items recommended for the chosen user.

Due to limitations with my computer's RAM I only worked with a ‘smaller’ dataset in only one specific category.

<a name="limits"></a>

### Limitations of the data

As mentioned in the sections above, no matter how good is the implementation of the recommender system the nature of the data I was working with poses a limitation to how accurate it can be.

The biggest discrepancy between the data extracted from the web crawler versus the data that Amazon possesses is that not all users are being accounted for, because not all users rate products. Secondly, there is no information provided on the past purchases of users, which is the most important data point on which eCommerce recommendation systems are built upon.

<a name="sparsity"></a>

### Matrix Sparsity

A common challenge with real-world ratings data is that most users will not have interacted with (in our case rated) most items, and most items will only have been rated by a small number of users. This results in a very empty or sparse user-item matrix.

The formula for calculating matrix sparsity is the following:

#(total no. of elements) = #(no. of unique items) \* #(no. of unique users)
Sparsity = 1 - #(of ratings)/ #(total no. of elements)

There occur several problems with very sparse matrices. Firstly, there arises a problem regarding time and space complexity. A very large matrix requires a lot of memory storage and if it can fit into memory there is still a problem with time complexity, since most computations performed will involve adding or multiplying zero values together. There are some workarounds to these two issues, for example using alternate data structures to represent the data. However, due to time restraints I did not implement such solutions. [22]

Furthermore, there is an issue that if the sparsity above is 99.5% collaborative filtering might not be the right option. This is a number researchers have chosen based on past personal experience. [24]

<a name="reduce"></a>

### Reducing Sparsity of the data

The analysis of the small subset of data in the Amazon Fashion category revealed a matrix sparsity of 99.99937% far from the 99.5% threshold

| Name              | Value       |
| ----------------- | ----------- |
| Number of ratings | 883636      |
| Number of items   | 186189      |
| Number of users   | 749233      |
| Matrix sparsity   | 99.999367 % |

There are three approaches we can take to reduce matrix sparsity. Either we reduce users to those who have rated more items than a certain threshold or we reduce items that have been rated a certain number of times or we reduce both.

In trying out different approaches I found that the best approach is to implement an item threshold if you want to implement an item-item based recommender system and a user threshold if you want to implement an user-user based system. For example, if you want to compare different items together you eliminate those who have been rated a certain number of times. Doing this also results in more accurate predictions.

In the dataset I was working with I have implemented an item threshold above 300. That meant that every item had to be rated by at least 300 different users. Only that change resulted in an decrease of matrix sparsity to 99.4105%.

| Name              | Value       |
| ----------------- | ----------- |
| Number of ratings | 138200      |
| Number of items   | 197         |
| Number of users   | 113357      |
| Matrix sparsity   | 99.410545 % |

This simple change also resulted in a big change in the average rating of an item distribution to a more gaussian like distribution.

However, the change of the distribution in the average rating per user worsened. It is possible to correct this with further reduction of user data. Although, that would bring a further reduction in data, which is not a desirable outcome.

<a name="machine"></a>

### Implementing the machine learning algorithm

In the implementation of the machine learning algorithm I tried two main approaches. The first one uses Pearson's correlation coefficient and the second one Cosine similarity.

<a name="pearsons"></a>

#### Pearson's correlation

The Pearson’s Correlation coefficient is a statistic that measures linear correlation between two variables X and Y [23]. In the case of collaborative filtering those two variables are the two rows of the item-user matrix. Representing two ratings from two different items.

Its formula is the following.

<a name="cosine"></a>

#### Cosine similarity

Cosine similarity on the other hand is a measurement between two non-zero vectors of an inner product space. It is defined to be equal to the cosine of the angle between them. (source: https://en.wikipedia.org/wiki/Cosine_similarity )

The formula is the following.

With cosine similarity you have to be careful that all of your values are defined. All the non-zero values should be changed into zero values. With doing so arises an issue in time complexity as now almost all values have to be computed, although only several of them are defined.
<a name="results"></a>

## Results

Interestingly, both methods couldn’t produce new recommendations for the majority of users. To speed up the process I removed all the users that only made one rating, because they were more likely not to have any predictions at all.

If I were to compute all of the data on my computer for all users using Pearson’s correlation it would have taken approx. 7 hours and even more for Cosine corr. That is why the data was tested with a different approach. The list of all unique users was shuffled to prevent unnecessary similarities and then it was computed for only ten thousand of users.

In testing both approaches it was overwhelmingly evident that Pearson's correlation was a much faster method. This is mostly due to me not optimizing the code and to increased time complexity due to the additional calculations with zero values.

From a glance at the graphs above it is evident that Pearsons similarity was able to recommend more items to users in comparison to the Cosine similarity. This is most likely due to harsher restrictions Cosine similarity was given.

<a name="accuracy"></a>

### Accuracy

Implementing accuracy measurements is not as straightforward in collaborative filtering as it is for example in calculating the accuracy of neural networks. There you can simply split the data into testing and training data and then compare your solution to the training data. However, in collaborative filtering doing so is a bit harder, since the data is interdependent. The best approach is to remove random data points and use the algorithm to predict them. However, this dataset was unfortunately too sparse and this method would not be the most reliable.

<a name="conclusion"></a>

## Conclusion

The most issues I had implementing the recommender system was with memory. Generally that is because the whole dataset is loaded simultaneously into the RAM. That is also one of the reasons I chose to work with a smaller dataset subcategory. If you have even bigger datasets it is crucial to implement dimensionality reduction for the user-item matrix. A good workaround with RAM issues is to use Dask - a python library used for storing your dataset in the memory on your hard disk.

Many improvements could be done to this recommendation system. One of the things is introducing more data points and with them more advanced methods. These range from deep learning, LSTMs to adding in sentiment analysis for turning comments into keywords and then using those as data points.

In retrospect it would be better implementing a content-based filtering system for such sparse data or a combination of both.

## References

- Collaborative system with implicit feedback
  https://github.com/AudreyGermain/Game-Recommendation-System#als
- https://medium.com/radon-dev/als-implicit-collaborative-filtering-5ed653ba39fe
- implicit vs explicit feedback
  https://www.digitalvidya.com/blog/collaborative-filtering/
- all you need to know about collaborative filtering
  https://www.digitalvidya.com/blog/collaborative-filtering /
- cosine similarity
  https://www.youtube.com/watch?v=h9gpufJFF-0&feature=emb_logo
- centered cosine similarity
  https://towardsdatascience.com/using-cosine-similarity-to-build-a-movie-recommendation-system-ae7f20842599 - cosine similarity
  https://www.analyticsvidhya.com/blog/2020/08/recommendation-system-k-nearest-neighbors/
- k nearest neighbour
  https://github.com/mandeep147/Amazon-Product-Recommender-System/blob/master/Recommender%20System/Recommender%20System.ipynb
- product recommender system na mojih podatkih
  https://www.youtube.com/watch?v=v_mONWiFv0k - recommender systems with neural networks

My dataset

- https://nijianmo.github.io/amazon/index.html
  Ups and downs: Modeling the visual evolution of fashion trends with one-class collaborative filtering, R. He, J. McAuley, WWW, 2016 pdf
- Image-based recommendations on styles and substitutes, J. McAuley, C. Targett, J. Shi, A. van den Hengel, SIGIR, 2015
  https://sigmoidal.io/recommender-systems-recommendation-engine/
- https://www.mckinsey.com/industries/retail/our-insights/
- how-retailers-can-keep-up-with-consumers
- https://www.analyticsvidhya.com/blog/2020/08/recommendation-system-k-nearest-neighbors/
- https://sigmoidal.io/recommender-systems-recommendation-engine/
- https://www.youtube.com/watch?v=bLhq63ygoU8&feature=emb_logo
- https://www.cs.umd.edu/~samir/498/Amazon-Recommendations.pdf
- https://stackoverflow.com/questions/2323768/how-does-the-amazon-recommendation-feature-work
- https://www.cs.umd.edu/~samir/498/Amazon-Recommendations.pdf
- https://machinelearningmastery.com/sparse-matrices-for-machine-learning/
- https://en.wikipedia.org/wiki/Pearson_correlation_coefficient
- https://stats.stackexchange.com/questions/367380/why-the-maximum-sparsity-is-99-5-in-collaborative-filtering
