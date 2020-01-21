# Book-recommendation-system-using-Pyspark

The book recommendation system is based on the Item based collaborative filtering technique. The script is written using pyspark on top of Spark's built in cluster manager. It is used to recommend similar books to each other based on the ratings and the strength of the ratings. 

This is based on the concept that "Users who liked this item also liked …”   

Steps followed are as follows:

1. It will take a book and find the users who liked that book.
2. It then finds other books that similar users liked and form pairs of books that were read by a user
2. It then measures the similarity of thir ratings across all the users who read both
3. It takes items and outputs other items as recommendations sorting by strength of similarity.

Metric used:
Cosine similarity: Compute how similar two non-zero vectors (of ratings) are in order to determine the similarity score between two books.	

Future enhancements: 
- Adjust the thresholds for the number of co-raters and the minimum score
- The quality of the similarities can be improved with different similarity metrics (Pearson correlation coeffient/ Jaccard similarity metric etc.)
- Invent a new similarity metric that takes number of co-raters into consideration.
- Take the author of the books into consideration to boost the scores.

Dataset:
•BX-Books - Books are identified by their respective ISBN. Invalid ISBNs have already been removed from the dataset. Moreover, some content-based information is obtained from Amazon Web Services. 
•BX-Book-Ratings - Contains the book rating information. Ratings are either explicit, expressed on a scale from 1-10 (higher values denoting higher appreciation), or implicit, expressed by 0.

This is based on the publicly available Books crossing dataset pulled from :http://www2.informatik.uni-freiburg.de/~cziegler/BX/

Keywords- collaborative filtering,  recommendation systems, pyspark,cache, persist, broadcast variables, cosine similarity
