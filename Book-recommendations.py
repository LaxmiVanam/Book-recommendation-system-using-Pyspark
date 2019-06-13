"""
__author = 'Laxmi Vanam'
__email = 'laxmivanam05@gmail.com'
"""

import sys
from pyspark import SparkConf, SparkContext
from math import sqrt


#Function to create a dictionary of Book IDs (ISBNs) and Book names
def loadBookNames():
    BookNames = {}
    with open("Book-recommendations/BX-Books.csv", encoding='ascii', errors='ignore') as f:
        next (f)
        for line in f:
            fields = line.strip('"').split('","')
            BookNames[fields[0]] = fields[1]
    return BookNames

#Function to filter duplicates that are resulted after self-join operation
def filterDuplicates( userRatings ):
    ratings = userRatings[1]
    (book1, rating1) = ratings[0]
    (book2, rating2) = ratings[1]
    return book1 < book2


#Function to explicitly extract the ratings as key value RDD where Key - Book pair and value is rating pair
def makePairs( userRatings ):
    ratings = userRatings[1]
    (book1, rating1) = ratings[0]
    (book2, rating2) = ratings[1]
    return ((book1, book2), (rating1, rating2))

#Function to compute cosine similarity metric for item based collaborative filtering
def computeCosineSimilarity(ratingPairs):
    numPairs = 0
    sum_xx = sum_yy = sum_xy = 0
    for ratingX, ratingY in ratingPairs:
        sum_xx += ratingX * ratingX
        sum_yy += ratingY * ratingY
        sum_xy += ratingX * ratingY
        numPairs += 1
    numerator = sum_xy
    denominator = sqrt(sum_xx) * sqrt(sum_yy)
    score = 0
    if (denominator):
        score = (numerator / (float(denominator)))
    return (score, numPairs)

#configuation to use all the cores of the computer and run as seperate executor using Spark's built-in cluster manager
conf = SparkConf().setMaster("local[*]").setAppName("BookSimilarities")
sc = SparkContext(conf = conf)

print("\nLoading Book names...")
nameDict = sc.broadcast(loadBookNames())

print("\nLoading Book data...")
ratingsDatawheader = sc.textFile("file:///SparkCourse/Book-recommendations/BX-Book-Ratings.csv")
header = ratingsDatawheader.first() #to exclude header row 

ratingsData =  ratingsDatawheader.filter(lambda line: line != header).map(lambda line: line.strip('"'))   

# Map ratings data to key-value pairs: user ID =>bookID, rating
ratings = ratingsData.map(lambda l: l.split(';')).map(lambda l: (l[0].strip('"'), (l[1].strip('"'), float(l[2].strip('"')))))

  
# find every pair of book rated by the same user using self-join (to find every combination).
joinedRatings = ratings.join(ratings)
 

# Filter out duplicate  pairs resulted from self join ( from the rdd format: userID => ((bookID, rating), (bookID, rating))

uniqueJoinedRatings = joinedRatings.filter(filterDuplicates)

# Now key by book pair and strip out the user information.
bookPairs = uniqueJoinedRatings.map(makePairs)

# We now have (book1, book2) => (rating1, rating2)
# Grouping by the book pair for all the ratings across the data set
bookPairRatings = bookPairs.groupByKey()


# Compute similarities on the RDD ((book1, book2) = > (rating1, rating2), (rating1, rating2) format)
bookPairSimilarities = bookPairRatings.mapValues(computeCosineSimilarity).cache()

# Extract similarities for the book we care about that are "good".

if (len(sys.argv) > 1):
   
    scoreThreshold = 0.95
    coOccurenceThreshold = 100

    bookID = int(sys.argv[1])

    # Filter for Books with this sim that are "good" as defined by
    # our quality thresholds above
    filteredResults = bookPairSimilarities.filter(lambda pairSim: \
        (pairSim[0][0] == bookID or pairSim[0][1] == bookID) \
        and pairSim[1][0] > scoreThreshold and pairSim[1][1] > coOccurenceThreshold)

    # Sort by quality score.
    results = filteredResults.map(lambda pairSim: (pairSim[1], pairSim[0])).sortByKey(ascending = False).take(10)

    print("Top 10 similar books for " + nameDict.value[bookID[1]])
    for result in results:
        (sim, pair) = result
        # Display the similarity result that isn't the book we're looking at
        similarBookID = pair[0]
        if (similarBookID == bookID):
            similarBookID = pair[1]
        print(nameDict.value[similarBookID[1]] + "\tscore: " + str(sim[0]) + "\tstrength: " + str(sim[1]))
        print(similarBookID + "\tscore: " + str(sim[0]) + "\tstrength: " + str(sim[1]))
