"""
/* ******************************************************************************************
 * Name - Siddharth Baronia 
 *
 * Problems - 1. Use a word2vec model on larger corpus
 *            2. Find vec representation of unique words in review
 *            3. Find synonyms of few words
 *            4. Run KMeans on vectors to cluster them on 2000 clusters
 *            5. Explore some clusters to see what words are in them
 * ********************************************************************************************/ 
"""

import sys,json,string,random
from pyspark import SparkConf, SparkContext
from pyspark.sql import SQLContext, DataFrame
from pyspark.sql.types import StructType, StructField, StringType
from pyspark.mllib.feature import Word2Vec,Word2VecModel
from pyspark.mllib.clustering import KMeans, KMeansModel

'''sbaronia - remove punctuation from every review'''
def clean_string_to_words(line):

	line_review = (line).lower()

	for st in string.punctuation:
		line_review = line_review.replace(st, ' ')

	words_list = line_review.lower().split(' ')	
	words_list = filter(None,words_list)

	return words_list


def main():
	inputs = sys.argv[1]
	output = sys.argv[2] 

	conf = SparkConf().setAppName('Word2Vec and KMeans')
	sc = SparkContext(conf=conf)
	assert sc.version >= '1.5.1'

	sqlContext = SQLContext(sc)

	'''sbaronia - get reviews from json and filter those with no reviews'''
   	review = sqlContext.read.json(inputs).select('reviewText').cache()
   	review_df = review.filter(review.reviewText != "").cache()

   	clean_words_rdd = review_df.map(lambda review: clean_string_to_words(review.reviewText)).cache()

   	'''sbaronia - fit word2vec model on our vocabulary of clean words'''
   	word2vec = Word2Vec()
   	word2vec_model = word2vec.fit(clean_words_rdd) # this will use all rdd and will make vocabulary

   	unique_words = word2vec_model.getVectors().keys()

   	'''sbaronia - save the model in word2vec directory of output directory'''
   	word2vec_model.save(sc,output + '/word2vec')

   	'''sbaronia - for some hardcoded words find 5 synonyms'''
   	words = ['dog','happy','litter']
   	for word in words:
   		print "The synonyms for " + word + " are :"
		syn = word2vec_model.findSynonyms(word,5)
		for word_x in syn:
			print str(word_x)

	'''sbaronia - randomly select 4 words and find their synonyms'''
	for i in range(4):
		rand_word = random.choice(unique_words)
		print "The synonyms for " + rand_word + " are :"
		syn = word2vec_model.findSynonyms(rand_word,5)
		for word_x in syn:
			print str(word_x)
   		
	'''sbaronia - make and rdd of vectors for each word in vocabulary'''
	vectors = []
	for word in unique_words:
		vectors.append(word2vec_model.transform(word))

	vectors_rdd = sc.parallelize(vectors).cache()

	'''sbaronia - run KMeans on it to form 2000 clusters 
	and save model in kmean directory of output directory'''
	kmeans_model = KMeans.train(vectors_rdd,2000)

	kmeans_model.save(sc,output + '/kmean')

	'''sbaronia - create a dictionary with keys as cluster and values 
	as words falling in that cluster'''
	word_index = {}
	clusters = []
	for word in unique_words:
		clus = kmeans_model.predict(word2vec_model.transform(word))
		if clus in clusters:
			word_index[clus].append(str(word))
		else:
			wordlist = []
			wordlist.append(str(word))
			word_index[clus] = wordlist
			clusters.append(clus)

	print word_index

	'''sbaronia - randomly select few clusters and print their words'''
	for i in range(8):
		cluster = random.choice(clusters)
		print cluster, word_index[cluster]

if __name__ == "__main__":
	main()


'''
To run on local -

spark-submit --master local[*] word2vec_q45.py data/reviews_Pet_Supplies_p2.json output45

model will be saved in word2vec and kmean folders in output45

Output - 

Using reviews_Pet_Supplies_p1.json only

The synonyms for dog are :
pup: 1.38317501162
pooch: 1.37720318286
cat: 1.28533465289
puppy: 1.1653157311
ferret: 1.09895399789

The synonyms for happy are :
pleased: 2.22089014794
satisfied: 2.19366785389
impressed: 1.95318911505
unhappy: 1.8864321259
thrilled: 1.83690929979

The synonyms for kitten are :
kitty: 1.45319940685
cat: 1.42840548995
girl: 1.29793940579
bengal: 1.24138813814
ferret: 1.22397970295

The synonyms for litter are :
box: 1.90698814725
litterbox: 1.88490670409
slaving: 1.76672134258
scoop: 1.75982686355
littler: 1.75184262171

========================

Using reviews_Pet_Supplies_p2.json

The synonyms for dog are :
(u'pooch', 1.4016182595588218)
(u'pup', 1.3044839400496864)
(u'cat', 1.0509527490714636)
(u'furbaby', 1.0503207236715408)
(u'horse', 1.0480330767334711)

The synonyms for happy are :
(u'pleased', 2.3426664676296838)
(u'satisfied', 2.3144723664435185)
(u'unhappy', 2.0786175837086351)
(u'impressed', 1.9957862330042622)
(u'statisfied', 1.9841318477722309)

The synonyms for kitten are :
(u'cat', 1.4783837084719882)
(u'kitty', 1.3880662648589148)
(u'bengal', 1.3770017550130971)
(u'siamese', 1.3717691957112115)
(u'kittens', 1.3481719358413875)

The synonyms for litter are :
(u'littler', 2.4640223631346432)
(u'liter', 2.2136412146107687)
(u'litterbox', 2.0605192329370303)
(u'sifting', 2.0424742595651004)
(u'catbox', 1.9809967068440599)

The synonyms for good are :
(u'decent', 1.5123157119995243)
(u'great', 1.4330993137383294)
(u'terrific', 1.1960650606868015)
(u'fantastic', 1.1871541610006433)
(u'nice', 1.1676777251829813)

The synonyms for bystander are :
(u'innocent', 0.31127552978340395)
(u'instigator', 0.30738969938510519)
(u'ordinance', 0.30736813889466569)
(u'runaway', 0.30349453710126661)
(u'screech', 0.30340137985047538)

The synonyms for instructive are :
(u'intimidating', 0.4429196598152958)
(u'vague', 0.4381684148941381)
(u'disconcerting', 0.42388494605327542)
(u'cryptic', 0.40761615485371711)
(u'unclear', 0.40647547469357631)

The synonyms for earz are :
(u'frontosa', 0.13951092931695147)
(u'phisohex', 0.12953862705534652)
(u'arowanas', 0.11966135412107774)
(u'softies', 0.11914127946098721)
(u'cheeseburger', 0.11776974974356795)

The synonyms for hammering are :
(u'pounding', 0.89357256230017457)
(u'compressing', 0.87097648555902907)
(u'straightening', 0.85540325232037628)
(u'snapping', 0.84702610119285204)
(u'banging', 0.84215300660452808)


From KMeans some random clusters and words in it:
===================================================

0: ['microlactin', 'dkh', '444', 'liberate', 'graduations', '335', 'iodoform', '147', '9s', 'mgyucca', '223', 'mgchondroitin', 'puddin', '75mg', 'atcc', '324', 'stats', '6000', 'enrollment', '030', 'cals', 'translates', '343', 'kilograms', 'rhf', '6ml', 'cored', 'canon', 'nucleic', 'microbiology', '273', 'fractions', 'felis', '670', 'boron', 'inflamma', 'coature', 'proline', '227', 'vb', 'ursi', '224', '25mg', '1the', 'chopped1', '460', 'cu', '5cc', 'mgl', '390', '125mg', '025', 'poopday', '2tsp', 'sachets', '325', 'linolenic', 'iuvitamin', '1391', 'dailyover', '192', '5901', '4mg', '0mg', 'basal', '480', '18oz', '000mg', 'zeolites', 'bagfor', '6mm', 'pipettes', 'surg', '3w', '144', 'regeneration', '340', '720', '35oz', '14oz', '347', 'mgascorbic', 'fibre', 'mgn', '510', 'b5', 'mgtaurine', '10ml', 'brights', 'milligrams', '290', 'bd', 'vol', '8000', 'sg', '9oz', 'lipoic', '1mg', 'mgmanganese', 'bottlei', 'minimumomega', '30p', 'reesei', 'kj', 'exposures', '525', 'gd', 'hemicellulase', 'counti', '760', 'herndon', '630', '354', 'multiply', '430', 'decaf', 'lactase', 'minimumcrude', 'nicotinic', 'mgcranberry', '1985', 'cfu', '24oz', 'pints', 'yields', '345', 'mardel', 'maracyn', '12the', 'kgtaurine', '2ml', '211', 'zn', 'cl', '0z', '380', '9632', '440', 'mgzinc', 'fip', 'fa', '1600', '1tsp', 'pipette', 'dgh', 'ichon', 'equaled']
1: ['consistency', 'thickness', 'shape', 'softness', 'texture']
1998: ['tempted'], 1999: ['outages', 'failures', 'outage']}
1818 ['buildings', 'keyboard', 'stairway', 'staircase', 'adjacent', 'bookcase', 'farthest', 'overlooking', 'adjoining', 'railings', 'fireplace', 'weaving', 'bridge', 'furthest', 'piano', 'railing', 'walkway', 'doorways', 'accross', 'cinder', 'beams', 'strategic', 'strategically', 'elevator', 'drawers', 'block', 'accessing', 'obstacles', 'archway', 'lanai', 'mantle', 'vertically', 'baseboard', 'cellar', 'radiator', 'blinds', 'backdoor', 'entrances', 'windowsill', 'hallways', 'horizontally', 'unprotected']
636 ['masters', 'lactating', 'compete', 'tnr', 'cousins', 'wormed', 'lives', 'domesticated', 'seniors', 'orphan', 'fostered', 'reside', 'rescuing', 'brood', 'herd', 'residents', 'companionship', 'citizens', 'generations', 'share', 'ferral', 'departed', 'queens', 'littermates', 'kittehs', 'diggers', 'deceased', 'cattery', 'orphaned', 'adoptions', 'appetites', 'mates']
950 ['fluffier', 'looser', 'drier', 'firmer', 'denser', 'softer']
1556 ['butternut', 'salad', 'husk', 'choy', 'kale', 'grape', 'bok', 'hulls', 'pot', 'mustard', 'pineapple', 'raisins', 'dandelion', 'leafy', 'chard', 'chili', 'cabbage', 'finely', 'squash', 'sprouts', 'lettuce', 'walnuts', 'celery', 'berries', 'leaf', 'chia', 'grapes', 'pasta', 'chopped', 'millet', 'groats', 'almonds', 'peppers', 'papaya']
87 ['cells']
8 ['widest', 'hardest']
1477 ['bit', 'tiny', 'lil', 'teeny', 'little', 'tad']
198 ['pliers', 'needle']
1999: ['outages', 'failures', 'outage']
'''

