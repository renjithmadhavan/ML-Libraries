from subprocess import Popen, PIPE, STDOUT

classpath = '~/weka.jar'
wekaFilter = 'weka.filters.unsupervised.attribute.AddCluster'
methode = 'weka.clusterers.SimpleKMeans -N 2'
inputFile = '../data/two_cluster.arff'

p = Popen(['java', '-classpath', classpath,
			'weka.filters.unsupervised.attribute.AddCluster', 
			'-W', methode, '-i', inputFile],
			shell=False, bufsize=1, stdout=PIPE, stderr=STDOUT, close_fds=True)

for line in iter(p.stdout.readline, b''):
    print line,
p.stdout.close()
p.wait()