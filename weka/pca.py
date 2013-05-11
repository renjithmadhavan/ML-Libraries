from subprocess import Popen, PIPE, STDOUT

# Settings to perform PCA.
classpath = '~/weka.jar'
filterClass = 'weka.filters.supervised.attribute.AttributeSelection'
searchClass = 'weka.attributeSelection.Ranker'
evaluationClass = 'weka.attributeSelection.PrincipalComponents -R 1.0 -C -A -1'
inputFile = '../data/ingredients.arff'

# Perform PCA.
p = Popen(['java', '-classpath', classpath,
			filterClass, 
			'-S', searchClass, '-E', evaluationClass, '-i', inputFile],
			shell=False, bufsize=1, stdout=PIPE, stderr=STDOUT, close_fds=True)

# Show transform data into eigenvector basis.
for line in iter(p.stdout.readline, b''):
    print line,
p.stdout.close()
p.wait()