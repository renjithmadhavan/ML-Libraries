from subprocess import Popen, PIPE, STDOUT

# Settings to perform Naive Bayes classifier.
classpath = '/Applications/weka-3-6-9/weka.jar'
evaluationClass = 'weka.classifiers.bayes.NaiveBayes'
trainFile = '../data/fisheriris.arff'
testFile = '../data/fisheriris.arff'

# Naive Bayes classifier.
p = Popen(['java', '-classpath', classpath,
			evaluationClass, 
			'-t', trainFile, '-T', testFile, '-p', '1'],
			shell=False, bufsize=1, stdout=PIPE, stderr=STDOUT, close_fds=True)

# Show predict labels.
for line in iter(p.stdout.readline, b''):
    print line,
p.stdout.close()
p.wait()