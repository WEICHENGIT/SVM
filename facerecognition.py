###############################################################################
#               Face Recognition Task
###############################################################################
"""
The dataset used in this example is a preprocessed excerpt
of the "Labeled Faces in the Wild", aka LFW_:

  http://vis-www.cs.umass.edu/lfw/lfw-funneled.tgz (233MB)

  _LFW: http://vis-www.cs.umass.edu/lfw/

"""

####################################################################
# Download the data (if not already on disk); load it as numpy arrays
lfw_people = fetch_lfw_people(min_faces_per_person=70, resize=0.4,
                              color=True, funneled=False, slice_=None,
                              download_if_missing=False)
# data_home='.'
# introspect the images arrays to find the shapes (for plotting)
images = lfw_people.images / 255.
n_samples, h, w, n_colors = images.shape
# the label to predict is the id of the person
target_names = lfw_people.target_names.tolist()

####################################################################
# Pick a pair to classify such as
# names = ['Tony Blair', 'Colin Powell']
names = ['Donald Rumsfeld', 'Colin Powell']
idx0 = (lfw_people.target == target_names.index(names[0]))
idx1 = (lfw_people.target == target_names.index(names[1]))
images = np.r_[images[idx0], images[idx1]]
n_samples = images.shape[0]
y = np.r_[np.zeros(np.sum(idx0)), np.ones(np.sum(idx1))].astype(np.int)

####################################################################
# Extract features
# features using only illuminations
X = (np.mean(images, axis=3)).reshape(n_samples, -1)
# or compute features using colors (3 times more features)
# X = images.copy().reshape(n_samples, -1)
# Scale features
X -= np.mean(X, axis=0)
X /= np.std(X, axis=0)

####################################################################
# Split data into a half training and half test set
# X_train, X_test, y_train, y_test, images_train, images_test = \
#    train_test_split(X, y, images, test_size=0.5, random_state=0)
# X_train, X_test, y_train, y_test = \
#    train_test_split(X, y, test_size=0.5, random_state=0)
indices = np.random.permutation(X.shape[0])
train_idx, test_idx = indices[:X.shape[0] / 2], indices[X.shape[0] / 2:]
X_train, X_test = X[train_idx, :], X[test_idx, :]
y_train, y_test = y[train_idx], y[test_idx]
images_train, images_test = images[train_idx, :, :, :], images[test_idx, :, :, :]

####################################################################
# Quantitative evaluation of the model quality on the test set
print ("Fitting the classifier to the training set")
t0 = time()
# fit a classifier,
clf = SVC(kernel='linear')
clf.fit(X_train, y_train)
print ("Predicting the people names on the testing set")
t0 = time()
# predict labels for the X_test images
y_pred = clf.predict(X_test)

print ("done in %0.3fs" % (time() - t0))
print ("Chance level : %s" % max(np.mean(y), 1. - np.mean(y)))
print ("Accuracy : %s" % clf.score(X_test, y_test))

####################################################################
# Look at the coefficients
pl.figure()
pl.imshow(np.reshape(clf.coef_, (h, w)))
pl.title("The coefficients estimated by SVC")

####################################################################
# Qualitative evaluation of the predictions using matplotlib
def plot_gallery(images, titles, n_row=3, n_col=4):
    """Helper function to plot a gallery of portraits"""
    pl.figure(figsize=(1.8 * n_col, 2.4 * n_row))
    pl.title("Classification results of Donald Rumsfeld and Colin Powell")
    pl.subplots_adjust(bottom=0, left=.01, right=.99, top=.90,
                       hspace=.35)
    for i in range(n_row * n_col):
        pl.subplot(n_row, n_col, i + 1)
        pl.imshow(images[i])
        pl.title(titles[i], size=12)
        pl.xticks(())
        pl.yticks(())

def title(y_pred, y_test, names):
    pred_name = names[int(y_pred)].rsplit(' ', 1)[-1]
    true_name = names[int(y_test)].rsplit(' ', 1)[-1]
    return 'predicted: %s\ntrue:      %s' % (pred_name, true_name)

prediction_titles = [title(y_pred[i], y_test[i], names)
                     for i in range(y_pred.shape[0])]

plot_gallery(images_test, prediction_titles)
pl.show()

###############################################################################
# QUESTION 5 : try various values for C in the SVC function and plot the error curve
###############################################################################

SCORE=[]
logC=[]
for i in range(-5,6):
    print("Fitting and predicting with linear kernel, C=%f" % 10**i)
    t0 = time()
    clf = SVC(kernel='linear',C=10**i)
    clf.fit(X_train, y_train)
    SCORE.append(clf.score(X_test, y_test))
    logC.append(i)
    print ("done in %0.3fs" % (time() - t0))
plt.close('all')
plt.figure(1)
plt.plot(logC,SCORE)
plt.xlabel("log C")
plt.ylabel("Score")
plt.title("The classification score with varying C, linear kernel")
plt.show()

###############################################################################
# QUESTION 6 : en ajoutant des variables de nuisances, montrez que la performance chute
###############################################################################

noise=np.random.normal(0,1,X_test.shape[0]*10000)
noise=np.reshape(noise,(X_test.shape[0],10000))
SCORE=[]
NOISE=[]
for i in range(0,10000,500):
    print("Fitting and predicting after adding %d nuisance features" % i)
    t0 = time()
    X_train_noise=np.c_[X_train,noise[0:X_train.shape[0],0:i]]
    X_test_noise=np.c_[X_test,noise[:,0:i]]
    clf = SVC(kernel='linear')
    clf.fit(X_train_noise, y_train)
    SCORE.append(clf.score(X_test_noise, y_test))
    NOISE.append(i)
    print ("done in %0.3fs" % (time() - t0))
plt.figure(2)
plt.plot(NOISE, SCORE)
plt.xlabel("Number of noise features added to training set")
plt.ylabel("Score")
plt.title("The classification score with different number of nuisances variables")
plt.show()

###############################################################################
# QUESTION 8.1 : quel est l’eﬀet du choix d’un noyau non-linéaire RBF sur la prédiction?
###############################################################################

SCORE=[]
logC=[]
for i in range(-5,6):
    print("Fitting and predicting with RBF kernel, C=%f" % 10**i)
    t0 = time()
    clf = SVC(kernel='rbf',C=10**i)
    clf.fit(X_train, y_train)
    SCORE.append(clf.score(X_test, y_test))
    logC.append(i)
    print ("done in %0.3fs" % (time() - t0))
plt.close('all')
plt.figure(3)
plt.plot(logC,SCORE)
plt.xlabel("log C")
plt.ylabel("Score")
plt.title("The classification score with varying C, RBF kernel")
plt.show()

###############################################################################
# QUESTION 8.2 : try reducing the dimension using sklearn.decomposition.RandomizedPCA and compute again an SVM classifer
###############################################################################

SCORE=[]
N_C=[]
for n_components in range(10,X_train.shape[0],20):
    t0 = time()
    pca = PCA( n_components=n_components, svd_solver='randomized',whiten=True)##problem!
#     print("Extracting the %d features from %d" % (pca.components_[0], X_train.shape[1])," with RandomizedPCA")
    X_train_PCA=pca.fit_transform(X_train)
    X_test_PCA=pca.transform(X_test)
    clf = SVC(kernel='linear',C=1)
    clf.fit(X_train_PCA, y_train)
    SCORE.append(clf.score(X_test_PCA, y_test))
    N_C.append(n_components)
    print ("done in %0.3fs" % (time() - t0))
plt.close('all')
plt.figure(4)
plt.plot(N_C,SCORE)
plt.xlabel("Number of components with PCA")
plt.ylabel("Score")
plt.title("The classification score with different feature dimension")
plt.show()

###############################################################################
# QUESTION 9 :
###############################################################################
clf = svm.SVC(kernel='linear',tol=0.001)
clf.fit(X_train, y_train)
plt.figure(5)
plt.plot(clf.coeffs_)
plt.show()
