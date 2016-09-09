# CS4246

## Topic
Stress Detection with Gaussian Process

##  Develop
1. Setup three main branches `documentation`, `technical` and `program`. Individual work should further branch from these branches.
2. Edit your work and push any work in progress into the respective branches. This is so that any rebase does not affect other's work.
3. Push to master once you are complete.

## Folder Structure
* **learner**
  * all codes are written in python here.
  * mltools: python code adapted by UCI Machine Learning Class
  * data: datas that used in the code. X is the features consisted of MFCC and Magnitude Spectrum; Y is the outcome, obtained using PHQ8 test.
  * forests.py: code written to do prediction by random forests using development set after training the model using training set
  * forests_crossValidation.py: code written to do 10-fold cross-validation test by random forests using the data provided in training set
* **features_extractor/AudioFeaturesExtractor**
  * all codes are written in java here.
  * it is developed using Eclipse (Version: Luna Service Release 2 (4.4.2))
  * lib/: libraries included in the java project
  * src/: the java codes written by me using template provided in CS2108 class.
