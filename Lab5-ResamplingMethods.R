# *****************************************************************************
# Lab 5: Resampling Methods ----
#
# Course Code: BBT4206
# Course Name: Business Intelligence II
# Semester Duration: 21st August 2023 to 28th November 2023
#
# Lecturer: Allan Omondi
# Contact: aomondi [at] strathmore.edu
#
# Note: The lecture contains both theory and practice. This file forms part of
#       the practice. It has required lab work submissions that are graded for
#       coursework marks.
#
# License: GNU GPL-3.0-or-later
# See LICENSE file for licensing information.
# *****************************************************************************

# **[OPTIONAL] Initialization: Install and use renv ----
# The R Environment ("renv") package helps you create reproducible environments
# for your R projects. This is helpful when working in teams because it makes
# your R projects more isolated, portable and reproducible.

# Further reading:
#   Summary: https://rstudio.github.io/renv/
#   More detailed article: https://rstudio.github.io/renv/articles/renv.html

# "renv" It can be installed as follows:
# if (!is.element("renv", installed.packages()[, 1])) {
# install.packages("renv", dependencies = TRUE,
# repos = "https://cloud.r-project.org") # nolint
# }
# require("renv") # nolint

# Once installed, you can then use renv::init() to initialize renv in a new
# project.

# The prompt received after executing renv::init() is as shown below:
# This project already has a lockfile. What would you like to do?

# 1: Restore the project from the lockfile.
# 2: Discard the lockfile and re-initialize the project.
# 3: Activate the project without snapshotting or installing any packages.
# 4: Abort project initialization.

# Select option 1 to restore the project from the lockfile
# renv::init() # nolint

# This will set up a project library, containing all the packages you are
# currently using. The packages (and all the metadata needed to reinstall
# them) are recorded into a lockfile, renv.lock, and a .Rprofile ensures that
# the library is used every time you open the project.

# Consider a library as the location where packages are stored.
# Execute the following command to list all the libraries available in your
# computer:
.libPaths()

# One of the libraries should be a folder inside the project if you are using
# renv

# Then execute the following command to see which packages are available in
# each library:
lapply(.libPaths(), list.files)

# This can also be configured using the RStudio GUI when you click the project
# file, e.g., "BBT4206-R.Rproj" in the case of this project. Then
# navigate to the "Environments" tab and select "Use renv with this project".

# As you continue to work on your project, you can install and upgrade
# packages, using either:
# install.packages() and update.packages or
# renv::install() and renv::update()

# You can also clean up a project by removing unused packages using the
# following command: renv::clean()

# After you have confirmed that your code works as expected, use
# renv::snapshot(), AT THE END, to record the packages and their
# sources in the lockfile.

# Later, if you need to share your code with someone else or run your code on
# a new machine, your collaborator (or you) can call renv::restore() to
# reinstall the specific package versions recorded in the lockfile.

# [OPTIONAL]
# Execute the following code to reinstall the specific package versions
# recorded in the lockfile (restart R after executing the command):
# renv::restore() # nolint

# [OPTIONAL]
# If you get several errors setting up renv and you prefer not to use it, then
# you can deactivate it using the following command (restart R after executing
# the command):
# renv::deactivate() # nolint

# If renv::restore() did not install the "languageserver" package (required to
# use R for VS Code), then it can be installed manually as follows (restart R
# after executing the command):

if (require("languageserver")) {
  require("languageserver")
} else {
  install.packages("languageserver", dependencies = TRUE,
                   repos = "https://cloud.r-project.org")
}

# Introduction ----
# Resampling methods are techniques that can be used to improve the performance
# and reliability of machine learning algorithms. They work by creating
# multiple training sets from the original training set. The model is then
# trained on each training set, and the results are averaged. This helps to
# reduce overfitting and improve the model's generalization performance.

# Resampling methods include:
## Splitting the dataset into train and test sets ----
## Bootstrapping (sampling with replacement) ----
## Basic k-fold cross validation ----
## Repeated cross validation ----
## Leave One Out Cross-Validation (LOOCV) ----

# STEP 1. Install and Load the Required Packages ----
## caret ----
if (require("caret")) {
  require("caret")
} else {
  install.packages("caret", dependencies = TRUE,
                   repos = "https://cloud.r-project.org")
}

## klaR ----
if (require("klaR")) {
  require("klaR")
} else {
  install.packages("klaR", dependencies = TRUE,
                   repos = "https://cloud.r-project.org")
}

## e1071 ----
if (require("e1071")) {
  require("e1071")
} else {
  install.packages("e1071", dependencies = TRUE,
                   repos = "https://cloud.r-project.org")
}

## readr ----
if (require("readr")) {
  require("readr")
} else {
  install.packages("readr", dependencies = TRUE,
                   repos = "https://cloud.r-project.org")
}

## LiblineaR ----
if (require("LiblineaR")) {
  require("LiblineaR")
} else {
  install.packages("LiblineaR", dependencies = TRUE,
                   repos = "https://cloud.r-project.org")
}

## naivebayes ----
if (require("naivebayes")) {
  require("naivebayes")
} else {
  install.packages("naivebayes", dependencies = TRUE,
                   repos = "https://cloud.r-project.org")
}




if (!is.element("mlbench", installed.packages()[, 1])) {
  install.packages("mlbench", dependencies = TRUE)
}
require("mlbench")


# Load the dataset
data(PimaIndiansDiabetes)
dataset <- PimaIndiansDiabetes

# Split the dataset
set.seed(123)
trainIndex <- createDataPartition(dataset$diabetes, p = 0.75, list = FALSE)
trainData <- dataset[trainIndex, ]
testData <- dataset[-trainIndex, ]


# Naive Bayes Training
model_nb <- naiveBayes(diabetes ~ ., data = trainData)

# Linear Model Training
model_lm <- train(diabetes ~ ., data = trainData, method = "glm", trControl = trainControl(method = "none"))

# Naive Bayes Testing
predictions_nb <- predict(model_nb, testData)

# Linear Model Testing
predictions_lm <- predict(model_lm, testData)


# Naive Bayes Evaluation
confMatrix_nb <- confusionMatrix(predictions_nb, testData$diabetes)
print(confMatrix_nb)

# Linear Model Evaluation
confMatrix_lm <- confusionMatrix(predictions_lm, testData$diabetes)
print(confMatrix_lm)



# k-fold Cross Validation for Linear Model
trainControl <- trainControl(method = "cv", number = 10)
model_lm_cv <- train(diabetes ~ ., data = trainData, method = "glm", trControl = trainControl)
print(model_lm_cv)

# Leave One Out Cross Validation (LOOCV) for Naive Bayes
trainControl <- trainControl(method = "LOOCV")
model_nb_loocv <- train(diabetes ~ ., data = trainData, method = "naive_bayes", trControl = trainControl)
print(model_nb_loocv)

# Repeated k-fold Cross Validation for Naive Bayes
trainControl <- trainControl(method = "repeatedcv", number = 10, repeats = 3)
model_nb_repeatedcv <- train(diabetes ~ ., data = trainData, method = "naive_bayes", trControl = trainControl)
print(model_nb_repeatedcv)

# Bootstrapping for Naive Bayes
trainControl <- trainControl(method = "boot", number = 500)
model_nb_boot <- train(diabetes ~ ., data = trainData, method = "naive_bayes", trControl = trainControl)
print(model_nb_boot)





# [OPTIONAL] **Deinitialization: Create a snapshot of the R environment ----
# Lastly, as a follow-up to the initialization step, record the packages
# installed and their sources in the lockfile so that other team-members can
# use renv::restore() to re-install the same package version in their local
# machine during their initialization step.
# renv::snapshot() # nolint

# References ----
## Brown, M. (2014). Dow Jones index (Version 1) [Dataset]. University of California, Irvine (UCI) Machine Learning Repository. https://doi.org/10.24432/C5788V # nolint ----

## Ferreira, R., Martiniano, A., Ferreira, A., Ferreira, A., & Sassi, R. (2017). Daily demand forecasting orders (Version 1) [Dataset]. University of California, Irvine (UCI) Machine Learning Repository. https://doi.org/10.24432/C5BC8T # nolint ----

## Iranian churn dataset (Version 1). (2020). [Dataset]. University of California, Irvine (UCI) Machine Learning Repository. https://doi.org/10.24432/C5JW3Z # nolint ----

## National Institute of Diabetes and Digestive and Kidney Diseases. (1999). Pima Indians Diabetes Dataset [Dataset]. UCI Machine Learning Repository. https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database # nolint ----

## Yeh, I.-C. (2016). Default of credit card clients (Version 1) [Dataset]. University of California, Irvine (UCI) Machine Learning Repository. https://doi.org/10.24432/C55S3H # nolint ----

# **Required Lab Work Submission** ----
## Part A ----
# Create a new file called
# "Lab5-Submission-ResamplingMethods.R".
# Provide all the code you have used to perform all the resampling methods we
# have gone through in this lab on the. This should be done on the
# "Pima Indians Diabetes" dataset provided in the "mlbench" package.

## Part B ----
# Upload *the link* to your
# "Lab5-Submission-ResamplingMethods.R" hosted
# on Github (do not upload the .R file itself) through the submission link
# provided on eLearning.

## Part C ----
# Create a markdown file called "Lab-Submission-Markdown.Rmd"
# and place it inside the folder called "markdown". Use R Studio to ensure the
# .Rmd file is based on the "GitHub Document (Markdown)" template when it is
# being created.

# Refer to the following file in Lab 1 for an example of a .Rmd file based on
# the "GitHub Document (Markdown)" template:
#     https://github.com/course-files/BBT4206-R-Lab1of15-LoadingDatasets/blob/main/markdown/BIProject-Template.Rmd # nolint

# Include Line 1 to 14 of BIProject-Template.Rmd in your .Rmd file to make it
# displayable on GitHub when rendered into its .md version

# It should have code chunks that explain all the steps performed on the
# dataset.

## Part D ----
# Render the .Rmd (R markdown) file into its .md (markdown) version by using
# knitR in RStudio.

# You need to download and install "pandoc" to render the R markdown.
# Pandoc is a file converter that can be used to convert the following files:
#   https://pandoc.org/diagram.svgz?v=20230831075849

# Documentation:
#   https://pandoc.org/installing.html and
#   https://github.com/REditorSupport/vscode-R/wiki/R-Markdown

# By default, Rmd files are open as Markdown documents. To enable R Markdown
# features, you need to associate *.Rmd files with rmd language.
# Add an entry Item "*.Rmd" and Value "rmd" in the VS Code settings,
# "File Association" option.

# Documentation of knitR: https://www.rdocumentation.org/packages/knitr/

# Upload *the link* to "Lab-Submission-Markdown.md" (not .Rmd)
# markdown file hosted on Github (do not upload the .Rmd or .md markdown files)
# through the submission link provided on eLearning.