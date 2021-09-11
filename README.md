# Heart_Failure_Prediction
Classification algos to predict Heart Failure.

Points to note:The dataset doesn't contain any missing values, so no need for imputer.
The dataset does have categorical values, so will have to use encoder for it. I have used OneHotEncoder.

Steps:
Read the dataset.
Encode the categorical values.
Split the dataset.
Create a scaler to scale the data, but remember to ignore the binary values added after encoding.
Fit the scaler using training values and then apply this scale to both training and testing values.
Use the various algos to predict the result and check the accuracy.

Output:
I had the best accuracy using support vector classification with rbf kernel.
