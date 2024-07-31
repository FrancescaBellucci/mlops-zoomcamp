# Models

I used an ensemble of Isolation Forests and XGBoost to compute predictions. 
After noticing that the _Age_ variable was the only one presenting outliers, and that its observations seemed to come from two distinct distributions, I decided to run an Anomaly Detection algorithm to try and separate them.
In this way, it is possible to isolate a smaller subgroup of clients that will certainly return. 
Both Isolation Forests and XGBoost are robust with respect to class imbalance and different scaling, which simplifies the preprocessing procedure significantly.
Furthermore, XGBoost is also robust with respect to outliers. 

The two models are trained and stored separately, and they are applied sequentially for computing predictions. 
More specifically, Isolation Forests are run first on the data, and XGBoost is only used to compute predictions for data points that don't allign with the subgroup of certainly returning customers.
