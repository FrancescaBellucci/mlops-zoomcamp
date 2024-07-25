# Data Dictionary

_customer_churn_records.parquet_ is the main dataset used for this project. It consists of 10,000 records and 18 columns. 
The content of the columns is the following:
* __RowNumber__ — (integer) — Row index in the dataset.
* __CustomerId__ — (integer) — 8-digit number that serves as a unique identifier for customers.
* __Surname__ — (string) — Surname of a customer.
* __CreditScore__ — (integer) — Credit score attributed to the customer by the bank. (It is a measure of likelyhood to pay back a loan). It can range between 0 and 1000.
* __Geography__ — (string) — Categorical variable indicating the customer’s location. It can be one of three values: _France_, _Germany_, or _Spain_.
* __Gender__ — (string) — Categorical variable indicating the customer's gender. The two genders are encoded as __Male__ and __Female__.
* __Age__ — (integer) — Customer's age.
* __Tenure__ — (integer) — Number of years that the customer has been a client of the bank. 
* __Balance__ — (float) — Balance of the customer's bank account. It ranges between 0 and 250.000.
* __NumOfProducts__ — (integer) — Number of products that a customer has purchased from the bank. It can't be higher than 4.
* __HasCrCard__ — (integer) —  Binary variable that indicates whether the customer has a credit card associated to their account. 
* __IsActiveMember__ — (integer) — Binary variable that indicates whether the customer has been active recently.
* __EstimatedSalary__ — (float) — Estimated salary of the customer, ranging between 0 and 200,000.
* __Exited__ — (integer) — Binary variable that indicates whether or not the customer left the bank. This is our __target variable__. 
* __Complain__ — (integer) — Binary variable that indicates whether the customer has forwarded a complaint to the bank or not.
* __Satisfaction Score__ — (integer) — Score provided by the customer for their complaint resolution. It goes from 1 to 5.
* __Card Type__ — (string) — Categorical variable indicating the type of card held by the customer. It can be one of four values: _SILVER_, _GOLD_, _PLATINUM_, and _DIAMOND_.
* __Points Earned__ — (integer) — Reward points earned by the customer for using their credit card. The maximum number of attainable points is 1,000.

