# Kaggle
## Google Analytics Customer Revenue Prediction
### Data Fields
•	fullVisitorId- A unique identifier for each user of the Google Merchandise Store.\
•	channelGrouping - The channel via which the user came to the Store.\
•	date - The date on which the user visited the Store.\
•	device - The specifications for the device used to access the Store.\
•	geoNetwork - This section contains information about the geography of the user.\
•	sessionId - A unique identifier for this visit to the store.\
•	socialEngagementType - Engagement type, either "Socially Engaged" or "Not Socially Engaged".\
•	totals - This section contains aggregate values across the session.\
•	trafficSource - This section contains information about the Traffic Source from which the session originated.\
•	visitId - An identifier for this session. This is part of the value usually stored as the _utmb cookie. This is only unique to the user. For a completely unique ID, you should use a combination of fullVisitorId and visitId.\
•	visitNumber - The session number for this user. If this is the first session, then this is set to 1.\
•	visitStartTime - The timestamp (expressed as POSIX time).

### Model select
 LightGBM + Catboost
