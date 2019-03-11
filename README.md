# Disaster Response Pipeline Project

### Motivation of the project
In disasters, quick and accurate reponse to people's messages is crucial. This project builds a disaster reponse pipeline to analyze message data and classify them. There are all in total 36 categories, including disaster types like `fire`, `floods` and people's needs like `water`, `food` etc. It helps to alarm aid centers to quickly respond in case of a disaster.

### Description of files

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/
