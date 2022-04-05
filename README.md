# Point Spread Prediction
This project targets on raising money from NBA Sports betting market.

# description of all directories
1. "data" in data directory downloaded from kaggle.
2. "checkpoint" directory works for DNN model.
3. "models" directory saves all models.
4. "New MS results" directory saves all temporaty or final accuracy results (csv files storing accuracy measures or graphs of data visulization).
5. "predictions" saves all predictions of all models.
6. all "runs" store all DNN models which will be stored in models if they are good models.
7. The rest directories do not have much meaning. 

# correct order of executing .py files.
0. Don't Run main.py!!!!!!! It has nothing!!!!
1. featureSelection.py: use this file to select features from original dataset.
2. PCA.py: use this file to do PCA transformation to get PCA dataset.
3. Now, we got original, feature selected, and PCA datasets ready. However, you don't have to do 1 and 2 becasue these datasets has already stored in "data" directory.
4. MLModels.py: use this file to generate all models and predictions from different algorithms (defined in LR.py, RF.py, and SVM.py; other algorithms imported in MLModels.py.) 
However, to change datasets (original, feature selected, and PCA), you have to manually change based on my comments in MLModels.py.
5. DNN_player_and_team_data.py:  the main file to execute DNN algorithm. example_web.py aggregates all functions and classes that used in DNN_player_and_team_data.py.
6. profitability_simulation.py: for each algorithm, choose the best model's prediction to simulate how much money we can get if we follow these models.
