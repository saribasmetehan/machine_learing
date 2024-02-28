import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, ShuffleSplit, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale
from sklearn import model_selection
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor


def main():
    class Data_analyzer:
        def __init__(self, data: pd.DataFrame):
            self.data = data

        def summarize_data(self) -> None:
            print(self.data.head())
            print(self.data.info())
            print(self.data.describe().T)
            print(self.data.isnull().sum())
            print(self.data.corr())

        def train_knn_model(self, x_train, y_train) -> None:
            self.knn_model = KNeighborsRegressor().fit(x_train, y_train)
            return self.knn_model
        def predict_knn_model(self, x_test, y_test) -> None:
            y_pred = self.knn_model.predict(x_test)
            mse = np.sqrt(mean_squared_error(y_test, y_pred))
            print(mse)

        def tune_knn_model(self, x_train, y_train, k_values, cv=10) -> int:
            knn = KNeighborsRegressor()
            parameters = {'n_neighbors': k_values}
            self.cv_model = GridSearchCV(knn, parameters, cv=cv)
            print(self.cv_model.fit(x_train, y_train))
            print(self.cv_model.best_params_["n_neighbors"])

        def svr_model(self, x_train, x_test, y_train):
            x_train = pd.DataFrame(x_train["Hits"])
            x_test = pd.DataFrame(x_test["Hits"])
            self.svr_model = SVR("linear").fit(x_train, y_train)
            print(self.svr_model)

        def predict_svr_model(self, x_test, y_test) -> None:
            prediction = self.svr_model.predict(x_test)
            mse = np.sqrt(mean_squared_error(y_test, prediction))
            print(mse)

        def tune_svr_model(self, svr_params, x_train, y_train, cv=10) -> None:
            self.cv_model = GridSearchCV(SVR(), svr_params, cv=cv).fit(x_train, y_train)
            print(self.cv_model.best_params_)
            svr_tuned = SVR("linear", C=pd.Series(cv_model.best_params_)[0]).fit(x_train, y_train)
            print(self.svr_tuned)

        def svr_rbf_model(self, xtrain, y_train) -> None:
            self.svr_rbf_model = SVR("rbf").fit(xtrain, y_train)
            print(self.svr_rbf_model)

        def predict_svr_rbf_model(self, x_test, y_test):
            prediction = self.svr_rbf_model(x_test)
            mse = np.sqrt(mean_squared_error(y_test, prediction))

        def tune_svr_rbf_model(self, x_train, y_train, svr_rbf_params, cv=10):
            svr_rbf = SVR(kernel='rbf')
            svr_rbf_cv_model = GridSearchCV(svr_rbf, svr_rbf_params, cv=cv)
            svr_rbf_cv_model.fit(x_train, y_train)

            print(svr_rbf_cv_model.best_params_)

            self.svr_rbf_model = svr_rbf_cv_model.best_estimator_
            print(self.svr_rbf_model)

        def mlp_model(self, x_train, x_test, y_train):
            scaler = StandardScaler()
            scaler.fit(x_train)
            x_train_scaled = scaler.transform(x_train)
            x_test_scaled = scaler.transform(x_test)

            mlp_model = MLPRegressor().fit(x_train_scaled, y_train)

            return mlp_model
        def predict_mlp_model(self):
            mlp_model.predict(self.x_train_scaled)
            y_pred =mlp.predic(self.x_test_scaled)

        def tune_mlp_model(self,mlp_params, cv =10,y_train:
            mlp_cv_model = GridSearchCV(mlp_model,mlp_params,cv=cv)
            mlp_cv_model =fit(x_train_scaled,y_train)





    hit = pd.read_csv("Hitters.csv")
    df_hit = hit.copy()
    df_hit = df_hit.dropna()
    df_hit = df_hit.iloc[:, 1:len(df_hit)]
    dms = pd.get_dummies(df_hit[["League", "Division", "NewLeague"]])
    y = df_hit["Salary"]
    x_ = df_hit.drop(["Salary", "League", "Division", "NewLeague"],axis= 1)
    x = pd.concat([x_, dms[["League_N", "Division_W", "NewLeague_N"]]], axis=1)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)

    analyzer = Data_analyzer(df_hit)
    analyzer.summarize_data()


if __name__ == "__main__":
    main()
