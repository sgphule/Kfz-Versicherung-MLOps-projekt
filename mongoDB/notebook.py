import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from unicodedata import category
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
import pickle
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import precision_recall_curve
import logging
logger = logging.getLogger(__name__)

def create_bar_plot(df):
    value_counts = df['Response'].value_counts()
    plt.figure(figsize=(7, 5))
    value_counts.plot(kind='bar')
    plt.xlabel('Response')
    plt.ylabel('Count')
    plt.title('Distribution of Responses')
    # plt.show()
    plt.savefig("mongoDB/graphs/1. Distribution of Responses.png")
    logger.info("graph stored at location - mongoDB/graphs/1. Distribution of Responses.png")


def plot_age_distribution(df):
    # df['Age'].hist(bins=20)
    plt.figure(figsize=(7, 5))
    df['Age'].plot(kind='hist', bins=20)
    plt.xlabel('Age')
    # plt.show()
    plt.title('Age Histogram')
    plt.savefig("mongoDB/graphs/2. Age Distribution.png")
    logger.info("graph stored at location - mongoDB/graphs/2. Age Distribution.png")

def plot_age_vs_annual_premium_scatter_plot(df):
    plt.figure(figsize=(7, 4))
    sns.scatterplot(x='Age', y='Annual_Premium', data=df)
    plt.xlabel('Age')
    plt.ylabel('Annual Premium')
    plt.title('Age vs Annual Premium')
    # plt.show()
    plt.savefig("mongoDB/graphs/3. Age Vs Annual Premium.png")
    logger.info("graph stored at location - mongoDB/graphs/3. Age Vs Annual Premium.png")

def plot_gender_response_comparison(df):
    value_counts = df['Gender'].value_counts()
    plt.figure(figsize=(8, 6))
    value_counts.plot(kind='bar')
    plt.xticks(rotation=0)
    plt.xlabel('Gender')
    plt.ylabel('Count')
    plt.title('Distribution of Gender (Gender Vs Responses)')
    # plt.show()
    plt.savefig("mongoDB/graphs/4. Distribution of Gender - Gender Vs Responses.png")
    logger.info("graph stored at location - mongoDB/graphs/4. Distribution of Gender - Gender Vs Responses.png")

def plot_gender_response_groupings(df):
    data = df.groupby(['Gender', 'Response'])['id'].count().to_frame().rename(columns={'id': 'count'}).reset_index()
    logger.info("Gender Response data:")
    logger.info(data)
    sns.catplot(x="Gender", y="count", col="Response", data=data, kind="bar", height=4, aspect=.7)
    # plt.show()
    plt.savefig("mongoDB/graphs/5. Gender Response groupings.png")
    logger.info("graph stored at location - mongoDB/graphs/5. Gender Response groupings.png")

def plot_driving_license_by_gender(df):
    data = df.groupby(['Gender'])['Driving_License'].count().to_frame().reset_index()
    sns.catplot(x="Gender", y="Driving_License", data=data, kind="bar")
    # plt.show()
    plt.title("Driving License Holders by Gender")
    plt.savefig("mongoDB/graphs/6. Driving License by Gender.png")
    logger.info("graph stored at location - mongoDB/graphs/6. Driving License by Gender.png")

def plot_license_holders_per_gender(df):
    license_holders = df[df['Driving_License'] == 1]
    gender_counts = license_holders['Gender'].value_counts()
    plt.figure(figsize=(8, 4))
    gender_counts.plot(kind='bar', color=['blue', 'pink'])
    plt.title('Number of Male vs Female License Holders')
    plt.xlabel('Gender')
    plt.xticks(rotation=0)
    plt.ylabel('Count of License Holders')
    # plt.show()
    plt.savefig("mongoDB/graphs/7. Number of Male vs Female License Holders.png")
    logger.info("graph stored at location - mongoDB/graphs/7. Number of Male vs Female License Holders.png")

def plot_already_insured_customers(df):
    plt.figure(figsize=(7, 4))
    df["Previously_Insured"] = df["Previously_Insured"].astype("category")
    sns.countplot(x="Previously_Insured", data=df)
    plt.title("Distribution of Already Insured Customers")
    plt.xlabel("Previously Insured")
    plt.ylabel("Count")
    # plt.show()
    plt.savefig("mongoDB/graphs/8. Distribution of Already Insured Customers.png")
    logger.info("graph stored at location - mongoDB/graphs/8. Distribution of Already Insured Customers.png")

def plot_vehicle_age_distribution(df):
    plt.figure(figsize=(7, 6))
    sns.countplot(x='Vehicle_Age', data=df)
    plt.xlabel('Vehicle Age')
    plt.ylabel('Count')
    plt.title('Distribution of Vehicle Age')
    # plt.show()
    plt.savefig("mongoDB/graphs/9. Distribution of Vehicle Age.png")
    logger.info("graph stored at location - mongoDB/graphs/9. Distribution of Vehicle Age.png")

def plot_vehicle_age_vs_responces_in_colors(df):
    data = df.groupby(['Vehicle_Age', 'Response'])['id'].count().to_frame().rename(columns={'id': 'count'}).reset_index()
    logger.info("Vehicle Age Vs Responces data:")
    logger.info(data)
    sns.catplot(x="Vehicle_Age", y="count", col="Response",
                    data=data, kind="bar",
                    height=4, aspect=.7);
    # plt.show()
    plt.savefig("mongoDB/graphs/10. Vehicle Age Vs Responces.png")
    logger.info("graph stored at location - mongoDB/graphs/10. Vehicle Age Vs Responces.png")

def plot_vehicle_damage_distribution(df):
    plt.figure(figsize=(8, 5))
    sns.countplot(data=df, x='Vehicle_Damage')
    plt.title('Distribution of Vehicle Damage')
    plt.xlabel('Vehicle Damage')
    plt.ylabel('Count')
    # plt.show()
    plt.savefig("mongoDB/graphs/11. Distribution of Vehicle Damage.png")
    logger.info("graph stored at location - mongoDB/graphs/11. Distribution of Vehicle Damage.png")

def plot_vehicle_damage_responces_count(df):
    data = df.groupby(['Vehicle_Damage', 'Response'])['id'].count().to_frame().rename(
        columns={'id': 'count'}).reset_index()
    sns.catplot(x="Vehicle_Damage", y="count", col="Response", data=data, kind="bar", height=4, aspect=.7)
    # plt.show()
    plt.savefig("mongoDB/graphs/12. Vehicle Damage Response Groupings.png")
    logger.info("graph stored at location - mongoDB/graphs/12. Vehicle Damage Response Groupings.png")

def plot_annual_premium_stats(df):
    plt.figure(figsize=(7, 5))
    df['Annual_Premium'].describe()
    df['Annual_Premium'].hist(bins=10)
    plt.xlabel('Annual_Premium')
    # plt.show()
    plt.title('Annual Premium Stats')
    plt.savefig("mongoDB/graphs/13. Annual Premium Stats.png")
    logger.info("graph stored at location - mongoDB/graphs/13. Annual Premium Stats.png")

def have_A_look_at_null_values(df):
    logger.info(df.info())


def get_numerical_and_categorical_features(df):
    numerical_features = ['Age', 'Vintage']
    categorical_features = ['Gender', 'Driving_License', 'Previously_Insured', 'Vehicle_Age_lt_1_Year',
                'Vehicle_Age_gt_2_Years', 'Vehicle_Damage_Yes', 'Region_Code', 'Policy_Sales_Channel']
    return numerical_features, categorical_features


def map_gender_column_to_zero_one_values(df):
    df['Gender'] = df['Gender'].map({'Female': 0, 'Male': 1}).astype(int)
    df.head(2)
    return df


def check_data_types_of_all_columns(df):
    for col in df.columns:
        logger.info(f"{col} >> {df[col].dtype}")


def create_dummy_columns_for_categorical_columns(df):
    # creating dummy cols for categorical features Vehicle_Age and Vehicle_Damage
    df = pd.get_dummies(df, drop_first=True)
    df.head(2)
    return df


def rename_columns_and_keep_int_data_type_for_numerical_features(df, categorical_features):
    # cols renaming and keeping dtype as int
    df = df.rename(
        columns={"Vehicle_Age_< 1 Year": "Vehicle_Age_lt_1_Year", "Vehicle_Age_> 2 Years": "Vehicle_Age_gt_2_Years"})
    df['Vehicle_Age_lt_1_Year'] = df['Vehicle_Age_lt_1_Year'].astype('int')
    df['Vehicle_Age_gt_2_Years'] = df['Vehicle_Age_gt_2_Years'].astype('int')
    df['Vehicle_Damage_Yes'] = df['Vehicle_Damage_Yes'].astype('int')

    for column in categorical_features:
        df[column] = df[column].astype('str')
    logger.info(df.head())
    return df, categorical_features


def drop_unnecessary_column(df):
    # id = df.id
    df = df.drop('id', axis=1)
    logger.info(df.head(2))
    return df


def scale_numerical_features(df, numerical_features):
    ss = StandardScaler()
    df[numerical_features] = ss.fit_transform(df[numerical_features])

    mm = MinMaxScaler()
    df[['Annual_Premium']] = mm.fit_transform(df[['Annual_Premium']])
    logger.info(df.head(2))
    return df


def get_train_test_split(df):
    train_target = df['Response']
    logger.info(train_target)
    train = df.drop(['Response'], axis=1)
    x_train, x_test, y_train, y_test = train_test_split(train, train_target, random_state=9)
    logger.info("displaying input data")
    logger.info(train.head(3))
    logger.info("displaying target column")
    logger.info(train_target.head(3))
    return x_train, x_test, y_train, y_test


def model_training_with_random_forest_classifier(x_train, y_train):
    random_search = {'criterion': ['entropy', 'gini'],
                     'max_depth': [2, 3, 4, 5, 6, 7, 10],
                     'min_samples_leaf': [4, 6, 8],
                     'min_samples_split': [5, 7, 10],
                     'n_estimators': [300]}
    clf = RandomForestClassifier()
    model = RandomizedSearchCV(estimator=clf, param_distributions=random_search, n_iter=10,
                               cv=4, verbose=1, random_state=101, n_jobs=-1)
    start = time.time()
    model.fit(x_train, y_train)
    stop = time.time()
    duration_in_minutes = (stop - start)/60
    logger.info(f"Training time: {duration_in_minutes}minutes")
    best_params = model.best_params_
    logger.info("Best Hyperparameters:")
    logger.info(best_params)
    return model, model.best_estimator_


def save_model(model):
    filename = 'rf_model.pkl'
    pickle.dump(model, open(filename, 'wb'))
    return filename


def load_pickle_file(filename):
    rf_load = pickle.load(open(filename, 'rb'))


def get_classification_report_by_evaluating_model_and_save_confusion_matrix(model, x_test, y_test, clf):
    y_pred = model.predict(x_test)
    logger.info(classification_report(y_test, y_pred, zero_division=1))
    cm = confusion_matrix(y_test, y_pred, labels=clf.classes_)
    plt.figure(figsize=(7, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=clf.classes_)
    disp.plot()
    # plt.show()
    plt.savefig("mongoDB/graphs/14. Confusion Matrix1.png")
    logger.info("graph stored at location - mongoDB/graphs/14. Confusion Matrix1.png")
    # ______________________________________________________________________
    y_scores = model.predict_proba(x_test)[:, 1]  # Probability of class 1
    precision, recall, thresholds = precision_recall_curve(y_test, y_scores)

    # Slice precision and recall to match thresholds
    precision = precision[:-1]
    recall = recall[:-1]

    # Example: prioritize precision but keep recall above 0.6
    valid = recall > 0.6
    score = precision[valid] * recall[valid]  # or just use precision[valid]
    best_threshold = thresholds[valid][np.argmax(score)]
    logger.info(f"best_threshold: {best_threshold}")

    y_pred_custom = (y_scores >= best_threshold).astype(int)

    plt.figure(figsize=(7, 5))
    plt.plot(recall, precision)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    # plt.show()
    plt.savefig("mongoDB/graphs/15. Precision-Recall Curve.png")
    logger.info("graph stored at location - mongoDB/graphs/15. Precision-Recall Curve.png")
    # _______________________________________________________________________

    logger.info(classification_report(y_test, y_pred_custom, zero_division=1))
    cm = confusion_matrix(y_test, y_pred_custom, labels=clf.classes_)
    plt.figure(figsize=(7, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=clf.classes_)
    disp.plot()
    # plt.show()
    plt.savefig("mongoDB/graphs/16. Confusion Matrix2.png")
    logger.info("graph stored at location - mongoDB/graphs/16. Confusion Matrix2.png")


def data_preprocessing(df):
    have_A_look_at_null_values(df)
    numerical_features, categorical_features = get_numerical_and_categorical_features(df)
    df = map_gender_column_to_zero_one_values(df)
    check_data_types_of_all_columns(df)
    df = create_dummy_columns_for_categorical_columns(df)
    df, categorical_features = rename_columns_and_keep_int_data_type_for_numerical_features(df, categorical_features)
    df = drop_unnecessary_column(df)
    df = scale_numerical_features(df, numerical_features)
    x_train, x_test, y_train, y_test = get_train_test_split(df)
    return x_train, x_test, y_train, y_test


def train_save_model_and_load_pickle_file(x_train, y_train):
    model, clf = model_training_with_random_forest_classifier(x_train, y_train)
    filename = save_model(model)
    load_pickle_file(filename)
    return model, clf


def main():
    logging.basicConfig(level=logging.INFO)
    df = pd.read_csv("mongoDB/data.csv")
    while True:
        logger.info("1. DISPLAY FIRST 5 RECORDS IN DATA")
        logger.info("2. DISPLAY SHAPE OF DATA")
        logger.info("3. VERIFY NULL VALUE EXISTENCE IN DATA")
        logger.info("4. DISPLAY GENERAL DATA INFORMATION")
        logger.info("5. SUMMARIZE  STATISTICS")
        logger.info("6. DISPLAY RESPONSE VALUE COUNTS")
        logger.info("7. PLOT BAR GRAPH SHOWING DISTRIBUTION OF RESPONCES")
        logger.info("8. PLOT AGE DISTRIBUTION GRAPH")
        logger.info("9. PLOT AGE VS ANNUAL PREMIUM GRAPH")
        logger.info("10. PLOT GENDER VS RESPONCES GRAPH")
        logger.info("11. PLOT GENDER RESPONSE GROUPING GRAPH")
        logger.info("12. PLOT DRIVING LICENSE BY GENDER GRAPH")
        logger.info("13. PLOT MALE VS FEMALE LICENSE HOLDERS GRAPH")
        logger.info("14. PLOT DISTRIBUTION OF ALREADY INSURED CUSTOMERS GRAPH")
        logger.info("15. PLOT VEHICLE AGE DISTRIBUTION GRAPH")
        logger.info("16. PLOT VEHICLE AGE VS RESPONCES GRAPH")
        logger.info("17. PLOT VEHICLE DAMAGE DISTRIBUTION GRAPH")
        logger.info("18. PLOT VEHICLE DAMAGE RESPONSE GROUPINGS GRAPH")
        logger.info("19. PLOT ANNUAL PREMIUM STATS GRAPH")
        logger.info("20. PERFORM DATA PREPROCESSING")
        logger.info("21. PERFORM MODEL TRAINING AFTER DATA PREPROCESSING")
        logger.info("22. GENERATE CLASSIFICATION REPORT AND CONFUSION MATRIX")
        logger.info("23. EXIT APPLICATION")
        logger.info("ENTER YOUR CHOICE: ")
        choice = int(input())
        match choice:
            case 1:
                logger.info(df.head())
            case 2:
                logger.info(df.shape)
            case 3:
                logger.info(df.isnull().sum())
            case 4:
                logger.info(df.info())
            case 5:
                logger.info(df.describe())
            case 6:
                logger.info(df['Response'].value_counts())
            case 7:
                create_bar_plot(df)
            case 8:
                plot_age_distribution(df)
            case 9:
                plot_age_vs_annual_premium_scatter_plot(df)
            case 10:
                plot_gender_response_comparison(df)
            case 11:
                plot_gender_response_groupings(df)
            case 12:
                plot_driving_license_by_gender(df)
            case 13:
                plot_license_holders_per_gender(df)
            case 14:
                plot_already_insured_customers(df)
            case 15:
                plot_vehicle_age_distribution(df)
            case 16:
                plot_vehicle_age_vs_responces_in_colors(df)
            case 17:
                plot_vehicle_damage_distribution(df)
            case 18:
                plot_vehicle_damage_responces_count(df)
            case 19:
                plot_annual_premium_stats(df)
            case 20:
                x_train, x_test, y_train, y_test = data_preprocessing(df)
            case 21:
                answer = input("HAVE YOU PERFORMED DATA PREPROCESSING (YES/NO)?")
                if answer == "NO" or answer == "no":
                    x_train, x_test, y_train, y_test = data_preprocessing(df)
                model, clf = train_save_model_and_load_pickle_file(x_train, y_train)
            case 22:
                answer = input("HAVE YOU PERFORMED DATA PREPROCESSING AND MODEL TRAINING (YES/NO)?")
                if answer == "NO" or answer == "no":
                    logging.info("PLEASE PERFORM DATA PREPROCESSING AND MODEL TRAINING FIRST!")
                else:
                    get_classification_report_by_evaluating_model_and_save_confusion_matrix(model, x_test, y_test, clf)
            case 23:
                logger.info("Exiting the loop. Goodbye!")
                break
            case _:
                logger.info("Invalid option. Try again.")



if __name__ == "__main__":
    main()