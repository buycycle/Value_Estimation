import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def plot_pred(y_test, preds, score, file_name="plot/bins.png"):
    fig, ax = plt.subplots()
    # Convert to numpy array and ensure float
    y_test_np = np.array(y_test).astype(float)
    preds_np = np.array(preds).astype(float)
    # Change kdeplot to histplot for bin plot
    sns.histplot(y_test_np, color="blue", ax=ax, label="Actual Values", bins=30)
    sns.histplot(preds_np, color="red", ax=ax, label="Predictions", bins=30)
    ax.legend(loc="upper right")
    ax.set_xlabel("Sales price")  # set x label
    # Insert score and sample size text
    sample_size = len(y_test_np)
    textstr = "MAP Error: {:.2f}\nSample Size: {}".format(score, sample_size)
    props = dict(boxstyle="round", facecolor="wheat", alpha=0.5)
    ax.text(0.05, 0.05, textstr, transform=ax.transAxes, fontsize=14, verticalalignment="bottom", bbox=props)
    plt.savefig(file_name)


def plot_difference(y_test, preds, file_name="plot/difference.png"):
    fig, ax = plt.subplots()
    # Convert to numpy array and ensure float
    y_test_np = np.array(y_test).astype(float)
    preds_np = np.array(preds).astype(float)
    # Calculate relative error in %
    rel_error = ((y_test_np - preds_np) / y_test_np) * 100
    # Create scatter plot
    ax.scatter(y_test_np, rel_error, color="blue", alpha=0.5)
    ax.axhline(0, color="red")  # Add horizontal line at 0
    ax.set_xlabel("Actual Sales Price")
    ax.set_ylabel("Relative Error % (Actual - Predicted)")
    plt.savefig(file_name)


def plot_pred_kde(y_test, preds, score, file_name="plot/kde.png"):
    fig, ax = plt.subplots()
    # Convert to numpy array and ensure float
    y_test_np = np.array(y_test).astype(float)
    preds_np = np.array(preds).astype(float)

    sns.kdeplot(y_test_np, color="blue", ax=ax, label="Actual Values")
    sns.kdeplot(preds_np, color="red", ax=ax, label="Predictions")

    ax.legend(loc="upper right")
    ax.set_xlabel("Sales price")  # set x label

    # Insert score text
    textstr = "MAP Error: {:.2f}".format(score)
    props = dict(boxstyle="round", facecolor="wheat", alpha=0.5)
    ax.text(0.05, 0.05, textstr, transform=ax.transAxes, fontsize=14, verticalalignment="bottom", bbox=props)

    plt.savefig(file_name)


def analyze_results(input_path="data/", plot_file_name="plot/pair.pdf"):
    X_test = pd.read_pickle(input_path + "X_test.pkl")
    y_test = pd.read_pickle(input_path + "y_test.pkl")
    preds = pd.read_pickle(input_path + "preds.pkl")

    preds = preds.reset_index(drop=True)
    preds.columns = ["prediction"]

    X_test = X_test.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)
    result = pd.concat([X_test, y_test, preds], axis=1)

    result["difference"] = result["sales_price"] - result["prediction"]
    result["difference_%"] = result["difference"] / result["sales_price"]

    # calculate the difference % and multiply it by 100
    result["difference_%"] = result["difference_%"] * 100
    # add a new column for difference bins
    result["difference_bin"] = pd.cut(
        result["difference_%"],
        bins=[-float("inf"), -100, -50, -20, -10, 0, 10, 20, 50, 100, float("inf")],
        labels=[
            "Overestimate <-100%",
            "-100% to -50%",
            "-50% to -20%",
            "-20% to -10%",
            "-10% to 0%",
            "0% to 10%",
            "10% to 20%",
            "20% to 50%",
            "50% to 100%",
            "Underestimate >100%",
        ],
    )

    result = result.dropna()

    # change numeric to int, necessary for some reason
    numeric_cols = result.select_dtypes(include=[np.number]).columns
    result[numeric_cols] = result[numeric_cols].fillna(0).astype(int)

    palette = sns.color_palette(
        ["grey", "darkblue", "blue", "lightblue", "green", "green", "lightcoral", "red", "darkred", "grey"]
    )

    # produce a sns pairplot
    sns.pairplot(result, hue="difference_bin", palette=palette)

    # save in plots folder
    plt.savefig(plot_file_name)


def prediction_interval(result, plot_file_name="plot/interval.pdf"):
    result = result.sort_values("sales_price")

    # Reset the index of the dataframe
    result.reset_index(inplace=True)

    # Plot the data
    # Define the interval width
    interval_width = 0.5  # Adjust this value as needed
    # Plot the data
    plt.plot(result["sales_price"], "ro", label="Sales Price")  # Add label for legend
    plt.plot(result["prediction"], "bo", label="Prediction")  # Add label for legend
    for i in range(len(result)):
        plt.fill_between(
            [i - interval_width / 2, i + interval_width / 2], result["low"].iloc[i], result["high"].iloc[i], alpha=0.2, color="r"
        )
    plt.xlabel("Ordered samples.")
    plt.ylabel("Sales Price [Eur]")  # Rename y-axis
    plt.legend()  # Add legend
    plt.show()
    plt.savefig(plot_file_name)
