from pathlib import Path

import pandas as pd


INPUT_PATH = Path(
    "data/processed/clean_business_data.csv"
)

REPORTS_DIR = Path(
    "data/processed/reports"
)


def load_clean_data(file_path):
    """Load the cleaned business dataset."""
    return pd.read_csv(file_path)


def filter_valid_records(dataframe):
    """Return only records marked as valid."""
    return dataframe[
        dataframe["data_quality_status"] == "Valid"
    ].copy()


def calculate_summary(dataframe):
    """Calculate overall business performance."""
    if dataframe.empty:
        return {
            "transactions": 0,
            "total_sales": 0,
            "total_expenses": 0,
            "total_profit": 0,
            "profit_margin": 0,
            "average_sales": 0,
            "average_expenses": 0,
            "average_profit": 0,
        }

    total_sales = dataframe["sales"].sum()
    total_expenses = dataframe["expenses"].sum()
    total_profit = total_sales - total_expenses

    profit_margin = (
        total_profit / total_sales * 100
        if total_sales != 0
        else 0
    )

    return {
        "transactions": len(dataframe),
        "total_sales": total_sales,
        "total_expenses": total_expenses,
        "total_profit": total_profit,
        "profit_margin": profit_margin,
        "average_sales": dataframe["sales"].mean(),
        "average_expenses": dataframe["expenses"].mean(),
        "average_profit": (
            dataframe["sales"] - dataframe["expenses"]
        ).mean(),
    }


def summarize_by_service(dataframe):
    """Summarize sales, expenses and profit by service."""
    if dataframe.empty:
        return pd.DataFrame(
            columns=[
                "service",
                "transactions",
                "sales",
                "expenses",
                "profit",
                "profit_margin",
            ]
        )

    summary = (
        dataframe.groupby("service")
        .agg(
            transactions=("id", "count"),
            sales=("sales", "sum"),
            expenses=("expenses", "sum"),
        )
        .reset_index()
    )

    summary["profit"] = (
        summary["sales"] - summary["expenses"]
    )

    summary["profit_margin"] = (
        summary["profit"]
        / summary["sales"].where(
            summary["sales"] != 0
        )
        * 100
    ).fillna(0)

    return summary.sort_values(
        "profit",
        ascending=False,
    ).reset_index(drop=True)


def summarize_by_country(dataframe):
    """Summarize sales, expenses and profit by country."""
    if dataframe.empty:
        return pd.DataFrame(
            columns=[
                "country",
                "transactions",
                "sales",
                "expenses",
                "profit",
                "profit_margin",
            ]
        )

    summary = (
        dataframe.groupby("country")
        .agg(
            transactions=("id", "count"),
            sales=("sales", "sum"),
            expenses=("expenses", "sum"),
        )
        .reset_index()
    )

    summary["profit"] = (
        summary["sales"] - summary["expenses"]
    )

    summary["profit_margin"] = (
        summary["profit"]
        / summary["sales"].where(
            summary["sales"] != 0
        )
        * 100
    ).fillna(0)

    return summary.sort_values(
        "profit",
        ascending=False,
    ).reset_index(drop=True)


def identify_top_transaction(dataframe):
    """Return the most profitable transaction."""
    if dataframe.empty:
        return None

    analysis = dataframe.copy()

    analysis["profit"] = (
        analysis["sales"] - analysis["expenses"]
    )

    return analysis.loc[
        analysis["profit"].idxmax()
    ]


def identify_lowest_transaction(dataframe):
    """Return the least profitable transaction."""
    if dataframe.empty:
        return None

    analysis = dataframe.copy()

    analysis["profit"] = (
        analysis["sales"] - analysis["expenses"]
    )

    return analysis.loc[
        analysis["profit"].idxmin()
    ]


def identify_outliers(dataframe):
    """Identify sales outliers using the IQR method."""
    if dataframe.empty:
        return dataframe.copy()

    sales = dataframe["sales"]

    q1 = sales.quantile(0.25)
    q3 = sales.quantile(0.75)
    iqr = q3 - q1

    lower_bound = q1 - (1.5 * iqr)
    upper_bound = q3 + (1.5 * iqr)

    outliers = dataframe[
        (sales < lower_bound)
        | (sales > upper_bound)
    ].copy()

    outliers["outlier_type"] = outliers[
        "sales"
    ].apply(
        lambda value: (
            "Low"
            if value < lower_bound
            else "High"
        )
    )

    outliers["lower_bound"] = lower_bound
    outliers["upper_bound"] = upper_bound

    return outliers


def generate_analysis_report(dataframe):
    """Generate the complete exploratory analysis."""
    valid_records = filter_valid_records(dataframe)

    return {
        "summary": calculate_summary(valid_records),
        "service_summary": summarize_by_service(
            valid_records
        ),
        "country_summary": summarize_by_country(
            valid_records
        ),
        "top_transaction": identify_top_transaction(
            valid_records
        ),
        "lowest_transaction": identify_lowest_transaction(
            valid_records
        ),
        "outliers": identify_outliers(
            valid_records
        ),
    }


def export_analysis_report(report, output_dir):
    """Export analysis results as CSV files."""
    output_dir = Path(output_dir)
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary = pd.DataFrame(
        [report["summary"]]
    )

    service_summary = report["service_summary"]
    country_summary = report["country_summary"]
    outliers = report["outliers"]

    summary.to_csv(
        output_dir / "business_summary.csv",
        index=False,
    )

    service_summary.to_csv(
        output_dir / "service_summary.csv",
        index=False,
    )

    country_summary.to_csv(
        output_dir / "country_summary.csv",
        index=False,
    )

    outliers.to_csv(
        output_dir / "sales_outliers.csv",
        index=False,
    )

    return [
        output_dir / "business_summary.csv",
        output_dir / "service_summary.csv",
        output_dir / "country_summary.csv",
        output_dir / "sales_outliers.csv",
    ]


def main():
    """Run analysis and export reusable reports."""
    dataframe = load_clean_data(INPUT_PATH)

    report = generate_analysis_report(dataframe)

    summary = report["summary"]

    print("Business Exploratory Analysis")
    print("=============================")
    print(f"Valid Transactions: {summary['transactions']}")
    print(f"Total Sales: {summary['total_sales']:,.2f}")
    print(f"Total Expenses: {summary['total_expenses']:,.2f}")
    print(f"Total Profit: {summary['total_profit']:,.2f}")
    print(f"Profit Margin: {summary['profit_margin']:.2f}%")
    print(f"Average Sales: {summary['average_sales']:,.2f}")
    print(f"Average Expenses: {summary['average_expenses']:,.2f}")
    print(f"Average Profit: {summary['average_profit']:,.2f}")

    print()
    print("Performance by Service")
    print("======================")
    print(
        report["service_summary"].to_string(
            index=False
        )
    )

    print()
    print("Performance by Country")
    print("=======================")
    print(
        report["country_summary"].to_string(
            index=False
        )
    )

    for title, key in [
        ("Top Transaction", "top_transaction"),
        ("Lowest Transaction", "lowest_transaction"),
    ]:
        transaction = report[key]

        if transaction is not None:
            profit = (
                transaction["sales"]
                - transaction["expenses"]
            )

            print()
            print(title)
            print("=" * len(title))
            print(f"ID: {int(transaction['id'])}")
            print(f"Customer: {transaction['customer']}")
            print(f"Service: {transaction['service']}")
            print(f"Sales: {transaction['sales']:,.2f}")
            print(f"Expenses: {transaction['expenses']:,.2f}")
            print(f"Profit: {profit:,.2f}")

    print()
    print("Sales Outliers")
    print("==============")

    outliers = report["outliers"]

    if outliers.empty:
        print("No sales outliers detected.")
    else:
        print(
            outliers[
                [
                    "id",
                    "customer",
                    "service",
                    "sales",
                    "outlier_type",
                ]
            ].to_string(index=False)
        )

    exported_files = export_analysis_report(
        report,
        REPORTS_DIR,
    )

    print()
    print("Exported Reports")
    print("================")

    for file_path in exported_files:
        print(f"Saved: {file_path}")


if __name__ == "__main__":
    main()
