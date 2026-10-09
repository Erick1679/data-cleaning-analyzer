from pathlib import Path

import pandas as pd


REQUIRED_COLUMNS = {
    "id",
    "date",
    "customer",
    "service",
    "sales",
    "expenses",
    "country",
}


def load_data(file_path):
    """Load a CSV file into a pandas DataFrame."""
    return pd.read_csv(file_path)


def validate_columns(dataframe):
    """Check that all required columns are present."""
    missing_columns = REQUIRED_COLUMNS - set(dataframe.columns)

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {sorted(missing_columns)}"
        )


def standardize_text(dataframe):
    """Standardize text fields."""
    text_columns = [
        "customer",
        "service",
        "country",
    ]

    cleaned = dataframe.copy()

    for column in text_columns:
        cleaned[column] = (
            cleaned[column]
            .fillna("")
            .astype(str)
            .str.strip()
        )

    cleaned["service"] = cleaned["service"].str.title()
    cleaned["country"] = cleaned["country"].str.title()

    return cleaned


def clean_numeric_columns(dataframe):
    """Convert sales and expenses into numeric values."""
    cleaned = dataframe.copy()

    for column in ["sales", "expenses"]:
        cleaned[column] = (
            cleaned[column]
            .astype(str)
            .str.replace(",", "", regex=False)
            .replace("", pd.NA)
        )

        cleaned[column] = pd.to_numeric(
            cleaned[column],
            errors="coerce",
        )

    return cleaned


def clean_dates(dataframe):
    """Convert dates into proper datetime values."""
    cleaned = dataframe.copy()

    cleaned["date"] = pd.to_datetime(
        cleaned["date"],
        errors="coerce",
        format="mixed",
    )

    return cleaned


def identify_duplicates(dataframe):
    """Return duplicate business records while ignoring record ID."""
    duplicate_columns = [
        "date",
        "customer",
        "service",
        "sales",
        "expenses",
        "country",
    ]

    return dataframe[
        dataframe.duplicated(
            subset=duplicate_columns,
            keep=False,
        )
    ].copy()


def identify_missing_values(dataframe):
    """Return the number of missing values in each column."""
    return dataframe.isna().sum()


def identify_invalid_sales(dataframe):
    """Return records with missing or non-positive sales."""
    return dataframe[
        dataframe["sales"].isna()
        | (dataframe["sales"] <= 0)
    ].copy()


def add_quality_status(dataframe):
    """Add a data-quality status to every record."""
    cleaned = dataframe.copy()

    duplicate_columns = [
        "date",
        "customer",
        "service",
        "sales",
        "expenses",
        "country",
    ]

    duplicate_mask = cleaned.duplicated(
        subset=duplicate_columns,
        keep=False,
    )

    text_missing_mask = (
        cleaned[
            [
                "customer",
                "service",
                "country",
            ]
        ]
        .eq("")
        .any(axis=1)
    )

    missing_data_mask = cleaned[
        [
            "date",
            "sales",
            "expenses",
        ]
    ].isna().any(axis=1)

    invalid_sales_mask = (
        cleaned["sales"].isna()
        | (cleaned["sales"] <= 0)
    )

    cleaned["data_quality_status"] = "Valid"

    cleaned.loc[
        duplicate_mask,
        "data_quality_status",
    ] = "Duplicate"

    cleaned.loc[
        text_missing_mask | missing_data_mask,
        "data_quality_status",
    ] = "Needs Review"

    cleaned.loc[
        invalid_sales_mask,
        "data_quality_status",
    ] = "Needs Review"

    return cleaned


def clean_data(dataframe):
    """Run the complete data-cleaning pipeline."""
    validate_columns(dataframe)

    cleaned = dataframe.copy()

    cleaned = standardize_text(cleaned)
    cleaned = clean_numeric_columns(cleaned)
    cleaned = clean_dates(cleaned)
    cleaned = add_quality_status(cleaned)

    return cleaned


def generate_cleaning_report(original, cleaned):
    """Generate a summary of data-quality issues."""
    duplicate_rows = identify_duplicates(original)

    report = {
        "original_rows": len(original),
        "cleaned_rows": len(cleaned),
        "duplicate_rows": len(duplicate_rows),
        "missing_values": int(
            original.isna().sum().sum()
        ),
        "invalid_dates": int(
            cleaned["date"].isna().sum()
        ),
        "invalid_sales": len(
            identify_invalid_sales(cleaned)
        ),
        "valid_rows": int(
            (cleaned["data_quality_status"] == "Valid").sum()
        ),
        "duplicate_status_rows": int(
            (
                cleaned["data_quality_status"]
                == "Duplicate"
            ).sum()
        ),
        "needs_review_rows": int(
            (
                cleaned["data_quality_status"]
                == "Needs Review"
            ).sum()
        ),
    }

    return report


def save_cleaned_data(dataframe, output_path):
    """Save cleaned data to a CSV file."""
    output_path = Path(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    dataframe.to_csv(
        output_path,
        index=False,
    )


def main():
    """Run the data-cleaning process from the command line."""
    input_path = Path(
        "data/raw/messy_business_data.csv"
    )

    output_path = Path(
        "data/processed/clean_business_data.csv"
    )

    original = load_data(input_path)

    cleaned = clean_data(original)

    report = generate_cleaning_report(
        original,
        cleaned,
    )

    save_cleaned_data(
        cleaned,
        output_path,
    )

    print("Data Cleaning Report")
    print("====================")
    print(
        f"Original rows: {report['original_rows']}"
    )
    print(
        f"Cleaned rows: {report['cleaned_rows']}"
    )
    print(
        f"Duplicate rows: {report['duplicate_rows']}"
    )
    print(
        f"Missing values: {report['missing_values']}"
    )
    print(
        f"Invalid dates: {report['invalid_dates']}"
    )
    print(
        f"Invalid sales: {report['invalid_sales']}"
    )
    print(
        f"Valid rows: {report['valid_rows']}"
    )
    print(
        f"Duplicate status rows: "
        f"{report['duplicate_status_rows']}"
    )
    print(
        f"Needs review rows: "
        f"{report['needs_review_rows']}"
    )
    print()
    print(
        f"Cleaned data saved to: {output_path}"
    )


if __name__ == "__main__":
    main()
