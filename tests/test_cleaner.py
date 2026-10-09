import pandas as pd
import pytest

from src.cleaner import (
    clean_data,
    generate_cleaning_report,
    identify_duplicates,
    identify_invalid_sales,
    identify_missing_values,
    validate_columns,
)


@pytest.fixture
def sample_data():
    return pd.DataFrame(
        {
            "id": [1, 2, 3],
            "date": [
                "2026-01-01",
                "2026/01/02",
                "2026-01-03",
            ],
            "customer": [
                "Acme Ltd",
                " BlueTech ",
                "Green Solutions",
            ],
            "service": [
                "Consulting",
                "data analysis",
                " training ",
            ],
            "sales": [
                "15,000",
                "10000",
                "5000",
            ],
            "expenses": [
                "9000",
                "6000",
                "4,000",
            ],
            "country": [
                "Kenya",
                "KENYA",
                " kenya ",
            ],
        }
    )


def test_validate_columns(sample_data):
    validate_columns(sample_data)


def test_validate_columns_raises_error():
    dataframe = pd.DataFrame(
        {
            "id": [1],
            "date": ["2026-01-01"],
        }
    )

    with pytest.raises(ValueError):
        validate_columns(dataframe)


def test_clean_data_standardizes_text(sample_data):
    cleaned = clean_data(sample_data)

    assert cleaned.loc[1, "customer"] == "BlueTech"
    assert cleaned.loc[2, "service"] == "Training"
    assert cleaned.loc[1, "country"] == "Kenya"


def test_clean_data_converts_numeric_values(sample_data):
    cleaned = clean_data(sample_data)

    assert cleaned.loc[0, "sales"] == 15000
    assert cleaned.loc[2, "expenses"] == 4000


def test_clean_data_converts_dates(sample_data):
    cleaned = clean_data(sample_data)

    assert pd.notna(cleaned.loc[0, "date"])
    assert pd.notna(cleaned.loc[1, "date"])
    assert pd.notna(cleaned.loc[2, "date"])


def test_identify_duplicates():
    dataframe = pd.DataFrame(
        {
            "id": [1, 2],
            "date": [
                "2026-01-01",
                "2026-01-01",
            ],
            "customer": [
                "Acme Ltd",
                "Acme Ltd",
            ],
            "service": [
                "Consulting",
                "Consulting",
            ],
            "sales": [10000, 10000],
            "expenses": [5000, 5000],
            "country": [
                "Kenya",
                "Kenya",
            ],
        }
    )

    duplicates = identify_duplicates(dataframe)

    assert len(duplicates) == 2


def test_identify_invalid_sales():
    dataframe = pd.DataFrame(
        {
            "sales": [
                10000,
                0,
                -500,
                pd.NA,
            ]
        }
    )

    invalid = identify_invalid_sales(dataframe)

    assert len(invalid) == 3


def test_identify_missing_values():
    dataframe = pd.DataFrame(
        {
            "customer": [
                "Acme Ltd",
                None,
                "BlueTech",
            ],
            "sales": [
                10000,
                5000,
                None,
            ],
        }
    )

    missing = identify_missing_values(dataframe)

    assert missing["customer"] == 1
    assert missing["sales"] == 1


def test_generate_cleaning_report(sample_data):
    cleaned = clean_data(sample_data)

    report = generate_cleaning_report(
        sample_data,
        cleaned,
    )

    assert report["original_rows"] == 3
    assert report["cleaned_rows"] == 3
    assert report["duplicate_rows"] == 0
    assert report["invalid_dates"] == 0
    assert report["invalid_sales"] == 0
