import pandas as pd
import pytest

from src.analyzer import (
    calculate_summary,
    filter_valid_records,
    generate_analysis_report,
    identify_lowest_transaction,
    identify_outliers,
    identify_top_transaction,
    load_clean_data,
    summarize_by_country,
    summarize_by_service,
)


@pytest.fixture
def sample_data():
    return pd.DataFrame(
        {
            "id": [1, 2, 3, 4],
            "date": [
                "2026-01-01",
                "2026-01-02",
                "2026-01-03",
                "2026-01-04",
            ],
            "customer": [
                "Acme Ltd",
                "BlueTech",
                "Green Solutions",
                "Delta Corp",
            ],
            "service": [
                "Consulting",
                "Data Analysis",
                "Training",
                "Consulting",
            ],
            "sales": [
                15000,
                10000,
                5000,
                9000,
            ],
            "expenses": [
                9000,
                6000,
                4000,
                5000,
            ],
            "country": [
                "Kenya",
                "Kenya",
                "Uganda",
                "Kenya",
            ],
            "data_quality_status": [
                "Valid",
                "Valid",
                "Needs Review",
                "Valid",
            ],
        }
    )


def test_load_clean_data(tmp_path, sample_data):
    file_path = tmp_path / "clean_data.csv"

    sample_data.to_csv(
        file_path,
        index=False,
    )

    loaded = load_clean_data(file_path)

    assert len(loaded) == 4
    assert "data_quality_status" in loaded.columns


def test_filter_valid_records(sample_data):
    valid_records = filter_valid_records(sample_data)

    assert len(valid_records) == 3
    assert all(
        valid_records["data_quality_status"]
        == "Valid"
    )


def test_calculate_summary(sample_data):
    valid_records = filter_valid_records(sample_data)

    summary = calculate_summary(valid_records)

    assert summary["transactions"] == 3
    assert summary["total_sales"] == 34000
    assert summary["total_expenses"] == 20000
    assert summary["total_profit"] == 14000
    assert summary["profit_margin"] == pytest.approx(
        41.1764705882
    )
    assert summary["average_sales"] == pytest.approx(
        11333.3333333
    )
    assert summary["average_expenses"] == pytest.approx(
        6666.6666667
    )
    assert summary["average_profit"] == pytest.approx(
        4666.6666667
    )


def test_summarize_by_service(sample_data):
    valid_records = filter_valid_records(sample_data)

    summary = summarize_by_service(valid_records)

    consulting = summary[
        summary["service"] == "Consulting"
    ].iloc[0]

    assert consulting["transactions"] == 2
    assert consulting["sales"] == 24000
    assert consulting["expenses"] == 14000
    assert consulting["profit"] == 10000
    assert consulting["profit_margin"] == pytest.approx(
        41.6666667
    )


def test_summarize_by_country(sample_data):
    valid_records = filter_valid_records(sample_data)

    summary = summarize_by_country(valid_records)

    kenya = summary[
        summary["country"] == "Kenya"
    ].iloc[0]

    assert kenya["transactions"] == 3
    assert kenya["sales"] == 34000
    assert kenya["expenses"] == 20000
    assert kenya["profit"] == 14000


def test_identify_top_transaction(sample_data):
    valid_records = filter_valid_records(sample_data)

    top = identify_top_transaction(valid_records)

    assert top["id"] == 1
    assert top["customer"] == "Acme Ltd"
    assert top["sales"] == 15000
    assert top["expenses"] == 9000


def test_identify_lowest_transaction(sample_data):
    valid_records = filter_valid_records(sample_data)

    lowest = identify_lowest_transaction(
        valid_records
    )
    assert lowest["id"] == 2
    assert lowest["customer"] == "BlueTech"
    assert lowest["sales"] == 10000
    assert lowest["expenses"] == 6000

def test_identify_outliers(sample_data):
    valid_records = filter_valid_records(sample_data)

    outliers = identify_outliers(valid_records)

    assert len(outliers) == 0


def test_generate_analysis_report(sample_data):
    report = generate_analysis_report(sample_data)

    assert report["summary"]["transactions"] == 3
    assert report["summary"]["total_sales"] == 34000
    assert report["summary"]["total_profit"] == 14000

    assert not report["service_summary"].empty
    assert not report["country_summary"].empty

    assert report["top_transaction"]["id"] == 1
    assert report["lowest_transaction"]["id"] == 2

    assert report["outliers"].empty
