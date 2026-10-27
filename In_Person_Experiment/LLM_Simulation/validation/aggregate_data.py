"""Prepare aggregated simulation data for the probit analyses."""


def make_clean_df(combined_df):
    clean_df = combined_df.copy()
    clean_df["rank_information"] = 1
    clean_df["location"] = "GPT-5.4"
    clean_df["subject_pool"] = "Custom Pool"
    clean_df["modality"] = "AI"

    treatment_map = {"Q00": 7, "Q05": 2, "Q50": 3, "Q'00": 7, "Q'05": 2, "Q'50": 3}
    clean_df["treatment_name"] = clean_df["treatment"]
    clean_df["treatment"] = clean_df["treatment"].map(treatment_map)

    usc_columns = [
        "subject_id", "cohort_id", "profit", "treatment",
        "subject_rank", "subject_type", "product_value", "wait_time",
        "subject_choice", "queuing_time", "shipping_time", "rank_information", "treatment_name", "location",
        "subject_pool", "modality"
    ]
    return clean_df[usc_columns]
